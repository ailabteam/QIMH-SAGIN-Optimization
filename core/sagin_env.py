import torch

class SAGINEnv:
    def __init__(self, config, scenario_name="urban_iot", seed=42):
        self.config = config
        self.scenario = config.SCENARIOS[scenario_name]
        self.device = config.DEVICE

        # Thiết lập seed để đảm bảo tính tái lập (Reproducibility)
        torch.manual_seed(seed)

        # 1. Khởi tạo vị trí các thực thể
        self._initialize_positions()

    def _initialize_positions(self):
        """
        Khởi tạo tọa độ (x, y, z) cho UE, UAV và LEO
        Sử dụng Tensor để đẩy thẳng lên GPU
        """
        num_ue = self.scenario["num_ue"]
        num_uav = self.config.NUM_UAV

        # UE: Phân bố ngẫu nhiên trên mặt đất (z=0) trong vùng AREA_SIZE x AREA_SIZE
        # Tọa độ UE: [num_ue, 3] -> (x, y, 0)
        self.ue_pos = torch.zeros((num_ue, 3), device=self.device)
        self.ue_pos[:, :2] = torch.rand((num_ue, 2), device=self.device) * self.config.AREA_SIZE

        # UAV: Giả định ban đầu đứng ở độ cao UAV_HEIGHT, phân bố đều hoặc ngẫu nhiên
        # Tọa độ UAV: [num_uav, 3] -> (x, y, h_uav)
        self.uav_pos = torch.zeros((num_uav, 3), device=self.device)
        self.uav_pos[:, :2] = torch.rand((num_uav, 2), device=self.device) * self.config.AREA_SIZE
        self.uav_pos[:, 2] = self.config.UAV_HEIGHT

        # LEO: Giả định vệ tinh ở ngay trung tâm vùng phủ sóng ở độ cao cực lớn
        # Tọa độ LEO: [1, 3] -> (x_center, y_center, h_leo)
        self.leo_pos = torch.tensor([
            [self.config.AREA_SIZE/2, self.config.AREA_SIZE/2, self.config.LEO_HEIGHT]
        ], device=self.device)

    def get_distances(self):
        """
        Tính toán ma trận khoảng cách Euclidean
        Sử dụng torch.cdist để tính toán vectorized cực nhanh
        """
        # Khoảng cách UE tới các UAV: Kết quả là ma trận [num_ue, num_uav]
        dist_ue_uav = torch.cdist(self.ue_pos, self.uav_pos)

        # Khoảng cách UE tới LEO: Kết quả là ma trận [num_ue, 1]
        dist_ue_leo = torch.cdist(self.ue_pos, self.leo_pos)

        # Khoảng cách UAV tới LEO: Kết quả là ma trận [num_uav, 1]
        dist_uav_leo = torch.cdist(self.uav_pos, self.leo_pos)

        return dist_ue_uav, dist_ue_leo, dist_uav_leo

    def get_channel_rates(self):
        dist_ue_uav, dist_ue_leo, _ = self.get_distances()

        # 1. Path Loss (FSPL)
        def calc_pl(dist, fc):
            return 20 * torch.log10(dist) + 20 * torch.log10(torch.tensor(fc)) + \
                   20 * torch.log10(torch.tensor(4 * 3.14159 / self.config.C_LIGHT))

        pl_ue_uav = calc_pl(dist_ue_uav, self.config.FC_UAV)
        pl_ue_leo = calc_pl(dist_ue_leo, self.config.FC_LEO)

        # 2. Received Power (dBm) = P_tx + G_ue + G_rx - PL
        p_tx_dbm = 10 * torch.log10(torch.tensor(self.config.P_TRANSMIT_UE * 1000))

        p_rx_uav_dbm = p_tx_dbm + self.config.G_UE + self.config.G_UAV - pl_ue_uav
        p_rx_leo_dbm = p_tx_dbm + self.config.G_UE + self.config.G_LEO - pl_ue_leo

        # Watts
        p_rx_uav = 10**(p_rx_uav_dbm / 10) / 1000
        p_rx_leo = 10**(p_rx_leo_dbm / 10) / 1000

        # 3. Noise & Rate
        noise = self.config.BOLTZMANN * self.config.TEMPERATURE * self.config.BANDWIDTH

        rate_uav = self.config.BANDWIDTH * torch.log2(1 + p_rx_uav / noise)
        rate_leo = self.config.BANDWIDTH * torch.log2(1 + p_rx_leo / noise)

        return rate_uav, rate_leo
    
    def generate_tasks(self):
        """
        Tạo task cho từng UE dựa trên kịch bản
        Mỗi task có: Data Size (bits) và Total Cycles cần thiết
        """
        min_size, max_size = self.scenario["task_data_size"]
        num_ue = self.scenario["num_ue"]
        
        # Kích thước dữ liệu ngẫu nhiên (chuyển từ MB sang bits)
        # [num_ue]
        task_data_mb = min_size + (max_size - min_size) * torch.rand(num_ue, device=self.device)
        self.task_data_bits = task_data_mb * 1024 * 1024 * 8
        
        # Tổng số chu kỳ CPU cần để xử lý các bits này
        self.task_cycles = self.task_data_bits * self.config.CYCLES_PER_BIT
        
        return self.task_data_bits, self.task_cycles
    
    def compute_cost(self, decisions):
        """
        Tính toán chi phí cho một quần thể các lời giải.
        decisions: Tensor [population_size, num_ue] chứa các giá trị {0, 1, 2}
        """
        pop_size, num_ue = decisions.shape
        rate_uav, rate_leo = self.get_channel_rates() # [num_ue, num_uav] và [num_ue, 1]
        dist_ue_uav, dist_ue_leo, _ = self.get_distances()

        # Giả sử mỗi UE kết nối với UAV gần nhất nếu chọn offload lên UAV
        best_rate_uav, _ = torch.max(rate_uav, dim=1) # [num_ue]
        best_dist_uav, _ = torch.min(dist_ue_uav, dim=1)

        # 1. Tính Độ trễ (Latency) cho 3 trường hợp [pop_size, num_ue]
        # T_local
        t_local = self.task_cycles / self.config.F_UE
        
        # T_uav = truyền dẫn + xử lý
        t_uav = (self.task_data_bits / best_rate_uav) + (self.task_cycles / self.config.F_UAV)
        
        # T_leo = truyền dẫn + xử lý + 2*trễ lan truyền
        t_prop_leo = dist_ue_leo.squeeze() / self.config.C_LIGHT
        t_leo = (self.task_data_bits / rate_leo.squeeze()) + (self.task_cycles / self.config.F_LEO) + 2 * t_prop_leo

        # 2. Tính Năng lượng (Energy) [pop_size, num_ue]
        # E_local = kappa * f^2 * cycles
        e_local = self.config.KAPPA * (self.config.F_UE**2) * self.task_cycles
        
        # E_uav = P_tx * T_upload
        e_uav = self.config.P_TRANSMIT_UE * (self.task_data_bits / best_rate_uav)
        
        # E_leo = P_tx * T_upload
        e_leo = self.config.P_TRANSMIT_UE * (self.task_data_bits / rate_leo.squeeze())

        # 3. Tổng hợp dựa trên decisions
        # Tạo mask cho từng loại quyết định
        mask_local = (decisions == 0).float()
        mask_uav = (decisions == 1).float()
        mask_leo = (decisions == 2).float()

        total_latency = (mask_local * t_local + mask_uav * t_uav + mask_leo * t_leo)
        total_energy = (mask_local * e_local + mask_uav * e_uav + mask_leo * e_leo)

        # 4. Tính Weighted Cost cho từng UE
        w_l = self.scenario["w_latency"]
        w_e = self.scenario["w_energy"]
        
        # Cost trung bình của toàn mạng cho mỗi cá thể trong quần thể
        # Kết quả: [pop_size]
        individual_costs = torch.mean(w_l * total_latency + w_e * total_energy, dim=1)
        
        return individual_costs
