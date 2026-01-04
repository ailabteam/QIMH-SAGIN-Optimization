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
