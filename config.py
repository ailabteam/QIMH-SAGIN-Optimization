# config.py
import torch

class BaseConfig:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    SEEDS = [i for i in range(30)]

    # --- Thông số Vật lý & Kênh truyền ---
    C_LIGHT = 3e8
    BOLTZMANN = 1.38e-23
    TEMPERATURE = 290
    BANDWIDTH = 20 * 1e6      # 20 MHz

    FC_UAV = 2e9              # 2 GHz (S-band)
    FC_LEO = 20e9             # 20 GHz (Ka-band)

    P_TRANSMIT_UE = 0.5       # 0.5 Watts (27 dBm)

    # Độ lợi Anten (dBi) - Rất quan trọng cho link vệ tinh
    G_UE = 0                  # UE dùng anten đẳng hướng
    G_UAV = 5                 # UAV có anten định hướng nhẹ
    G_LEO = 40                # Vệ tinh LEO có anten độ lợi cực cao (Beamforming)

    # --- Quy mô mạng mặc định ---
    NUM_UAV = 3
    NUM_LEO = 1
    AREA_SIZE = 10000         # 10km
    UAV_HEIGHT = 200
    LEO_HEIGHT = 600000       # 600km


    # Khả năng tính toán (CPU Cycles/s)
    F_UE = 0.6 * 1e9          # 0.6 GHz
    F_UAV = 4.0 * 1e9         # 4.0 GHz
    F_LEO = 15.0 * 1e9        # 15.0 GHz (Vệ tinh có bộ xử lý mạnh hơn)
    
    # Tham số năng lượng
    KAPPA = 1e-28             # Hệ số kiến trúc chip (để tính Power = kappa * f^3)
    
    # Tham số Task (Số chu kỳ CPU cần để xử lý 1 bit)
    CYCLES_PER_BIT = 1000     # Ví dụ: 1000 cycles/bit



# Định nghĩa các kịch bản
SCENARIOS = {
    "urban_iot": {
        "num_ue": 200,
        "task_data_size": (0.5, 2.0),
        "w_latency": 0.4,
        "w_energy": 0.6
    },
    "industrial_remote": {
        "num_ue": 30,
        "task_data_size": (5.0, 15.0),
        "w_latency": 0.5,
        "w_energy": 0.5
    },
    "emergency_rescue": {
        "num_ue": 50,
        "task_data_size": (1.0, 5.0),
        "w_latency": 0.9,
        "w_energy": 0.1
    }
}
