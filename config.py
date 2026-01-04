# config.py
import torch

class BaseConfig:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # --- Thông số Vật lý ---
    C_LIGHT = 3e8
    BOLTZMANN = 1.38e-23
    TEMPERATURE = 290
    BANDWIDTH = 20 * 1e6
    FC_UAV = 2e9
    FC_LEO = 20e9
    P_TRANSMIT_UE = 0.5
    G_UE = 0
    G_UAV = 5
    G_LEO = 40
    
    # --- Thông số Tính toán ---
    F_UE = 0.6 * 1e9
    F_UAV = 4.0 * 1e9
    F_LEO = 15.0 * 1e9
    KAPPA = 1e-28
    CYCLES_PER_BIT = 1000
    
    # --- Quy mô mạng ---
    NUM_UAV = 3
    NUM_LEO = 1
    AREA_SIZE = 10000
    UAV_HEIGHT = 200
    LEO_HEIGHT = 600000

    # --- Tối ưu hóa ---
    POPULATION_SIZE = 100

    # Đưa SCENARIOS vào đây
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
