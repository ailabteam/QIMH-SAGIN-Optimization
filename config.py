# config.py
import torch

class BaseConfig:
    DEVICE = "cuda:0"
    SEEDS = [i for i in range(30)]  # 30 seeds khác nhau để lấy thống kê
    
    # Thông số vật lý chung
    BANDWIDTH = 20 * 1e6
    NOISE_POWER = -174 
    TASK_CYCLES_BIT = 500

# Định nghĩa các kịch bản cụ thể
SCENARIOS = {
    "urban_iot": {
        "num_ue": 200,
        "task_data_size": (0.5, 2.0), # MB
        "w_latency": 0.4,
        "w_energy": 0.6
    },
    "industrial_remote": {
        "num_ue": 30,
        "task_data_size": (5.0, 15.0), # MB
        "w_latency": 0.5,
        "w_energy": 0.5
    },
    "emergency_rescue": {
        "num_ue": 50,
        "task_data_size": (1.0, 5.0), # MB
        "w_latency": 0.9, # Rất ưu tiên độ trễ
        "w_energy": 0.1
    }
}
