# test_env.py
from config import BaseConfig, SCENARIOS
from core.sagin_env import SAGINEnv

class TestConfig(BaseConfig):
    SCENARIOS = SCENARIOS
    NUM_UAV = 3
    AREA_SIZE = 10000
    UAV_HEIGHT = 200
    LEO_HEIGHT = 600000
    NUM_LEO = 1

config = TestConfig()
env = SAGINEnv(config, scenario_name="urban_iot")
d_ue_uav, d_ue_leo, d_uav_leo = env.get_distances()

print(f"UE-UAV Distance Shape: {d_ue_uav.shape}") # Mong đợi: [200, 3]
print(f"UE-LEO Distance Shape: {d_ue_leo.shape}") # Mong đợi: [200, 1]
print(f"Ví dụ khoảng cách UE_0 tới LEO: {d_ue_leo[0].item():.2f} mét")
