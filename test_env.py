# test_env.py cập nhật
from config import BaseConfig, SCENARIOS
from core.sagin_env import SAGINEnv
import torch

class TestConfig(BaseConfig):
    SCENARIOS = SCENARIOS

config = TestConfig()
env = SAGINEnv(config, scenario_name="urban_iot")
env.generate_tasks()

pop_size = 5
num_ue = config.SCENARIOS["urban_iot"]["num_ue"]

# Kịch bản 1: Tất cả UE xử lý tại chỗ (Decisions toàn là 0)
dec_local = torch.zeros((pop_size, num_ue), device=config.DEVICE, dtype=torch.long)
cost_local = env.compute_cost(dec_local)

# Kịch bản 2: Quyết định ngẫu nhiên (0, 1, hoặc 2)
dec_rand = torch.randint(0, 3, (pop_size, num_ue), device=config.DEVICE)
cost_rand = env.compute_cost(dec_rand)

print(f"Cost (All Local): {cost_local[0].item():.4f}")
print(f"Cost (Random): {cost_rand[0].item():.4f}")

