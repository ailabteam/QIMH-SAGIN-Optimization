# main.py
from config import BaseConfig, SCENARIOS
from core.sagin_env import SAGINEnv
from models.qga_optimizer import QGAOptimizer
import matplotlib.pyplot as plt

class RunConfig(BaseConfig):
    SCENARIOS = SCENARIOS
    POPULATION_SIZE = 100
    MAX_ITER = 100

# Khởi tạo kịch bản
config = RunConfig()
env = SAGINEnv(config, scenario_name="urban_iot")
env.generate_tasks()

# Khởi tạo thuật toán
optimizer = QGAOptimizer(config, num_ue=config.SCENARIOS["urban_iot"]["num_ue"], env=env)

# Chạy tối ưu
print("Starting Quantum-Inspired Optimization...")
history = optimizer.run(max_iter=config.MAX_ITER)

# Vẽ biểu đồ hội tụ (Convergence Plot)
plt.figure(figsize=(8, 5))
plt.plot(history, label='QGA')
plt.xlabel('Generation')
plt.ylabel('System Cost')
plt.title('Convergence Analysis on RTX 4090')
plt.grid(True)
plt.legend()
plt.savefig('results/convergence_test.pdf', format='pdf', bbox_inches='tight')
print(f"Final Best Cost: {optimizer.best_cost:.4f}")
