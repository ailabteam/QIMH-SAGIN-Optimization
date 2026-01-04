# main.py cập nhật
from config import BaseConfig, SCENARIOS
from core.sagin_env import SAGINEnv
from models.qga_optimizer import QGAOptimizer
from models.pso_optimizer import PSOOptimizer
import matplotlib.pyplot as plt
import torch

class RunConfig(BaseConfig):
    SCENARIOS = SCENARIOS
    POPULATION_SIZE = 100
    MAX_ITER = 100

config = RunConfig()
env = SAGINEnv(config, scenario_name="urban_iot")
env.generate_tasks()

# 1. Chạy QGA
print("--- Running QGA ---")
qga = QGAOptimizer(config, num_ue=config.SCENARIOS["urban_iot"]["num_ue"], env=env)
qga_history = qga.run(max_iter=config.MAX_ITER)

# 2. Chạy PSO
print("\n--- Running PSO ---")
pso = PSOOptimizer(config, num_ue=config.SCENARIOS["urban_iot"]["num_ue"], env=env)
pso_history = pso.run(max_iter=config.MAX_ITER)

# 3. Vẽ biểu đồ so sánh
plt.figure(figsize=(10, 6))
plt.plot(qga_history, label='Proposed Hybrid QGA', color='blue', linewidth=2)
plt.plot(pso_history, label='Classical PSO', color='red', linestyle='--', linewidth=2)

plt.xlabel('Generation')
plt.ylabel('Average System Cost')
plt.title('Convergence Comparison: QGA vs PSO in 6G SAGIN')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.savefig('results/qga_vs_pso.pdf', format='pdf', bbox_inches='tight')

print(f"\nFinal Cost - QGA: {qga.best_cost:.4f}")
print(f"Final Cost - PSO: {pso.gbest_cost:.4f}")
