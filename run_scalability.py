import torch
import time
import pandas as pd
from config import BaseConfig
from core.sagin_env import SAGINEnv
from models.qga_optimizer import QGAOptimizer
from models.pso_optimizer import PSOOptimizer

def run_scalability():
    config = BaseConfig()
    ue_scales = [50, 100, 200, 500, 1000] # Tăng quy mô mạng
    results = []
    
    for num_ue in ue_scales:
        print(f"Testing scalability with {num_ue} UEs...")
        
        # Cập nhật config tạm thời
        config.SCENARIOS["urban_iot"]["num_ue"] = num_ue
        env = SAGINEnv(config, scenario_name="urban_iot", seed=42)
        env.generate_tasks()
        
        # Đo thời gian QGA
        start_qga = time.time()
        qga = QGAOptimizer(config, num_ue=num_ue, env=env)
        qga.run(max_iter=100, verbose=False)
        time_qga = time.time() - start_qga
        
        # Đo thời gian PSO
        start_pso = time.time()
        pso = PSOOptimizer(config, num_ue=num_ue, env=env)
        pso.run(max_iter=100, verbose=False)
        time_pso = time.time() - start_pso
        
        results.append({
            "Num_UE": num_ue,
            "QGA_Time": time_qga,
            "PSO_Time": time_pso,
            "QGA_Cost": qga.best_cost,
            "PSO_Cost": pso.gbest_cost
        })

    df = pd.DataFrame(results)
    df.to_csv("results/scalability_results.csv", index=False)
    print("\nScalability Study Results:")
    print(df)

if __name__ == "__main__":
    run_scalability()
