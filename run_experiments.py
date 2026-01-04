import torch
import pandas as pd
import numpy as np
from config import BaseConfig, SCENARIOS
from core.sagin_env import SAGINEnv
from models.qga_optimizer import QGAOptimizer
from models.pso_optimizer import PSOOptimizer
from tqdm import tqdm

def run_suite():
    config = BaseConfig()
    results = []
    
    # Chúng ta sẽ chạy trên 10 seeds để lấy thống kê nhanh (có thể tăng lên 30 sau)
    seeds = range(10)
    
    for sc_name, sc_params in SCENARIOS.items():
        print(f"\n>>> Evaluating Scenario: {sc_name} <<<")
        
        for seed in tqdm(seeds):
            # Khởi tạo môi trường với seed cụ thể
            env = SAGINEnv(config, scenario_name=sc_name, seed=seed)
            env.generate_tasks()
            
            # 1. Chạy QGA
            qga = QGAOptimizer(config, num_ue=sc_params["num_ue"], env=env)
            qga_hist = qga.run(max_iter=100)
            
            # 2. Chạy PSO
            pso = PSOOptimizer(config, num_ue=sc_params["num_ue"], env=env)
            pso_hist = pso.run(max_iter=100)
            
            # Lưu kết quả cuối cùng của mỗi seed
            results.append({
                "Scenario": sc_name,
                "Seed": seed,
                "QGA_Final_Cost": qga.best_cost,
                "PSO_Final_Cost": pso.gbest_cost,
                "Gain_%": (pso.gbest_cost - qga.best_cost) / pso.gbest_cost * 100
            })

    # Chuyển sang DataFrame và lưu
    df = pd.DataFrame(results)
    df.to_csv("results/experimental_results.csv", index=False)
    
    # In báo cáo tóm tắt
    summary = df.groupby("Scenario")[["QGA_Final_Cost", "PSO_Final_Cost", "Gain_%"]].mean()
    print("\n--- EXPERIMENT SUMMARY (Averaged over seeds) ---")
    print(summary)

if __name__ == "__main__":
    run_suite()
