import torch
import pandas as pd
import numpy as np
import os
from config import BaseConfig
from core.sagin_env import SAGINEnv
from models.qga_optimizer import QGAOptimizer
from models.pso_optimizer import PSOOptimizer
from tqdm import tqdm

def run_suite():
    config = BaseConfig()
    if not os.path.exists("results"):
        os.makedirs("results")
        
    results = []
    seeds = range(10) # Chạy 10 seeds để lấy thống kê
    
    # Truy cập SCENARIOS thông qua config
    for sc_name in config.SCENARIOS.keys():
        sc_params = config.SCENARIOS[sc_name]
        print(f"\n>>> Evaluating Scenario: {sc_name} <<<")
        
        for seed in tqdm(seeds, desc=f"Seeds for {sc_name}"):
            env = SAGINEnv(config, scenario_name=sc_name, seed=seed)
            env.generate_tasks()
            
            # Chạy QGA (verbose=False)
            qga = QGAOptimizer(config, num_ue=sc_params["num_ue"], env=env)
            _ = qga.run(max_iter=100, verbose=False)
            
            # Chạy PSO (verbose=False)
            pso = PSOOptimizer(config, num_ue=sc_params["num_ue"], env=env)
            _ = pso.run(max_iter=100, verbose=False)
            
            results.append({
                "Scenario": sc_name,
                "Seed": seed,
                "QGA_Cost": qga.best_cost,
                "PSO_Cost": pso.gbest_cost,
                "Gain_Percentage": (pso.gbest_cost - qga.best_cost) / pso.gbest_cost * 100
            })

    df = pd.DataFrame(results)
    df.to_csv("results/experimental_results.csv", index=False)
    
    # Tính toán bảng tóm tắt
    summary = df.groupby("Scenario")[["QGA_Cost", "PSO_Cost", "Gain_Percentage"]].agg(['mean', 'std'])
    print("\n" + "="*60)
    print("FINAL EXPERIMENT SUMMARY")
    print("="*60)
    print(summary)
    print("="*60)
    print("Results saved to results/experimental_results.csv")

if __name__ == "__main__":
    run_suite()
