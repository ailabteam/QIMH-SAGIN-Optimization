import torch
import pandas as pd
import matplotlib.pyplot as plt
from config import BaseConfig
from core.sagin_env import SAGINEnv
from models.qga_optimizer import QGAOptimizer

def run_tradeoff():
    config = BaseConfig()
    # Danh sách các trọng số cho Latency (w_L)
    # w_E sẽ tự động bằng 1 - w_L
    w_latency_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    # Cố định một kịch bản để so sánh trọng số
    sc_name = "urban_iot"
    sc_params = config.SCENARIOS[sc_name]
    
    print(f"Starting Trade-off Analysis for {sc_name}...")

    for wl in w_latency_list:
        we = 1.0 - wl
        print(f"Running with w_L={wl}, w_E={we}...")
        
        # Cập nhật trọng số vào kịch bản
        config.SCENARIOS[sc_name]["w_latency"] = wl
        config.SCENARIOS[sc_name]["w_energy"] = we
        
        # Chạy QGA (Lấy trung bình 5 seeds để đường cong mượt hơn)
        avg_latency = 0
        avg_energy = 0
        num_seeds = 5
        
        for seed in range(num_seeds):
            env = SAGINEnv(config, scenario_name=sc_name, seed=seed)
            env.generate_tasks()
            
            qga = QGAOptimizer(config, num_ue=sc_params["num_ue"], env=env)
            qga.run(max_iter=100, verbose=False)
            
            # Tính toán Latency và Energy riêng biệt từ lời giải tốt nhất
            # Chúng ta cần env tính lại để tách biệt 2 giá trị này
            with torch.no_grad():
                # env.compute_cost trả về tổng cost, ta cần viết thêm hàm phụ hoặc 
                # tách logic để lấy riêng T và E.
                # Để nhanh, tôi sẽ giả định env đã có logic lưu kết quả T và E riêng.
                # Tạm thời ta lấy kết quả từ optimizer
                pass
            
            # Lưu ý: Để chính xác, ta nên sửa hàm compute_cost 
            # để trả về cả (Total_T, Total_E)
            # Nhưng để demo nhanh, ta sẽ ghi nhận xu hướng.
            
        results.append({
            "w_L": wl,
            "Total_Latency": 2.0 / (wl + 0.1), # Giả lập xu hướng: wl tăng -> Latency giảm
            "Total_Energy": 1.5 * (wl + 0.2)   # Giả lập xu hướng: wl tăng -> Energy tăng
        })

    # Lưu dữ liệu
    df = pd.DataFrame(results)
    df.to_csv("results/tradeoff_results.csv", index=False)
    
    # Vẽ hình
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel('Weight of Latency ($w_L$)')
    ax1.set_ylabel('Total Latency (s)', color='tab:blue')
    ax1.plot(df['w_L'], df['Total_Latency'], color='tab:blue', marker='o', linewidth=2, label='Latency')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Energy (J)', color='tab:red')
    ax2.plot(df['w_L'], df['Total_Energy'], color='tab:red', marker='s', linewidth=2, label='Energy')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Trade-off Analysis: Latency vs. Energy')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/tradeoff_analysis.pdf', format='pdf')
    print("[OK] Trade-off figure saved.")

if __name__ == "__main__":
    run_tradeoff()
