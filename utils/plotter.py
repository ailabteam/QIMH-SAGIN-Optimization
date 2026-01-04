import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_results():
    # Đọc dữ liệu
    df = pd.read_csv("results/experimental_results.csv")
    
    # Chuyển đổi dữ liệu sang dạng long-format để vẽ seaborn
    df_melted = df.melt(id_vars=['Scenario', 'Seed'], 
                        value_vars=['QGA_Cost', 'PSO_Cost'],
                        var_name='Algorithm', value_name='System Cost')
    
    # 1. Vẽ Box Plot so sánh Cost
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    palette = {"QGA_Cost": "#1f77b4", "PSO_Cost": "#d62728"}
    
    ax = sns.boxplot(data=df_melted, x='Scenario', y='System Cost', 
                     hue='Algorithm', palette=palette, width=0.6)
    
    plt.title('Performance Comparison: Hybrid QGA vs Classical PSO', fontsize=14)
    plt.xlabel('Simulation Scenarios', fontsize=12)
    plt.ylabel('Average System Cost (Latency + Energy)', fontsize=12)
    plt.legend(title='Algorithm')
    
    # Lưu file PDF chất lượng cao
    plt.savefig('results/scenario_boxplots.pdf', format='pdf', bbox_inches='tight')
    print("[OK] Figure saved: results/scenario_boxplots.pdf")

if __name__ == "__main__":
    plot_results()
