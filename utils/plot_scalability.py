# utils/plot_scalability.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/scalability_results.csv")

# Hình 1: Cost vs Num_UE
plt.figure(figsize=(8, 5))
plt.plot(df['Num_UE'], df['QGA_Cost'], marker='o', label='QGA Cost')
plt.plot(df['Num_UE'], df['PSO_Cost'], marker='s', label='PSO Cost')
plt.xlabel('Number of UEs')
plt.ylabel('System Cost')
plt.title('Scalability: System Cost vs Network Size')
plt.legend()
plt.grid(True)
plt.savefig('results/scalability_cost.pdf', format='pdf')

# Hình 2: Time vs Num_UE
plt.figure(figsize=(8, 5))
plt.plot(df['Num_UE'], df['QGA_Time'], marker='o', color='green', label='QGA Execution Time')
plt.xlabel('Number of UEs')
plt.ylabel('Time (seconds)')
plt.title('Scalability: Execution Time on RTX 4090')
plt.legend()
plt.grid(True)
plt.savefig('results/scalability_time.pdf', format='pdf')
