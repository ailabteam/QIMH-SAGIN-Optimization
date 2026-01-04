import torch
import numpy as np

class QGAOptimizer:
    def __init__(self, config, num_ue, env):
        self.config = config
        self.num_ue = num_ue
        self.env = env
        self.device = config.DEVICE
        self.pop_size = config.POPULATION_SIZE
        
        # Khởi tạo pha theta ngẫu nhiên để tăng tính khám phá (Exploration)
        self.theta = torch.rand((self.pop_size, self.num_ue), device=self.device) * (np.pi/2)
        
        self.best_cost = float('inf')
        self.best_sol = None

    def observe(self):
        # Đo đạc trạng thái lượng tử
        probs = torch.sin(self.theta)**2
        rand_vals = torch.rand_like(probs)
        decisions = torch.zeros_like(probs, dtype=torch.long)
        decisions[rand_vals > 0.33] = 1
        decisions[rand_vals > 0.66] = 2
        return decisions

    def evolve(self, current_decisions, costs, best_sol, iteration, max_iter):
        """
        Cổng quay lượng tử thích nghi (Adaptive Quantum Rotation Gate)
        """
        # Giảm dần bước quay theo thời gian để hội tụ sâu
        base_step = 0.05 * np.pi
        step_size = base_step * (1 - iteration / max_iter)
        
        # Logic cổng quay: 
        # Nếu chi phí của cá thể > chi phí tốt nhất, quay theta về hướng best_sol
        best_sol_repeated = best_sol.repeat(self.pop_size, 1)
        
        # Hướng quay (Direction)
        direction = torch.zeros_like(self.theta)
        direction[current_decisions < best_sol_repeated] = 1.0
        direction[current_decisions > best_sol_repeated] = -1.0
        
        # Cập nhật pha với bước quay thích nghi
        self.theta += direction * step_size
        self.theta = torch.clamp(self.theta, 0.01, np.pi/2 - 0.01)

    def run(self, max_iter):
        history = []
        for i in range(max_iter):
            decisions = self.observe()
            costs = self.env.compute_cost(decisions)
            
            min_val, min_idx = torch.min(costs, dim=0)
            
            if min_val < self.best_cost:
                self.best_cost = min_val.item()
                self.best_sol = decisions[min_idx].clone()
            
            self.evolve(decisions, costs, self.best_sol, i, max_iter)
            history.append(self.best_cost)
            
            if i % 20 == 0:
                print(f"Iteration {i}: Best Cost = {self.best_cost:.4f}")
        
        return history
