import torch

class PSOOptimizer:
    def __init__(self, config, num_ue, env):
        self.config = config
        self.num_ue = num_ue
        self.env = env
        self.device = config.DEVICE
        self.pop_size = config.POPULATION_SIZE
        
        # Vị trí [0, 2.99] để sau này làm tròn thành {0, 1, 2}
        self.pos = torch.rand((self.pop_size, self.num_ue), device=self.device) * 2.99
        self.vel = torch.randn((self.pop_size, self.num_ue), device=self.device) * 0.1
        
        self.pbest_pos = self.pos.clone()
        self.pbest_cost = torch.full((self.pop_size,), float('inf'), device=self.device)
        
        self.gbest_pos = None
        self.gbest_cost = float('inf')

    def run(self, max_iter):
        history = []
        w = 0.7  # Quán tính
        c1 = 1.5 # Học hỏi cá nhân
        c2 = 1.5 # Học hỏi cộng đồng
        
        for i in range(max_iter):
            # 1. Chuyển vị trí liên tục sang quyết định rời rạc
            decisions = torch.clamp(self.pos.long(), 0, 2)
            
            # 2. Tính chi phí
            costs = self.env.compute_cost(decisions)
            
            # 3. Cập nhật Personal Best
            better_mask = costs < self.pbest_cost
            self.pbest_pos[better_mask] = self.pos[better_mask].clone()
            self.pbest_cost[better_mask] = costs[better_mask]
            
            # 4. Cập nhật Global Best
            min_val, min_idx = torch.min(costs, dim=0)
            if min_val < self.gbest_cost:
                self.gbest_cost = min_val.item()
                self.gbest_pos = self.pos[min_idx].clone()
            
            # 5. Cập nhật Vận tốc và Vị trí
            r1, r2 = torch.rand(2, self.pop_size, self.num_ue, device=self.device)
            self.vel = w * self.vel + \
                       c1 * r1 * (self.pbest_pos - self.pos) + \
                       c2 * r2 * (self.gbest_pos - self.pos)
            self.pos += self.vel
            self.pos = torch.clamp(self.pos, 0, 2.99)
            
            history.append(self.gbest_cost)
            if i % 20 == 0:
                print(f"PSO Iteration {i}: Best Cost = {self.gbest_cost:.4f}")
                
        return history
