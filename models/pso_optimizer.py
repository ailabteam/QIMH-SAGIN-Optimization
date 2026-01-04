import torch

class PSOOptimizer:
    def __init__(self, config, num_ue, env):
        self.config = config
        self.num_ue = num_ue
        self.env = env
        self.device = config.DEVICE
        self.pop_size = config.POPULATION_SIZE
        self.pos = torch.rand((self.pop_size, self.num_ue), device=self.device) * 2.99
        self.vel = torch.randn((self.pop_size, self.num_ue), device=self.device) * 0.1
        self.pbest_pos = self.pos.clone()
        self.pbest_cost = torch.full((self.pop_size,), float('inf'), device=self.device)
        self.gbest_pos = None
        self.gbest_cost = float('inf')

    def run(self, max_iter, verbose=True): # Đã thêm verbose
        history = []
        w, c1, c2 = 0.7, 1.5, 1.5
        for i in range(max_iter):
            decisions = torch.clamp(self.pos.long(), 0, 2)
            costs = self.env.compute_cost(decisions)
            better_mask = costs < self.pbest_cost
            self.pbest_pos[better_mask] = self.pos[better_mask].clone()
            self.pbest_cost[better_mask] = costs[better_mask]
            min_val, min_idx = torch.min(costs, dim=0)
            if min_val < self.gbest_cost:
                self.gbest_cost = min_val.item()
                self.gbest_pos = self.pos[min_idx].clone()
            r1, r2 = torch.rand(2, self.pop_size, self.num_ue, device=self.device)
            self.vel = w * self.vel + c1 * r1 * (self.pbest_pos - self.pos) + c2 * r2 * (self.gbest_pos - self.pos)
            self.pos += self.vel
            self.pos = torch.clamp(self.pos, 0, 2.99)
            history.append(self.gbest_cost)
            if verbose and i % 20 == 0:
                print(f"PSO Iteration {i}: Best Cost = {self.gbest_cost:.4f}")
        return history
