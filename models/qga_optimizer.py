import torch
import numpy as np

class QGAOptimizer:
    def __init__(self, config, num_ue, env):
        self.config = config
        self.num_ue = num_ue
        self.env = env
        self.device = config.DEVICE
        self.pop_size = config.POPULATION_SIZE
        self.theta = torch.full((self.pop_size, self.num_ue, 2), np.pi/4, device=self.device)
        self.best_cost = float('inf')
        self.best_sol_bits = None

    def observe(self):
        probs = torch.sin(self.theta)**2
        rand_vals = torch.rand_like(probs)
        bits = (rand_vals < probs).long()
        decisions = bits[:, :, 0] * 2 + bits[:, :, 1]
        decisions = torch.clamp(decisions, 0, 2)
        return bits, decisions

    def evolve(self, current_bits, costs, iteration, max_iter):
        min_val, min_idx = torch.min(costs, dim=0)
        if min_val < self.best_cost:
            self.best_cost = min_val.item()
            self.best_sol_bits = current_bits[min_idx].clone()

        base_step = 0.02 * np.pi
        step_size = base_step * (1 - iteration / max_iter)
        best_bits = self.best_sol_bits.unsqueeze(0)
        
        direction = torch.zeros_like(self.theta)
        direction[(current_bits == 0) & (best_bits == 1)] = 1.0
        direction[(current_bits == 1) & (best_bits == 0)] = -1.0
        
        mutation = (torch.rand_like(self.theta) < 0.01).float() * (torch.rand_like(self.theta) - 0.5)
        self.theta += (direction * step_size) + (mutation * 0.01)
        self.theta = torch.clamp(self.theta, 0.01, np.pi/2 - 0.01)

    def run(self, max_iter, verbose=True): # Đã thêm verbose
        history = []
        for i in range(max_iter):
            bits, decisions = self.observe()
            costs = self.env.compute_cost(decisions)
            self.evolve(bits, costs, i, max_iter)
            history.append(self.best_cost)
            if verbose and i % 20 == 0:
                print(f"QGA Iteration {i}: Best Cost = {self.best_cost:.4f}")
        return history
