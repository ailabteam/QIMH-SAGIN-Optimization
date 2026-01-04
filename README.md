# Joint Task Offloading and Resource Orchestration in 6G SAGINs: A Hybrid Quantum-Inspired Meta-Heuristic Approach

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.9+](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of our research paper on **6G Space-Air-Ground Integrated Networks (SAGIN)** optimization. We propose a **Hybrid Quantum-Inspired Meta-Heuristic (QIMH)** algorithm to solve the joint task offloading and resource allocation problem.

## 🌟 Key Features
- **3-Layer NTN Architecture:** Detailed modeling of Space (LEO), Air (UAV), and Ground (UE) layers.
- **Quantum-Inspired Logic:** Implements 2-Qubit encoding and adaptive Quantum Rotation Gates for superior exploration.
- **GPU Acceleration:** Leveraging **NVIDIA RTX 4090** via PyTorch vectorization to achieve near-constant execution time for up to 1000 users.
- **Multi-Scenario Support:** Urban IoT, Industrial Remote, and Emergency Rescue scenarios.

## 🏗 Project Structure
```text
QIMH-SAGIN-Optimization/
├── core/               # System models (Channel, Latency, Energy)
│   └── sagin_env.py    # Main SAGIN Environment logic
├── models/             # Optimization algorithms
│   ├── qga_optimizer.py # Proposed Hybrid QGA
│   └── pso_optimizer.py # Classical PSO Benchmark
├── results/            # Experimental data (.csv) and Publication figures (.pdf)
├── utils/              # Plotting and helper scripts
├── config.py           # System parameters and scenario definitions
├── main.py             # Quick execution and comparison script
├── run_experiments.py  # Statistical analysis (Multi-seed)
└── run_scalability.py  # GPU performance and scalability test
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Conda
- NVIDIA GPU (RTX 30-series or 40-series recommended for best performance)

### Installation
```bash
# Clone the repository
git clone https://github.com/ailabteam/QIMH-SAGIN-Optimization.git
cd QIMH-SAGIN-Optimization

# Create and activate the environment
conda create -n qimh_env python=3.11 -y
conda activate qimh_env

# Install dependencies
pip install torch torchvision torchaudio numpy scipy matplotlib pandas tqdm seaborn
```

## 📊 Running Experiments

1. **Quick Comparison:** Run a single trial of QGA vs PSO.
   ```bash
   python main.py
   ```

2. **Full Statistical Study:** Run multi-seed experiments across 3 scenarios (Urban, Industrial, Emergency).
   ```bash
   python run_experiments.py
   ```

3. **Scalability Test:** Evaluate execution time and cost for networks up to 1000 UEs.
   ```bash
   python run_scalability.py
   ```

4. **Trade-off Analysis:** Analyze the impact of weighting factors ($w_L$ vs $w_E$).
   ```bash
   python run_tradeoff.py
   ```

## 📈 Results Preview
Our experiments on an **NVIDIA RTX 4090** demonstrate:
- **Efficiency:** Up to **11% cost reduction** in latency-critical scenarios compared to PSO.
- **Speed:** Scalability to 1000 users with a constant execution time of **~0.25 seconds**.
- **Robustness:** Lower standard deviation across multiple random seeds.

All figures are saved in the `results/` directory in high-quality PDF format.

## 📝 Citation
If you find this work useful in your research, please cite:
```bibtex
@inproceedings{haodp2026sagin,
  title={Joint Task Offloading and Resource Orchestration in 6G SAGINs: A Hybrid Quantum-Inspired Meta-Heuristic Approach},
  author={Phuc Hao Do},
  booktitle={2026 IEEE 11th International Conference on Communications and Electronics (ICCE)},
  year={2026}
}
```
