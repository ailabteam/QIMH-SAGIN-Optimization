import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os

def check_environment():
    print("="*50)
    print("CHECKING RESEARCH ENVIRONMENT")
    print("="*50)

    # 1. Kiểm tra Python & Thư viện cơ bản
    print(f"[+] Python version: {os.sys.version.split()[0]}")
    print(f"[+] NumPy version: {np.__version__}")
    print(f"[+] SciPy version: {scipy.__version__}")
    print(f"[+] Pandas version: {pd.__version__}")

    # 2. Kiểm tra PyTorch và GPU
    print(f"[+] PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"[+] CUDA Available: {cuda_available}")

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"[+] Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"    - GPU {i}: {props.name}")
            print(f"      VRAM: {props.total_memory / 1024**2:.0f} MB")
            print(f"      Compute Capability: {props.major}.{props.minor}")
        
        # Thử nghiệm một phép tính ma trận nhỏ trên GPU 0 và GPU 1
        try:
            for i in range(num_gpus):
                device = torch.device(f'cuda:{i}')
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                c = torch.matmul(a, b)
                print(f"[OK] Matrix multiplication on {props.name} (GPU {i}) successful.")
        except Exception as e:
            print(f"[ERROR] GPU Computation failed: {e}")
    else:
        print("[!] WARNING: CUDA not found. Please check your installation.")

    # 3. Kiểm tra khả năng xuất file PDF (cho Paper)
    print("\n[+] Checking PDF plotting capability...")
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(np.random.randn(50).cumsum(), label='Random Walk (Test)')
        plt.title("Environment Test Plot")
        plt.legend()
        plt.grid(True)
        
        # Thiết lập font chuẩn academic
        plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'serif'})
        
        test_pdf = 'test_check_figure.pdf'
        plt.savefig(test_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_pdf):
            print(f"[OK] PDF figure saved successfully: {test_pdf}")
            # os.remove(test_pdf) # Xóa file test sau khi xong (tùy chọn)
        else:
            print("[ERROR] PDF file was not created.")
    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")

    # 4. Kiểm tra logic Quantum-inspired cơ bản bằng Torch
    print("\n[+] Testing Quantum-inspired logic (Superposition simulation)...")
    try:
        # Giả lập 1000 Q-bits trạng thái chồng chập trên GPU
        q_bits = torch.full((1000, 2), 1.0/np.sqrt(2), device='cuda:0' if cuda_available else 'cpu')
        # Kiểm tra tính chuẩn hóa |alpha|^2 + |beta|^2 = 1
        norm = torch.sum(torch.abs(q_bits)**2, dim=1)
        if torch.allclose(norm, torch.ones_like(norm)):
            print("[OK] Quantum-inspired tensor operations successful.")
    except Exception as e:
        print(f"[ERROR] Quantum-inspired logic test failed: {e}")

    print("\n" + "="*50)
    print("ENVIRONMENT CHECK COMPLETED")
    print("="*50)

if __name__ == "__main__":
    check_environment()
