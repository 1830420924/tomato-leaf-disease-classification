import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("GPU不可用，训练脚本会回退到CPU。请检查PyCharm使用的Python解释器是否为CUDA环境。")
        return

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        memory_gb = properties.total_memory / 1024**3
        print(f"GPU {index}: {properties.name} ({memory_gb:.1f} GB)")

    device = torch.device("cuda")
    x = torch.randn(2048, 2048, device=device)
    y = x @ x
    torch.cuda.synchronize()
    print(f"CUDA smoke test passed: {y.shape}")


if __name__ == "__main__":
    main()
