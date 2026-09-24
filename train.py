"""训练番茄叶片三分类模型，优先使用CUDA并支持MobileNetV2与SimpleCNN。"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm import tqdm


TOMATO_CLASSES = {
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
}


class SimpleCNN(nn.Module):
    """用于论文对比实验的轻量基线网络。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).flatten(1)
        return self.classifier(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    num_classes: int,
    pretrained: bool = True,
    model_name: str = "mobilenet_v2",
) -> nn.Module:
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes)
    if model_name != "mobilenet_v2":
        raise ValueError(f"不支持的模型：{model_name}")

    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def build_loaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_set = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_set = datasets.ImageFolder(data_dir / "test", transform=eval_transform)
    if (
        set(train_set.classes) != TOMATO_CLASSES
        or val_set.classes != train_set.classes
        or test_set.classes != train_set.classes
    ):
        raise RuntimeError(
            "当前项目只支持番茄三分类，请使用 data_tomato（Tomato___healthy、"
            "Tomato___Early_blight、Tomato___Late_blight）。"
        )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_set, shuffle=True, **loader_kwargs),
        DataLoader(val_set, shuffle=False, **loader_kwargs),
        DataLoader(test_set, shuffle=False, **loader_kwargs),
        train_set.classes,
    )


def run_epoch(model, loader, criterion, optimizer, scaler, device, training: bool):
    model.train(training)
    total_loss = 0.0
    all_targets, all_predictions = [], []
    use_cuda = device.type == "cuda"

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=use_cuda)
        targets = targets.to(device, non_blocking=use_cuda)
        if training:
            optimizer.zero_grad(set_to_none=True)

        amp_context = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()
        with amp_context:
            logits = model(images)
            loss = criterion(logits, targets)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    average_loss = total_loss / max(1, len(loader.dataset))
    accuracy = float(np.mean(np.array(all_targets) == np.array(all_predictions)))
    macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)
    return average_loss, accuracy, macro_f1


def plot_history(history: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(epochs)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticks(epochs)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data_tomato"))
    parser.add_argument("--model", choices=["mobilenet_v2", "simple_cnn"], default="mobilenet_v2")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0, help="Windows/PyCharm建议保持为0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tomato_3class"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tomato_3class_best.pth"))
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Device: cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Device: cpu（未检测到CUDA）")

    train_loader, val_loader, _, classes = build_loaders(
        args.data_dir, args.image_size, args.batch_size, args.workers, device
    )
    if len(classes) < 2:
        raise RuntimeError("至少需要两个类别才能进行分类训练。")
    print(f"Model: {args.model}")
    print(f"Classes: {len(classes)} | train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)}")

    model = build_model(
        len(classes),
        pretrained=(not args.no_pretrained and args.model == "mobilenet_v2"),
        model_name=args.model,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    history_path = args.output_dir / "history.json"
    curve_path = args.output_dir / "training_curves.png"
    best_f1 = -1.0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, training=True
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, training=False
        )
        scheduler.step(val_f1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)
        plot_history(history, curve_path)
        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            stale_epochs = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "classes": classes,
                "image_size": args.image_size,
                "architecture": args.model,
                "best_val_macro_f1": best_f1,
            }
            torch.save(checkpoint, args.checkpoint)
            print(f"已保存最佳模型：{args.checkpoint}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"验证集连续{args.patience}轮没有提升，提前停止。")
                break

    print(f"训练完成，历史记录：{history_path}")


if __name__ == "__main__":
    main()
