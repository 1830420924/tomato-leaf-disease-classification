"""在测试集上输出分类指标和混淆矩阵。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data_tomato"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tomato_3class_best.pth"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tomato_3class"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model_name = checkpoint.get("architecture", "mobilenet_v2")
    image_size = checkpoint.get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data_dir / "test", transform=transform)
    if dataset.classes != classes:
        raise RuntimeError(
            "测试集类别顺序与模型不一致："
            f"dataset={dataset.classes}, checkpoint={classes}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(len(classes), pretrained=False, model_name=model_name)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    targets, predictions = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            targets.extend(labels.numpy().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(classes))))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        args.output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )

    figure, axis = plt.subplots(figsize=(max(10, len(classes) * 0.35), max(8, len(classes) * 0.30)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Tomato Leaf Disease Confusion Matrix")
    axis.set_xticks(range(len(classes)))
    axis.set_yticks(range(len(classes)))
    axis.set_xticklabels(classes, rotation=90, fontsize=6)
    axis.set_yticklabels(classes, fontsize=6)
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    print(report)
    print(f"Model used for evaluation: {model_name}")
    print(f"Device used for evaluation: {device}")


if __name__ == "__main__":
    main()
