"""命令行单张图片预测，可用于论文展示和调试。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tomato_3class_best.pth"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model_name = checkpoint.get("architecture", "mobilenet_v2")
    image_size = checkpoint.get("image_size", 224)
    model = build_model(len(classes), pretrained=False, model_name=model_name)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(5, len(classes)))
    print(f"Device: {device}")
    print(f"Image: {args.image}")
    for value, index in zip(values, indices):
        print(f"{classes[int(index)]}: {float(value) * 100:.2f}%")


if __name__ == "__main__":
    main()
