"""本地叶片病害可视化识别页面。"""

from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from train import build_model


CHECKPOINT = Path("models/tomato_3class_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISPLAY_NAMES = {
    "Tomato___healthy": "番茄健康叶片",
    "Tomato___Early_blight": "番茄早疫病",
    "Tomato___Late_blight": "番茄晚疫病",
}


@st.cache_resource
def load_model():
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    classes = checkpoint["classes"]
    model_name = checkpoint.get("architecture", "mobilenet_v2")
    image_size = checkpoint.get("image_size", 224)
    model = build_model(len(classes), pretrained=False, model_name=model_name)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, classes, transform


st.set_page_config(page_title="番茄叶片三分类识别", page_icon="🍅", layout="centered")
st.title("🍅 番茄叶片三分类识别系统")
st.caption(f"当前推理设备：{DEVICE}")

if not CHECKPOINT.exists():
    st.warning("尚未找到 models/tomato_3class_best.pth，请先运行番茄三分类训练。")
else:
    model, classes, transform = load_model()
    uploaded = st.file_uploader("上传叶片图片", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="待识别图片", use_container_width=True)
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        values, indices = torch.topk(probabilities, k=min(3, len(classes)))
        top_class = classes[int(indices[0])]
        st.success(f"预测结果：{DISPLAY_NAMES.get(top_class, top_class)}")
        st.write(f"置信度：{float(values[0]) * 100:.2f}%")
        st.subheader("Top-3预测结果")
        for value, index in zip(values, indices):
            class_name = classes[int(index)]
            st.write(f"{DISPLAY_NAMES.get(class_name, class_name)}：{float(value) * 100:.2f}%")
