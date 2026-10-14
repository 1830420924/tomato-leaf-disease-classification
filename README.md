# 基于 MobileNetV2 的番茄叶片五分类识别系统

本项目是本科毕业设计，使用 PlantVillage 中的番茄叶片子集，识别以下五类：

- 番茄细菌性斑点病（`Tomato___Bacterial_spot`）
- 番茄早疫病（`Tomato___Early_blight`）
- 番茄晚疫病（`Tomato___Late_blight`）
- 番茄叶霉病（`Tomato___Leaf_Mold`）
- 番茄健康叶片（`Tomato___healthy`）

当前正式主线为番茄五分类。原来的番茄三分类数据、模型和实验结果保留为基线，不会被五分类数据覆盖。

项目使用 Python、PyTorch、MobileNetV2 和 Streamlit。

## 运行环境

- Python 3.9+
- PyTorch CUDA版
- NVIDIA GPU
- PyCharm

已训练的五分类 MobileNetV2 模型随仓库提供，可直接用于预测和可视化识别。没有 NVIDIA GPU 时，预测和评价会自动使用 CPU。

安装 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

如需使用 GPU，请先安装与本机环境兼容的 CUDA 版 PyTorch，再安装其余依赖。当前验证环境为 Python 3.9.25、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121 和 Streamlit 1.50.0。

## 1. 检查GPU

在 PyCharm Terminal 中运行：

```bash
python check_gpu.py
```

训练时应看到：

```text
Device: cuda
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
```

## 2. 准备番茄五分类数据

原始图片位于 `archive_38class/data_38class/raw/color`。运行下面的命令，筛选五个番茄类别并生成新的 `data_tomato_5class`：

```bash
python prepare_dataset.py --source archive_38class/data_38class/raw/color --output data_tomato_5class --seed 42 --overwrite
```

生成结构：

```text
data_tomato_5class/
├── train/
├── val/
└── test/
```

划分比例为70%/15%/15%。`--overwrite` 只会清空并重新生成 `data_tomato_5class`，不会覆盖原来的 `data_tomato` 三分类数据。

当前五分类数据共7,579张，其中训练集5,303张、验证集1,135张、测试集1,141张。

## 3. 训练MobileNetV2正式模型

```bash
python train.py --data-dir data_tomato_5class --model mobilenet_v2 --epochs 12 --batch-size 64 --workers 0 --patience 3 --output-dir outputs/tomato_5class --checkpoint models/tomato_5class_best.pth
```

正式输出：

- `models/tomato_5class_best.pth`
- `outputs/tomato_5class/history.json`
- `outputs/tomato_5class/training_curves.png`

## 4. 训练SimpleCNN基线模型

```bash
python train.py --data-dir data_tomato_5class --model simple_cnn --epochs 12 --batch-size 64 --workers 0 --patience 3 --output-dir outputs/tomato_5class_simple_cnn --checkpoint models/tomato_5class_simple_cnn_best.pth --no-pretrained
```

论文中可以比较SimpleCNN和MobileNetV2的准确率、Macro-F1和混淆矩阵。

## 5. 评价正式模型

```bash
python evaluate.py --data-dir data_tomato_5class --checkpoint models/tomato_5class_best.pth --batch-size 64 --workers 0 --output-dir outputs/tomato_5class
```

结果包括：

- `classification_report.txt`
- `confusion_matrix.csv`
- `confusion_matrix.png`

## 6. 单张图片预测

```bash
python predict.py --image "data_tomato_5class/test/Tomato___Leaf_Mold/图片文件.jpg"
```

## 7. 启动可视化系统

```bash
python -m streamlit run app.py
```

页面显示五类番茄叶片结果，推理设备会显示为CUDA或CPU。

## 原三分类基线结果

原三分类实验结果保留如下，后续可作为五分类实验的基线：

| 模型 | 测试集准确率 | 测试集Macro-F1 |
|---|---:|---:|
| SimpleCNN基线 | 90.10% | 88.31% |
| MobileNetV2基线 | 99.85% | 99.83% |

原三分类模型为 `models/tomato_3class_best.pth`，结果位于 `outputs/tomato_3class`。

## 五分类正式实验结果

MobileNetV2 已完成12轮训练，最佳验证集 Macro-F1 为99.76%。使用 `models/tomato_5class_best.pth` 对全部1,141张测试图片进行评价，正确识别1,138张，结果如下：

| 模型 | 测试集准确率 | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| MobileNetV2（五分类） | 99.74% | 99.72% | 99.60% | 99.66% |
| SimpleCNN（五分类） | 待训练 | 待评价 | 待评价 | 待评价 |

三张错误预测均来自早疫病：两张被识别为晚疫病，一张被识别为叶霉病。各类别详细指标见 `outputs/tomato_5class/classification_report.txt`，混淆矩阵见同目录下的 `confusion_matrix.csv` 和 `confusion_matrix.png`。

五分类 SimpleCNN 对比实验尚未完成；上面的三分类基线使用不同的类别和测试集，不应直接用于比较五分类任务的性能。上述结果仅反映当前 PlantVillage 测试集表现。

## 论文写作重点

1. 说明从PlantVillage番茄叶片类别中选择五类的原因；
2. 介绍图像预处理、数据增强和MobileNetV2迁移学习；
3. 对比SimpleCNN和MobileNetV2；
4. 使用Accuracy、Precision、Recall、Macro-F1和混淆矩阵评价；
5. 说明PlantVillage背景较规范，真实田间泛化能力属于局限性；
6. 论文提纲见 `论文写作提纲.md`。

## Dataset source

本项目使用PlantVillage数据集的 `color` 配置。

- Hugging Face dataset: https://huggingface.co/datasets/mohanty/PlantVillage
- Original project: https://github.com/spMohanty/PlantVillage-Dataset
- Selected classes: `Tomato___Bacterial_spot`, `Tomato___Early_blight`, `Tomato___Late_blight`, `Tomato___Leaf_Mold`, `Tomato___healthy`
- Expected source directory: `archive_38class/data_38class/raw/color`

数据集本身不纳入仓库。下载后将类别文件夹放在上述目录，并运行数据准备命令即可。
