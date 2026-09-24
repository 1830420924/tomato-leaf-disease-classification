# 基于 MobileNetV2 的番茄叶片三分类识别系统

本项目是本科毕业设计，使用 PlantVillage 中的番茄叶片子集，识别以下三类：

- 番茄健康叶片
- 番茄早疫病
- 番茄晚疫病

项目使用 Python、PyTorch、MobileNetV2 和 Streamlit。当前运行主线只保留番茄三分类；旧的38类文件如存在于 `archive_38class/`，仅作为备份，不参与训练、评估或推理。

## 运行环境

- Python 3.9+
- PyTorch CUDA版
- NVIDIA GPU
- PyCharm

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

## 2. 准备番茄三分类数据

原始图片备份位于 `archive_38class/data_38class/raw/color`。运行下面的命令，只筛选三个番茄类别并生成 `data_tomato`：

```bash
python prepare_dataset.py --source archive_38class/data_38class/raw/color --output data_tomato --seed 42 --overwrite
```

生成结构：

```text
data_tomato/
├── train/
├── val/
└── test/
```

划分比例为70%/15%/15%，当前项目只使用 `data_tomato`。
使用 `--overwrite` 会先清空输出目录下已有的 `train/val/test`，再重新划分；数据集、模型和训练输出已被 `.gitignore` 排除，克隆到 GitHub 后需要重新准备数据并训练模型。

## 3. 训练MobileNetV2正式模型

```bash
python train.py --data-dir data_tomato --model mobilenet_v2 --epochs 12 --batch-size 64 --workers 0 --patience 3 --output-dir outputs/tomato_3class --checkpoint models/tomato_3class_best.pth
```

正式输出：

- `models/tomato_3class_best.pth`
- `outputs/tomato_3class/history.json`
- `outputs/tomato_3class/training_curves.png`

## 4. 训练SimpleCNN基线模型

```bash
python train.py --data-dir data_tomato --model simple_cnn --epochs 12 --batch-size 64 --workers 0 --patience 3 --output-dir outputs/tomato_simple_cnn --checkpoint models/tomato_simple_cnn_best.pth --no-pretrained
```

论文中比较SimpleCNN和MobileNetV2的准确率、Macro-F1和混淆矩阵。

## 5. 评价正式模型

```bash
python evaluate.py --data-dir data_tomato --checkpoint models/tomato_3class_best.pth --batch-size 64 --workers 0 --output-dir outputs/tomato_3class
```

结果包括：

- `classification_report.txt`
- `confusion_matrix.csv`
- `confusion_matrix.png`

## 6. 单张图片预测

```bash
python predict.py --image "data_tomato/test/Tomato___Late_blight/图片文件.jpg"
```

## 7. 启动可视化系统

```bash
python -m streamlit run app.py
```

页面只显示番茄健康叶片、早疫病和晚疫病三类结果，推理设备会显示为CUDA。

## 当前正式实验结果

| 模型 | 测试集准确率 | 测试集Macro-F1 |
|---|---:|---:|
| SimpleCNN基线 | 90.10% | 88.31% |
| MobileNetV2正式模型 | 99.85% | 99.83% |

MobileNetV2正式模型已保存为 `models/tomato_3class_best.pth`，详细报告位于 `outputs/tomato_3class`。

## 论文写作重点

1. 说明为什么将完整PlantVillage数据集收缩为番茄三分类；
2. 介绍图像预处理、数据增强和MobileNetV2迁移学习；
3. 对比SimpleCNN和MobileNetV2；
4. 使用Accuracy、Precision、Recall、Macro-F1和混淆矩阵评价；
5. 说明PlantVillage背景较规范，真实田间泛化能力属于局限性；
6. 论文提纲见 `论文写作提纲.md`。

## Public repository artifacts

The current public repository supersedes the earlier note above: trained model checkpoints and experiment outputs are included, while the dataset remains excluded.

## Dataset source

This project uses the PlantVillage dataset with the `color` configuration.

- Hugging Face dataset: https://huggingface.co/datasets/mohanty/PlantVillage
- Original project: https://github.com/spMohanty/PlantVillage-Dataset
- Selected classes: `Tomato___healthy`, `Tomato___Early_blight`, and `Tomato___Late_blight`
- Expected source directory: `archive_38class/data_38class/raw/color`

After downloading the dataset, keep the class folders under the expected source directory and run the preparation command below. The repository does not include the dataset itself.

