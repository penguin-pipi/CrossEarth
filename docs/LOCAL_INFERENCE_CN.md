# CrossEarth 本地语义分割推理指南

本文档详细介绍如何在本地运行 CrossEarth 模型实现遥感图像语义分割。

## 目录
- [环境配置](#环境配置)
- [模型权重下载](#模型权重下载)
- [数据准备](#数据准备)
- [运行推理](#运行推理)
- [单张图片推理](#单张图片推理)
- [常见问题](#常见问题)

## 环境配置

### 1. 创建 Conda 环境

```bash
# 创建新的 conda 环境
conda create -n CrossEarth python=3.9 -y
conda activate CrossEarth

# 安装 PyTorch（根据你的 CUDA 版本选择合适的版本）
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# 安装 mmcv 相关库
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"

# 安装 xformers
pip install xformers=='0.0.20'

# 安装其他依赖
pip install -r requirements.txt
pip install future tensorboard
```

### 2. 验证安装

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import mmcv; print(f'MMCV version: {mmcv.__version__}')"
python -c "import mmseg; print(f'MMSegmentation version: {mmseg.__version__}')"
```

## 模型权重下载

### 1. 下载预训练权重

你可以从以下途径下载模型权重：

- **Hugging Face**: https://huggingface.co/Cusyoung/CrossEarth/tree/main
- **百度网盘**: https://pan.baidu.com/s/1nOQlZUTeySn6-jiqjheWhQ?pwd=eq40

### 2. 必需的权重文件

**重要**: 以下两个 backbone 权重文件是推理时必需的，请先下载并放置到 `CrossEarth/checkpoints/` 目录：

- `dinov2_converted.pth` - 用于 512x512 输入分辨率
- `dinov2_converted_1024x1024.pth` - 用于 1024x1024 输入分辨率

### 3. 任务特定权重

根据你的任务需求，下载对应的任务权重文件：

| 任务 | 权重文件 |
|------|----------|
| ISPRS Potsdam(i) → Vaihingen | Potsdam(i)-source.pth |
| ISPRS Potsdam(r) → Vaihingen | Potsdam(r)-source.pth |
| ISPRS Vaihingen → Potsdam | Vaihingen-source.pth |
| WHU Building A → S | WHU-Building-A2S.pth |
| WHU Building S → A | WHU-Building-S2A.pth |
| DeepGlobe → Massachusetts | D2M-Rein.pth 或 D2M-MTP.pth |
| Potsdam → RescueNet | Potsdam(r)2RescueNet.pth 或 Potsdam(i)2RescueNet.pth |
| CASID 子集 | casid-sub-source.pth, casid-tem-source.pth 等 |

### 4. 权重文件存放

创建 checkpoints 目录并放置权重：

```bash
mkdir -p CrossEarth/checkpoints/

# 将下载的权重文件放入该目录
mv dinov2_converted.pth CrossEarth/checkpoints/
mv dinov2_converted_1024x1024.pth CrossEarth/checkpoints/
mv your_task_weights.pth CrossEarth/checkpoints/
```

## 数据准备

### 1. 数据集目录结构

CrossEarth 支持多种遥感数据集，数据应按以下结构组织：

```
data/
├── potsdam/
│   ├── img_dir/
│   │   ├── train/
│   │   └── val/
│   └── ann_dir/
│       ├── train/
│       └── val/
├── vaihingen/
│   ├── img_dir/
│   │   ├── train/
│   │   └── val/
│   └── ann_dir/
│       ├── train/
│       └── val/
├── rescuenet/
│   ├── test-org-img/
│   └── test-label-img/
└── ...
```

### 2. 修改配置文件中的数据路径

在运行推理前，需要修改配置文件中的数据路径：

**a) 修改数据集配置 (`configs/_base_/datasets/xxx.py`)**

例如，修改 `configs/_base_/datasets/rescue_512x512.py`：

```python
# 将 Rescue_root 修改为你的数据集实际路径
Rescue_root = "/path/to/your/rescuenet/data"  # 修改这里
```

**b) 修改类别数量（如需要）**

在 `configs/_base_/models/CrossEarth_dinov2_mask2former.py` 中：

```python
# 根据你使用的数据集调整类别数
loveda_classes = 7
casid_classes = 5
isprs_classes = 6
road_buliding_classes = 2
rescue_classes = 5

# 设置当前使用的类别数
num_classes = rescue_classes  # 根据你的数据集修改
```

## 运行推理

### 1. 基本推理命令

```bash
# 使用 512x512 输入分辨率进行推理
python tools/test.py configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py ./checkpoints/your_model.pth

# 使用 1024x1024 输入分辨率进行推理
python tools/test.py configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_1024x1024_bs4x2.py ./checkpoints/your_model.pth
```

### 2. 指定工作目录

```bash
python tools/test.py configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py ./checkpoints/your_model.pth --work-dir ./work_dirs/my_inference
```

### 3. 保存预测结果

```bash
python tools/test.py configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py ./checkpoints/your_model.pth --out ./output/predictions
```

### 4. 可视化预测结果

```bash
python tools/test.py configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py ./checkpoints/your_model.pth --show-dir ./output/visualizations
```

### 5. 完整参数说明

```bash
python tools/test.py [CONFIG] [CHECKPOINT] [OPTIONS]

必需参数:
  CONFIG        配置文件路径
  CHECKPOINT    模型权重路径

可选参数:
  --work-dir    结果保存目录
  --out         预测输出目录
  --show        显示预测结果（需要GUI环境）
  --show-dir    可视化结果保存目录
  --wait-time   可视化显示间隔时间（秒）
  --cfg-options 覆盖配置文件中的设置
  --launcher    分布式启动方式 (none, pytorch, slurm, mpi)
  --tta         启用测试时增强
```

## 单张图片推理

如果你只想对单张图片进行推理，可以使用仓库中提供的 `tools/inference_single.py` 脚本。

### 使用方法

```bash
# 基本用法
python tools/inference_single.py configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py ./checkpoints/your_model.pth ./test_image.png

# 指定输出目录和设备
python tools/inference_single.py configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py ./checkpoints/your_model.pth ./test_image.png --output ./my_results --device cuda:0

# 使用特定的调色板进行可视化
python tools/inference_single.py configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py ./checkpoints/your_model.pth ./test_image.png --palette isprs
```

### 可用的调色板选项

| 调色板名称 | 适用数据集 | 类别数 |
|-----------|-----------|-------|
| `default` | 通用 | 8 |
| `isprs` | ISPRS Potsdam/Vaihingen | 6 |
| `loveda` | LoveDA | 7 |
| `building` | WHU Building | 2 |
| `rescue` | RescueNet | 5 |

### 输出文件

脚本会在输出目录生成两个文件：
- `{图片名}_pred.png`: 原始预测标签（灰度图）
- `{图片名}_color.png`: 彩色可视化结果

## 常见问题

### Q1: 出现 "CUDA out of memory" 错误

**解决方案：**
- 减小 batch_size
- 使用较小的输入分辨率 (512x512)
- 使用 CPU 进行推理：`--device cpu`

### Q2: 找不到 dinov2_converted.pth

**解决方案：**
确保已下载 backbone 权重并放置在正确路径：
```bash
ls CrossEarth/checkpoints/
# 应该看到 dinov2_converted.pth
```

### Q3: 数据集加载失败

**解决方案：**
1. 检查数据路径是否正确
2. 确认数据集目录结构是否符合要求
3. 检查配置文件中的 `data_root` 设置

### Q4: 类别数量不匹配

**解决方案：**
修改 `configs/_base_/models/CrossEarth_dinov2_mask2former.py` 中的 `num_classes`，确保与你的数据集类别数一致。

### Q5: 如何使用自己的数据集？

**步骤：**
1. 按照支持的数据集格式准备数据
2. 在 `rs_dataset/` 中创建数据集类（可参考现有数据集）
3. 在 `configs/_base_/datasets/` 中创建配置文件
4. 修改训练/测试配置使用新数据集

### Q6: 如何在没有 GPU 的机器上运行？

```bash
# 使用 CPU 进行推理（速度较慢）
python tools/test.py configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py ./checkpoints/your_model.pth --cfg-options model.backbone.init_cfg.device=cpu
```

## 联系方式

如有更多问题，请访问：
- GitHub Issues: https://github.com/Cuzyoung/CrossEarth/issues
- 项目主页: https://cuzyoung.github.io/CrossEarth-Homepage/
