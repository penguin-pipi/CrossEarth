# Copyright (c) OpenMMLab. All rights reserved.
"""
单张图片语义分割推理脚本
Usage:
    python tools/inference_single.py configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py ./checkpoints/your_model.pth ./test_image.png
"""
import argparse
import os
import os.path as osp
import sys

# 设置路径 (与 test.py 保持一致的风格)
os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))
sys.path.append(os.curdir)

import numpy as np
from PIL import Image

from mmengine.config import Config
from mmseg.apis import init_model, inference_model

import CrossEarth
import rs_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='CrossEarth 单张图片语义分割推理')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型权重路径')
    parser.add_argument('image', help='输入图片路径')
    parser.add_argument('--output', '-o', default='./output', help='输出目录')
    parser.add_argument('--device', default='cuda:0', help='推理设备 (cuda:0 或 cpu)')
    parser.add_argument('--palette', default='default', 
                       choices=['default', 'isprs', 'loveda', 'building', 'rescue'],
                       help='调色板类型')
    return parser.parse_args()


# 定义不同数据集的调色板
PALETTES = {
    'default': [
        [0, 0, 0],       # 背景
        [255, 0, 0],     # 类别1
        [0, 255, 0],     # 类别2
        [0, 0, 255],     # 类别3
        [255, 255, 0],   # 类别4
        [255, 0, 255],   # 类别5
        [0, 255, 255],   # 类别6
        [128, 128, 128], # 类别7
    ],
    'isprs': [
        [255, 255, 255], # Impervious surfaces
        [0, 0, 255],     # Building
        [0, 255, 255],   # Low vegetation
        [0, 255, 0],     # Tree
        [255, 255, 0],   # Car
        [255, 0, 0],     # Clutter
    ],
    'loveda': [
        [255, 255, 255], # Background
        [255, 0, 0],     # Building
        [255, 255, 0],   # Road
        [0, 0, 255],     # Water
        [159, 129, 183], # Barren
        [0, 255, 0],     # Forest
        [255, 195, 128], # Agricultural
    ],
    'building': [
        [0, 0, 0],       # Background
        [255, 255, 255], # Building
    ],
    'rescue': [
        [0, 0, 0],       # Background
        [255, 0, 0],     # Building-flooded
        [255, 255, 0],   # Building-non-flooded
        [0, 255, 0],     # Road-flooded
        [0, 0, 255],     # Road-non-flooded
    ],
}


def colorize_mask(mask, palette_name='default'):
    """将预测的类别标签转换为彩色图片
    
    Args:
        mask: 预测的分割掩码，包含类别索引
        palette_name: 调色板名称
        
    Returns:
        彩色RGB图像数组
        
    Note:
        如果类别索引超出调色板大小，将使用灰色 [128, 128, 128] 表示未知类别
    """
    palette = PALETTES.get(palette_name, PALETTES['default'])
    palette_array = np.array(palette, dtype=np.uint8)
    
    # 创建输出图像
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # 对每个类别进行着色
    for class_idx in np.unique(mask):
        if class_idx < len(palette_array):
            color_mask[mask == class_idx] = palette_array[class_idx]
        else:
            # 未知类别使用灰色
            color_mask[mask == class_idx] = [128, 128, 128]
    
    return color_mask


def main():
    args = parse_args()
    
    # 检查输入文件是否存在
    if not osp.exists(args.config):
        print(f'错误: 配置文件不存在: {args.config}')
        sys.exit(1)
    
    if not osp.exists(args.checkpoint):
        print(f'错误: 模型权重不存在: {args.checkpoint}')
        sys.exit(1)
    
    if not osp.exists(args.image):
        print(f'错误: 输入图片不存在: {args.image}')
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载配置和模型
    print(f'正在加载配置文件: {args.config}')
    cfg = Config.fromfile(args.config)
    
    print(f'正在加载模型: {args.checkpoint}')
    print(f'使用设备: {args.device}')
    model = init_model(cfg, args.checkpoint, device=args.device)
    
    # 进行推理
    print(f'正在处理图片: {args.image}')
    result = inference_model(model, args.image)
    
    # 获取预测结果
    pred_mask = result.pred_sem_seg.data.cpu().numpy().squeeze()
    
    # 保存结果
    basename = osp.splitext(osp.basename(args.image))[0]
    
    # 保存原始预测标签
    label_path = osp.join(args.output, f'{basename}_pred.png')
    Image.fromarray(pred_mask.astype(np.uint8)).save(label_path)
    print(f'预测标签已保存到: {label_path}')
    
    # 保存彩色可视化结果
    color_mask = colorize_mask(pred_mask, args.palette)
    color_path = osp.join(args.output, f'{basename}_color.png')
    Image.fromarray(color_mask).save(color_path)
    print(f'彩色结果已保存到: {color_path}')
    
    # 打印统计信息
    unique_classes = np.unique(pred_mask)
    print(f'\n预测统计:')
    print(f'  图片尺寸: {pred_mask.shape}')
    print(f'  检测到的类别: {unique_classes}')
    for cls in unique_classes:
        count = np.sum(pred_mask == cls)
        ratio = count / pred_mask.size * 100
        print(f'  类别 {cls}: {count} 像素 ({ratio:.2f}%)')
    
    print('\n推理完成!')


if __name__ == '__main__':
    main()
