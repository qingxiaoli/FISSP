#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_dataset.py

脚本：对已经下载的 FISSP 数据集进行数据过滤与质量控制，包括：
1. Outlier Data Filtering (基于CNN特征聚类，剔除非细粒度/异常图片)
2. Low-quality Data Filtering (分辨率/亮度/对比度/清晰度 等指标)
3. Sensitive Data Filtering (利用敏感图像识别模型过滤NSFW/暴力/违法等内容)
4. Final Quality Check (最终交叉检查，确保满足FGIR数据需求)

使用方法：
    python scripts/prepare_dataset.py --data_dir ./data
"""

import os
import argparse
import shutil
import glob

import numpy as np
import cv2  # 假设我们用 opencv-python
from PIL import Image

import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50  # 仅作示例，可根据需求更换

# ==============  1. Outlier Data Filtering  ==============
def extract_features_with_cnn(image_path, model, transform, device="cpu"):
    """
    使用给定的 CNN 模型对图像进行特征提取。
    这里以 ResNet50 的 Global Average Pooling 后的向量作为示例。
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARNING] Failed to open image {image_path}: {e}")
        return None

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img_tensor)
    # 特征 shape: (batch_size=1, feature_dim)
    features = features.cpu().numpy().squeeze()
    return features

def cluster_and_filter_outliers(image_features, image_paths, z_threshold=3.0):
    """
    简单示例：对特征做均值方差检测，将超出一定范围的样本视为异常/离群点。
    更复杂的方式可以用KMeans或DBSCAN来聚类，然后根据类别大小或相似性过滤。

    参数:
    - image_features: (N, D) 矩阵，N为图像数量，D为特征维度
    - image_paths: 对应每张图像的文件路径
    - z_threshold: 离群值阈值；例如>3视为离群
    """
    # 计算每个维度上的 mean 和 std
    mean_vec = np.mean(image_features, axis=0)
    std_vec = np.std(image_features, axis=0) + 1e-6

    # 计算每个样本的Z分数
    z_scores = np.abs((image_features - mean_vec) / std_vec)
    # 这里简化：先对所有维度做平均Z分数
    z_scores_mean = np.mean(z_scores, axis=1)

    removed_count = 0
    for idx, z_val in enumerate(z_scores_mean):
        if z_val > z_threshold:
            # 认为这张图片是离群样本
            try:
                os.remove(image_paths[idx])
                removed_count += 1
            except Exception as e:
                print(f"[WARNING] Failed to remove outlier image {image_paths[idx]}: {e}")
    return removed_count

# ==============  2. Low-quality Data Filtering  ==============
def check_image_quality(image_path,
                        min_resolution=(200, 200),
                        brightness_threshold=(0.2, 0.95),
                        blur_threshold=100.0):
    """
    检查图像质量，返回 True 表示图像质量足够好，False 表示需要剔除。
    - 分辨率：宽高都 >= min_resolution
    - 亮度：简单估计亮度在一定范围
    - 清晰度(Blur)：通过Laplacian算子估计图像的清晰度
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        h, w, c = img.shape

        # 1) 分辨率过滤
        if (w < min_resolution[0]) or (h < min_resolution[1]):
            return False

        # 2) 亮度过滤(简单估计：把图片转换成灰度后，取[0..1]区间的平均值)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray) / 255.0
        if mean_val < brightness_threshold[0] or mean_val > brightness_threshold[1]:
            return False

        # 3) 清晰度(Blur)检测：通过Laplacian的方差来判断
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap < blur_threshold:
            return False

        return True

    except Exception as e:
        print(f"[WARNING] check_image_quality failed for {image_path}: {e}")
        return False

def filter_low_quality_images(root_dir,
                              min_resolution=(200, 200),
                              brightness_threshold=(0.2, 0.95),
                              blur_threshold=100.0):
    removed_count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                if not check_image_quality(file_path,
                                           min_resolution=min_resolution,
                                           brightness_threshold=brightness_threshold,
                                           blur_threshold=blur_threshold):
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except Exception as e:
                        print(f"[WARNING] Failed to remove low-quality image {file_path}: {e}")
    return removed_count

# ==============  3. Sensitive Data Filtering  ==============
def load_sensitive_recognition_model(model_path):
    """
    加载你训练好的敏感图像识别模型（NSFW / PEDA376k 等），
    此处仅示意。如果你用的是 PyTorch，可以像加载普通分类器一样加载。
    """
    # 占位示例，需根据你的实际模型结构实现
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model

def is_sensitive_image(image_path, model, transform, threshold=0.5, device="cpu"):
    """
    使用敏感图像识别模型对图片进行预测，若预测为敏感内容则返回 True。
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except:
        return True  # 无法打开的图片也作为可疑处理

    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        # 这里假设模型输出一个对 NSFW / 敏感类的概率
        prob = model(img_tensor)  # 形状假设为 (1,)
    prob = prob.item()

    return (prob >= threshold)

def filter_sensitive_images(root_dir, model, transform, threshold=0.5, device="cpu"):
    removed_count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, file)
                if is_sensitive_image(file_path, model, transform, threshold=threshold, device=device):
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except Exception as e:
                        print(f"[WARNING] Failed to remove sensitive image {file_path}: {e}")
    return removed_count

# ==============  4. Final Quality Check  ==============
def final_quality_check(root_dir):
    """
    最终交叉检查，确保符合FGIR任务需求。这里可以包括：
    - 手动抽样检查 (人工评审)
    - 基于业务逻辑的类别检查
    - 统计一些指标 (图像数量, 平均分辨率, 类别分布等)
    """
    print("[INFO] Performing final quality check...")

    # 示例：统计剩余图片数量
    total_images = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                total_images += 1

    print(f"[INFO] Final dataset contains {total_images} images in total.")
    # 你可以在此处添加更多统计或可视化，如类别分布、平均分辨率等
    # ...

# ==============  Main Pipeline  ==============
def prepare_fissp_dataset(data_dir: str,
                          device="cpu",
                          sensitive_model_path=None):
    """
    根据四个主要步骤来过滤和整理 FISSP 数据集。
    1. Outlier Data Filtering (CNN特征+聚类/离群检测)
    2. Low-quality Data Filtering (分辨率、亮度、对比度、清晰度)
    3. Sensitive Data Filtering (敏感内容检测)
    4. Final Quality Check
    """
    # 1) 初始化一个CNN用于特征提取 (示例：ResNet50)
    print("[INFO] Initializing CNN model (ResNet50) for outlier detection...")
    base_model = resnet50(pretrained=True)
    # 去掉最后一层分类器，只保留全局平均池化输出
    feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])
    feature_extractor.eval()
    # 定义图像预处理
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    feature_extractor.to(device)

    # 2) 扫描数据文件夹，提取特征
    all_image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        all_image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    print(f"[INFO] Extracting features from {len(all_image_paths)} images for outlier detection...")
    image_features_list = []
    valid_image_paths = []  # 保存真正成功提取到特征的

    with torch.no_grad():
        for img_path in all_image_paths:
            features = extract_features_with_cnn(img_path, feature_extractor, transform, device=device)
            if features is not None:
                image_features_list.append(features)
                valid_image_paths.append(img_path)

    if len(image_features_list) == 0:
        print("[WARNING] No valid features extracted. Skipping outlier filtering.")
    else:
        image_features_array = np.array(image_features_list)
        removed_outliers = cluster_and_filter_outliers(image_features_array, valid_image_paths, z_threshold=3.0)
        print(f"[INFO] Outlier data filtering removed {removed_outliers} images.")

    # 3) 过滤低质量图像
    print("[INFO] Filtering low-quality images...")
    removed_low_quality = filter_low_quality_images(data_dir,
                                                    min_resolution=(200, 200),
                                                    brightness_threshold=(0.2, 0.95),
                                                    blur_threshold=100.0)
    print(f"[INFO] Removed {removed_low_quality} low-quality images.")

    # 4) 过滤敏感图像（若提供了敏感模型路径）
    if sensitive_model_path is not None and os.path.exists(sensitive_model_path):
        print("[INFO] Filtering sensitive images...")
        # 加载敏感图像识别模型
        sensitive_model = load_sensitive_recognition_model(sensitive_model_path)
        # 这里假设敏感模型输入也需要相同的 transform；根据你的实际情况可能需要改动
        removed_sensitive = filter_sensitive_images(data_dir,
                                                    model=sensitive_model,
                                                    transform=transform,
                                                    threshold=0.5,
                                                    device=device)
        print(f"[INFO] Removed {removed_sensitive} sensitive images.")
    else:
        print("[INFO] No sensitive model path provided or path is invalid. Skipping sensitive filtering.")

    # 5) 最终质量检查
    final_quality_check(data_dir)

def main():
    parser = argparse.ArgumentParser(description="FISSP Data Preparation Script")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory where the downloaded FISSP dataset is stored.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run CNN for outlier/sensitive detection (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--sensitive_model", type=str, default=None,
                        help="Path to the trained sensitive image recognition model (e.g., NSFW).")
    args = parser.parse_args()

    prepare_fissp_dataset(data_dir=args.data_dir,
                          device=args.device,
                          sensitive_model_path=args.sensitive_model)

if __name__ == "__main__":
    main()