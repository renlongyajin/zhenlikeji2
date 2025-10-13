#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析当前检测问题 - 找出漏检的真正医学图像和误检的文字块
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_image_features(image_path):
    """分析图像特征"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    height, width = img.shape[:2]
    aspect_ratio = width / height
    area = width * height

    # 计算更多特征
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)

    # 边缘密度
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (height * width)

    return {
        'path': image_path,
        'width': width,
        'height': height,
        'aspect_ratio': aspect_ratio,
        'area': area,
        'variance': variance,
        'edge_density': edge_density
    }

def main():
    """主函数"""
    print("=== 分析当前检测结果 ===")

    # 分析当前检测的图像
    current_images = [
        "data/optimized_processed/images/page_5_region_1_tumor_or_organ.png",
        "data/optimized_processed/images/page_17_region_0_medical_diagram.png",
        "data/optimized_processed/images/page_17_region_1_medical_diagram.png"
    ]

    print("\n当前检测到的图像特征:")
    for img_path in current_images:
        if Path(img_path).exists():
            features = analyze_image_features(img_path)
            if features:
                print(f"\n{Path(img_path).name}:")
                print(f"  尺寸: {features['width']}x{features['height']}")
                print(f"  长宽比: {features['aspect_ratio']:.3f}")
                print(f"  面积: {features['area']:,} 像素")
                print(f"  方差: {features['variance']:.1f}")
                print(f"  边缘密度: {features['edge_density']:.4f}")

                # 判断是否为文字块的依据
                is_likely_text = (features['aspect_ratio'] > 4.0 or features['aspect_ratio'] < 0.25 or
                                features['area'] < 50000)
                print(f"  疑似文字块: {is_likely_text}")

    # 对比之前成功检测的图像
    print("\n=== 对比之前成功检测的图像 ===")

    previous_successful = [
        "data/test_pages/images/page_5_portrait_area_portrait.png",
        "data/test_pages/images/page_5_main_image_area_medical_diagram.png",
        "data/test_pages/images/page_16_portrait_area_portrait.png",
        "data/test_pages/images/page_16_main_image_area_medical_diagram.png"
    ]

    print("\n之前成功检测的图像特征:")
    for img_path in previous_successful:
        if Path(img_path).exists():
            features = analyze_image_features(img_path)
            if features:
                print(f"\n{Path(img_path).name}:")
                print(f"  尺寸: {features['width']}x{features['height']}")
                print(f"  长宽比: {features['aspect_ratio']:.3f}")
                print(f"  面积: {features['area']:,} 像素")
                print(f"  方差: {features['variance']:.1f}")
                print(f"  边缘密度: {features['edge_density']:.4f}")

if __name__ == "__main__":
    main()