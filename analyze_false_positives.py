#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析误检的图像，重新调整参数
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FalsePositiveAnalyzer:
    """误检分析器"""

    def __init__(self, extracted_dir: str):
        self.extracted_dir = Path(extracted_dir)

    def analyze_suspicious_images(self, suspicious_pages: list):
        """分析可疑页面的提取结果"""
        print("=== 分析可疑页面的提取结果 ===\n")

        for page_num in suspicious_pages:
            image_files = list(self.extracted_dir.glob(f"page_{page_num}_image_*.png"))

            if not image_files:
                print(f"第{page_num}页: 无提取图像")
                continue

            print(f"第{page_num}页: 提取了{len(image_files)}张图像")

            for img_file in image_files:
                self._analyze_single_image(img_file)

    def _analyze_single_image(self, img_path: Path):
        """分析单张图像的特征"""
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ❌ 无法读取: {img_path.name}")
            return

        height, width = img.shape[:2]
        aspect_ratio = width / height if height > 0 else 1.0
        area = width * height

        # 颜色特征
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        variance = np.var(gray)

        # 边缘特征
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (width * height)

        # 复杂度评估
        complexity_score = self._calculate_complexity(gray)

        print(f"  {img_path.name}:")
        print(f"    尺寸: {width}x{height} (比例: {aspect_ratio:.2f})")
        print(f"    面积: {area:,} 像素")
        print(f"    平均亮度: {mean_brightness:.1f}")
        print(f"    方差: {variance:.1f}")
        print(f"    边缘密度: {edge_density:.4f}")
        print(f"    复杂度评分: {complexity_score:.2f}")

        # 判断是否为有效医学图像
        is_valid = self._is_valid_medical_image(width, height, variance, edge_density, complexity_score)

        if is_valid:
            print(f"    ✅ 可能是有效医学图像")
        else:
            print(f"    ❌ 可能是误检")
        print()

    def _calculate_complexity(self, gray: np.ndarray) -> float:
        """计算图像复杂度"""
        # 使用拉普拉斯算子检测整体变化
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.var(laplacian)

        # 归一化到0-1范围
        complexity = min(variance / 1000, 1.0)
        return complexity

    def _is_valid_medical_image(self, width: int, height: int, variance: float, edge_density: float, complexity: float) -> bool:
        """判断是否为有效医学图像"""

        # 基于分析结果的建议阈值
        criteria = []

        # 1. 尺寸要求
        if width >= 200 and height >= 200:
            criteria.append("尺寸足够")
        else:
            criteria.append("尺寸太小")

        # 2. 方差要求（内容复杂度）
        if variance > 100:
            criteria.append("内容有变化")
        else:
            criteria.append("内容太平淡")

        # 3. 边缘密度要求
        if 0.02 < edge_density < 0.3:
            criteria.append("边缘密度合适")
        else:
            criteria.append("边缘密度异常")

        # 4. 复杂度要求
        if complexity > 0.1:
            criteria.append("复杂度足够")
        else:
            criteria.append("复杂度太低")

        # 综合判断
        valid_criteria = [c for c in criteria if not c.startswith("尺寸") and not c.startswith("内容") and not c.startswith("边缘") and not c.startswith("复杂度")]

        # 如果大部分标准都通过，则认为是有效图像
        return len([c for c in criteria if not c.endswith("太小") and not c.endswith("太平淡") and not c.endswith("异常") and not c.endswith("太低")]) >= 3

    def generate_optimized_parameters(self):
        """基于分析结果生成优化参数"""
        print("\n=== 基于分析结果的建议参数 ===")

        print("建议的严格过滤参数:")
        print("1. 最小尺寸: width >= 200, height >= 200")
        print("2. 最小面积: 40,000 像素")
        print("3. 方差范围: 100 - 5000")
        print("4. 边缘密度: 0.02 - 0.30")
        print("5. 复杂度评分: > 0.1")
        print("6. 矩形度: >= 0.6")
        print("7. 长宽比: 0.5 - 3.0")

if __name__ == "__main__":
    analyzer = FalsePositiveAnalyzer("data/cropped_images_optimized")

    # 分析可疑页面
    suspicious_pages = [1, 11, 12, 13, 14, 15]  # 第11-15页不应该有图像
    analyzer.analyze_suspicious_images(suspicious_pages)

    analyzer.generate_optimized_parameters()