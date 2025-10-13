#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析页码特征，优化过滤算法
"""

import cv2
import numpy as np
from pathlib import Path

class PageNumberAnalyzer:
    """页码分析器"""

    def __init__(self):
        self.page_num_samples = [
            "data/extracted_images/page_16_detected_1.png",
            "data/extracted_images/page_17_detected_2.png"
        ]

    def analyze_page_number_features(self):
        """分析页码图像的特征"""
        print("=== 页码特征分析 ===\n")

        page_num_features = []

        for img_path in self.page_num_samples:
            if not Path(img_path).exists():
                print(f"文件不存在: {img_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue

            features = self._extract_features(img, Path(img_path).name)
            page_num_features.append(features)

        # 总结特征模式
        if page_num_features:
            self._summarize_features(page_num_features)

        return page_num_features

    def _extract_features(self, img, filename):
        """提取图像特征"""
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

        # 文字特征（页码通常是简单的数字）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # 小组件分析（文字特征）
        small_components = 0
        large_components = 0
        for i in range(1, num_labels):
            comp_area = stats[i, cv2.CC_STAT_AREA]
            if 10 < comp_area < 100:  # 小组件，可能是文字笔画
                small_components += 1
            elif comp_area > 500:  # 大组件，可能是图形
                large_components += 1

        features = {
            'filename': filename,
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'mean_brightness': mean_brightness,
            'variance': variance,
            'edge_density': edge_density,
            'small_components': small_components,
            'large_components': large_components,
            'text_likeness': small_components / max(num_labels - 1, 1)
        }

        self._print_features(features)
        return features

    def _print_features(self, features):
        """打印特征"""
        print(f"文件: {features['filename']}")
        print(f"  尺寸: {features['width']}x{features['height']}")
        print(f"  长宽比: {features['aspect_ratio']:.2f}")
        print(f"  面积: {features['area']:,} 像素")
        print(f"  平均亮度: {features['mean_brightness']:.1f}")
        print(f"  方差: {features['variance']:.1f}")
        print(f"  边缘密度: {features['edge_density']:.4f}")
        print(f"  小组件数量: {features['small_components']}")
        print(f"  大组件数量: {features['large_components']}")
        print(f"  文字相似度: {features['text_likeness']:.3f}")
        print()

    def _summarize_features(self, features_list):
        """总结特征模式"""
        print("=== 特征模式总结 ===")

        # 计算平均值
        avg_width = np.mean([f['width'] for f in features_list])
        avg_height = np.mean([f['height'] for f in features_list])
        avg_aspect = np.mean([f['aspect_ratio'] for f in features_list])
        avg_area = np.mean([f['area'] for f in features_list])
        avg_text_likeness = np.mean([f['text_likeness'] for f in features_list])

        print(f"平均尺寸: {avg_width:.0f}x{avg_height:.0f}")
        print(f"平均长宽比: {avg_aspect:.2f}")
        print(f"平均面积: {avg_area:.0f} 像素")
        print(f"平均文字相似度: {avg_text_likeness:.3f}")

        # 页码判断标准
        print("\n=== 页码判断标准 ===")
        print("基于分析结果，建议的过滤条件:")
        print(f"1. 尺寸过滤: 宽度 < 200 且 高度 < 150")
        print(f"2. 面积过滤: 面积 < 30,000 像素")
        print(f"3. 长宽比过滤: 长宽比 > 1.5 (横向矩形)")
        print(f"4. 位置过滤: 位于页面底部 20% 区域")
        print(f"5. 文字特征: 文字相似度 > 0.3")

    def generate_filtering_rules(self):
        """生成过滤规则代码"""
        rules_code = '''
    def _is_likely_page_number(self, roi: np.ndarray, bbox: tuple, page_width: int, page_height: int) -> bool:
        """判断是否为页码"""
        try:
            x, y, w, h = bbox

            # 1. 尺寸检查 - 页码通常较小
            if w > 200 or h > 150:
                return False

            # 2. 面积检查
            if w * h > 30000:
                return False

            # 3. 长宽比检查 - 页码通常是横向矩形
            aspect_ratio = w / h if h > 0 else 1.0
            if aspect_ratio < 1.5:
                return False

            # 4. 位置检查 - 页码通常在页面底部
            relative_y = (y + h) / page_height
            if relative_y < 0.8:  # 不在底部20%区域
                return False

            # 5. 文字特征检查
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            small_components = 0
            for i in range(1, num_labels):
                if 10 < stats[i, cv2.CC_STAT_AREA] < 100:
                    small_components += 1

            text_likeness = small_components / max(num_labels - 1, 1)

            # 6. 简单性检查 - 页码通常结构简单
            variance = np.var(gray)
            if variance > 1000:  # 太复杂
                return False

            return text_likeness > 0.3

        except Exception:
            return False
'''
        return rules_code

def main():
    """主函数"""
    analyzer = PageNumberAnalyzer()

    # 分析页码特征
    features = analyzer.analyze_page_number_features()

    # 生成过滤规则代码
    print("=== 生成的过滤规则代码 ===")
    print(analyzer.generate_filtering_rules())

if __name__ == "__main__":
    main()