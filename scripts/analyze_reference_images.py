#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析参考图像特征 - 为优化PDF图像提取提供参考
分析人物半身像和肿瘤图像的特征
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class ReferenceImageAnalyzer:
    """参考图像特征分析器"""

    def __init__(self):
        self.portrait_path = "data/人物半身像.png"
        self.tumor_path = "data/肿瘤图形.png"

    def analyze_portrait_features(self):
        """分析人物半身像特征"""
        img = cv2.imread(self.portrait_path)
        if img is None:
            print(f"无法读取图像: {self.portrait_path}")
            return None

        features = self._extract_image_features(img, "人物半身像")
        return features

    def analyze_tumor_features(self):
        """分析肿瘤图像特征"""
        img = cv2.imread(self.tumor_path)
        if img is None:
            print(f"无法读取图像: {self.tumor_path}")
            return None

        features = self._extract_image_features(img, "肿瘤图像")
        return features

    def _extract_image_features(self, img, img_type):
        """提取图像特征"""
        height, width = img.shape[:2]

        # 基本几何特征
        aspect_ratio = width / height
        area = width * height

        # 颜色空间分析
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 颜色特征
        features = {
            'type': img_type,
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'gray_variance': np.var(gray),
            'color_features': self._analyze_color_features(hsv),
            'edge_features': self._analyze_edge_features(gray),
            'border_features': self._analyze_border_features(img, gray),
            'shape_features': self._analyze_shape_features(gray)
        }

        return features

    def _analyze_color_features(self, hsv):
        """分析颜色特征"""
        height, width = hsv.shape[:2]

        # 人物半身像：肤色检测
        # 肿瘤图像：医学染色颜色检测

        # 肤色范围 (HSV)
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.count_nonzero(skin_mask) / (width * height)

        # 红色范围 (血液、组织)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_ratio = (np.count_nonzero(red_mask1) + np.count_nonzero(red_mask2)) / (width * height)

        # 蓝色范围 (医学染色、边框)
        lower_blue = np.array([100, 30, 50])
        upper_blue = np.array([130, 200, 200])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.count_nonzero(blue_mask) / (width * height)

        # 紫色范围 (某些医学染色)
        lower_purple = np.array([140, 30, 50])
        upper_purple = np.array([160, 255, 255])
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        purple_ratio = np.count_nonzero(purple_mask) / (width * height)

        # 灰色/中性色 (边框)
        lower_gray = np.array([0, 0, 100])
        upper_gray = np.array([180, 50, 200])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        gray_ratio = np.count_nonzero(gray_mask) / (width * height)

        return {
            'skin_ratio': skin_ratio,
            'red_ratio': red_ratio,
            'blue_ratio': blue_ratio,
            'purple_ratio': purple_ratio,
            'gray_ratio': gray_ratio,
            'hsv_variance': np.var(hsv, axis=(0,1))
        }

    def _analyze_edge_features(self, gray):
        """分析边缘特征"""
        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])

        # Sobel边缘检测
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_density = np.count_nonzero(sobel_magnitude > 50) / (gray.shape[0] * gray.shape[1])

        return {
            'canny_edge_density': edge_density,
            'sobel_edge_density': sobel_density,
            'edge_variance': np.var(edges)
        }

    def _analyze_border_features(self, img, gray):
        """分析边框特征 - 重点分析肿瘤图像的方框"""
        height, width = img.shape[:2]

        # 提取边缘区域 (10% 边界)
        border_width = int(min(width, height) * 0.1)

        # 四个边框区域
        top_border = img[0:border_width, :]
        bottom_border = img[height-border_width:height, :]
        left_border = img[:, 0:border_width]
        right_border = img[:, width-border_width:width]

        borders = [top_border, bottom_border, left_border, right_border]
        border_names = ['top', 'bottom', 'left', 'right']

        border_features = {}

        for i, (border, name) in enumerate(zip(borders, border_names)):
            border_gray = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)
            border_hsv = cv2.cvtColor(border, cv2.COLOR_BGR2HSV)

            # 检测边框颜色 (蓝色/灰色)
            lower_blue = np.array([100, 30, 50])
            upper_blue = np.array([130, 200, 200])
            blue_mask = cv2.inRange(border_hsv, lower_blue, upper_blue)
            blue_ratio = np.count_nonzero(blue_mask) / (border.shape[0] * border.shape[1])

            # 检测灰色边框
            lower_gray = np.array([0, 0, 100])
            upper_gray = np.array([180, 50, 200])
            gray_mask = cv2.inRange(border_hsv, lower_gray, upper_gray)
            gray_ratio = np.count_nonzero(gray_mask) / (border.shape[0] * border.shape[1])

            # 边缘密度
            border_edges = cv2.Canny(border_gray, 50, 150)
            border_edge_density = np.count_nonzero(border_edges) / (border.shape[0] * border.shape[1])

            border_features[name] = {
                'blue_ratio': blue_ratio,
                'gray_ratio': gray_ratio,
                'edge_density': border_edge_density,
                'uniformity': np.var(border_gray)
            }

        # 计算整体边框特征
        avg_blue_ratio = np.mean([border_features[name]['blue_ratio'] for name in border_names])
        avg_gray_ratio = np.mean([border_features[name]['gray_ratio'] for name in border_names])
        avg_edge_density = np.mean([border_features[name]['edge_density'] for name in border_names])

        # 判断是否有明显边框
        has_border = (avg_blue_ratio > 0.1) or (avg_gray_ratio > 0.15)

        return {
            'border_details': border_features,
            'avg_blue_ratio': avg_blue_ratio,
            'avg_gray_ratio': avg_gray_ratio,
            'avg_edge_density': avg_edge_density,
            'has_border': has_border
        }

    def _analyze_shape_features(self, gray):
        """分析形状特征"""
        # 阈值分割
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 查找连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if num_labels <= 1:
            return {'num_components': 0, 'largest_area': 0, 'shape_complexity': 0}

        # 找到最大的组件
        largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_area = stats[largest_idx, cv2.CC_STAT_AREA]

        # 计算形状复杂度（周长面积比）
        component_mask = (labels == largest_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0 and largest_area > 0:
                circularity = 4 * np.pi * largest_area / (perimeter ** 2)
                shape_complexity = 1.0 - circularity  # 复杂度越高，圆形度越低
            else:
                shape_complexity = 0
        else:
            shape_complexity = 0

        return {
            'num_components': num_labels - 1,  # 排除背景
            'largest_area': largest_area,
            'shape_complexity': shape_complexity,
            'circularity': circularity if 'circularity' in locals() else 0
        }

    def compare_features(self):
        """对比两种图像的特征"""
        print("=== 参考图像特征分析 ===\n")

        portrait_features = self.analyze_portrait_features()
        tumor_features = self.analyze_tumor_features()

        if not portrait_features or not tumor_features:
            return

        # 打印对比分析
        print("人物半身像特征:")
        self._print_features(portrait_features)

        print("\n" + "="*50 + "\n")

        print("肿瘤图像特征:")
        self._print_features(tumor_features)

        print("\n" + "="*50 + "\n")

        print("特征对比总结:")
        self._compare_summary(portrait_features, tumor_features)

        return portrait_features, tumor_features

    def _print_features(self, features):
        """打印特征信息"""
        print(f"类型: {features['type']}")
        print(f"尺寸: {features['width']}x{features['height']} (比例: {features['aspect_ratio']:.3f})")
        print(f"面积: {features['area']:,} 像素")
        print(f"灰度方差: {features['gray_variance']:.1f}")

        color = features['color_features']
        print(f"颜色特征:")
        print(f"  肤色比例: {color['skin_ratio']:.3f}")
        print(f"  红色比例: {color['red_ratio']:.3f}")
        print(f"  蓝色比例: {color['blue_ratio']:.3f}")
        print(f"  紫色比例: {color['purple_ratio']:.3f}")
        print(f"  灰色比例: {color['gray_ratio']:.3f}")

        edge = features['edge_features']
        print(f"边缘特征:")
        print(f"  Canny边缘密度: {edge['canny_edge_density']:.4f}")
        print(f"  Sobel边缘密度: {edge['sobel_edge_density']:.4f}")

        border = features['border_features']
        print(f"边框特征:")
        print(f"  有边框: {border['has_border']}")
        print(f"  平均蓝色比例: {border['avg_blue_ratio']:.3f}")
        print(f"  平均灰色比例: {border['avg_gray_ratio']:.3f}")
        print(f"  平均边缘密度: {border['avg_edge_density']:.3f}")

        shape = features['shape_features']
        print(f"形状特征:")
        print(f"  组件数量: {shape['num_components']}")
        print(f"  最大组件面积: {shape['largest_area']:,}")
        print(f"  形状复杂度: {shape['shape_complexity']:.3f}")
        print(f"  圆形度: {shape['circularity']:.3f}")

    def _compare_summary(self, portrait, tumor):
        """对比总结"""
        print("关键区别:")

        # 长宽比
        p_ratio = portrait['aspect_ratio']
        t_ratio = tumor['aspect_ratio']
        print(f"长宽比 - 人物: {p_ratio:.3f}, 肿瘤: {t_ratio:.3f}")

        # 肤色 vs 医学颜色
        p_skin = portrait['color_features']['skin_ratio']
        t_red = tumor['color_features']['red_ratio']
        t_blue = tumor['color_features']['blue_ratio']
        print(f"颜色特征 - 人物肤色: {p_skin:.3f}, 肿瘤红色: {t_red:.3f}, 肿瘤蓝色: {t_blue:.3f}")

        # 边框特征
        p_border = portrait['border_features']['has_border']
        t_border = tumor['border_features']['has_border']
        print(f"边框特征 - 人物有边框: {p_border}, 肿瘤有边框: {t_border}")

        if t_border:
            t_blue_border = tumor['border_features']['avg_blue_ratio']
            t_gray_border = tumor['border_features']['avg_gray_ratio']
            print(f"  肿瘤边框 - 蓝色: {t_blue_border:.3f}, 灰色: {t_gray_border:.3f}")

        print("\n识别要点:")
        print("1. 人物半身像：肤色比例高，长宽比接近1，无明显边框")
        print("2. 肿瘤图像：有清晰边框（蓝色/灰色），医学染色颜色，长宽比适中")

def main():
    """主函数"""
    analyzer = ReferenceImageAnalyzer()
    analyzer.compare_features()

if __name__ == "__main__":
    main()