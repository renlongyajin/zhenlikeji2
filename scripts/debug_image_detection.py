#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试图片检测 - 详细分析每个步骤
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import fitz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebugImageDetector:
    """调试版图片检测器"""

    def __init__(self):
        pass

    def analyze_page(self, page_image: np.ndarray, page_num: int):
        """详细分析页面"""
        logger.info(f"\n=== 详细分析第{page_num}页 ===")

        height, width = page_image.shape[:2]
        logger.info(f"页面尺寸: {width}x{height}")

        # 1. 检测淡蓝色边框
        blue_regions = self._detect_blue_borders_debug(page_image, page_num)

        # 2. 检测人物头像
        portrait_regions = self._detect_portrait_regions_debug(page_image, page_num)

        # 3. 检测医学图像
        medical_regions = self._detect_medical_images_debug(page_image, page_num)

        logger.info(f"\n总结:")
        logger.info(f"淡蓝色边框区域: {len(blue_regions)}")
        logger.info(f"人物头像区域: {len(portrait_regions)}")
        logger.info(f"医学图像区域: {len(medical_regions)}")

        return blue_regions + portrait_regions + medical_regions

    def _detect_blue_borders_debug(self, image: np.ndarray, page_num: int) -> list:
        """调试版淡蓝色边框检测"""
        logger.info("\n--- 检测淡蓝色边框 ---")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 更宽泛的蓝色范围
        lower_blue = np.array([90, 30, 30])
        upper_blue = np.array([140, 200, 220])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        logger.info(f"蓝色像素数量: {np.count_nonzero(blue_mask)} / {blue_mask.size}")

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blue_morphed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(blue_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"找到{len(contours)}个蓝色轮廓")

        candidates = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            logger.info(f"  轮廓{i}: 面积={area}, 位置=({x},{y}), 尺寸={w}x{h}")

            if area > 2000:  # 降低阈值
                # 检查形状规整性
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)

                shape_score = area / box_area if box_area > 0 else 0
                logger.info(f"    形状评分: {shape_score:.2f}")

                if shape_score > 0.3:  # 降低要求
                    candidates.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': w / h if h > 0 else 1.0,
                        'roi': image[y:y+h, x:x+w],
                        'detection_type': 'blue_border',
                        'confidence': 0.8,
                        'shape_score': shape_score
                    })
                    logger.info(f"    ✓ 保留为候选区域")

        return candidates

    def _detect_portrait_regions_debug(self, image: np.ndarray, page_num: int) -> list:
        """调试版人物头像检测"""
        logger.info("\n--- 检测人物头像区域 ---")

        height, width = image.shape[:2]

        # 定义候选区域（右上角）
        x, y, w, h = int(width * 0.7), int(height * 0.05), int(width * 0.25), int(height * 0.25)
        roi = image[y:y+h, x:x+w]

        logger.info(f"分析右上角区域: 位置({x},{y}), 尺寸{w}x{h}")

        # 分析是否为人物头像
        is_portrait, reason = self._is_likely_portrait_debug(roi)
        logger.info(f"人物头像检测结果: {is_portrait}, 原因: {reason}")

        if is_portrait:
            return [{
                'bbox': (x, y, w, h),
                'area': w * h,
                'aspect_ratio': w / h,
                'roi': roi,
                'detection_type': 'portrait_region',
                'confidence': 0.7
            }]
        return []

    def _is_likely_portrait_debug(self, roi: np.ndarray) -> tuple:
        """判断是否为人物头像（调试版）"""
        height, width = roi.shape[:2]

        # 1. 检查宽高比
        aspect_ratio = width / height
        logger.info(f"  宽高比: {aspect_ratio:.2f} (期望: 0.8-1.2)")
        if not (0.7 <= aspect_ratio <= 1.3):  # 放宽要求
            return False, f"宽高比不合适: {aspect_ratio:.2f}"

        # 2. 分析颜色分布
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        std_hsv = np.std(hsv, axis=(0, 1))
        color_variation = np.mean(std_hsv)
        logger.info(f"  颜色变化: {color_variation:.1f} (期望: <60)")

        if color_variation > 60:  # 放宽要求
            return False, f"颜色变化太大: {color_variation:.1f}"

        # 3. 边缘检测
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (width * height)
        logger.info(f"  边缘密度: {edge_density:.3f} (期望: 0.03-0.25)")

        if not (0.03 <= edge_density <= 0.25):  # 放宽要求
            return False, f"边缘密度不合适: {edge_density:.3f}"

        return True, "符合人物头像特征"

    def _detect_medical_images_debug(self, image: np.ndarray, page_num: int) -> list:
        """调试版医学图像检测"""
        logger.info("\n--- 检测医学图像 ---")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用自适应阈值
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 5)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"找到{len(contours)}个轮廓")

        candidates = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            logger.info(f"  轮廓{i}: 面积={area}, 位置=({x},{y}), 尺寸={w}x{h}, 长宽比={aspect_ratio:.2f}")

            if area > 10000:  # 降低面积阈值
                logger.info(f"    面积足够大: {area}")

                if 0.2 <= aspect_ratio <= 4.0:  # 放宽宽高比要求
                    logger.info(f"    长宽比合适: {aspect_ratio:.2f}")

                    roi = image[y:y+h, x:x+w]

                    if self._is_meaningful_medical_image_debug(roi):
                        logger.info(f"    ✓ 是有意义的医学图像")
                        candidates.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'roi': roi,
                            'detection_type': 'medical_image',
                            'confidence': 0.6
                        })
                    else:
                        logger.info(f"    ✗ 不是有意义的医学图像")

        return candidates

    def _is_meaningful_medical_image_debug(self, roi: np.ndarray) -> bool:
        """判断是否为有意义的医学图像（调试版）"""
        height, width = roi.shape[:2]

        # 1. 不能太均匀
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        logger.info(f"    灰度方差: {variance:.1f} (期望: >20)")

        if variance < 20:  # 降低要求
            return False

        # 2. 检查医学相关颜色
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 红色区域
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        total_red = np.count_nonzero(red_mask1) + np.count_nonzero(red_mask2)
        red_ratio = total_red / (width * height)

        # 蓝色区域
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.count_nonzero(blue_mask) / (width * height)

        # 紫色区域
        lower_purple = np.array([140, 50, 50])
        upper_purple = np.array([160, 255, 255])
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        purple_ratio = np.count_nonzero(purple_mask) / (width * height)

        logger.info(f"    颜色比例 - 红: {red_ratio:.3f}, 蓝: {blue_ratio:.3f}, 紫: {purple_ratio:.3f}")
        logger.info(f"    阈值 - 每种颜色 > 0.01")

        return (red_ratio > 0.01) or (blue_ratio > 0.01) or (purple_ratio > 0.01)

def main():
    """主函数"""
    # 测试样本页面
    pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"
    sample_pages = [5, 10, 13, 16, 17]

    doc = fitz.open(pdf_path)
    detector = DebugImageDetector()

    for page_num in sample_pages:
        logger.info(f"\n{'='*60}")

        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        pix = None

        nparr = np.frombuffer(img_data, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 详细分析
        detected_images = detector.analyze_page(img_cv, page_num)

        # 保存可视化结果
        result_img = img_cv.copy()
        for i, detected in enumerate(detected_images):
            x, y, w, h = detected['bbox']
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

            label = f"{detected['detection_type']}"
            cv2.putText(result_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)

        result_path = f"data/processed/debug_page_{page_num}.png"
        cv2.imwrite(result_path, result_img)
        logger.info(f"保存调试结果: {result_path}")

    doc.close()

if __name__ == "__main__":
    main()