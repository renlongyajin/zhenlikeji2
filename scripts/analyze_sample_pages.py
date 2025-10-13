#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析指定样本页面的图片检测问题
"""

import fitz
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sample_pages():
    """分析样本页面"""
    pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"
    sample_pages = [5, 10, 13, 16, 17]  # 第5,10,13,16,17页（1-indexed）

    doc = fitz.open(pdf_path)

    for page_num in sample_pages:
        logger.info(f"\n=== 分析第{page_num}页 ===")

        page = doc.load_page(page_num - 1)  # 0-indexed

        # 获取页面图片
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        pix = None

        # 转换为OpenCV格式
        nparr = np.frombuffer(img_data, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        height, width = img_cv.shape[:2]
        logger.info(f"页面尺寸: {width}x{height}")

        # 分析页面布局
        analyze_page_layout(img_cv, page_num)

        # 保存完整页面图片用于参考
        output_path = f"data/processed/sample_page_{page_num}.png"
        cv2.imwrite(output_path, img_cv)
        logger.info(f"保存完整页面: {output_path}")

def analyze_page_layout(image: np.ndarray, page_num: int):
    """分析页面布局"""
    height, width = image.shape[:2]

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger.info(f"检测到{len(contours)}个轮廓")

    # 分析每个轮廓
    meaningful_regions = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 1000:  # 只考虑较大的区域
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # 提取区域
            roi = image[y:y+h, x:x+w]

            # 分析区域特征
            features = analyze_region_features(roi, (x, y, w, h))

            if features['is_meaningful']:
                meaningful_regions.append({
                    'index': i,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'features': features
                })

                logger.info(f"  区域{i}: 位置({x},{y}) 尺寸{w}x{h} 面积{area} 长宽比{aspect_ratio:.2f}")
                logger.info(f"    特征: {features}")

    # 保存分析结果图片
    result_img = image.copy()
    for region in meaningful_regions:
        x, y, w, h = region['bbox']
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_img, f"R{region['index']}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    result_path = f"data/processed/sample_page_{page_num}_analysis.png"
    cv2.imwrite(result_path, result_img)
    logger.info(f"保存分析结果: {result_path}")

    return meaningful_regions

def analyze_region_features(roi: np.ndarray, bbox: tuple) -> dict:
    """分析区域特征"""
    x, y, w, h = bbox

    # 基础统计
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)

    # 边缘密度
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (w * h)

    # 颜色分析
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 检查红色区域（可能是医学图像）
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_ratio = (np.count_nonzero(red_mask1) + np.count_nonzero(red_mask2)) / (w * h)

    # 检查是否为纯色（可能是文字背景）
    is_uniform = variance < 50

    # 检查纹理复杂度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    texture_complexity = np.mean(sobel_combined)

    # 判断是否为有意义的内容
    is_meaningful = (
        variance > 20 and  # 不是太均匀
        edge_density > 0.005 and  # 有一定的边缘
        not is_uniform  # 不是纯色的
    )

    return {
        'variance': variance,
        'edge_density': edge_density,
        'red_ratio': red_ratio,
        'is_uniform': is_uniform,
        'texture_complexity': texture_complexity,
        'is_meaningful': is_meaningful
    }

def load_reference_images():
    """加载参考图像"""
    reference_dir = Path("data")

    portrait_path = reference_dir / "人物头像.png"
    tumor_path = reference_dir / "肿瘤图形.png"

    references = {}

    if portrait_path.exists():
        portrait_img = cv2.imread(str(portrait_path))
        if portrait_img is not None:
            references['portrait'] = portrait_img
            logger.info(f"加载人物头像参考: {portrait_path}")

    if tumor_path.exists():
        tumor_img = cv2.imread(str(tumor_path))
        if tumor_img is not None:
            references['tumor'] = tumor_img
            logger.info(f"加载肿瘤图形参考: {tumor_path}")

    return references

if __name__ == "__main__":
    # 首先加载参考图像
    references = load_reference_images()
    logger.info(f"加载了{len(references)}个参考图像")

    # 分析样本页面
    analyze_sample_pages()