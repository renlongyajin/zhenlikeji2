#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅提取裁剪图像 - 全文档测试版本
只生成切割出来的图片，不提取完整页面图像
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CroppedImageExtractor:
    """仅提取裁剪图像的提取器"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.output_dir = Path("data/cropped_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_cropped_images_all_pages(self):
        """从所有页面提取裁剪的图像"""
        logger.info(f"开始提取裁剪图像: {self.pdf_path}")

        try:
            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF总页数: {total_pages}")

            all_extracted_images = []
            total_cropped = 0

            # 处理所有页面
            for page_num in range(1, total_pages + 1):
                logger.info(f"\n处理第{page_num}/{total_pages}页...")

                page = doc.load_page(page_num - 1)

                # 只进行图像区域检测和裁剪提取
                cropped_images = self._extract_cropped_images_from_page(page, page_num)

                all_extracted_images.extend(cropped_images)
                total_cropped += len(cropped_images)

                if len(cropped_images) > 0:
                    logger.info(f"第{page_num}页提取完成: {len(cropped_images)}张裁剪图像")
                else:
                    logger.info(f"第{page_num}页: 未检测到有效图像")

            doc.close()

            # 最终统计
            logger.info(f"\n=== 全文档提取完成 ===")
            logger.info(f"总页数: {total_pages}")
            logger.info(f"提取裁剪图像: {total_cropped}张")

            self._print_final_statistics(all_extracted_images)

            return all_extracted_images

        except Exception as e:
            logger.error(f"提取失败: {e}")
            if 'doc' in locals():
                doc.close()
            raise

    def _extract_cropped_images_from_page(self, page, page_num: int) -> list:
        """从单个页面提取裁剪的图像"""
        try:
            # 获取页面截图
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            pix = None

            # 转换为OpenCV格式
            nparr = np.frombuffer(img_data, np.uint8)
            page_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 检测图像区域
            detected_regions = self._detect_medical_image_regions(page_image, page_num)

            # 提取裁剪的图像
            cropped_images = []
            for i, region in enumerate(detected_regions):
                x, y, w, h = region['bbox']

                # 提取ROI（裁剪图像）
                roi = page_image[y:y+h, x:x+w]

                # 保存裁剪图像
                filename = f"page_{page_num}_image_{i}.png"
                filepath = self.output_dir / filename
                cv2.imwrite(str(filepath), roi)

                cropped_images.append({
                    'page': page_num,
                    'filename': filename,
                    'path': str(filepath),
                    'size': roi.shape,
                    'bbox': region['bbox'],
                    'confidence': region['confidence'],
                    'image_type': region.get('image_type', 'unknown')
                })

            return cropped_images

        except Exception as e:
            logger.error(f"第{page_num}页处理失败: {e}")
            return []

    def _detect_medical_image_regions(self, image: np.ndarray, page_num: int) -> list:
        """检测医学图像区域 - 专门针对医学文档优化"""
        try:
            page_height, page_width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 多尺度边缘检测 - 针对医学图像优化
            edges1 = cv2.Canny(gray, 30, 100)   # 检测弱边缘
            edges2 = cv2.Canny(gray, 50, 150)   # 中等边缘
            edges = cv2.bitwise_or(edges1, edges2)

            # 形态学操作增强医学图像特征
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_regions = []

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                # 基础筛选 - 医学图像通常较大
                if area < 15000:  # 医学图像面积阈值
                    continue

                # 极小小图过滤 - 关键改进
                if w < 300 or h < 200 or w * h < 40000:
                    logger.debug(f"第{page_num}页: 过滤小图 {w}x{h}")
                    continue

                aspect_ratio = w / h if h > 0 else 1.0
                if not (0.3 <= aspect_ratio <= 3.0):  # 医学图像合理的宽高比范围
                    continue

                # 矩形度检测
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)
                rectangularity = area / box_area if box_area > 0 else 0

                if rectangularity < 0.6:  # 医学图像通常有规整的边界
                    continue

                # 页码过滤
                roi = image[y:y+h, x:x+w]
                if self._is_likely_page_number(roi, (x, y, w, h), page_width, page_height):
                    logger.info(f"第{page_num}页: 过滤页码 {w}x{h} at position ({x},{y})")
                    continue

                # 医学内容验证
                if self._is_medical_content(roi):
                    # 分类医学图像类型
                    image_type = self._classify_medical_image(roi)

                    confidence = rectangularity * 0.6 + min(area / 200000, 0.4)

                    detected_regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'rectangularity': rectangularity,
                        'confidence': confidence,
                        'image_type': image_type
                    })

            # 按置信度排序，每页最多保留4个图像
            detected_regions.sort(key=lambda x: x['confidence'], reverse=True)
            return detected_regions[:4]

        except Exception as e:
            logger.error(f"医学图像检测失败: {e}")
            return []

    def _is_likely_page_number(self, roi: np.ndarray, bbox: tuple, page_width: int, page_height: int) -> bool:
        """判断是否为页码"""
        try:
            x, y, w, h = bbox

            # 快速尺寸过滤 - 页码通常很小
            if w > 250 or h > 180:
                return False

            # 小图过滤 - 关键
            if w < 350 and h < 200 and w * h < 60000:
                # 检查是否是页码特征
                aspect_ratio = w / h if h > 0 else 1.0
                relative_y = (y + h) / page_height

                # 页码特征：小尺寸 + 横向 + 底部位置
                if (aspect_ratio > 1.4 and relative_y > 0.75) or (w < 200 and h < 120):
                    return True

            return False

        except Exception:
            return False

    def _is_medical_content(self, roi: np.ndarray) -> bool:
        """验证是否为医学相关内容"""
        try:
            height, width = roi.shape[:2]

            # 基本尺寸检查
            if width < 100 or height < 100:
                return False

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 方差检查 - 医学图像应有适中复杂度
            variance = np.var(gray)
            if variance < 100 or variance > 5000:
                return False

            # 边缘密度检查
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (width * height)

            if edge_density < 0.02 or edge_density > 0.3:
                return False

            return True

        except Exception:
            return False

    def _classify_medical_image(self, roi: np.ndarray) -> str:
        """简单分类医学图像类型"""
        try:
            height, width = roi.shape[:2]
            aspect_ratio = width / height if height > 0 else 1.0

            # 基于长宽比的简单分类
            if 0.8 <= aspect_ratio <= 2:
                # 接近正方形 - 可能是人物半身像
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # 肤色检测
                lower_skin = np.array([0, 20, 70])
                upper_skin = np.array([20, 255, 255])
                skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                skin_ratio = np.count_nonzero(skin_mask) / (width * height)

                if skin_ratio > 0.05:
                    return 'portrait'
                else:
                    return 'medical_image'

            elif aspect_ratio > 1.5:
                # 横向矩形 - 可能是肿瘤图像或医学图表
                return 'tumor_or_chart'

            else:
                return 'medical_image'

        except Exception:
            return 'unknown'

    def _print_final_statistics(self, extracted_images):
        """打印最终统计信息"""
        if not extracted_images:
            logger.info("未提取到任何图像")
            return

        # 按页面统计
        by_page = {}
        by_type = {}

        for img in extracted_images:
            page = img['page']
            img_type = img['image_type']

            if page not in by_page:
                by_page[page] = 0
            by_page[page] += 1

            if img_type not in by_type:
                by_type[img_type] = 0
            by_type[img_type] += 1

        logger.info(f"\n=== 提取统计 ===")
        logger.info(f"总提取图像数: {len(extracted_images)}")

        logger.info(f"\n按页面分布:")
        for page in sorted(by_page.keys()):
            logger.info(f"  第{page}页: {by_page[page]}张")

        logger.info(f"\n按类型分布:")
        for img_type, count in by_type.items():
            logger.info(f"  {img_type}: {count}张")

def main():
    """主函数"""
    pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"

    if not os.path.exists(pdf_path):
        logger.error(f"PDF文件不存在: {pdf_path}")
        return

    extractor = CroppedImageExtractor(pdf_path)

    # 提取所有页面的裁剪图像
    extracted_images = extractor.extract_cropped_images_all_pages()

    logger.info(f"\n提取完成！所有裁剪图像保存在: {extractor.output_dir.absolute()}")

if __name__ == "__main__":
    main()