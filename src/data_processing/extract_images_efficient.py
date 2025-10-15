#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高效图像提取工具 - 专门提取第5、6、16、17页的图像
跳过OCR处理，专注于图像提取
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientImageExtractor:
    """高效图像提取器"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.output_dir = Path("data/extracted_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_images_from_pages(self, target_pages: list):
        """从指定页面提取图像"""
        logger.info(f"开始提取PDF图像: {self.pdf_path}")
        logger.info(f"目标页面: {target_pages}")

        try:
            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF总页数: {total_pages}")

            extracted_images = []

            for page_num in target_pages:
                if page_num > total_pages:
                    logger.warning(f"PDF只有{total_pages}页，跳过第{page_num}页")
                    continue

                logger.info(f"\n提取第{page_num}页图像...")

                page = doc.load_page(page_num - 1)  # 页面索引从0开始

                # 方法1: 提取PDF中的嵌入图像
                # embedded_images = self._extract_embedded_images(page, page_num)

                # 方法2: 通过页面截图检测图像区域
                screenshot_images = self._extract_from_screenshot(page, page_num)

                # 合并结果
                # page_images = embedded_images + screenshot_images
                page_images = screenshot_images
                extracted_images.extend(page_images)

                logger.info(f"第{page_num}页提取完成: {len(page_images)}张图像")

            doc.close()

            logger.info(f"\n提取完成！总共提取 {len(extracted_images)} 张图像")
            return extracted_images

        except Exception as e:
            logger.error(f"提取失败: {e}")
            if 'doc' in locals():
                doc.close()
            raise

    def _extract_embedded_images(self, page, page_num: int) -> list:
        """提取页面中嵌入的图像"""
        images = []
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)

                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    # 转换为numpy数组
                    img_data = pix.tobytes("png")
                    nparr = np.frombuffer(img_data, np.uint8)
                    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    # 保存图像
                    filename = f"page_{page_num}_embedded_{img_index}.png"
                    filepath = self.output_dir / filename
                    cv2.imwrite(str(filepath), img_cv)

                    images.append({
                        'page': page_num,
                        'type': 'embedded',
                        'index': img_index,
                        'filename': filename,
                        'path': str(filepath),
                        'size': img_cv.shape
                    })

                    logger.info(f"  提取嵌入图像 {img_index}: {img_cv.shape}")

                pix = None

            except Exception as e:
                logger.warning(f"  提取嵌入图像 {img_index} 失败: {e}")
                continue

        return images

    def _extract_from_screenshot(self, page, page_num: int) -> list:
        """通过页面截图检测和提取图像区域"""
        try:
            # 获取高质量页面截图
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            pix = None

            # 转换为OpenCV格式
            nparr = np.frombuffer(img_data, np.uint8)
            page_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 检测图像区域
            detected_regions = self._detect_image_regions(page_image, page_num)

            # 提取检测到的区域
            extracted_images = []
            for i, region in enumerate(detected_regions):
                x, y, w, h = region['bbox']

                # 提取ROI
                roi = page_image[y:y+h, x:x+w]

                # 保存提取的图像
                filename = f"page_{page_num}_detected_{i}.png"
                filepath = self.output_dir / filename
                cv2.imwrite(str(filepath), roi)

                extracted_images.append({
                    'page': page_num,
                    'type': 'detected',
                    'index': i,
                    'filename': filename,
                    'path': str(filepath),
                    'size': roi.shape,
                    'bbox': region['bbox'],
                    'confidence': region.get('confidence', 1.0)
                })

                logger.info(f"  检测提取图像 {i}: {roi.shape} (置信度: {region.get('confidence', 1.0):.2f})")

            return extracted_images

        except Exception as e:
            logger.error(f"页面截图提取失败: {e}")
            return []

    def _detect_image_regions(self, image: np.ndarray, page_num: int) -> list:
        """检测图像区域 - 智能过滤页码"""
        try:
            page_height, page_width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 使用我们优化过的矩形检测方法
            edges = cv2.Canny(gray, 50, 150)

            # 形态学操作增强矩形特征
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_regions = []

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                # 基础筛选条件 - 加强尺寸过滤
                if area < 15000:  # 面积太小
                    continue

                # 新增：极小小图直接过滤 - 页码通常是这个尺寸范围
                if w < 200 or h < 200 or w * h < 40000:  # 极小的矩形，很可能是页码
                    logger.info(f"  过滤极小小图: 尺寸{w}x{h}, 面积{w*h}")
                    continue

                aspect_ratio = w / h if h > 0 else 1.0
                if not (0.4 <= aspect_ratio <= 2.5):  # 宽高比不合适
                    continue

                # 矩形度检测
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)

                if box_area > 0:
                    rectangularity = area / box_area
                else:
                    rectangularity = 0

                if rectangularity < 0.7:  # 矩形度不够
                    continue

                # 页码过滤 - 关键改进
                roi = image[y:y+h, x:x+w]
                if self._is_likely_page_number(roi, (x, y, w, h), page_width, page_height):
                    logger.info(f"  过滤掉可能的页码: 位置({x},{y}), 尺寸{w}x{h}")
                    continue

                # 内容验证
                if self._validate_content(roi):
                    detected_regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'rectangularity': rectangularity,
                        'confidence': rectangularity * 0.8 + min(area / 100000, 0.2)
                    })

            # 按置信度排序
            detected_regions.sort(key=lambda x: x['confidence'], reverse=True)

            # 每页最多保留3个最可能的图像
            return detected_regions[:3]

        except Exception as e:
            logger.error(f"图像区域检测失败: {e}")
            return[]

    def _validate_content(self, roi: np.ndarray) -> bool:
        """验证ROI内容"""
        try:
            height, width = roi.shape[:2]

            # 基本尺寸检查
            if width < 50 or height < 50:
                return False

            # 转换为灰度图
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 方差检查 - 确保内容有足够的变化
            variance = np.var(gray)
            if variance < 200:
                return False

            # 边缘密度检查
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (width * height)

            if edge_density < 0.01 or edge_density > 0.5:
                return False

            return True

        except Exception:
            return False

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

            # 综合判断 - 放宽文字要求，加强其他特征
            is_page_number = (
                (aspect_ratio > 1.5 and relative_y > 0.8 and w * h < 30000) or  # 位置+尺寸特征
                (text_likeness > 0.2 and w < 200 and h < 150) or  # 文字特征+小尺寸
                (aspect_ratio > 1.7 and w * h < 20000)  # 极端长宽比+小面积
            )

            if is_page_number:
                logger.info(f"    检测到页码特征: 尺寸{w}x{h}, 位置{relative_y:.2f}, 文字相似度{text_likeness:.2f}")

            return is_page_number

        except Exception:
            return False

def main():
    """主函数"""
    pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"

    if not os.path.exists(pdf_path):
        logger.error(f"PDF文件不存在: {pdf_path}")
        return

    extractor = EfficientImageExtractor(pdf_path)

    # 提取第5、6、16、17页的图像
    # target_pages = [5, 6, 16, 17]
    target_pages = list(range(1,225))
    extracted_images = extractor.extract_images_from_pages(target_pages)

    # 输出结果
    logger.info(f"\n=== 提取结果总结 ===")
    logger.info(f"总共提取了 {len(extracted_images)} 张图像")

    # 按页面分组
    by_page = {}
    for img in extracted_images:
        page = img['page']
        if page not in by_page:
            by_page[page] = []
        by_page[page].append(img)

    for page in sorted(by_page.keys()):
        logger.info(f"\n第{page}页:")
        for img in by_page[page]:
            logger.info(f"  {img['type']} {img['index']}: {img['filename']} ({img['size'][1]}x{img['size'][0]})")

    # 显示文件位置
    logger.info(f"\n提取的图像保存在: {extractor.output_dir.absolute()}")

if __name__ == "__main__":
    main()