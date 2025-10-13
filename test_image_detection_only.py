#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门测试图像检测功能 - 跳过OCR处理
"""

import sys
sys.path.append('src')

import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDetectionTester:
    """图像检测测试器 - 跳过OCR"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.output_dir = Path("data/detection_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)

    def test_detection_on_pages(self, target_pages: list):
        """在指定页面测试图像检测"""
        logger.info(f"测试PDF图像检测: {self.pdf_path}")
        logger.info(f"目标页面: {target_pages}")

        try:
            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF总页数: {total_pages}")

            results = []

            for page_num in target_pages:
                if page_num > total_pages:
                    logger.warning(f"PDF只有{total_pages}页，跳过第{page_num}页")
                    continue

                logger.info(f"\n处理第{page_num}页...")

                # 获取页面图像
                page = doc.load_page(page_num - 1)
                pix = page.get_pixmap(dpi=300)  # 适中的分辨率
                img_data = pix.tobytes("png")
                pix = None

                # 转换为OpenCV格式
                nparr = np.frombuffer(img_data, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # 测试新的矩形边框检测
                logger.info("  应用新的矩形边框检测...")
                rectangular_regions = self._test_rectangular_detection(img_cv, page_num)

                # 测试区域合并
                logger.info("  测试区域合并...")
                merged_regions = self._test_region_merging(rectangular_regions)

                # 验证完整性
                logger.info("  验证区域完整性...")
                validated_regions = self._test_completeness_validation(img_cv, merged_regions)

                # 保存结果
                page_result = {
                    'page_num': page_num,
                    'rectangular_regions': len(rectangular_regions),
                    'merged_regions': len(merged_regions),
                    'validated_regions': len(validated_regions),
                    'detected_images': validated_regions
                }
                results.append(page_result)

                logger.info(f"  结果: {len(rectangular_regions)}个矩形区域, "
                          f"{len(merged_regions)}个合并区域, "
                          f"{len(validated_regions)}个验证通过")

                # 保存可视化结果
                self._save_visualization(img_cv, validated_regions, page_num)

            doc.close()
            return results

        except Exception as e:
            logger.error(f"测试失败: {e}")
            if 'doc' in locals():
                doc.close()
            raise

    def _test_rectangular_detection(self, image: np.ndarray, page_num: int) -> list:
        """测试矩形边框检测"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 简化版的矩形检测
            edges = cv2.Canny(gray, 50, 150)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rectangular_regions = []

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                # 基础筛选
                if area < 5000:  # 面积太小
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

                rectangular_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'rectangularity': rectangularity
                })

            logger.info(f"    检测到 {len(rectangular_regions)} 个矩形区域")
            return rectangular_regions

        except Exception as e:
            logger.error(f"矩形检测失败: {e}")
            return []

    def _test_region_merging(self, regions: list) -> list:
        """测试区域合并"""
        if not regions:
            return []

        # 简单的合并策略
        merged = []
        for region in regions:
            x1, y1, w1, h1 = region['bbox']

            # 检查是否与已合并的区域重叠
            overlap_found = False
            for i, kept in enumerate(merged):
                x2, y2, w2, h2 = kept['bbox']

                # 计算重叠面积
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                if overlap_area > 0:
                    # 有重叠，合并边界
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y

                    merged[i]['bbox'] = (new_x, new_y, new_w, new_h)
                    overlap_found = True
                    break

            if not overlap_found:
                merged.append(region)

        return merged

    def _test_completeness_validation(self, image: np.ndarray, regions: list) -> list:
        """测试完整性验证"""
        validated = []

        for region in regions:
            x, y, w, h = region['bbox']
            roi = image[y:y+h, x:x+w]

            # 简单的完整性检查
            if self._simple_completeness_check(roi):
                validated.append(region)

        return validated

    def _simple_completeness_check(self, roi: np.ndarray) -> bool:
        """简单的完整性检查"""
        try:
            height, width = roi.shape[:2]

            # 基本尺寸检查
            if width < 50 or height < 50:
                return False

            # 方差检查
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray)

            if variance < 100:  # 内容太平淡
                return False

            return True

        except Exception:
            return False

    def _save_visualization(self, image: np.ndarray, regions: list, page_num: int):
        """保存可视化结果"""
        result_img = image.copy()

        # 绘制检测到的区域
        for i, region in enumerate(regions):
            x, y, w, h = region['bbox']
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

            # 添加标签
            label = f"Region {i+1}"
            cv2.putText(result_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)

        # 保存结果
        output_path = self.output_dir / f"page_{page_num}_detection.png"
        cv2.imwrite(str(output_path), result_img)
        logger.info(f"    保存可视化结果: {output_path}")

def main():
    """主函数"""
    pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"

    tester = ImageDetectionTester(pdf_path)

    # 测试第5页和第16页
    results = tester.test_detection_on_pages([5, 16])

    # 输出总结
    logger.info("\n=== 测试总结 ===")
    for result in results:
        logger.info(f"第{result['page_num']}页: "
                   f"{result['rectangular_regions']}矩形区域 → "
                   f"{result['merged_regions']}合并区域 → "
                   f"{result['validated_regions']}验证通过")

if __name__ == "__main__":
    main()