#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门调试漏检页面 - 分析第5页和第56页
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SpecificPageDebugger:
    """专门页面调试器"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.debug_dir = Path("data/debug_specific")
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def debug_pages(self, target_pages: list):
        """调试特定页面"""
        logger.info(f"开始调试特定页面: {target_pages}")

        try:
            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF总页数: {total_pages}")

            for page_num in target_pages:
                if page_num > total_pages:
                    logger.warning(f"PDF只有{total_pages}页，无法处理第{page_num}页")
                    continue

                logger.info(f"\n=== 调试第{page_num}页 ===")

                page = doc.load_page(page_num - 1)

                # 获取页面截图
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                pix = None

                nparr = np.frombuffer(img_data, np.uint8)
                page_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # 详细分析
                self._detailed_analysis(page_image, page_num)

            doc.close()

        except Exception as e:
            logger.error(f"调试失败: {e}")
            if 'doc' in locals():
                doc.close()
            raise

    def _detailed_analysis(self, image: np.ndarray, page_num: int):
        """详细分析单个页面"""
        page_height, page_width = image.shape[:2]
        logger.info(f"页面尺寸: {page_width}x{page_height}")

        # 保存原始页面图像用于分析
        original_path = self.debug_dir / f"page_{page_num}_original.png"
        cv2.imwrite(str(original_path), image)
        logger.info(f"保存原始页面: {original_path}")

        # 1. 多尺度边缘检测
        logger.info("1. 开始边缘检测...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 80, 200)
        combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))

        # 保存边缘检测结果
        edges_path = self.debug_dir / f"page_{page_num}_edges.png"
        cv2.imwrite(str(edges_path), combined_edges)
        logger.info(f"保存边缘检测结果: {edges_path}")

        # 2. 形态学操作
        logger.info("2. 形态学操作...")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morphed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)

        morphed_path = self.debug_dir / f"page_{page_num}_morphed.png"
        cv2.imwrite(str(morphed_path), morphed)
        logger.info(f"保存形态学结果: {morphed_path}")

        # 3. 轮廓检测与分析
        logger.info("3. 轮廓检测...")
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"检测到 {len(contours)} 个轮廓")

        # 详细分析每个轮廓
        valid_candidates = []
        debug_image = image.copy()

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            logger.info(f"\n  轮廓 {i}:")
            logger.info(f"    位置: ({x}, {y}), 尺寸: {w}x{h}")
            logger.info(f"    面积: {area}")
            logger.info(f"    长宽比: {w/h if h > 0 else 1.0:.2f}")

            # 检查基础条件
            if area < 5000:
                logger.info(f"    ❌ 面积太小 (<5000)")
                continue

            # 极小小图过滤（放宽条件）
            if w < 100 and h < 100:
                logger.info(f"    ❌ 尺寸太小 (<100x100)")
                continue

            aspect_ratio = w / h if h > 0 else 1.0
            logger.info(f"    长宽比: {aspect_ratio:.2f}")

            # 矩形度检测
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box_area = cv2.contourArea(box)
            rectangularity = area / box_area if box_area > 0 else 0
            logger.info(f"    矩形度: {rectangularity:.3f}")

            if rectangularity < 0.5:
                logger.info(f"    ❌ 矩形度太低 (<0.5)")
                continue

            # 绘制轮廓
            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 2)
            cv2.putText(debug_image, f"{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 内容分析
            roi = image[y:y+h, x:x+w]
            content_valid = self._analyze_content_detailed(roi, (x, y, w, h), page_width, page_height)

            if content_valid:
                logger.info(f"    ✅ 通过内容验证")
                valid_candidates.append({
                    'index': i,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'rectangularity': rectangularity
                })
            else:
                logger.info(f"    ❌ 内容验证失败")

        # 保存调试图像
        debug_path = self.debug_dir / f"page_{page_num}_debug.png"
        cv2.imwrite(str(debug_path), debug_image)
        logger.info(f"保存调试图像: {debug_path}")

        # 保存有效候选区域
        if valid_candidates:
            logger.info(f"\n✅ 找到 {len(valid_candidates)} 个有效候选区域:")
            for candidate in valid_candidates:
                logger.info(f"  候选 {candidate['index']}: {candidate['bbox']}")

                # 保存候选图像
                x, y, w, h = candidate['bbox']
                candidate_img = image[y:y+h, x:x+w]
                candidate_path = self.debug_dir / f"page_{page_num}_candidate_{candidate['index']}.png"
                cv2.imwrite(str(candidate_path), candidate_img)
                logger.info(f"  保存候选图像: {candidate_path}")
        else:
            logger.warning(f"第{page_num}页: 未找到有效候选区域")

    def _analyze_content_detailed(self, roi: np.ndarray, bbox: tuple, page_width: int, page_height: int) -> bool:
        """详细内容分析"""
        try:
            x, y, w, h = bbox

            # 基本尺寸检查（放宽条件）
            if w < 80 or h < 80:
                return False

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 方差检查（放宽）
            variance = np.var(gray)
            logger.info(f"    方差: {variance:.1f}")
            if variance < 50:
                return False

            # 边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (w * h)
            logger.info(f"    边缘密度: {edge_density:.4f}")

            if edge_density < 0.01 or edge_density > 0.4:
                return False

            # 位置分析（针对医学文档）
            relative_y = (y + h/2) / page_height
            relative_x = (x + w/2) / page_width
            logger.info(f"    相对位置: x={relative_x:.2f}, y={relative_y:.2f}")

            # 医学图像通常在页面中部或特定位置，不在极边缘
            if relative_x < 0.05 or relative_x > 0.95 or relative_y < 0.05 or relative_y > 0.95:
                return False

            return True

        except Exception as e:
            logger.error(f"内容分析失败: {e}")
            return False

def main():
    """主函数"""
    pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"

    debugger = SpecificPageDebugger(pdf_path)

    # 调试有问题的页面
    problem_pages = [5, 56]  # 第5页和第56页
    debugger.debug_pages(problem_pages)

if __name__ == "__main__":
    main()