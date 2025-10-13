#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的特定页面提取
"""

import sys
sys.path.append('src')

from extract_cropped_images_optimized import OptimizedCroppedImageExtractor
import logging
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_pages():
    """测试特定页面"""
    pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"

    # 创建优化提取器
    extractor = OptimizedCroppedImageExtractor(pdf_path)

    # 只测试关键页面
    test_pages = [5, 56, 16, 17]  # 之前漏检的页面

    logger.info(f"测试优化后的提取 - 页面: {test_pages}")

    try:
        # 手动处理这些页面
        doc = fitz.open(pdf_path)

        for page_num in test_pages:
            logger.info(f"\n=== 测试第{page_num}页 ===")

            page = doc.load_page(page_num - 1)
            cropped_images = extractor._extract_cropped_images_from_page(page, page_num)

            if cropped_images:
                logger.info(f"✅ 第{page_num}页提取成功: {len(cropped_images)}张图像")
                for i, img in enumerate(cropped_images):
                    logger.info(f"  图像{i}: {img['size'][1]}x{img['size'][0]} ({img['image_type']})")
            else:
                logger.warning(f"❌ 第{page_num}页未提取到图像")

        doc.close()

    except Exception as e:
        logger.error(f"测试失败: {e}")
        if 'doc' in locals():
            doc.close()
        raise

if __name__ == "__main__":
    test_specific_pages()