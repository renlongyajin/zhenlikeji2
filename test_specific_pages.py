#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特定页面的PDF处理效果
用于分析第5、16、17页的图像识别效果
"""

import sys
sys.path.append('src')

from src.data_processing.pdf_scan_processor import PDFScanProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_pages():
    """测试特定页面"""
    pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"

    # 创建处理器，只处理特定页面
    processor = PDFScanProcessor(pdf_path, output_dir="data/test_pages")

    logger.info("开始测试特定页面处理...")

    try:
        # 只处理第5、16、17页（页面索引从0开始）
        target_pages = [4, 15, 16]  # 第5、16、17页

        processor.doc = fitz.open(pdf_path)
        total_pages = len(processor.doc)
        logger.info(f"PDF总页数: {total_pages}")

        # 处理目标页面
        for page_num in target_pages:
            if page_num < total_pages:
                logger.info(f"处理第 {page_num + 1} 页")

                try:
                    page_result = processor._process_page(page_num)
                    if page_result:
                        processor.processed_results.append(page_result)
                        processor._save_page_result(page_num, page_result)

                        # 打印详细信息
                        logger.info(f"第{page_num + 1}页处理结果:")
                        logger.info(f"  OCR文本长度: {len(page_result['ocr_text'])}")
                        logger.info(f"  OCR置信度: {page_result['ocr_confidence']:.3f}")
                        logger.info(f"  检测到图片: {len(page_result['detected_images'])}")

                        for i, img in enumerate(page_result['detected_images']):
                            logger.info(f"    图片{i+1}: {img['image_type']} (区域: {img['region_name']}, 置信度: {img['confidence']})")

                except Exception as e:
                    logger.error(f"处理第{page_num + 1}页失败: {e}")
                    continue
            else:
                logger.warning(f"PDF只有{total_pages}页，无法处理第{page_num + 1}页")

        # 生成最终结果
        result = processor._generate_final_result()

        logger.info("测试完成！")
        return result

    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise
    finally:
        if processor.doc:
            processor.doc.close()

if __name__ == "__main__":
    import fitz
    result = test_specific_pages()