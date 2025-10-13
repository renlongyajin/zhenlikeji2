#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的PDF处理器 - 快速验证关键改进
"""

import sys
sys.path.append('src')

from src.data_processing.optimized_pdf_scan_processor import OptimizedPDFScanProcessor
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_optimized_processor():
    """测试优化后的处理器"""
    pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"

    # 创建处理器，使用更快的参数
    processor = OptimizedPDFScanProcessor(pdf_path, output_dir="data/optimized_test")

    try:
        logger.info("开始测试优化后的PDF处理器...")
        start_time = time.time()

        # 只测试第5页和第16页，使用简化的处理流程
        result = processor.process_pdf(target_pages=[5, 16])

        processing_time = time.time() - start_time
        logger.info(f"处理完成，耗时: {processing_time:.2f}秒")

        # 分析结果
        logger.info("\n=== 测试结果分析 ===")

        if result and 'pages' in result:
            for page_result in result['pages']:
                page_num = page_result['page_num']
                logger.info(f"\n第{page_num}页:")
                logger.info(f"  OCR文本长度: {len(page_result['ocr_text'])}")
                logger.info(f"  OCR置信度: {page_result['ocr_confidence']:.3f}")
                logger.info(f"  检测到图片: {len(page_result['detected_images'])}")

                # 详细分析检测到的图片
                for i, img in enumerate(page_result['detected_images']):
                    logger.info(f"    图片{i+1}: {img['image_type']} (置信度: {img['confidence']:.3f})")

                    # 检查是否是经过验证的完整区域
                    if img.get('completeness_validated'):
                        logger.info(f"      ✓ 已通过完整性验证")

                    # 检查边框特征
                    if 'rectangular_border' in img.get('analysis', {}):
                        logger.info(f"      ✓ 检测到矩形边框特征")

        # 检查统计信息
        if result and 'statistics' in result:
            stats = result['statistics']
            logger.info(f"\n=== 统计信息 ===")
            logger.info(f"总页数: {result['metadata']['total_pages']}")
            logger.info(f"检测到图片: {stats['total_images']}")
            logger.info(f"人物头像: {stats['portrait_images']}")
            logger.info(f"肿瘤图像: {stats['tumor_images']}")
            logger.info(f"医学图表: {stats['medical_diagrams']}")
            logger.info(f"高置信度图片: {stats['high_confidence_images']}")

        return result

    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise
    finally:
        if hasattr(processor, 'doc') and processor.doc:
            processor.doc.close()

def analyze_test_results():
    """分析测试结果"""
    logger.info("\n=== 优化效果分析 ===")

    # 检查生成的图片
    import os
    from pathlib import Path

    image_dir = Path("data/optimized_test/images")
    if image_dir.exists():
        image_files = list(image_dir.glob("*.png"))
        logger.info(f"生成的图片文件: {len(image_files)}")

        for img_file in image_files:
            logger.info(f"  - {img_file.name}")

            # 简单的文件大小检查
            file_size = img_file.stat().st_size
            if file_size < 1000:  # 小于1KB的图片可能有问题
                logger.warning(f"    ⚠️ 文件过小，可能提取不完整")
            else:
                logger.info(f"    ✓ 文件大小正常 ({file_size}字节)")
    else:
        logger.warning("未找到生成的图片目录")

if __name__ == "__main__":
    # 运行测试
    result = test_optimized_processor()

    # 分析结果
    analyze_test_results()

    logger.info("\n测试完成！")