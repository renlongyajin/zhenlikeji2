#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF抽取和章节分析统一工具
先抽取文本，再分析章节结构
"""

import argparse
import sys
from pathlib import Path
import logging

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from pdf_text_extractor import PDFTextExtractor
from chapter_structure_analyzer import ChapterStructureAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_only(pdf_path: str, output_dir: str = "data/extracted/text_stable"):
    """只抽取文本，不分析章节"""
    extractor = PDFTextExtractor(output_dir=output_dir)
    return extractor.extract_text_from_pdf(pdf_path)

def analyze_structure_only(text_file_path: str):
    """只分析章节结构"""
    analyzer = ChapterStructureAnalyzer()
    return analyzer.analyze_text_file(text_file_path)

def extract_and_analyze(pdf_path: str, output_dir: str = "data/extracted/text_stable"):
    """抽取文本并分析章节结构"""
    logger.info("=== 步骤1: 抽取PDF文本 ===")

    # 1. 抽取文本
    extractor = PDFTextExtractor(output_dir=output_dir)
    extract_result = extractor.extract_text_from_pdf(pdf_path)

    logger.info(f"文本抽取完成: {extract_result['text_output_path']}")

    logger.info("=== 步骤2: 分析章节结构 ===")

    # 2. 分析章节结构
    analyzer = ChapterStructureAnalyzer()
    analyze_result = analyzer.analyze_text_file(extract_result['text_output_path'])

    logger.info(f"章节分析完成: {analyze_result['json_output_path']}")

    return {
        'extract_result': extract_result,
        'analyze_result': analyze_result
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PDF抽取和章节分析工具")
    parser.add_argument("input", help="输入PDF文件路径或文本文件路径")
    parser.add_argument("-m", "--mode", choices=['extract', 'analyze', 'both'], default='both',
                       help="运行模式: extract(只抽取), analyze(只分析), both(抽取+分析)")
    parser.add_argument("-o", "--output", default="data/extracted/text_stable",
                       help="文本输出目录 (默认: data/extracted/text_stable)")
    parser.add_argument("-t", "--text-file",
                       help="当mode=analyze时，指定要分析的文本文件路径")

    args = parser.parse_args()

    try:
        if args.mode == 'extract':
            # 只抽取文本
            result = extract_text_only(args.input, args.output)
            print(f"\n文本抽取完成！")
            print(f"输出文件: {result['text_output_path']}")
            print(f"总页数: {result['total_pages']}")
            print(f"有文本页面: {result['pages_with_text']}")
            print(f"文本块数量: {result['text_blocks']}")

        elif args.mode == 'analyze':
            # 只分析章节结构
            text_file = args.text_file or args.input
            result = analyze_structure_only(text_file)
            print(f"\n章节分析完成！")
            print(f"输入文件: {result['text_file']}")
            print(f"总章节数: {result['total_chapters']}")
            print(f"JSON输出: {result['json_output_path']}")

        elif args.mode == 'both':
            # 抽取并分析
            result = extract_and_analyze(args.input, args.output)
            extract_result = result['extract_result']
            analyze_result = result['analyze_result']

            print(f"\n=== 完整处理完成 ===")
            print(f"\n文本抽取结果:")
            print(f"  输出文件: {extract_result['text_output_path']}")
            print(f"  总页数: {extract_result['total_pages']}")
            print(f"  有文本页面: {extract_result['pages_with_text']}")
            print(f"  文本块数量: {extract_result['text_blocks']}")

            print(f"\n章节分析结果:")
            print(f"  总章节数: {analyze_result['total_chapters']}")
            print(f"  JSON输出: {analyze_result['json_output_path']}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())