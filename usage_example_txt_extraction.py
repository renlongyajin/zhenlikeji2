#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF纯文本提取器使用示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.pdf_txt_extracted import PDFTextExtractor

def simple_usage_example():
    """简单使用示例"""
    print("PDF纯文本提取器使用示例")
    print("=" * 50)

    # 创建提取器
    extractor = PDFTextExtractor(output_dir="example_output")

    # 示例1: 提取单个PDF文件
    pdf_path = "your_document.pdf"  # 替换为实际的PDF文件路径

    print(f"准备提取PDF: {pdf_path}")
    print("注意: 请确保PDF文件存在，或者修改路径为实际文件")

    try:
        # 执行提取
        result = extractor.extract_text_from_pdf(pdf_path)

        print(f"\n提取结果:")
        print(f"- 文件路径: {result['pdf_path']}")
        print(f"- 总页数: {result['total_pages']}")
        print(f"- 提取文本块: {result['extracted_text_blocks']}")
        print(f"- 处理时间: {result['processing_time']}")

    except FileNotFoundError:
        print(f"\n文件不存在: {pdf_path}")
        print("请替换为实际的PDF文件路径")
    except Exception as e:
        print(f"\n提取失败: {e}")

def batch_processing_example():
    """批量处理示例"""
    print("\n" + "=" * 50)
    print("批量处理示例")
    print("=" * 50)

    extractor = PDFTextExtractor(output_dir="batch_output")

    # 示例2: 批量处理目录中的所有PDF
    pdf_directory = "path/to/pdf/directory"  # 替换为实际的PDF目录

    print(f"准备批量处理目录: {pdf_directory}")
    print("注意: 请确保目录存在且包含PDF文件")

    try:
        results = extractor.process_multiple_pdfs(pdf_directory)

        print(f"\n批量处理结果:")
        print(f"- 成功处理文件数: {len(results)}")

        for i, result in enumerate(results, 1):
            print(f"\n文件 {i}:")
            print(f"  - 文件: {os.path.basename(result['pdf_path'])}")
            print(f"  - 页数: {result['total_pages']}")
            print(f"  - 文本块: {result['extracted_text_blocks']}")

    except FileNotFoundError:
        print(f"\n目录不存在: {pdf_directory}")
        print("请替换为实际的PDF文件目录")
    except Exception as e:
        print(f"\n批量处理失败: {e}")

def advanced_usage_example():
    """高级使用示例 - 处理提取的文本内容"""
    print("\n" + "=" * 50)
    print("高级使用示例 - 文本内容处理")
    print("=" * 50)

    # 创建提取器
    extractor = PDFTextExtractor(output_dir="advanced_output")

    # 示例PDF路径（需要替换为实际文件）
    pdf_path = "sample_medical.pdf"

    try:
        # 提取文本
        result = extractor.extract_text_from_pdf(pdf_path)

        # 获取生成的文本内容
        text_content = result['text_content']

        print("文本内容分析:")
        print("-" * 30)

        # 分析文本内容
        lines = text_content.split('\n')
        title_lines = [line for line in lines if line.startswith('#')]
        content_lines = [line for line in lines if line.strip() and not line.startswith('#')]

        print(f"总行数: {len(lines)}")
        print(f"标题行数: {len(title_lines)}")
        print(f"内容行数: {len(content_lines)}")

        # 显示标题层级分布
        chapter_titles = [t for t in title_lines if t.startswith('# ')]
        section_titles = [t for t in title_lines if t.startswith('## ')]
        subsection_titles = [t for t in title_lines if t.startswith('### ')]

        print(f"\n标题层级分布:")
        print(f"- 章节标题: {len(chapter_titles)}")
        print(f"- 节标题: {len(section_titles)}")
        print(f"- 小节标题: {len(subsection_titles)}")

        # 显示前几个标题
        if title_lines:
            print(f"\n前5个标题:")
            for i, title in enumerate(title_lines[:5], 1):
                print(f"{i}. {title}")

        # 保存提取的文本到不同格式
        output_base = os.path.splitext(os.path.basename(pdf_path))[0]

        # 保存为纯文本文件
        txt_file = f"advanced_output/{output_base}_plain.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f"\n已保存纯文本: {txt_file}")

        # 提取并保存所有标题
        titles_file = f"advanced_output/{output_base}_titles.txt"
        with open(titles_file, 'w', encoding='utf-8') as f:
            for title in title_lines:
                f.write(title + '\n')
        print(f"已保存标题: {titles_file}")

        # 提取并保存主要内容（非标题行）
        content_file = f"advanced_output/{output_base}_content.txt"
        with open(content_file, 'w', encoding='utf-8') as f:
            for content in content_lines:
                if content.strip():  # 跳过空行
                    f.write(content + '\n')
        print(f"已保存主要内容: {content_file}")

    except FileNotFoundError:
        print(f"\n文件不存在: {pdf_path}")
        print("请替换为实际的PDF文件路径")
    except Exception as e:
        print(f"\n高级处理失败: {e}")

def main():
    """主函数"""
    print("PDF纯文本提取器使用示例")
    print("=" * 60)
    print("这个示例展示了如何使用新的纯文本提取器")
    print("特点:")
    print("- ✅ 仅提取文本内容，忽略所有图片")
    print("- ✅ 自动识别章节、节、小节等标题")
    print("- ✅ 生成结构化的Markdown格式文本")
    print("- ✅ 支持批量处理多个PDF文件")
    print("=" * 60)

    # 显示使用说明
    print("\n基本命令行使用方法:")
    print("1. 单个文件: python src/data_processing/pdf_txt_extracted.py document.pdf")
    print("2. 批量处理: python src/data_processing/pdf_txt_extracted.py pdf_directory --batch")
    print("3. 指定输出: python src/data_processing/pdf_txt_extracted.py document.pdf -o output_dir")

    print("\n编程使用方法:")
    print("见下面的示例代码")
    print("-" * 40)

    try:
        # 运行示例
        simple_usage_example()
        batch_processing_example()
        advanced_usage_example()

        print("\n" + "=" * 60)
        print("示例完成！")
        print("要开始使用，请:")
        print("1. 确保有PDF文件需要处理")
        print("2. 修改示例中的文件路径为实际路径")
        print("3. 运行提取器或自定义代码")
        print("=" * 60)

    except Exception as e:
        print(f"示例运行失败: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())