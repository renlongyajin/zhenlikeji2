#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF分析器 - 用于分析扫描版PDF文件的结构和内容
"""

import fitz  # PyMuPDF
import os
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFAnalyzer:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = None
        self.metadata = {}
        self.page_info = []
        self.images_info = []

    def open_pdf(self):
        """打开PDF文件"""
        try:
            self.doc = fitz.open(self.pdf_path)
            logger.info(f"成功打开PDF文件: {self.pdf_path}")
            logger.info(f"总页数: {len(self.doc)}")
            return True
        except Exception as e:
            logger.error(f"打开PDF文件失败: {e}")
            return False

    def analyze_structure(self):
        """分析PDF结构"""
        if not self.doc:
            return False

        # 获取PDF元数据
        self.metadata = {
            'page_count': len(self.doc),
            'title': self.doc.metadata.get('title', ''),
            'author': self.doc.metadata.get('author', ''),
            'subject': self.doc.metadata.get('subject', ''),
            'creation_date': self.doc.metadata.get('creationDate', ''),
            'modification_date': self.doc.metadata.get('modDate', '')
        }

        logger.info(f"PDF元数据: {json.dumps(self.metadata, ensure_ascii=False, indent=2)}")

        # 分析每一页
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            page_info = self.analyze_page(page, page_num)
            self.page_info.append(page_info)

        return True

    def analyze_page(self, page, page_num):
        """分析单个页面"""
        page_dict = {
            'page_number': page_num + 1,
            'width': page.rect.width,
            'height': page.rect.height,
            'rotation': page.rotation,
            'text_blocks': [],
            'images': [],
            'is_scanned': False
        }

        # 获取文本块
        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])

        text_content = ""
        for block in blocks:
            if "lines" in block:  # 文本块
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                if block_text.strip():
                    page_dict['text_blocks'].append({
                        'text': block_text.strip(),
                        'bbox': block["bbox"],
                        'font_size': line["spans"][0]["size"] if line["spans"] else 0,
                        'font_flags': line["spans"][0]["flags"] if line["spans"] else 0
                    })
                    text_content += block_text + "\n"

        # 获取图片
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            img_info = self.extract_image_info(page, img, page_num, img_index)
            if img_info:
                page_dict['images'].append(img_info)

        # 判断是否为扫描版页面（文本内容很少但有图片）
        if len(text_content.strip()) < 50 and len(page_dict['images']) > 0:
            page_dict['is_scanned'] = True

        logger.info(f"第{page_num + 1}页: 文本块{len(page_dict['text_blocks'])}, 图片{len(page_dict['images'])}, 扫描版: {page_dict['is_scanned']}")

        return page_dict

    def extract_image_info(self, page, img, page_num, img_index):
        """提取图片信息"""
        try:
            xref = img[0]
            pix = fitz.Pixmap(self.doc, xref)

            if pix.n - pix.alpha > 3:  # CMYK转换
                pix = fitz.Pixmap(fitz.csRGB, pix)

            img_info = {
                'page': page_num + 1,
                'index': img_index,
                'xref': xref,
                'width': pix.width,
                'height': pix.height,
                'size': len(pix.samples),
                'colorspace': pix.colorspace.n if pix.colorspace else 0,
                'alpha': pix.alpha
            }

            # 保存图片到文件
            img_filename = f"page_{page_num + 1}_img_{img_index}.png"
            img_path = Path("data/raw/images") / img_filename
            img_path.parent.mkdir(parents=True, exist_ok=True)

            if pix.save(str(img_path)):
                img_info['file_path'] = str(img_path)

            pix = None
            return img_info

        except Exception as e:
            logger.error(f"提取图片失败 (页{page_num + 1}, 图{img_index}): {e}")
            return None

    def detect_chapter_structure(self):
        """检测章节结构"""
        chapters = []
        current_chapter = None

        for page_info in self.page_info:
            for text_block in page_info['text_blocks']:
                text = text_block['text']
                font_size = text_block.get('font_size', 0)

                # 检测章节标题（字体较大且包含章节关键词）
                if font_size > 14 and self.is_chapter_title(text):
                    if current_chapter:
                        chapters.append(current_chapter)

                    current_chapter = {
                        'title': text,
                        'page': page_info['page_number'],
                        'bbox': text_block['bbox'],
                        'font_size': font_size,
                        'content': []
                    }
                elif current_chapter:
                    current_chapter['content'].append({
                        'text': text,
                        'page': page_info['page_number'],
                        'bbox': text_block['bbox']
                    })

        if current_chapter:
            chapters.append(current_chapter)

        logger.info(f"检测到{len(chapters)}个章节")
        return chapters

    def is_chapter_title(self, text):
        """判断是否为章节标题"""
        chapter_keywords = [
            '第', '章', '节', '篇', '部分', '章', '节', '篇', 'Part', 'Chapter', 'Section'
        ]

        # 检查是否包含章节关键词
        for keyword in chapter_keywords:
            if keyword in text:
                # 检查格式，如"第1章"、"第一章"、"1. "等
                if any(pattern in text for pattern in ['第', '章', '节', '.', '、']):
                    return True

        return False

    def generate_analysis_report(self):
        """生成分析报告"""
        report = {
            'metadata': self.metadata,
            'summary': {
                'total_pages': len(self.page_info),
                'scanned_pages': sum(1 for p in self.page_info if p['is_scanned']),
                'total_images': sum(len(p['images']) for p in self.page_info),
                'total_text_blocks': sum(len(p['text_blocks']) for p in self.page_info)
            },
            'page_analysis': self.page_info,
            'chapters': self.detect_chapter_structure()
        }

        return report

    def close(self):
        """关闭PDF文件"""
        if self.doc:
            self.doc.close()

def main():
    """主函数"""
    pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"

    if not os.path.exists(pdf_path):
        logger.error(f"PDF文件不存在: {pdf_path}")
        return

    analyzer = PDFAnalyzer(pdf_path)

    try:
        # 打开PDF
        if not analyzer.open_pdf():
            return

        # 分析结构
        analyzer.analyze_structure()

        # 生成报告
        report = analyzer.generate_analysis_report()

        # 保存报告
        report_path = "data/processed/pdf_analysis_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"分析完成，报告已保存到: {report_path}")

        # 打印关键信息
        print("\n=== PDF分析报告 ===")
        print(f"总页数: {report['summary']['total_pages']}")
        print(f"扫描版页面: {report['summary']['scanned_pages']}")
        print(f"总图片数: {report['summary']['total_images']}")
        print(f"文本块数: {report['summary']['total_text_blocks']}")
        print(f"检测到章节数: {len(report['chapters'])}")

        if report['chapters']:
            print("\n章节列表:")
            for i, chapter in enumerate(report['chapters'][:10]):  # 只显示前10个章节
                print(f"  {i+1}. {chapter['title']} (第{chapter['page']}页)")

    finally:
        analyzer.close()

if __name__ == "__main__":
    main()