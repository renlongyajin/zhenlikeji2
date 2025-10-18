#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown to JSON Parser
将现有的markdown文件解析为结构化JSON格式
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarkdownToJSONParser:
    """将markdown文件解析为结构化JSON"""

    def __init__(self):
        self.content = []
        self.document_info = {}
        self.pages = []
        self.hierarchy = []

    def parse_markdown_file(self, markdown_path: str) -> Dict:
        """解析markdown文件"""
        markdown_path = Path(markdown_path)
        if not markdown_path.exists():
            raise FileNotFoundError(f"Markdown文件不存在: {markdown_path}")

        logger.info(f"开始解析markdown文件: {markdown_path}")

        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析文档信息
        self._parse_document_info(content)

        # 解析页面内容
        self._parse_page_content(content)

        # 构建层次结构
        self._build_hierarchy()

        result = {
            "document_info": self.document_info,
            "content_structure": {
                "hierarchy": self.hierarchy,
                "pages": self.pages
            },
            "text_content": {
                "chapters": self._build_chapters(),
                "raw_text": self._extract_raw_text()
            }
        }

        logger.info(f"解析完成，共{len(self.pages)}页，{len(self.hierarchy)}个章节")
        return result

    def _parse_document_info(self, content: str):
        """解析文档信息"""
        # 提取标题
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            self.document_info["title"] = title_match.group(1).strip()
        else:
            self.document_info["title"] = "Unknown Title"

        # 提取文档信息部分
        info_section = re.search(r'##\s+文档信息\s*\n(.*?)\n---', content, re.DOTALL)
        if info_section:
            info_text = info_section.group(1)

            # 提取原始PDF
            pdf_match = re.search(r'-\s+\*\*原始PDF\*\*:\s*(.+)', info_text)
            if pdf_match:
                self.document_info["original_pdf"] = pdf_match.group(1).strip()

            # 提取总页数
            pages_match = re.search(r'-\s+\*\*总页数\*\*:\s*(\d+)', info_text)
            if pages_match:
                self.document_info["total_pages"] = int(pages_match.group(1))

            # 提取提取时间
            time_match = re.search(r'-\s+\*\*提取时间\*\*:\s*(.+)', info_text)
            if time_match:
                self.document_info["extraction_time"] = time_match.group(1).strip()

            # 提取文本块数量
            blocks_match = re.search(r'-\s+\*\*文本块数量\*\*:\s*(\d+)', info_text)
            if blocks_match:
                self.document_info["text_blocks_count"] = int(blocks_match.group(1))

        self.document_info["processing_engine"] = "MarkdownToJSONParser"

    def _parse_page_content(self, content: str):
        """解析页面内容"""
        # 按页面分割
        page_pattern = r'####\s+第(\d+)页\s*\n'
        page_splits = re.split(page_pattern, content)

        # 跳过文档信息部分
        start_idx = 0
        for i, split in enumerate(page_splits):
            if split and split.isdigit():
                start_idx = i - 1
                break

        page_splits = page_splits[start_idx:]

        for i in range(0, len(page_splits) - 1, 2):
            if i + 1 < len(page_splits):
                try:
                    page_num = int(page_splits[i])
                    page_content = page_splits[i + 1]

                    page_data = self._parse_single_page(page_num, page_content)
                    if page_data:
                        self.pages.append(page_data)
                except ValueError:
                    continue

    def _parse_single_page(self, page_num: int, content: str) -> Optional[Dict]:
        """解析单个页面"""
        if not content.strip():
            return None

        lines = content.strip().split('\n')
        text_blocks = []
        titles = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测标题
            title_info = self._detect_title(line)

            text_block = {
                "text": line,
                "is_title": title_info["is_title"],
                "title_type": title_info["title_type"],
                "confidence": 0.9,
                "bbox": [0, 0, 100, 20]  # 模拟bbox
            }
            text_blocks.append(text_block)

            if title_info["is_title"]:
                titles.append({
                    "title": line,
                    "type": title_info["title_type"]
                })

        return {
            "page_number": page_num,
            "text_blocks": text_blocks,
            "titles": titles
        }

    def _detect_title(self, line: str) -> Dict:
        """检测标题类型"""
        # 章节模式 (第X章)
        if re.match(r'^#\s+第[一二三四五六七八九十\d]+章\s*', line):
            return {"is_title": True, "title_type": "chapter"}

        # 节模式 (第X节)
        if re.match(r'^##\s+第[一二三四五六七八九十\d]+节\s*', line):
            return {"is_title": True, "title_type": "section"}

        # 小节模式 (X、 or X.)
        if re.match(r'^###\s+[一二三四五六七八九十]+\.?\s*', line):
            return {"is_title": True, "title_type": "subsection"}

        # 其他标题模式
        if line.startswith('#'):
            return {"is_title": True, "title_type": "generic"}

        return {"is_title": False, "title_type": None}

    def _build_hierarchy(self):
        """构建层次结构"""
        current_chapter = None
        current_section = None

        for page in self.pages:
            for block in page["text_blocks"]:
                if not block["is_title"]:
                    continue

                title_type = block["title_type"]
                title_text = block["text"]

                if title_type == "chapter":
                    current_chapter = {
                        "type": "chapter",
                        "title": title_text,
                        "page": page["page_number"],
                        "sections": []
                    }
                    self.hierarchy.append(current_chapter)
                    current_section = None

                elif title_type == "section" and current_chapter:
                    current_section = {
                        "type": "section",
                        "title": title_text,
                        "page": page["page_number"],
                        "subsections": []
                    }
                    current_chapter["sections"].append(current_section)

                elif title_type == "subsection" and current_section:
                    subsection = {
                        "type": "subsection",
                        "title": title_text,
                        "page": page["page_number"]
                    }
                    current_section["subsections"].append(subsection)

    def _build_chapters(self) -> List[Dict]:
        """构建章节内容"""
        chapters = []
        current_chapter = None
        current_section = None
        current_chapter_text = []
        current_section_text = []

        for page in self.pages:
            for block in page["text_blocks"]:
                if block["is_title"]:
                    # 保存前一章的内容
                    if current_chapter and current_chapter_text:
                        if current_section and current_section_text:
                            current_section["content"] = "\n".join(current_section_text)
                            current_section_text = []
                        current_chapter["content"] = "\n".join(current_chapter_text)
                        chapters.append(current_chapter)
                        current_chapter_text = []

                    # 开始新的章节/节
                    title_type = block["title_type"]
                    title_text = block["text"]

                    if title_type == "chapter":
                        current_chapter = {
                            "title": title_text,
                            "page": page["page_number"],
                            "sections": []
                        }
                        current_section = None
                    elif title_type == "section" and current_chapter:
                        current_section = {
                            "title": title_text,
                            "page": page["page_number"],
                            "content": ""
                        }
                        current_chapter["sections"].append(current_section)
                    elif title_type == "subsection" and current_section:
                        subsection = {
                            "title": title_text,
                            "page": page["page_number"],
                            "content": ""
                        }
                        current_section["subsections"] = current_section.get("subsections", [])
                        current_section["subsections"].append(subsection)
                else:
                    # 普通文本内容
                    text_line = block["text"].strip()
                    if text_line:
                        current_chapter_text.append(text_line)
                        current_section_text.append(text_line)

        # 保存最后一章的内容
        if current_chapter and current_chapter_text:
            if current_section and current_section_text:
                current_section["content"] = "\n".join(current_section_text)
            current_chapter["content"] = "\n".join(current_chapter_text)
            chapters.append(current_chapter)

        return chapters

    def _extract_raw_text(self) -> List[str]:
        """提取原始文本"""
        raw_text = []
        for page in self.pages:
            for block in page["text_blocks"]:
                if not block["is_title"] and block["text"].strip():
                    raw_text.append(block["text"].strip())
        return raw_text

    def save_json(self, data: Dict, output_path: str):
        """保存JSON文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"JSON文件已保存: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Markdown to JSON Parser")
    parser.add_argument("input", help="输入markdown文件路径")
    parser.add_argument("-o", "--output", help="输出JSON文件路径")

    args = parser.parse_args()

    parser = MarkdownToJSONParser()

    try:
        # 解析markdown
        result = parser.parse_markdown_file(args.input)

        # 生成输出路径
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input)
            output_path = input_path.parent / f"{input_path.stem}_structured.json"

        # 保存JSON
        parser.save_json(result, output_path)

        print(f"解析完成！")
        print(f"输入文件: {args.input}")
        print(f"输出文件: {output_path}")
        print(f"总页数: {result['document_info'].get('total_pages', 0)}")
        print(f"章节数: {len(result['content_structure']['hierarchy'])}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())