#!/usr/bin/env python3
"""
章节内容提取器
从提取的文本文件中，按小节提取完整内容
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MedicalSection:
    """医学小节数据结构"""
    id: str
    chapter_number: int
    chapter_title: str
    section_number: int
    section_title: str
    page_range: List[int]
    content: str
    disease_name: str
    metadata: Dict[str, Any]

class SectionContentExtractor:
    """章节内容提取器"""

    def __init__(self):
        # 中文数字到阿拉伯数字的映射
        self.chinese_numbers = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
        }

        # 小节标题正则表达式
        self.section_pattern = re.compile(r'第([一二三四五六七八九十]+)节(.+)')

        # 章节标题正则表达式
        self.chapter_pattern = re.compile(r'第([一二三四五六七八九十]+)章(.+)')

        # 页码标记正则表达式
        self.page_pattern = re.compile(r'#### 第(\d+)页')

    def extract_sections_from_text(self, text_file_path: str) -> List[MedicalSection]:
        """
        从文本文件中提取小节内容

        Args:
            text_file_path: 文本文件路径

        Returns:
            小节列表
        """
        logger.info(f"开始提取小节内容: {text_file_path}")

        text_file_path = Path(text_file_path)
        if not text_file_path.exists():
            raise FileNotFoundError(f"文本文件不存在: {text_file_path}")

        # 读取文本内容
        with open(text_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 按页分割内容
        pages = self._split_by_pages(content)
        logger.info(f"共识别 {len(pages)} 页内容")

        # 提取小节
        sections = self._extract_sections_from_pages(pages)
        logger.info(f"共提取 {len(sections)} 个小节")

        return sections

    def _split_by_pages(self, content: str) -> List[Dict[str, Any]]:
        """按页分割内容"""
        pages = []
        page_blocks = re.split(self.page_pattern, content)

        for i in range(1, len(page_blocks), 2):  # 跳过分割符
            if i + 1 < len(page_blocks):
                page_num = int(page_blocks[i])
                page_content = page_blocks[i + 1].strip()

                if page_content:
                    pages.append({
                        'page_number': page_num,
                        'content': page_content
                    })

        return pages

    def _extract_sections_from_pages(self, pages: List[Dict[str, Any]]) -> List[MedicalSection]:
        """从页面内容中提取小节"""
        sections = []
        current_chapter = None
        current_chapter_num = None
        current_section = None
        current_section_num = None
        section_content = []
        section_start_page = None

        for page in pages:
            page_num = page['page_number']
            page_content = page['content']

            # 查找章节标题
            chapter_match = self.chapter_pattern.search(page_content)
            if chapter_match:
                chinese_num = chapter_match.group(1)
                chapter_title = chapter_match.group(2).strip()
                current_chapter_num = self.chinese_numbers.get(chinese_num, 0)
                current_chapter = chapter_title
                logger.info(f"找到章节: 第{current_chapter_num}章 {current_chapter}")

            # 查找小节标题
            section_matches = list(self.section_pattern.finditer(page_content))

            for i, match in enumerate(section_matches):
                chinese_num = match.group(1)
                section_title = match.group(2).strip()
                section_num = self.chinese_numbers.get(chinese_num, 0)

                # 如果已有当前小节，先保存
                if current_section and section_content:
                    self._save_section(sections, current_chapter_num, current_chapter,
                                     current_section_num, current_section, section_content,
                                     section_start_page, page_num - 1)

                # 开始新小节
                current_section_num = section_num
                current_section = section_title
                section_start_page = page_num
                section_content = []

                # 提取小节内容（从标题后到下一个标题前）
                start_pos = match.end()
                if i < len(section_matches) - 1:
                    # 不是最后一个小节，提取到下一个标题前
                    end_pos = section_matches[i + 1].start()
                    content = page_content[start_pos:end_pos].strip()
                else:
                    # 最后一个小节，提取到页面末尾
                    content = page_content[start_pos:].strip()

                if content:
                    section_content.append(content)

        # 保存最后一个小节
        if current_section and section_content and current_chapter:
            self._save_section(sections, current_chapter_num, current_chapter,
                             current_section_num, current_section, section_content,
                             section_start_page, page_num)

        return sections

    def _save_section(self, sections: List[MedicalSection], chapter_num: int, chapter_title: str,
                     section_num: int, section_title: str, content_parts: List[str],
                     start_page: int, end_page: int):
        """保存小节数据"""
        # 合并内容
        full_content = '\n'.join(content_parts)

        # 提取疾病名称（从小节标题中提取）
        disease_name = self._extract_disease_name(section_title)

        # 生成ID
        section_id = f"ch{chapter_num}_sec{section_num}_{self._generate_slug(disease_name)}"

        # 创建小节对象
        section = MedicalSection(
            id=section_id,
            chapter_number=chapter_num,
            chapter_title=chapter_title,
            section_number=section_num,
            section_title=section_title,
            page_range=[start_page, end_page],
            content=full_content,
            disease_name=disease_name,
            metadata={
                "content_type": "完整小节",
                "has_images": "图" in full_content,
                "content_length": len(full_content),
                "page_count": end_page - start_page + 1
            }
        )

        sections.append(section)
        logger.info(f"提取小节: {section_title} (页{start_page}-{end_page})")

    def _extract_disease_name(self, section_title: str) -> str:
        """从标题中提取疾病名称"""
        # 移除"第X节"前缀
        disease_name = re.sub(r'^第[一二三四五六七八九十]+节', '', section_title).strip()

        # 清理特殊字符
        disease_name = re.sub(r'[\.。,，;；:：\s]+$', '', disease_name)

        return disease_name

    def _generate_slug(self, text: str) -> str:
        """生成URL友好的slug"""
        # 转换为拼音（简化处理）
        import unicodedata

        # 移除特殊字符
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)

        return text.lower()

    def save_sections_to_json(self, sections: List[MedicalSection], output_path: str):
        """保存小节到JSON文件"""
        output_path = Path(output_path)

        # 转换为字典格式
        data = {
            "text_file": str(output_path).replace('_sections.json', '.txt'),
            "total_sections": len(sections),
            "sections": []
        }

        for section in sections:
            section_dict = {
                "id": section.id,
                "chapter_number": section.chapter_number,
                "chapter_title": section.chapter_title,
                "section_number": section.section_number,
                "section_title": section.section_title,
                "page_range": section.page_range,
                "content": section.content,
                "disease_name": section.disease_name,
                "metadata": section.metadata
            }
            data["sections"].append(section_dict)

        # 保存JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"小节数据已保存: {output_path}")
        return output_path

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="章节内容提取器")
    parser.add_argument("text_file", help="输入文本文件路径")
    parser.add_argument("--output", "-o", help="输出JSON文件路径")

    args = parser.parse_args()

    # 创建提取器
    extractor = SectionContentExtractor()

    try:
        # 提取小节内容
        sections = extractor.extract_sections_from_text(args.text_file)

        # 生成输出文件名
        if args.output:
            output_path = args.output
        else:
            text_path = Path(args.text_file)
            output_path = text_path.parent / f"{text_path.stem}_sections.json"

        # 保存结果
        extractor.save_sections_to_json(sections, output_path)

        print(f"✅ 章节内容提取完成！")
        print(f"📊 共提取 {len(sections)} 个小节")
        print(f"📁 输出文件: {output_path}")

        # 显示前几个小节作为示例
        if sections:
            print(f"\n📝 前3个小节示例:")
            for i, section in enumerate(sections[:3], 1):
                print(f"{i}. {section.section_title} (页{section.page_range[0]}-{section.page_range[1]})")
                print(f"   疾病: {section.disease_name}")
                print(f"   内容长度: {len(section.content)} 字符")
                print()

        return 0

    except Exception as e:
        logger.error(f"提取失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main())