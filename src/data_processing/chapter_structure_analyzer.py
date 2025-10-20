#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
章节结构分析器
基于已抽取的文本文件分析章节结构，生成JSON
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChapterInfo:
    """章节信息"""
    title: str
    page: int
    sections: List['SectionInfo']
    content: str = ""

@dataclass
class SectionInfo:
    """节信息"""
    title: str
    page: int
    subsections: List['SubsectionInfo']
    content: str = ""

@dataclass
class SubsectionInfo:
    """小节信息"""
    title: str
    page: int
    content: str = ""

class ChapterStructureAnalyzer:
    """章节结构分析器"""

    def __init__(self):
        self.chapters = []
        self.raw_text = []

    def analyze_text_file(self, text_file_path: str) -> dict:
        """分析文本文件，提取章节结构"""
        text_file_path = Path(text_file_path)
        if not text_file_path.exists():
            raise FileNotFoundError(f"文本文件不存在: {text_file_path}")

        logger.info(f"开始分析文本文件: {text_file_path}")

        # 读取文本内容
        with open(text_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析文本结构
        lines = self._parse_text_content(content)

        # 识别章节结构
        structure = self._identify_chapters(lines)

        # 构建层次结构
        hierarchy = self._build_hierarchy(structure)

        # 构建文本内容
        text_content = self._build_text_content(structure)

        result = {
            'text_file': str(text_file_path),
            'total_chapters': len(structure),  # 使用structure的长度，而不是self.chapters
            'content_structure': {
                'hierarchy': hierarchy,
                'pages': self._build_page_structure(lines)
            },
            'text_content': text_content
        }

        # 保存JSON文件
        json_output_path = self._save_json_content(text_file_path, result)
        result['json_output_path'] = json_output_path

        logger.info(f"章节结构分析完成，共识别 {len(structure)} 个章节")
        return result

    def _parse_text_content(self, content: str) -> List[Dict]:
        """解析文本内容，提取带页码信息的行"""
        lines = []
        current_page = None

        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line:
                continue

            # 检测页码标记
            page_match = re.match(r'^#### 第(\d+)页$', line)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            # 跳过文档信息部分
            if line.startswith('## 文档信息') or line.startswith('- **'):
                continue

            # 跳过分隔符
            if line == '---':
                continue

            if current_page and line:
                lines.append({
                    'text': line,
                    'page': current_page,
                    'line_num': line_num
                })

        return lines

    def _identify_chapters(self, lines: List[Dict]) -> List[Dict]:
        """识别章节结构 - 智能去重逻辑"""
        structure = []
        current_chapter = None
        current_section = None
        current_content = []

        # 用于去重的集合 - 存储标准化后的标题
        seen_chapters = set()
        seen_sections = set()

        for line_info in lines:
            text = line_info['text']
            page = line_info['page']

            # 尝试识别章节标题
            title_type = self._identify_title_type(text)

            if title_type == 'chapter':
                clean_title = self._clean_title(text)
                normalized_title = self._normalize_title(clean_title)

                # 智能去重：检查是否已经存在相似的章节
                if self._is_similar_title_exists(normalized_title, seen_chapters):
                    continue

                seen_chapters.add(normalized_title)

                # 保存前一章的内容
                if current_chapter:
                    if current_section:
                        current_section['content'] = '\n'.join(current_content)
                        current_chapter['sections'].append(current_section)
                        current_content = []
                    current_chapter['content'] = '\n'.join(current_content)
                    structure.append(current_chapter)
                    current_content = []

                # 开始新章节
                current_chapter = {
                    'type': 'chapter',
                    'title': clean_title,
                    'page': page,
                    'sections': [],
                    'content': ''
                }
                current_section = None

            elif title_type == 'section':
                # 如果没有当前章节，创建一个默认章节
                if not current_chapter:
                    current_chapter = {
                        'type': 'chapter',
                        'title': f'未命名章节（页{page}）',
                        'page': page,
                        'sections': [],
                        'content': ''
                    }

                clean_title = self._clean_title(text)
                normalized_title = self._normalize_title(clean_title)

                # 智能去重：检查是否已经存在相似的节
                section_key = f"{self._normalize_title(current_chapter['title'])}::{normalized_title}"
                if self._is_similar_title_exists(normalized_title, seen_sections, prefix=current_chapter['title']):
                    continue

                seen_sections.add(section_key)

                # 保存前一节的内容
                if current_section:
                    current_section['content'] = '\n'.join(current_content)
                    current_chapter['sections'].append(current_section)
                    current_content = []

                # 开始新节
                current_section = {
                    'type': 'section',
                    'title': clean_title,
                    'page': page,
                    'subsections': [],
                    'content': ''
                }

            elif title_type == 'subsection':
                if current_section:
                    subsection = {
                        'type': 'subsection',
                        'title': self._clean_title(text),
                        'page': page,
                        'content': ''
                    }
                    current_section['subsections'].append(subsection)
            else:
                # 普通内容
                if text.strip():
                    current_content.append(text.strip())

        # 保存最后一部分内容
        if current_chapter:
            if current_section:
                current_section['content'] = '\n'.join(current_content)
                current_chapter['sections'].append(current_section)
            else:
                current_chapter['content'] = '\n'.join(current_content)
            structure.append(current_chapter)

        return structure

    def _identify_title_type(self, text: str) -> Optional[str]:
        """识别标题类型"""
        if not text or len(text.strip()) < 2:
            return None

        # 去除Markdown标记
        clean_text = re.sub(r'^[#\s]+', '', text).strip()

        # 章节模式 (第X章)
        # 可能只有"第一章"三个字，后面没有标题（分两行显示）
        if re.match(r'^第[一二三四五六七八九十\d]+章', clean_text):
            return 'chapter'

        # 节模式 (第X节)
        if re.match(r'^第[一二三四五六七八九十\d]+节', clean_text):
            return 'section'

        # 小节模式 (X、 or X.)
        if re.match(r'^[一二三四五六七八九十]+[\.、]', clean_text):
            return 'subsection'

        return None

    def _normalize_title(self, title: str) -> str:
        """标准化标题用于去重"""
        # 移除空格和标点
        normalized = re.sub(r'[\s\.，。！？；：\-\•]', '', title)
        # 统一ROSE相关变体
        normalized = re.sub(r'ROSE?|ROSm?|ROSP?', 'ROSE', normalized)
        # 移除OCR错误字符
        normalized = re.sub(r'[占古]', '要', normalized)  # "要点"识别错误
        return normalized

    def _is_similar_title_exists(self, normalized_title: str, seen_titles: set, prefix: str = "") -> bool:
        """检查是否存在相似的标题"""
        # 完全匹配
        if normalized_title in seen_titles:
            return True

        # 检查相似度 - 如果新标题包含已有关键词，认为是重复
        for seen_title in seen_titles:
            # 如果两者有很长的公共子串，认为是重复
            if self._longest_common_substring(normalized_title, seen_title) >= 10:
                return True

            # 如果新标题是已存在标题的子集（章节号不同但内容相同）
            if normalized_title.replace('第', '').replace('章', '') in seen_title.replace('第', '').replace('章', ''):
                return True

        return False

    def _longest_common_substring(self, s1: str, s2: str) -> int:
        """计算两个字符串的最长公共子串长度"""
        if not s1 or not s2:
            return 0

        # 动态规划求解最长公共子串
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_length = max(max_length, dp[i][j])
                else:
                    dp[i][j] = 0

        return max_length

    def _clean_title(self, title: str) -> str:
        """清理标题文本"""
        # 移除Markdown格式
        title = re.sub(r'^[#\s]+', '', title)
        # 移除多余的标点符号
        title = re.sub(r'[\.。,，;；:：]+$', '', title)
        # 移除多余的空格
        title = re.sub(r'\s+', ' ', title).strip()
        return title

    def _build_hierarchy(self, structure: List[Dict]) -> List[Dict]:
        """构建层次结构用于JSON输出"""
        hierarchy = []

        for chapter in structure:
            chapter_dict = {
                "type": "chapter",
                "title": chapter['title'],
                "page": chapter['page'],
                "sections": []
            }

            for section in chapter['sections']:
                section_dict = {
                    "type": "section",
                    "title": section['title'],
                    "page": section['page'],
                    "subsections": []
                }

                for subsection in section.get('subsections', []):
                    subsection_dict = {
                        "type": "subsection",
                        "title": subsection['title'],
                        "page": subsection['page']
                    }
                    section_dict["subsections"].append(subsection_dict)

                chapter_dict["sections"].append(section_dict)

            hierarchy.append(chapter_dict)

        return hierarchy

    def _build_text_content(self, structure: List[Dict]) -> Dict:
        """构建文本内容"""
        chapters = []
        raw_text = []

        for chapter in structure:
            chapter_dict = {
                "title": chapter['title'],
                "page": chapter['page'],
                "sections": []
            }

            for section in chapter['sections']:
                section_dict = {
                    "title": section['title'],
                    "page": section['page'],
                    "content": section.get('content', '')
                }
                chapter_dict["sections"].append(section_dict)

            chapter_dict["content"] = chapter.get('content', '')
            chapters.append(chapter_dict)

        return {
            "chapters": chapters,
            "raw_text": raw_text
        }

    def _build_page_structure(self, lines: List[Dict]) -> List[Dict]:
        """构建页面结构"""
        pages_dict = {}

        for line_info in lines:
            page_num = line_info['page']
            if page_num not in pages_dict:
                pages_dict[page_num] = {
                    "page_number": page_num,
                    "text_blocks": [],
                    "titles": []
                }

            text = line_info['text']
            title_type = self._identify_title_type(text)

            pages_dict[page_num]["text_blocks"].append({
                "text": text,
                "is_title": title_type is not None,
                "title_type": title_type
            })

            if title_type:
                pages_dict[page_num]["titles"].append({
                    "title": self._clean_title(text),
                    "type": title_type
                })

        return list(pages_dict.values())

    def _save_json_content(self, text_file_path: Path, json_content: Dict) -> str:
        """保存JSON内容"""
        output_filename = f"{text_file_path.stem}_structured.json"
        # 保存到 data/extracted 目录
        json_output_dir = Path("data/extracted")
        json_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = json_output_dir / output_filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON内容已保存: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"保存JSON文件失败: {e}")
            raise


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="章节结构分析工具")
    parser.add_argument("input", help="输入文本文件路径")

    args = parser.parse_args()

    analyzer = ChapterStructureAnalyzer()

    try:
        result = analyzer.analyze_text_file(args.input)

        print(f"章节结构分析完成！")
        print(f"输入文件: {result['text_file']}")
        print(f"总章节数: {result['total_chapters']}")
        print(f"JSON输出文件: {result['json_output_path']}")

    except Exception as e:
        logger.error(f"分析失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())