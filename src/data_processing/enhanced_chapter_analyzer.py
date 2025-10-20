#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版章节结构分析器
专门处理医学文献的复杂章节结构
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
    content: str = ""

class EnhancedChapterAnalyzer:
    """增强版章节分析器 - 专门处理医学文献"""

    def __init__(self):
        self.medical_keywords = [
            '细胞', '肿瘤', '恶性', '肺脏', '肺部', 'ROSE', '组学', '特点',
            '分型', '要点', '特征', '少见病', '评价', '图谱'
        ]

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

        # 使用多阶段识别策略
        chapters = self._extract_chapters_multistage(lines)

        # 构建最终结构
        result = {
            'text_file': str(text_file_path),
            'total_chapters': len(chapters),
            'chapters': chapters,
            'extraction_summary': self._generate_summary(chapters)
        }

        # 保存JSON文件
        json_output_path = self._save_json_content(text_file_path, result)
        result['json_output_path'] = json_output_path

        logger.info(f"增强版章节分析完成，共识别 {len(chapters)} 个章节")
        return result

    def _parse_text_content(self, content: str) -> List[Dict]:
        """解析文本内容"""
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

    def _extract_chapters_multistage(self, lines: List[Dict]) -> List[Dict]:
        """多阶段章节提取策略"""

        # 第一阶段：识别所有可能的章节标题
        candidate_titles = self._find_candidate_titles(lines)

        # 第二阶段：验证和过滤候选标题
        valid_titles = self._validate_titles(candidate_titles)

        # 第三阶段：构建章节结构
        chapters = self._build_chapter_structure(lines, valid_titles)

        return chapters

    def _find_candidate_titles(self, lines: List[Dict]) -> List[Dict]:
        """寻找候选章节标题"""
        candidates = []

        for i, line_info in enumerate(lines):
            text = line_info['text']

            # 去除Markdown标记
            clean_text = re.sub(r'^[#\s]+', '', text).strip()

            # 模式1: 标准章节格式
            if re.match(r'^第[一二三四五六七八九十\d]+章', clean_text):
                candidates.append({
                    'title': clean_text,
                    'page': line_info['page'],
                    'line_idx': i,
                    'type': 'chapter',
                    'confidence': 1.0
                })

            # 模式2: 标准节格式
            elif re.match(r'^第[一二三四五六七八九十\d]+节', clean_text):
                candidates.append({
                    'title': clean_text,
                    'page': line_info['page'],
                    'line_idx': i,
                    'type': 'section',
                    'confidence': 1.0
                })

            # 模式3: 医学关键词组合（处理OCR错误）
            elif self._is_medical_title_candidate(clean_text):
                candidates.append({
                    'title': clean_text,
                    'page': line_info['page'],
                    'line_idx': i,
                    'type': 'possible_chapter',
                    'confidence': 0.7
                })

        return candidates

    def _is_medical_title_candidate(self, text: str) -> bool:
        """判断是否为医学标题候选"""
        # 长度检查
        if len(text) < 5 or len(text) > 50:
            return False

        # 关键词检查
        keyword_count = sum(1 for keyword in self.medical_keywords if keyword in text)

        # 必须包含至少2个医学关键词
        if keyword_count >= 2:
            return True

        # 或者包含ROSE相关词汇
        if re.search(r'ROSE?|ROSm?|ROSP?', text):
            return True

        return False

    def _validate_titles(self, candidates: List[Dict]) -> List[Dict]:
        """验证和过滤候选标题"""
        if not candidates:
            return []

        # 按置信度排序
        candidates.sort(key=lambda x: x['confidence'], reverse=True)

        # 去重：移除相似的标题
        unique_titles = []
        seen_normalized = set()

        for candidate in candidates:
            normalized = self._normalize_title(candidate['title'])

            # 检查是否与已存在的标题相似
            is_duplicate = False
            for seen in seen_normalized:
                if self._are_titles_similar(normalized, seen):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_titles.append(candidate)
                seen_normalized.add(normalized)

        # 按行号排序
        unique_titles.sort(key=lambda x: x['line_idx'])

        return unique_titles

    def _normalize_title(self, title: str) -> str:
        """标准化标题"""
        # 移除空格和标点
        normalized = re.sub(r'[\s\.，。！？；：\-\•]', '', title)
        # 统一ROSE相关变体
        normalized = re.sub(r'ROSE?|ROSm?|ROSP?', 'ROSE', normalized)
        # 统一章节号
        normalized = re.sub(r'^第[一二三四五六七八九十\d]+章', '第X章', normalized)
        normalized = re.sub(r'^第[一二三四五六七八九十\d]+节', '第X节', normalized)
        # 移除OCR错误字符
        normalized = re.sub(r'[占古]', '要', normalized)
        return normalized

    def _are_titles_similar(self, title1: str, title2: str) -> bool:
        """判断两个标题是否相似"""
        # 完全相同
        if title1 == title2:
            return True

        # 计算相似度
        common_chars = len(set(title1) & set(title2))
        total_chars = len(set(title1) | set(title2))
        similarity = common_chars / total_chars if total_chars > 0 else 0

        return similarity > 0.8

    def _build_chapter_structure(self, lines: List[Dict], valid_titles: List[Dict]) -> List[Dict]:
        """构建章节结构"""
        chapters = []

        # 按章节分组
        chapter_groups = self._group_by_chapters(valid_titles)

        for chapter_group in chapter_groups:
            if chapter_group['type'] != 'chapter':
                continue

            chapter = {
                'title': chapter_group['title'],
                'page': chapter_group['page'],
                'sections': []
            }

            # 提取该章节的内容
            start_idx = chapter_group['line_idx']
            end_idx = self._find_chapter_end(lines, valid_titles, start_idx)

            # 提取节
            chapter['sections'] = self._extract_sections(lines, start_idx, end_idx, valid_titles)

            # 提取内容
            chapter['content'] = self._extract_content(lines, start_idx, end_idx, valid_titles)

            chapters.append(chapter)

        return chapters

    def _group_by_chapters(self, titles: List[Dict]) -> List[Dict]:
        """按章节分组标题"""
        groups = []
        current_chapter = None

        for title in titles:
            if title['type'] == 'chapter':
                if current_chapter:
                    groups.append(current_chapter)
                current_chapter = {
                    'title': title['title'],
                    'page': title['page'],
                    'line_idx': title['line_idx'],
                    'type': 'chapter',
                    'sections': []
                }
            elif title['type'] == 'section' and current_chapter:
                current_chapter['sections'].append(title)

        if current_chapter:
            groups.append(current_chapter)

        return groups

    def _find_chapter_end(self, lines: List[Dict], titles: List[Dict], start_idx: int) -> int:
        """找到章节结束位置"""
        # 找到下一个章节开始位置
        for title in titles:
            if title['line_idx'] > start_idx and title['type'] == 'chapter':
                return title['line_idx'] - 1

        # 如果没有下一个章节，到文件末尾
        return len(lines) - 1

    def _extract_sections(self, lines: List[Dict], start_idx: int, end_idx: int, titles: List[Dict]) -> List[Dict]:
        """提取节信息"""
        sections = []

        for title in titles:
            if start_idx <= title['line_idx'] <= end_idx and title['type'] == 'section':
                # 找到节的结束位置
                section_end = self._find_section_end(lines, titles, title['line_idx'], end_idx)

                # 提取节内容
                section_content = self._extract_section_content(lines, title['line_idx'], section_end)

                sections.append({
                    'title': title['title'],
                    'page': title['page'],
                    'content': section_content
                })

        return sections

    def _find_section_end(self, lines: List[Dict], titles: List[Dict], start_idx: int, chapter_end: int) -> int:
        """找到节结束位置"""
        # 找到下一个节或章节开始位置
        for title in titles:
            if title['line_idx'] > start_idx and title['line_idx'] <= chapter_end:
                if title['type'] in ['section', 'chapter']:
                    return title['line_idx'] - 1

        return chapter_end

    def _extract_section_content(self, lines: List[Dict], start_idx: int, end_idx: int) -> str:
        """提取节内容"""
        content_lines = []

        for i in range(start_idx + 1, end_idx + 1):
            if i < len(lines):
                line_text = lines[i]['text']
                # 跳过其他标题
                if not self._is_title_line(line_text):
                    content_lines.append(line_text)

        return '\n'.join(content_lines)

    def _extract_content(self, lines: List[Dict], start_idx: int, end_idx: int, titles: List[Dict]) -> str:
        """提取章节内容"""
        content_lines = []

        for i in range(start_idx + 1, end_idx + 1):
            if i < len(lines):
                line_text = lines[i]['text']
                # 跳过标题行
                if not self._is_title_line(line_text):
                    content_lines.append(line_text)

        return '\n'.join(content_lines)

    def _is_title_line(self, text: str) -> bool:
        """判断是否为标题行"""
        clean_text = re.sub(r'^[#\s]+', '', text).strip()

        # 检查标准格式
        if re.match(r'^第[一二三四五六七八九十\d]+[章节]', clean_text):
            return True

        # 检查是否包含医学关键词组合
        keyword_count = sum(1 for keyword in self.medical_keywords if keyword in clean_text)
        return keyword_count >= 3 and len(clean_text) < 30

    def _generate_summary(self, chapters: List[Dict]) -> Dict:
        """生成提取摘要"""
        total_sections = sum(len(chapter.get('sections', [])) for chapter in chapters)

        return {
            'total_chapters': len(chapters),
            'total_sections': total_sections,
            'chapters_detail': [
                {
                    'title': chapter['title'],
                    'page': chapter['page'],
                    'sections_count': len(chapter.get('sections', []))
                }
                for chapter in chapters
            ]
        }

    def _save_json_content(self, text_file_path: Path, json_content: Dict) -> str:
        """保存JSON内容"""
        output_filename = f"{text_file_path.stem}_enhanced_structured.json"
        # 保存到 data/extracted 目录
        json_output_dir = Path("data/extracted")
        json_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = json_output_dir / output_filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)
            logger.info(f"增强版JSON已保存: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"保存JSON文件失败: {e}")
            raise


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="增强版章节结构分析工具")
    parser.add_argument("input", help="输入文本文件路径")

    args = parser.parse_args()

    analyzer = EnhancedChapterAnalyzer()

    try:
        result = analyzer.analyze_text_file(args.input)

        print(f"增强版章节分析完成！")
        print(f"输入文件: {result['text_file']}")
        print(f"总章节数: {result['total_chapters']}")
        print(f"JSON输出文件: {result['json_output_path']}")

        # 显示详细信息
        summary = result['extraction_summary']
        print(f"\n详细结构:")
        for chapter_info in summary['chapters_detail']:
            print(f"  - {chapter_info['title']} (页{chapter_info['page']}) - {chapter_info['sections_count']}节")

    except Exception as e:
        logger.error(f"分析失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())