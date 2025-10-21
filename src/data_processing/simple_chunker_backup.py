#!/usr/bin/env python3
"""
简化版数据切块脚本
策略：
1. 每个小节（## 标题）= 一个块
2. 如果小节内容 > 800字符，按500字符切块（在句号边界）
3. 保留100字符重叠
"""

import re
import json
from typing import List, Dict, Any
from pathlib import Path


class SimpleChunker:
    """简单切块器"""

    def __init__(self, max_chunk_size: int = 800, split_size: int = 500, overlap: int = 100):
        """初始化

        Args:
            max_chunk_size: 超过这个长度就切块
            split_size: 切块的目标大小
            overlap: 块之间的重叠字符数
        """
        self.max_chunk_size = max_chunk_size
        self.split_size = split_size
        self.overlap = overlap

    def process_markdown(self, md_file: str) -> List[Dict[str, Any]]:
        """处理Markdown文件，生成切块数据

        Args:
            md_file: Markdown文件路径

        Returns:
            切块数据列表
        """
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = []
        current_chapter = ""
        current_section = ""
        section_content = ""
        page_number = 1  # 简化版：从1开始，后续可以手动映射

        lines = content.split('\n')

        for line in lines:
            # 识别章标题 (# 第X章)
            if re.match(r'^#\s+第[一二三四五六七八九十百]+章', line):
                # 保存之前的小节
                if section_content.strip():
                    chunks.extend(self._create_chunks(
                        current_chapter, current_section, section_content, page_number
                    ))
                    section_content = ""

                current_chapter = line.replace('#', '').strip()
                current_section = ""
                page_number += 1

            # 识别节标题 (## 第X节)
            elif re.match(r'^##\s+第[一二三四五六七八九十百]+节', line):
                # 保存之前的小节
                if section_content.strip():
                    chunks.extend(self._create_chunks(
                        current_chapter, current_section, section_content, page_number
                    ))
                    section_content = ""

                current_section = line.replace('##', '').strip()
                page_number += 1

            # 识别其他二级标题 (##)
            elif re.match(r'^##\s+', line) and not re.match(r'^##\s+第[一二三四五六七八九十百]+节', line):
                # 保存之前的小节
                if section_content.strip():
                    chunks.extend(self._create_chunks(
                        current_chapter, current_section, section_content, page_number
                    ))
                    section_content = ""

                current_section = line.replace('##', '').strip()

            # 普通内容
            else:
                # 跳过图注行（以*（开头的）
                if line.strip().startswith('*（图'):
                    continue
                if line.strip():
                    section_content += line + "\n"

        # 保存最后一个小节
        if section_content.strip():
            chunks.extend(self._create_chunks(
                current_chapter, current_section, section_content, page_number
            ))

        # 为每个块添加唯一ID
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = f"chunk_{i+1:04d}"
            chunk['chunk_index'] = i

        return chunks

    def _create_chunks(self, chapter: str, section: str, content: str, page_number: int) -> List[Dict[str, Any]]:
        """创建切块

        Args:
            chapter: 章标题
            section: 节标题
            content: 内容
            page_number: 页码

        Returns:
            切块列表
        """
        content = content.strip()

        # 如果内容不超过最大长度，直接返回一个块
        if len(content) <= self.max_chunk_size:
            return [{
                'content': content,
                'chapter_title': chapter,
                'section_title': section,
                'page_number': page_number,
                'content_length': len(content)
            }]

        # 否则按句子边界切块
        return self._split_by_sentences(chapter, section, content, page_number)

    def _split_by_sentences(self, chapter: str, section: str, content: str, page_number: int) -> List[Dict[str, Any]]:
        """按句子边界切块

        Args:
            chapter: 章标题
            section: 节标题
            content: 内容
            page_number: 页码

        Returns:
            切块列表
        """
        chunks = []
        current_pos = 0
        chunk_count = 0

        while current_pos < len(content):
            # 确定当前块的结束位置
            end_pos = min(current_pos + self.split_size, len(content))

            # 如果不是最后，找句号边界
            if end_pos < len(content):
                # 在目标位置前后50字符范围内查找句号
                search_start = max(end_pos - 50, current_pos)
                search_end = min(end_pos + 50, len(content))
                search_text = content[search_start:search_end]

                # 查找句号位置
                sentence_ends = [m.start() for m in re.finditer(r'[。！？；]', search_text)]

                if sentence_ends:
                    # 找到最接近目标位置的句号
                    target_offset = end_pos - search_start
                    best_end = min(sentence_ends, key=lambda x: abs(x - target_offset))
                    end_pos = search_start + best_end + 1  # +1 包含句号
                else:
                    # 找不到句号，强制在目标位置切
                    pass

            # 提取块内容
            chunk_text = content[current_pos:end_pos].strip()

            if chunk_text:
                chunks.append({
                    'content': chunk_text,
                    'chapter_title': chapter,
                    'section_title': section,
                    'page_number': page_number,
                    'sub_chunk_index': chunk_count,
                    'content_length': len(chunk_text)
                })
                chunk_count += 1

            # 移动位置，保留重叠 - 但确保向前推进
            next_pos = end_pos - self.overlap

            # 防止无限循环：确保向前推进至少1个字符
            if next_pos <= current_pos:
                next_pos = current_pos + 1

            current_pos = next_pos

            # 避免无限循环
            if current_pos >= len(content) - 10:
                break

        return chunks

    def save_chunks(self, chunks: List[Dict[str, Any]], output_file: str):
        """保存切块结果

        Args:
            chunks: 切块数据
            output_file: 输出文件路径
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存 {len(chunks)} 个切块到: {output_file}")

        # 统计信息
        total_content_length = sum(c['content_length'] for c in chunks)
        avg_content_length = total_content_length / len(chunks) if chunks else 0

        print(f"\n📊 切块统计:")
        print(f"  - 总块数: {len(chunks)}")
        print(f"  - 总字符数: {total_content_length}")
        print(f"  - 平均块大小: {avg_content_length:.0f} 字符")
        print(f"  - 章节数: {len(set(c['chapter_title'] for c in chunks))}")
        print(f"  - 小节数: {len(set(c['section_title'] for c in chunks if c['section_title']))}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='简化版Markdown切块工具')
    parser.add_argument('input_file', default='data/clean_data.md', help='输入Markdown文件')
    parser.add_argument('--output', '-o', default='data/simple_chunks.json', help='输出JSON文件')
    parser.add_argument('--max-size', type=int, default=800, help='最大块大小（超过则切块）')
    parser.add_argument('--split-size', type=int, default=500, help='切块目标大小')
    parser.add_argument('--overlap', type=int, default=100, help='块重叠大小')

    args = parser.parse_args()

    print(f"🚀 开始处理: {args.input_file}")
    print(f"  - 最大块大小: {args.max_size} 字符")
    print(f"  - 切块目标大小: {args.split_size} 字符")
    print(f"  - 重叠大小: {args.overlap} 字符")
    print()

    chunker = SimpleChunker(
        max_chunk_size=args.max_size,
        split_size=args.split_size,
        overlap=args.overlap
    )

    chunks = chunker.process_markdown(args.input_file)
    chunker.save_chunks(chunks, args.output)

    # 显示示例
    if chunks:
        print(f"\n📝 示例块（前3个）:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n块 {i}:")
            print(f"  章: {chunk.get('chapter_title', 'N/A')}")
            print(f"  节: {chunk.get('section_title', 'N/A')}")
            print(f"  长度: {chunk['content_length']} 字符")
            print(f"  内容: {chunk['content'][:100]}...")


if __name__ == "__main__":
    main()
