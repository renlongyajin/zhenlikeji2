#!/usr/bin/env python3
"""基于目录驱动的章节/小节切块脚本"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CHAPTER_IN_TOC_PATTERN = re.compile(r"^#?\s*(第[一二三四五六七八九十百0-9]+章.*)")
SECTION_IN_TOC_PATTERN = re.compile(r"(第[一二三四五六七八九十百0-9]+节.*?)(?=第[一二三四五六七八九十百0-9]+节|$)")
CHAPTER_HEADING_PATTERN = re.compile(r"^#?\s*第[一二三四五六七八九十百0-9]+章")
SECTION_HEADING_PATTERN = re.compile(r"^#?\s*第[一二三四五六七八九十百0-9]+节")
IMAGE_INLINE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def clean_trailing_page(text: str) -> str:
    text = re.sub(r"[\.．。·\s]*\d+.*$", "", text)
    text = re.sub(r"[\.．。·]+\s*$", "", text)
    return text.strip()


def normalize_chapter(text: str) -> str:
    text = text.lstrip("# ").strip()
    text = clean_trailing_page(text)
    return text


def normalize_section(text: str) -> str:
    text = text.strip()
    text = clean_trailing_page(text)
    match = re.match(r"(第[一二三四五六七八九十百0-9]+节)(.*)", text)
    if match:
        head, tail = match.groups()
        tail = tail.strip()
        return f"{head} {tail}".strip()
    return text


def extract_sections_from_line(text: str) -> List[str]:
    sections = []
    for match in SECTION_IN_TOC_PATTERN.finditer(text):
        name = normalize_section(match.group(1))
        if name:
            sections.append(name)
    return sections


def parse_toc(lines: List[str]) -> Tuple[List[Dict[str, Any]], int, int]:
    toc: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    toc_start = toc_end = -1
    in_toc = False

    for idx, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue

        if text.startswith("# 目 录"):
            in_toc = True
            toc_start = idx
            continue

        if not in_toc:
            continue

        if text.startswith("参考文献"):
            if current:
                toc.append(current)
            toc.append({"chapter": "参考文献", "sections": []})
            toc_end = idx
            break

        chapter_match = CHAPTER_IN_TOC_PATTERN.match(text)
        if chapter_match:
            if current:
                toc.append(current)
            title = normalize_chapter(chapter_match.group(1))
            current = {"chapter": title, "sections": []}
            continue

        if current:
            sections = extract_sections_from_line(text)
            current["sections"].extend(sections)

    if toc_start == -1:
        raise ValueError("未找到目录段落 (# 目 录)")
    if toc_end == -1:
        raise ValueError("未在目录中找到‘参考文献’结束标记")

    return toc, toc_start, toc_end


def extract_images(text: str) -> Tuple[str, List[Dict[str, str]]]:
    images: List[Dict[str, str]] = []

    def repl(match: re.Match) -> str:
        alt = match.group(1).strip()
        url = match.group(2).strip()
        images.append({"url": url, "alt": alt})
        return ""

    without_images = IMAGE_INLINE_PATTERN.sub(repl, text)
    return without_images, images


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_content(
    lines: List[str],
    toc: List[Dict[str, Any]],
    toc_start: int,
    toc_end: int,
    remove_images_from_full: bool = False,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    page_number = 1

    def add_chunk(chapter: str, section: str, content_lines: List[str]):
        content_full = "".join(content_lines).strip()
        if not content_full:
            return

        stripped_text, images = extract_images(content_full)
        content_clean = clean_text(stripped_text)
        full_text = stripped_text if remove_images_from_full else content_full

        chunks.append(
            {
                "chunk_id": "",
                "chunk_index": len(chunks),
                "chapter_title": chapter,
                "section_title": section,
                "page_number": page_number,
                "content": full_text,
                "content_full": full_text,
                "content_clean": content_clean,
                "images": images,
            }
        )

    # 前言（目录之前）
    preface = lines[:toc_start]
    add_chunk("前言", "前言", preface)

    # 初始化状态
    chapter_idx = -1
    current_chapter_title = ""
    current_sections: List[str] = []
    section_idx = -1
    current_section_title = ""
    pending_intro: List[str] = []
    buffer: List[str] = []
    force_section = False

    def flush_section():
        nonlocal buffer
        if section_idx < 0:
            buffer = []
            return
        add_chunk(current_chapter_title or "未命名章节", current_section_title or current_chapter_title, buffer)
        buffer = []

    content_lines = lines[toc_end + 1 :]

    for line in content_lines:
        text = line.strip()

        if not text:
            if section_idx >= 0:
                buffer.append(line)
            else:
                pending_intro.append(line)
            continue

        if CHAPTER_HEADING_PATTERN.match(text):
            flush_section()
            chapter_idx += 1
            if chapter_idx < len(toc):
                current_chapter_title = toc[chapter_idx]["chapter"]
                current_sections = toc[chapter_idx]["sections"]
            else:
                current_chapter_title = normalize_chapter(text)
                current_sections = []
            section_idx = -1
            current_section_title = ""
            pending_intro = []
            force_section = False
            page_number += 1
            # 如果该章没有小节，自动进入单节模式
            if not current_sections:
                section_idx = 0
                current_section_title = current_chapter_title
                force_section = True
            continue

        if SECTION_HEADING_PATTERN.match(text):
            flush_section()
            section_idx += 1
            if section_idx < len(current_sections):
                current_section_title = current_sections[section_idx]
            else:
                current_section_title = normalize_section(text)
            buffer = pending_intro.copy()
            pending_intro = []
            page_number += 1
            continue

        target = buffer if section_idx >= 0 or force_section else pending_intro
        target.append(line)

    flush_section()

    for idx, chunk in enumerate(chunks, start=1):
        chunk["chunk_id"] = f"chunk_{idx:04d}"
        chunk["chunk_index"] = idx - 1

    return chunks


def main():
    parser = argparse.ArgumentParser(description="章节/小节切块脚本")
    parser.add_argument("input_file", help="输入 Markdown 文件路径")
    parser.add_argument(
        "-o", "--output", default="data/chapter_chunks.json", help="输出 JSON 文件路径"
    )
    parser.add_argument(
        "--no-picture-link",
        action="store_true",
        help="从 content_full 中移除 Markdown 图片链接（图片信息仅保留在 images 字段）",
    )
    args = parser.parse_args()

    lines = Path(args.input_file).read_text(encoding="utf-8").splitlines(keepends=True)
    toc, toc_start, toc_end = parse_toc(lines)
    chunks = chunk_content(
        lines,
        toc,
        toc_start,
        toc_end,
        remove_images_from_full=args.no_picture_link,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ 已生成 {len(chunks)} 个章节/小节块 -> {args.output}")

    lengths_full = [len(c["content_full"]) for c in chunks]
    lengths_clean = [len(c["content_clean"]) for c in chunks]
    if lengths_full:
        avg_full = sum(lengths_full) / len(lengths_full)
        avg_clean = sum(lengths_clean) / len(lengths_clean)
        print("\n📊 切块统计：")
        print(f"  - 块总数: {len(chunks)}")
        print(f"  - 平均长度(content_full): {avg_full:.1f} 字符")
        print(f"  - 平均长度(content_clean): {avg_clean:.1f} 字符")
        print("  - 各块长度:")
        for chunk, lf, lc in zip(chunks, lengths_full, lengths_clean):
            print(
                f"    · {chunk['chunk_id']}: full={lf} 字符, clean={lc} 字符"
            )


if __name__ == "__main__":
    main()
