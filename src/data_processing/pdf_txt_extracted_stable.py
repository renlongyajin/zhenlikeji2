#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Text Extractor - Stable OCR Version
稳定的PDF文本提取器，包含OCR功能但具有强大的错误处理
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import re
import time
from dataclasses import dataclass

# OCR相关导入
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR未安装，将仅使用PyMuPDF文本提取")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    """文本块信息"""
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    page_num: int
    is_title: bool = False
    title_type: Optional[str] = None  # chapter, section, subsection

class StablePDFTextExtractor:
    """稳定的PDF文本提取器 - 优先PyMuPDF，OCR备选"""

    def __init__(self, output_dir: str = "data/extracted/text_stable"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # OCR引擎
        self.ocr_engine = None
        if PADDLEOCR_AVAILABLE:
            self._init_ocr()

    def _init_ocr(self):
        """初始化OCR引擎 - 最简配置"""
        try:
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=False,  # 强制CPU
                det_limit_side_len=480,  # 降低分辨率
                drop_score=0.3,
                det_db_thresh=0.3,
                det_db_box_thresh=0.4,
                cpu_threads=1,  # 单线程
                enable_mkldnn=False,
                use_mp=False,
                max_batch_size=1
            )
            logger.info("OCR引擎初始化成功")
        except Exception as e:
            logger.error(f"OCR引擎初始化失败: {e}")
            self.ocr_engine = None

    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """从PDF中提取文本内容"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        logger.info(f"开始处理PDF: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF总页数: {total_pages}")

            all_text_blocks = []
            pages_with_text = 0

            # 处理每一页
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                logger.info(f"处理第 {page_num}/{total_pages} 页")

                try:
                    page = doc.load_page(page_idx)
                    page_blocks = self._extract_text_from_page_stable(page, page_num)

                    if page_blocks:
                        all_text_blocks.extend(page_blocks)
                        pages_with_text += 1
                        logger.info(f"第{page_num}页提取成功，共{len(page_blocks)}个文本块")
                    else:
                        logger.warning(f"第{page_num}页无文本内容")

                except Exception as e:
                    logger.error(f"处理第{page_num}页失败: {e}")
                    continue

            doc.close()

            # 生成文本内容
            text_content = self._generate_text_content(all_text_blocks, pdf_path, total_pages)

            result = {
                'pdf_path': str(pdf_path),
                'total_pages': total_pages,
                'pages_with_text': pages_with_text,
                'text_blocks': len(all_text_blocks),
                'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'text_content': text_content
            }

            # 保存文件
            output_path = self._save_text_content(pdf_path, text_content)
            result['output_path'] = output_path

            logger.info(f"文本提取完成，成功处理 {pages_with_text}/{total_pages} 页")
            return result

        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            raise

    def _extract_text_from_page_stable(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """稳定的单页文本提取"""
        text_blocks = []

        try:
            # 1. 优先尝试PyMuPDF
            pymupdf_blocks = self._try_pymupdf_extraction(page, page_num)
            if pymupdf_blocks:
                return pymupdf_blocks

            # 2. PyMuPDF失败，尝试OCR
            if self.ocr_engine:
                logger.info(f"第{page_num}页：尝试OCR提取")
                ocr_blocks = self._try_ocr_extraction(page, page_num)
                if ocr_blocks:
                    return ocr_blocks

            logger.warning(f"第{page_num}页：无有效文本内容")
            return []

        except Exception as e:
            logger.error(f"第{page_num}页提取失败: {e}")
            return []

    def _try_pymupdf_extraction(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """尝试PyMuPDF提取"""
        try:
            text_blocks = []
            page_text = page.get_text()

            if page_text and page_text.strip():
                # 按行处理文本
                lines = page_text.strip().split('\n')
                for line_num, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        # 检测标题
                        is_title, title_type = self._is_likely_title(line)

                        text_block = TextBlock(
                            text=line,
                            bbox=(0, line_num * 20, 100, (line_num + 1) * 20),  # 模拟位置
                            confidence=0.9,  # PyMuPDF文本置信度高
                            page_num=page_num,
                            is_title=is_title,
                            title_type=title_type
                        )
                        text_blocks.append(text_block)

                if text_blocks:
                    logger.info(f"第{page_num}页：PyMuPDF提取成功，{len(text_blocks)}个文本块")
                    return text_blocks

            return []

        except Exception as e:
            logger.error(f"PyMuPDF提取失败: {e}")
            return []

    def _try_ocr_extraction(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """尝试OCR提取"""
        try:
            # 获取页面图像
            pix = page.get_pixmap(dpi=200)  # 降低DPI以提高速度
            img_data = pix.tobytes("png")
            pix = None

            # 转换为numpy数组
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error(f"第{page_num}页：图像解码失败")
                return []

            # 执行OCR
            result = self.ocr_engine.ocr(img, cls=True)

            if not result or not result[0]:
                logger.warning(f"第{page_num}页：OCR无结果")
                return []

            # 解析OCR结果
            text_blocks = []
            for line_result in result[0]:
                if not line_result or len(line_result) < 2:
                    continue

                try:
                    bbox, (text, confidence) = line_result[0], line_result[1]

                    if not text or confidence < 0.2:
                        continue

                    # 检测标题
                    is_title, title_type = self._is_likely_title(text)

                    # 标准化bbox
                    norm_bbox = self._normalize_bbox(bbox, img.shape)

                    text_block = TextBlock(
                        text=text,
                        bbox=norm_bbox,
                        confidence=confidence,
                        page_num=page_num,
                        is_title=is_title,
                        title_type=title_type
                    )
                    text_blocks.append(text_block)

                except (IndexError, ValueError, TypeError) as e:
                    logger.warning(f"解析OCR行结果失败: {e}")
                    continue

            # 按位置排序
            text_blocks.sort(key=lambda x: (x.bbox[1], x.bbox[0]))

            logger.info(f"第{page_num}页：OCR提取成功，{len(text_blocks)}个文本块")
            return text_blocks

        except Exception as e:
            logger.error(f"OCR提取失败: {e}")
            return []

    def _normalize_bbox(self, bbox, image_shape) -> Tuple[int, int, int, int]:
        """标准化边界框"""
        try:
            height, width = image_shape[:2]

            # PaddleOCR格式：四个顶点
            if len(bbox) == 4 and isinstance(bbox[0], list):
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
            else:
                # 其他格式
                x_min, y_min = int(bbox[0]), int(bbox[1])
                if len(bbox) == 4:
                    x_max, y_max = x_min + int(bbox[2]), y_min + int(bbox[3])
                else:
                    x_max, y_max = int(bbox[2]), int(bbox[3])

            # 确保在图像范围内
            x_min = max(0, min(x_min, width))
            y_min = max(0, min(y_min, height))
            x_max = max(0, min(x_max, width))
            y_max = max(0, min(y_max, height))

            return (x_min, y_min, x_max - x_min, y_max - y_min)

        except Exception as e:
            logger.error(f"边界框标准化失败: {e}")
            return (0, 0, 100, 20)  # 默认bbox

    def _is_likely_title(self, text: str) -> tuple:
        """标题检测"""
        if not text or len(text.strip()) < 2:
            return False, None

        text = text.strip()

        # 章节模式 (第X章)
        if re.match(r'^第[一二三四五六七八九十\d]+章\s*', text):
            return True, 'chapter'

        # 节模式 (第X节)
        if re.match(r'^第[一二三四五六七八九十\d]+节\s*', text):
            return True, 'section'

        # 小节模式 (X、 or X.)
        if re.match(r'^[一二三四五六七八九十]+\.?\s*', text):
            return True, 'subsection'

        # 其他标题指示器
        title_indicators = [
            len(text) < 30,  # 短文本
            text.endswith('：') or text.endswith(':'),  # 以冒号结尾
        ]

        if sum(title_indicators) >= 2:
            return True, 'generic'

        return False, None

    def _generate_text_content(self, text_blocks: List[TextBlock], pdf_path: Path, total_pages: int) -> str:
        """生成文本内容"""
        doc_title = pdf_path.stem.replace('_', ' ').replace('-', ' ')

        content_lines = [
            f"# {doc_title}",
            "",
            "## 文档信息",
            "",
            f"- **原始PDF**: {pdf_path.name}",
            f"- **总页数**: {total_pages}",
            f"- **提取时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **文本块数量**: {len(text_blocks)}",
            "",
            "---",
            ""
        ]

        # 按页面分组
        pages_dict = {}
        for block in text_blocks:
            if block.page_num not in pages_dict:
                pages_dict[block.page_num] = []
            pages_dict[block.page_num].append(block)

        # 生成每页内容
        for page_num in sorted(pages_dict.keys()):
            page_blocks = pages_dict[page_num]

            content_lines.append(f"#### 第{page_num}页")
            content_lines.append("")

            for block in page_blocks:
                if block.is_title:
                    title_prefix = {
                        'chapter': '#',
                        'section': '##',
                        'subsection': '###',
                        'generic': '####'
                    }.get(block.title_type, '##')
                    content_lines.append(f"{title_prefix} {block.text}")
                else:
                    content_lines.append(block.text)
                content_lines.append("")

            content_lines.append("---")
            content_lines.append("")

        return "\n".join(content_lines)

    def _save_text_content(self, pdf_path: Path, content: str) -> str:
        """保存文本内容"""
        output_filename = f"{pdf_path.stem}_stable_extracted.txt"
        output_path = self.output_dir / output_filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"文本内容已保存: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"保存文本文件失败: {e}")
            raise


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="稳定版PDF文本提取工具")
    parser.add_argument("input", help="输入PDF文件路径")
    parser.add_argument("-o", "--output", default="data/extracted/text_stable",
                       help="输出目录 (默认: data/extracted/text_stable)")

    args = parser.parse_args()

    extractor = StablePDFTextExtractor(output_dir=args.output)

    try:
        result = extractor.extract_text_from_pdf(args.input)

        print(f"文本提取完成！")
        print(f"原始PDF: {result['pdf_path']}")
        print(f"总页数: {result['total_pages']}")
        print(f"有文本的页面: {result['pages_with_text']}")
        print(f"文本块数量: {result['text_blocks']}")
        print(f"输出文件: {result['output_path']}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())