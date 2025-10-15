#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Text Extractor - Pure Text Version
专注于纯文本提取，移除所有图片相关逻辑
"""

import fitz  # PyMuPDF
import os
import json
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import re
import time
from dataclasses import dataclass

# 中文NLP库
try:
    import jieba
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True

    # 加载自定义词典（医学相关）
    medical_terms = [
        '主任医师', '副主任医师', '主治医师', '医师', '教授', '博士', '硕士',
        '靳芳', '冯靖', '植丽佳', '腺癌', '分化', '病理', '细胞学', '肿瘤',
        '恶件肺脏疾病', '哺脏少见病', '快速现场评价', '组学图谱'
    ]
    for term in medical_terms:
        jieba.add_word(term)

except ImportError:
    JIEBA_AVAILABLE = False
    logging.warning("jieba未安装，将使用基础分词方法")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR未安装，将使用备用OCR方法")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    """文本块信息"""
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    is_title: bool = False
    is_caption: bool = False
    title_type: Optional[str] = None  # chapter, section, subsection

class PDFTextExtractor:
    """PDF纯文本提取器 - 使用验证过的稳定OCR逻辑"""

    def __init__(self, output_dir: str = "data/extracted/text"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ocr_engine = None
        self._init_ocr()

    def _init_ocr(self):
        """初始化OCR引擎 - 简化的稳定配置"""
        if PADDLEOCR_AVAILABLE:
            try:
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='ch',
                    use_gpu=False,  # 强制使用CPU避免稳定性问题
                    det_limit_side_len=960,  # 降低分辨率以提高速度
                    drop_score=0.3,  # 提高阈值减少误检
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.4,  # 提高box阈值
                    det_db_unclip_ratio=1.5,
                    cpu_threads=2,  # 减少线程数
                    enable_mkldnn=False,  # 禁用mkldnn避免稳定性问题
                    use_mp=False,  # 禁用多进程
                    max_batch_size=5  # 减少批处理大小
                )
                logger.info("PaddleOCR初始化成功（简化配置）")
            except Exception as e:
                logger.error(f"PaddleOCR初始化失败: {e}")
                self.ocr_engine = None
        else:
            logger.warning("PaddleOCR不可用，将使用PyMuPDF内置文本提取")

    def _check_gpu_available(self) -> bool:
        """检测GPU是否可用"""
        try:
            import paddle
            return paddle.is_compiled_with_cuda()
        except:
            return False

    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """
        从PDF中提取纯文本内容 - 主处理函数
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        logger.info(f"开始处理PDF: {pdf_path}")

        try:
            # 打开PDF文档
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF总页数: {total_pages}")

            all_text_blocks = []

            # 处理每一页
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                logger.info(f"处理第 {page_num}/{total_pages} 页")

                try:
                    page = doc.load_page(page_idx)
                    page_text_blocks = self._extract_text_from_page(page, page_num)
                    all_text_blocks.extend(page_text_blocks)
                except Exception as e:
                    logger.error(f"处理第{page_num}页失败: {e}")
                    continue

            doc.close()

            # 生成纯文本内容
            text_content = self._generate_text_content(all_text_blocks, pdf_path, total_pages)

            # 保存结果
            result = {
                'pdf_path': str(pdf_path),
                'total_pages': total_pages,
                'extracted_text_blocks': len(all_text_blocks),
                'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'text_content': text_content
            }

            # 保存文本文件
            self._save_text_content(pdf_path, text_content)

            logger.info(f"文本提取完成，共提取 {len(all_text_blocks)} 个文本块")
            return result

        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            raise

    def _extract_text_from_page(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """
        从单个页面提取文本块 - 优先使用PyMuPDF，OCR作为备选
        """
        text_blocks = []

        try:
            # 首先尝试PyMuPDF的内置文本提取（更快更稳定）
            text_blocks = self._extract_text_with_pymupdf_fallback(page, page_num)

            # 如果PyMuPDF提取到有效文本，直接返回
            if text_blocks:
                logger.info(f"第{page_num}页：PyMuPDF文本提取成功，共{len(text_blocks)}个文本块")
                return text_blocks

            # 如果PyMuPDF没有提取到文本，尝试OCR
            logger.info(f"第{page_num}页：PyMuPDF未提取到文本，尝试OCR")

            # 获取高质量页面图像（用于OCR）
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            pix = None

            # 转换为OpenCV格式
            nparr = np.frombuffer(img_data, np.uint8)
            page_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 提取文本
            text_content = self._extract_text_with_ocr(page_image, page_num)

            # 转换文本块格式
            for block_data in text_content.get('text_blocks', []):
                text_block = TextBlock(
                    text=block_data['text'],
                    bbox=block_data['bbox'],
                    confidence=block_data['confidence'],
                    is_title=block_data.get('is_title', False),
                    is_caption=block_data.get('is_caption', False),
                    title_type=block_data.get('title_type')
                )
                text_blocks.append(text_block)

            return text_blocks

        except Exception as e:
            logger.error(f"页面 {page_num} 文本提取失败: {e}")
            # 如果OCR失败，返回空列表
            return []

    def _extract_text_with_ocr(self, image: np.ndarray, page_num: int) -> Dict:
        """
        使用OCR提取文本 - 复用unified_pdf_processor的稳定逻辑
        """
        if not self.ocr_engine:
            return {'full_text': '', 'text_blocks': [], 'confidence': 0.0}

        try:
            # 图像预处理 - 使用验证过的预处理逻辑
            processed_img = self._preprocess_for_ocr(image)

            # 执行OCR
            result = self.ocr_engine.ocr(processed_img, cls=True)

            if not result or not result[0]:
                return {'full_text': '', 'text_blocks': [], 'confidence': 0.0}

            # 解析OCR结果 - 使用验证过的逻辑
            text_blocks = []
            full_text = ""
            total_confidence = 0.0

            for line in result[0]:
                if line and len(line) >= 2:
                    bbox, (text, confidence) = line[0], line[1]
                    if text and confidence > 0.2:  # 置信度阈值
                        # 判断文本类型 - 使用验证过的标题检测
                        is_title, title_type = self._is_likely_title(text)
                        is_caption = self._is_likely_caption(text)

                        # 标准化边界框
                        normalized_bbox = self._normalize_bbox(bbox, image.shape)

                        text_block = {
                            'text': text,
                            'bbox': normalized_bbox,
                            'confidence': confidence,
                            'is_title': is_title,
                            'is_caption': is_caption,
                            'title_type': title_type
                        }

                        text_blocks.append(text_block)
                        full_text += text + "\n"
                        total_confidence += confidence

            # 按位置排序文本块 - 从上到下，从左到右
            text_blocks.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))

            avg_confidence = total_confidence / len(text_blocks) if text_blocks else 0.0

            return {
                'full_text': full_text.strip(),
                'text_blocks': text_blocks,
                'confidence': avg_confidence
            }

        except Exception as e:
            logger.error(f"OCR文本提取失败 (第{page_num}页): {e}")
            return {'full_text': '', 'text_blocks': [], 'confidence': 0.0}

    def _extract_text_with_pymupdf_fallback(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """
        PyMuPDF备选文本提取方法
        """
        text_blocks = []

        try:
            page_text = page.get_text()
            if page_text.strip():
                # 将页面文本按行分割
                lines = page_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        # 检测标题
                        is_title, title_type = self._is_likely_title(line)
                        is_caption = self._is_likely_caption(line)

                        text_block = TextBlock(
                            text=line,
                            bbox=(0, 0, 0, 0),  # 没有bbox信息
                            confidence=0.8,  # 默认置信度
                            is_title=is_title,
                            is_caption=is_caption,
                            title_type=title_type
                        )
                        text_blocks.append(text_block)

            return text_blocks

        except Exception as e:
            logger.error(f"PyMuPDF备选文本提取失败 (第{page_num}页): {e}")
            return []

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """OCR预处理 - 使用验证过的逻辑"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # 降噪
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

            return denoised

        except Exception as e:
            logger.error(f"OCR预处理失败: {e}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _normalize_bbox(self, bbox, image_shape):
        """标准化边界框 - 使用验证过的逻辑"""
        height, width = image_shape[:2]

        # PaddleOCR返回的是四个点的坐标
        if len(bbox) == 4 and isinstance(bbox[0], list):
            # 四个顶点格式
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
        else:
            # 假设是 [x,y,w,h] 格式
            x_min, y_min, w, h = bbox
            x_max, y_max = x_min + w, y_min + h

        # 确保坐标在图像范围内
        x_min = max(0, min(x_min, width))
        y_min = max(0, min(y_min, height))
        x_max = max(0, min(x_max, width))
        y_max = max(0, min(y_max, height))

        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    def _is_likely_title(self, text: str) -> tuple:
        """
        标题检测 - 使用验证过的逻辑
        返回 (是否为标题, 标题类型)
        """
        if not text or len(text.strip()) < 2:
            return False, None

        text = text.strip()

        # 章节模式 (第X章)
        chapter_pattern = r'^第[一二三四五六七八九十\d]+章\s*'
        if re.match(chapter_pattern, text):
            return True, 'chapter'

        # 节模式 (第X节)
        section_pattern = r'^第[一二三四五六七八九十\d]+节\s*'
        if re.match(section_pattern, text):
            return True, 'section'

        # 小节模式 (X、 or X.)
        subsection_pattern = r'^[一二三四五六七八九十]+\.?\s*'
        if re.match(subsection_pattern, text):
            return True, 'subsection'

        # 其他标题指示器
        title_indicators = [
            len(text) < 30,  # 短文本
            text.endswith('：') or text.endswith(':'),  # 以冒号结尾
            text.isupper() or (sum(1 for c in text if c.isupper()) / len(text) > 0.3),  # 很多大写
        ]

        if sum(title_indicators) >= 2:
            return True, 'generic'

        return False, None

    def _is_likely_caption(self, text: str) -> bool:
        """判断是否为图注"""
        if not text:
            return False

        # 图注特征
        if (text.startswith('图') and re.match(r'图[\d\-]+', text)) or \
           (text.startswith('表') and re.match(r'表[\d\-]+', text)):
            return True

        return False

    def _generate_text_content(self, text_blocks: List[TextBlock], pdf_path: Path, total_pages: int) -> str:
        """
        生成纯文本内容
        """
        # 文档标题（使用PDF文件名）
        doc_title = pdf_path.stem.replace('_', ' ').replace('-', ' ')

        # 构建文档头部信息
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

        # 按页面分组文本块
        pages_dict = {}
        for block in text_blocks:
            if block.page_num not in pages_dict:
                pages_dict[block.page_num] = []
            pages_dict[block.page_num].append(block)

        # 生成每页的内容
        for page_num in sorted(pages_dict.keys()):
            page_blocks = pages_dict[page_num]

            # 添加页面标题
            content_lines.append(f"#### 第{page_num}页")
            content_lines.append("")

            # 添加该页的文本内容
            for block in page_blocks:
                if block.is_title:
                    # 根据标题级别添加相应的markdown标记
                    title_prefix = {
                        'chapter': '#',
                        'section': '##',
                        'subsection': '###',
                        'generic': '####'
                    }.get(block.title_type, '##')
                    content_lines.append(f"{title_prefix} {block.text}")
                else:
                    content_lines.append(block.text)

                content_lines.append("")  # 空行分隔

            content_lines.append("---")  # 页面分隔线
            content_lines.append("")

        return "\n".join(content_lines)

    def _save_text_content(self, pdf_path: Path, text_content: str) -> str:
        """
        保存文本内容到文件
        """
        # 生成输出文件名
        output_filename = f"{pdf_path.stem}_extracted.txt"
        output_path = self.output_dir / output_filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text_content)

            logger.info(f"文本内容已保存到: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"保存文本文件失败: {e}")
            raise

    def process_multiple_pdfs(self, pdf_dir: str) -> List[Dict[str, any]]:
        """
        批量处理多个PDF文件
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"目录不存在: {pdf_dir}")

        # 查找所有PDF文件
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"在目录 {pdf_dir} 中未找到PDF文件")
            return []

        logger.info(f"找到 {len(pdf_files)} 个PDF文件")

        results = []
        for pdf_file in pdf_files:
            try:
                logger.info(f"处理文件: {pdf_file}")
                result = self.extract_text_from_pdf(str(pdf_file))
                results.append(result)
            except Exception as e:
                logger.error(f"处理文件 {pdf_file} 失败: {e}")
                continue

        logger.info(f"批量处理完成，成功处理 {len(results)} 个文件")
        return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="PDF纯文本提取工具")
    parser.add_argument("input", help="输入PDF文件路径或包含PDF文件的目录")
    parser.add_argument("-o", "--output", default="data/extracted/text",
                       help="输出目录 (默认: data/extracted/text)")
    parser.add_argument("-b", "--batch", action="store_true",
                       help="批量处理模式（处理整个目录）")

    args = parser.parse_args()

    # 创建提取器
    extractor = PDFTextExtractor(output_dir=args.output)

    try:
        if args.batch or Path(args.input).is_dir():
            # 批量处理
            results = extractor.process_multiple_pdfs(args.input)
            print(f"批量处理完成，成功提取 {len(results)} 个文件的文本内容")
        else:
            # 单个文件处理
            result = extractor.extract_text_from_pdf(args.input)
            print(f"文本提取完成！")
            print(f"原始PDF: {result['pdf_path']}")
            print(f"总页数: {result['total_pages']}")
            print(f"文本块数量: {result['extracted_text_blocks']}")
            print(f"输出目录: {args.output}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())