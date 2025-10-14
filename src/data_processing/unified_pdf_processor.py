#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一PDF处理器 - 智能文本和图像提取
支持上下文感知的图像命名和结构化Markdown输出
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
from collections import defaultdict

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR未安装，将使用备用OCR方法")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedImage:
    """提取的图像信息"""
    page_num: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    image_type: str
    confidence: float
    context_text: str
    suggested_filename: str
    raw_image: np.ndarray

@dataclass
class TextBlock:
    """文本块信息"""
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    is_title: bool = False
    is_caption: bool = False

class UnifiedPDFProcessor:
    """统一PDF处理器 - 智能文本和图像提取"""

    def __init__(self, pdf_path: str, output_dir: str = "data/extracted"):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.output_dir / "texts").mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "markdown").mkdir(exist_ok=True)
        (self.output_dir / "debug").mkdir(exist_ok=True)

        self.doc = None
        self.ocr_engine = None
        self.processing_results = []

        # 人物名称识别模式
        self.name_patterns = [
            r'([^，。；：\s]{2,4})\s*[，。；：]\s*([^，。；：\s]{2,4})',  # 姓名模式
            r'([^，。；：\s]{2,3})[，。；：]',  # 单名模式
        ]

        # 图像标题模式
        self.caption_patterns = [
            r'图[\d\-]+[\s]*[^\n]+',  # 图2-1 腺癌
            r'表[\d\-]+[\s]*[^\n]+',  # 表1-2 统计数据
            r'[^，。；：\n]{2,20}图[\d\-]+',  # 反向匹配
        ]

        self._init_ocr()

    def _init_ocr(self):
        """初始化OCR引擎"""
        if PADDLEOCR_AVAILABLE:
            try:
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='ch',
                    use_gpu=self._check_gpu_available(),
                    det_limit_side_len=1280,
                    drop_score=0.2,
                    det_db_thresh=0.2,
                    det_db_box_thresh=0.3,
                    det_db_unclip_ratio=1.8
                )
                logger.info("PaddleOCR初始化成功")
            except Exception as e:
                logger.error(f"PaddleOCR初始化失败: {e}")
                self.ocr_engine = None
        else:
            logger.warning("PaddleOCR不可用，将使用备用文本提取方法")

    def _check_gpu_available(self) -> bool:
        """检测GPU是否可用"""
        try:
            import paddle
            return paddle.is_compiled_with_cuda()
        except:
            return False

    def process_pdf(self, target_pages: List[int] = None) -> Dict:
        """处理PDF文件 - 统一提取文本和图像"""
        logger.info(f"开始统一处理PDF: {self.pdf_path}")

        try:
            self.doc = fitz.open(self.pdf_path)
            total_pages = len(self.doc)
            logger.info(f"PDF总页数: {total_pages}")

            if target_pages:
                pages_to_process = [p-1 for p in target_pages if p <= total_pages]
            else:
                pages_to_process = range(total_pages)

            # 处理每一页
            for page_idx in pages_to_process:
                page_num = page_idx + 1
                logger.info(f"处理第 {page_num}/{total_pages} 页")

                try:
                    page_result = self._process_page_unified(page_idx)
                    if page_result:
                        self.processing_results.append(page_result)
                        self._save_page_result(page_idx, page_result)
                except Exception as e:
                    logger.error(f"处理第{page_num}页失败: {e}")
                    continue

            # 生成最终结果
            result = self._generate_final_result()

            logger.info("PDF统一处理完成")
            return result

        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            raise
        finally:
            if self.doc:
                self.doc.close()

    def _process_page_unified(self, page_idx: int) -> Optional[Dict]:
        """统一处理单个页面"""
        page = self.doc.load_page(page_idx)
        page_num = page_idx + 1

        # 获取高质量页面图像
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        pix = None

        # 转换为OpenCV格式
        nparr = np.frombuffer(img_data, np.uint8)
        page_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 提取文本
        text_content = self._extract_text_with_ocr(page_image, page_num)

        # 提取图像
        extracted_images = self._extract_images_intelligent(page_image, page_num, text_content)

        # 分析页面结构
        page_structure = self._analyze_page_structure(text_content, extracted_images, page_image)

        page_result = {
            'page_num': page_num,
            'text_content': text_content,
            'extracted_images': extracted_images,
            'page_structure': page_structure,
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        return page_result

    def _extract_text_with_ocr(self, image: np.ndarray, page_num: int) -> Dict:
        """使用OCR提取文本"""
        if not self.ocr_engine:
            return {'full_text': '', 'text_blocks': [], 'confidence': 0.0}

        try:
            # 图像预处理
            processed_img = self._preprocess_for_ocr(image)

            # 执行OCR
            result = self.ocr_engine.ocr(processed_img, cls=True)

            if not result or not result[0]:
                return {'full_text': '', 'text_blocks': [], 'confidence': 0.0}

            # 解析OCR结果
            text_blocks = []
            full_text = ""
            total_confidence = 0.0

            for line in result[0]:
                if line and len(line) >= 2:
                    bbox, (text, confidence) = line[0], line[1]
                    if text and confidence > 0.2:
                        # 判断文本类型
                        is_title = self._is_likely_title(text)
                        is_caption = self._is_likely_caption(text)

                        text_block = TextBlock(
                            text=text,
                            bbox=self._normalize_bbox(bbox, image.shape),
                            confidence=confidence,
                            is_title=is_title,
                            is_caption=is_caption
                        )

                        text_blocks.append(text_block)
                        full_text += text + "\n"
                        total_confidence += confidence

            # 按位置排序文本块
            text_blocks.sort(key=lambda x: (x.bbox[1], x.bbox[0]))

            avg_confidence = total_confidence / len(text_blocks) if text_blocks else 0.0

            return {
                'full_text': full_text.strip(),
                'text_blocks': [vars(block) for block in text_blocks],
                'confidence': avg_confidence
            }

        except Exception as e:
            logger.error(f"OCR文本提取失败 (第{page_num}页): {e}")
            return {'full_text': '', 'text_blocks': [], 'confidence': 0.0}

    def _extract_images_intelligent(self, image: np.ndarray, page_num: int, text_content: Dict) -> List[ExtractedImage]:
        """智能提取图像 - 使用准确切割逻辑"""
        extracted_images = []

        try:
            # 检测候选图像区域（使用修正后的准确检测）
            candidate_regions = self._detect_image_regions_intelligent(image, text_content)

            for i, region in enumerate(candidate_regions):
                x, y, w, h = region['bbox']
                roi = image[y:y+h, x:x+w]

                # 分析图像内容
                content_analysis = self._analyze_image_content(roi)

                # 获取上下文文本
                context_text = self._get_context_text(region['bbox'], text_content.get('text_blocks', []))

                # 智能命名
                suggested_filename = self._generate_intelligent_filename(
                    roi, region, context_text, page_num, i
                )

                # 分类图像
                image_type = self._classify_image_type(roi, content_analysis, context_text, region)

                extracted_image = ExtractedImage(
                    page_num=page_num,
                    bbox=region['bbox'],
                    image_type=image_type,
                    confidence=region['confidence'],
                    context_text=context_text,
                    suggested_filename=suggested_filename,
                    raw_image=roi.copy()
                )

                extracted_images.append(extracted_image)

                # 保存图像
                self._save_extracted_image(extracted_image)

            return extracted_images

        except Exception as e:
            logger.error(f"智能图像提取失败 (第{page_num}页): {e}")
            return []

    def _detect_image_regions_intelligent(self, image: np.ndarray, text_content: Dict) -> List[Dict]:
        """基于高效提取器的准确图像区域检测"""
        detected_regions = []

        try:
            page_height, page_width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 使用高效提取器优化过的矩形检测方法
            edges = cv2.Canny(gray, 50, 150)

            # 形态学操作增强矩形特征
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                # 基础筛选条件 - 加强尺寸过滤（与高效提取器一致）
                if area < 15000:  # 面积太小
                    continue

                # 新增：极小小图直接过滤 - 页码通常是这个尺寸范围
                if w < 200 or h < 200 or w * h < 40000:  # 极小的矩形，很可能是页码
                    logger.info(f"  过滤极小小图: 尺寸{w}x{h}, 面积{w*h}")
                    continue

                aspect_ratio = w / h if h > 0 else 1.0
                if not (0.4 <= aspect_ratio <= 2.5):  # 宽高比不合适
                    continue

                # 矩形度检测
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)

                if box_area > 0:
                    rectangularity = area / box_area
                else:
                    rectangularity = 0

                if rectangularity < 0.7:  # 矩形度不够
                    continue

                # 页码过滤 - 关键改进（使用高效提取器的逻辑）
                roi = image[y:y+h, x:x+w]
                if self._is_likely_page_number(roi, (x, y, w, h), page_width, page_height):
                    logger.info(f"  过滤掉可能的页码: 位置({x},{y}), 尺寸{w}x{h}")
                    continue

                # 内容验证（使用高效提取器的验证逻辑）
                if self._validate_content(roi):
                    detected_regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'rectangularity': rectangularity,
                        'confidence': rectangularity * 0.8 + min(area / 100000, 0.2)
                    })

            # 按置信度排序
            detected_regions.sort(key=lambda x: x['confidence'], reverse=True)

            # 每页最多保留3个最可能的图像（与高效提取器一致）
            return detected_regions[:3]

        except Exception as e:
            logger.error(f"准确图像区域检测失败: {e}")
            return []

    def _validate_content(self, roi: np.ndarray) -> bool:
        """验证ROI内容 - 来自高效提取器"""
        try:
            height, width = roi.shape[:2]

            # 基本尺寸检查
            if width < 50 or height < 50:
                return False

            # 转换为灰度图
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 方差检查 - 确保内容有足够的变化
            variance = np.var(gray)
            if variance < 200:
                return False

            # 边缘密度检查
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (width * height)

            if edge_density < 0.01 or edge_density > 0.5:
                return False

            return True

        except Exception:
            return False

    def _is_likely_page_number(self, roi: np.ndarray, bbox: tuple, page_width: int, page_height: int) -> bool:
        """判断是否为页码 - 来自高效提取器"""
        try:
            x, y, w, h = bbox

            # 1. 尺寸检查 - 页码通常较小
            if w > 200 or h > 150:
                return False

            # 2. 面积检查
            if w * h > 30000:
                return False

            # 3. 长宽比检查 - 页码通常是横向矩形
            aspect_ratio = w / h if h > 0 else 1.0
            if aspect_ratio < 1.5:
                return False

            # 4. 位置检查 - 页码通常在页面底部
            relative_y = (y + h) / page_height
            if relative_y < 0.8:  # 不在底部20%区域
                return False

            # 5. 文字特征检查
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            small_components = 0
            for i in range(1, num_labels):
                if 10 < stats[i, cv2.CC_STAT_AREA] < 100:
                    small_components += 1

            text_likeness = small_components / max(num_labels - 1, 1)

            # 6. 简单性检查 - 页码通常结构简单
            variance = np.var(gray)
            if variance > 1000:  # 太复杂
                return False

            # 综合判断 - 放宽文字要求，加强其他特征
            is_page_number = (
                (aspect_ratio > 1.5 and relative_y > 0.8 and w * h < 30000) or  # 位置+尺寸特征
                (text_likeness > 0.2 and w < 200 and h < 150) or  # 文字特征+小尺寸
                (aspect_ratio > 1.7 and w * h < 20000)  # 极端长宽比+小面积
            )

            if is_page_number:
                logger.info(f"    检测到页码特征: 尺寸{w}x{h}, 位置{relative_y:.2f}, 文字相似度{text_likeness:.2f}")

            return is_page_number

        except Exception:
            return False

    def _detect_medical_image_regions(self, image: np.ndarray) -> List[Dict]:
        """检测医学图像区域"""
        regions = []

        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            height, width = image.shape[:2]

            # 医学图像颜色特征
            # 蓝色边框（常见医学图像边框）
            lower_blue = np.array([100, 30, 50])
            upper_blue = np.array([130, 200, 200])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # 红色（血液、组织）
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            # 紫色（某些染色）
            lower_purple = np.array([140, 30, 50])
            upper_purple = np.array([160, 255, 255])
            purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

            # 合并医学颜色特征
            medical_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(red_mask, purple_mask))

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            medical_morphed = cv2.morphologyEx(medical_mask, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(medical_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                if area > 15000:
                    # 计算医学特征得分
                    roi = image[y:y+h, x:x+w]
                    medical_score = self._calculate_medical_score(roi)

                    regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': medical_score,
                        'detection_method': 'medical_color',
                        'medical_score': medical_score
                    })

            return regions

        except Exception as e:
            logger.error(f"医学图像区域检测失败: {e}")
            return []

    def _detect_rectangular_regions(self, image: np.ndarray) -> List[Dict]:
        """检测矩形区域"""
        regions = []

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)

            # 形态学操作增强矩形特征
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                if area > 20000:  # 较大的区域
                    # 计算矩形度
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box_area = cv2.contourArea(box)

                    rectangularity = area / box_area if box_area > 0 else 0

                    if rectangularity > 0.7:
                        regions.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'confidence': rectangularity,
                            'detection_method': 'rectangular_shape',
                            'rectangularity': rectangularity
                        })

            return regions

        except Exception as e:
            logger.error(f"矩形区域检测失败: {e}")
            return []

    def _detect_salient_regions_advanced(self, image: np.ndarray) -> List[Dict]:
        """高级显著性区域检测"""
        regions = []

        try:
            # 转换为LAB颜色空间
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # 计算显著性
            gaussian1 = cv2.GaussianBlur(l, (21, 21), 0)
            gaussian2 = cv2.GaussianBlur(l, (5, 5), 0)
            saliency = cv2.absdiff(gaussian1, gaussian2)

            # 二值化
            _, binary = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                if area > 20000:
                    # 计算显著性得分
                    roi = image[y:y+h, x:x+w]
                    saliency_score = np.var(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) / 1000

                    regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': min(saliency_score, 1.0),
                        'detection_method': 'saliency',
                        'saliency_score': saliency_score
                    })

            return regions

        except Exception as e:
            logger.error(f"显著性区域检测失败: {e}")
            return []

    def _detect_portrait_regions(self, image: np.ndarray) -> List[Dict]:
        """检测人物头像区域"""
        regions = []

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 使用OpenCV的人脸检测
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                # 扩展区域以包含完整的头像
                margin = int(max(w, h) * 0.5)
                x_new = max(0, x - margin)
                y_new = max(0, y - margin)
                w_new = min(image.shape[1] - x_new, w + 2 * margin)
                h_new = min(image.shape[0] - y_new, h + 2 * margin)

                # 确保宽高比接近1:1
                if 0.7 <= w_new/h_new <= 1.3:
                    regions.append({
                        'bbox': (x_new, y_new, w_new, h_new),
                        'area': w_new * h_new,
                        'confidence': 0.8,  # 人脸检测置信度
                        'detection_method': 'face_detection',
                        'is_portrait': True
                    })

            return regions

        except Exception as e:
            logger.error(f"人物头像检测失败: {e}")
            return []

    def _generate_intelligent_filename(self, image: np.ndarray, region: Dict, context_text: str,
                                     page_num: int, index: int) -> str:
        """生成智能文件名"""

        # 最高优先级：检测方法明确指示
        detection_method = region.get('detection_method', 'unknown')

        # 人物头像 - 多重方式尝试提取人名
        if region.get('is_portrait') or detection_method == 'face_detection':
            # 1. 尝试从上下文提取人名（使用更宽松的模式）
            name = self._extract_person_name_enhanced(context_text)
            if name:
                return f"人物_{name}.png"

            # 2. 尝试从整个页面文本中提取（扩展搜索范围）
            extended_name = self._extract_person_name_from_extended_context(context_text)
            if extended_name:
                return f"人物_{extended_name}.png"

            # 3. 如果检测到人脸但无法提取名字，使用页面信息
            return f"人物_头像_{page_num}_{index}.png"

        # 医学图像
        elif detection_method == 'medical_color':
            # 尝试提取图注
            caption = self._extract_figure_caption(context_text)
            if caption:
                # 清理文件名
                clean_caption = re.sub(r'[\\/:*?"\u003c\u003e|]', '', caption)
                clean_caption = clean_caption.strip()[:50]  # 限制长度
                return f"{clean_caption}.png"
            else:
                return f"医学图像_{page_num}_{index}.png"

        elif detection_method == 'rectangular_shape':
            return f"图表_{page_num}_{index}.png"

        elif detection_method == 'saliency':
            return f"图像_{page_num}_{index}.png"

        else:
            # 通用处理：尝试从上下文提取有意义的信息
            intelligent_name = self._extract_name_from_context(context_text, region)
            if intelligent_name:
                return intelligent_name
            else:
                return f"提取图像_{page_num}_{index}.png"

    def _extract_name_from_context(self, context_text: str, region: Dict) -> Optional[str]:
        """从上下文文本中提取名称"""
        if not context_text:
            return None

        # 人物名称模式
        for pattern in self.name_patterns:
            matches = re.findall(pattern, context_text)
            if matches:
                # 返回第一个匹配的名称
                if isinstance(matches[0], tuple):
                    name = ''.join(matches[0])
                else:
                    name = matches[0]

                # 验证名称合理性（2-4个字符）
                if 2 <= len(name) <= 4:
                    return name

        # 图注模式
        for pattern in self.caption_patterns:
            matches = re.findall(pattern, context_text)
            if matches:
                caption = matches[0]
                # 清理图注
                clean_caption = caption.strip()
                if len(clean_caption) > 3 and len(clean_caption) < 100:
                    return clean_caption

        return None

    def _extract_person_name_enhanced(self, text: str) -> Optional[str]:
        """增强版人名提取 - 更宽松的模式"""
        if not text:
            return None

        # 扩展的中文姓名模式
        name_patterns = [
            r'([^，。；：\s]{2,4})\s*[，。；：]\s*(?:主任医师|副主任医师|主治医师|医师|教授|博士|硕士)',  # 姓名, 职务
            r'([^，。；：\s]{2,4})\s*(?:医生|先生|女士|老师|主任)',  # 姓名 + 称谓
            r'(?:姓名[：:]\s*)?([^，。；：\s]{2,4})[，。；：\s]',  # 明确的姓名指示
            r'([^，。；：\s]{2,3})[，。；：]',  # 姓名后接标点（更宽松）
            r'^\s*([^，。；：\s]{2,4})\s*$',  # 单独的姓名
        ]

        # 扩展的姓氏列表
        extended_surnames = ['王', '李', '张', '刘', '陈', '杨', '黄', '赵', '周', '吴',
                           '徐', '孙', '胡', '朱', '高', '林', '何', '郭', '马', '罗',
                           '梁', '宋', '郑', '谢', '韩', '唐', '冯', '于', '董', '萧',
                           '程', '曹', '袁', '邓', '许', '傅', '沈', '曾', '彭', '吕',
                           '苏', '卢', '蒋', '蔡', '贾', '丁', '魏', '薛', '叶', '阎',
                           '余', '潘', '杜', '戴', '夏', '钟', '汪', '田', '任', '姜',
                           '范', '方', '石', '姚', '谭', '廖', '邹', '熊', '金', '陆',
                           '郝', '孔', '白', '崔', '康', '毛', '邱', '秦', '江', '史',
                           '顾', '侯', '邵', '孟', '龙', '万', '段', '雷', '钱', '汤',
                           '尹', '黎', '易', '常', '武', '乔', '贺', '赖', '龚', '文',
                           '靳', '冯', '植']  # 包含用户提到的姓氏

        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[0], tuple):
                    potential_name = matches[0][0]
                else:
                    potential_name = matches[0]

                # 更宽松的姓名验证
                if 2 <= len(potential_name) <= 4:
                    # 检查是否包含常见姓氏
                    if potential_name[0] in extended_surnames:
                        return potential_name
                    # 如果没有匹配姓氏但符合长度要求，也接受（更宽松）
                    elif len(potential_name) >= 2:
                        return potential_name

        return None

    def _extract_person_name_from_extended_context(self, text: str) -> Optional[str]:
        """从扩展上下文提取人名 - 更广泛的搜索"""
        if not text:
            return None

        # 更宽松的模式，可能包含更多假阳性但也更可能找到真实人名
        loose_patterns = [
            r'([\u4e00-\u9fff]{2,4})',  # 任意2-4个中文字符
        ]

        # 扩展的姓氏列表
        extended_surnames = ['王', '李', '张', '刘', '陈', '杨', '黄', '赵', '周', '吴',
                           '徐', '孙', '胡', '朱', '高', '林', '何', '郭', '马', '罗',
                           '梁', '宋', '郑', '谢', '韩', '唐', '冯', '于', '董', '萧',
                           '程', '曹', '袁', '邓', '许', '傅', '沈', '曾', '彭', '吕',
                           '苏', '卢', '蒋', '蔡', '贾', '丁', '魏', '薛', '叶', '阎',
                           '余', '潘', '杜', '戴', '夏', '钟', '汪', '田', '任', '姜',
                           '范', '方', '石', '姚', '谭', '廖', '邹', '熊', '金', '陆',
                           '郝', '孔', '白', '崔', '康', '毛', '邱', '秦', '江', '史',
                           '顾', '侯', '邵', '孟', '龙', '万', '段', '雷', '钱', '汤',
                           '尹', '黎', '易', '常', '武', '乔', '贺', '赖', '龚', '文',
                           '靳', '冯', '植']  # 包含用户提到的姓氏

        for pattern in loose_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2 and len(match) <= 4 and match[0] in extended_surnames:
                    return match

        return None

    def _extract_person_name(self, text: str) -> Optional[str]:
        """从文本中提取人名 - 原始版本，保持兼容性"""
        if not text:
            return None

        # 常见中文姓名模式
        name_patterns = [
            r'([^，。；：\s]{2,4})\s*[,，]\s*([^，。；：\s]{2,4})',  # 姓名, 职务
            r'([^，。；：\s]{2,3})[，。；：]',  # 姓名后接标点
            r'^([^，。；：\s]{2,4})$',  # 单独的姓名
        ]

        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[0], tuple):
                    potential_name = matches[0][0]
                else:
                    potential_name = matches[0]

                # 简单的姓名验证
                if 2 <= len(potential_name) <= 4:
                    # 检查是否包含常见姓氏
                    common_surnames = ['王', '李', '张', '刘', '陈', '杨', '黄', '赵', '周', '吴',
                                     '徐', '孙', '胡', '朱', '高', '林', '何', '郭', '马', '罗',
                                     '靳', '冯', '植']  # 包含用户提到的姓氏
                    if potential_name[0] in common_surnames:
                        return potential_name

        return None

    def _extract_figure_caption(self, text: str) -> Optional[str]:
        """提取图注"""
        if not text:
            return None

        # 图注模式
        caption_pattern = r'(图[\d\-]+[\s]*[^\n]+)'
        matches = re.findall(caption_pattern, text)

        if matches:
            return matches[0].strip()

        return None

    def _get_context_text(self, bbox: Tuple[int, int, int, int], text_blocks: List[Dict]) -> str:
        """获取图像周围的上下文文本"""
        x, y, w, h = bbox
        center_y = y + h // 2

        # 查找在图像上方和下方的文本
        context_text = []

        for block in text_blocks:
            block_bbox = block['bbox']
            block_x, block_y, block_w, block_h = block_bbox
            block_center_y = block_y + block_h // 2

            # 考虑在图像附近（上下200像素范围内）的文本
            if abs(block_center_y - center_y) < 200:
                # 检查是否有重叠
                if not (block_x + block_w < x or block_x > x + w):
                    context_text.append(block['text'])

        return ' '.join(context_text)

    def _classify_image_type(self, image: np.ndarray, content_analysis: Dict, context_text: str, region: Dict = None) -> str:
        """分类图像类型 - 优先识别人像"""

        # 最高优先级：基于上下文文本中的明确指示
        if '头像' in context_text or '照片' in context_text or '人物' in context_text:
            return 'portrait'
        elif '图' in context_text and ('癌' in context_text or '肿瘤' in context_text or '病理' in context_text):
            return 'medical_diagram'
        elif '表' in context_text and ('统计' in context_text or '数据' in context_text):
            return 'chart'

        # 第二优先级：人脸检测（最可靠的肖像识别方法）
        if content_analysis.get('has_face', False):
            return 'portrait'

        # 第三优先级：检测方法指示
        if region and region.get('is_portrait'):
            return 'portrait'
        elif region and region.get('detection_method') == 'face_detection':
            return 'portrait'

        # 第四优先级：医学特征（但避免将人像误判为医学图像）
        # 检查图像比例是否接近正方形（人像特征）
        height, width = image.shape[:2]
        aspect_ratio = width / height if height > 0 else 1.0
        is_square_like = 0.7 <= aspect_ratio <= 1.3

        # 如果图像接近正方形且没有明显的医学颜色特征，优先考虑人像
        medical_score = content_analysis.get('medical_score', 0)
        if medical_score > 0.7:  # 只有很高的医学得分才分类为医学图像
            return 'medical_diagram'
        elif medical_score > 0.4 and not is_square_like:  # 中等医学得分 + 非正方形
            return 'medical_diagram'
        elif is_square_like and content_analysis.get('color_variance', 0) > 100:  # 正方形 + 丰富色彩 = 人像
            return 'portrait'
        elif content_analysis.get('has_regular_pattern', False):
            return 'chart'
        elif medical_score > 0.2:  # 轻微医学特征
            return 'medical_diagram'
        elif is_square_like:
            return 'portrait'  # 默认正方形图像为人像

        return 'general_image'

    def _analyze_image_content(self, image: np.ndarray) -> Dict:
        """分析图像内容"""
        analysis = {
            'has_face': False,
            'medical_score': 0.0,
            'has_regular_pattern': False,
            'color_variance': 0.0,
            'edge_density': 0.0
        }

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 人脸检测
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            analysis['has_face'] = len(faces) > 0

            # 医学特征分析
            analysis['medical_score'] = self._calculate_medical_score(image)

            # 纹理分析
            analysis['color_variance'] = np.var(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1])

            # 边缘密度
            edges = cv2.Canny(gray, 50, 150)
            analysis['edge_density'] = np.count_nonzero(edges) / (image.shape[0] * image.shape[1])

            # 规律性检测（图表特征）
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
            analysis['has_regular_pattern'] = lines is not None and len(lines) > 5

        except Exception as e:
            logger.error(f"图像内容分析失败: {e}")

        return analysis

    def _calculate_medical_score(self, image: np.ndarray) -> float:
        """计算医学图像得分"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            height, width = image.shape[:2]

            # 医学颜色特征
            # 蓝色（医学边框）
            lower_blue = np.array([100, 30, 50])
            upper_blue = np.array([130, 200, 200])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            blue_ratio = np.count_nonzero(blue_mask) / (width * height)

            # 红色（血液、组织）
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_ratio = np.count_nonzero(red_mask) / (width * height)

            # 紫色（染色）
            lower_purple = np.array([140, 30, 50])
            upper_purple = np.array([160, 255, 255])
            purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
            purple_ratio = np.count_nonzero(purple_mask) / (width * height)

            # 综合医学得分
            medical_score = (blue_ratio * 0.4 + red_ratio * 0.4 + purple_ratio * 0.2)

            return min(medical_score * 10, 1.0)  # 归一化到0-1

        except Exception as e:
            logger.error(f"医学得分计算失败: {e}")
            return 0.0

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """OCR预处理"""
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
        """标准化边界框"""
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

    def _is_likely_title(self, text: str) -> bool:
        """判断是否为标题"""
        if not text:
            return False

        # 标题特征
        if len(text) < 50 and (text.endswith('章') or text.endswith('节') or
                                re.match(r'^[第]?[一二三四五六七八九十\d]+[章节、\.]', text)):
            return True

        # 大字体特征（如果包含数字或特殊符号）
        if re.match(r'^[\d\s\.\-]+[\u4e00-\u9fff]+', text) and len(text) < 30:
            return True

        return False

    def _is_likely_caption(self, text: str) -> bool:
        """判断是否为图注"""
        if not text:
            return False

        # 图注特征
        if (text.startswith('图') and re.match(r'图[\d\-]+', text)) or \
           (text.startswith('表') and re.match(r'表[\d\-]+', text)):
            return True

        return False

    def _merge_overlapping_regions_intelligent(self, regions: List[Dict]) -> List[Dict]:
        """智能合并重叠区域"""
        if not regions:
            return []

        # 按置信度排序
        regions.sort(key=lambda x: x['confidence'], reverse=True)

        merged = []
        for region in regions:
            x1, y1, w1, h1 = region['bbox']
            area1 = w1 * h1

            # 检查与已合并区域的重叠
            overlap_too_much = False

            for kept in merged:
                x2, y2, w2, h2 = kept['bbox']

                # 计算重叠面积
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                if overlap_area > 0.6 * area1:  # 重叠面积超过60%
                    overlap_too_much = True
                    break

            if not overlap_too_much:
                merged.append(region)

        return merged

    def _analyze_page_structure(self, text_content: Dict, extracted_images: List[ExtractedImage], page_image: np.ndarray) -> Dict:
        """分析页面结构"""
        structure = {
            'has_title': False,
            'has_captions': False,
            'image_positions': [],
            'text_flow': 'normal',
            'page_type': 'content'
        }

        # 分析文本
        text_blocks = text_content.get('text_blocks', [])
        for block in text_blocks:
            if block.get('is_title', False):
                structure['has_title'] = True
            if block.get('is_caption', False):
                structure['has_captions'] = True

        # 分析图像位置
        for img in extracted_images:
            x, y, w, h = img.bbox
            structure['image_positions'].append({
                'position': 'top' if y < page_image.shape[0] * 0.3 else
                           'bottom' if y > page_image.shape[0] * 0.7 else 'middle',
                'type': img.image_type,
                'filename': img.suggested_filename
            })

        # 判断页面类型
        if structure['has_title'] and len(text_blocks) < 10:
            structure['page_type'] = 'title_page'
        elif len(extracted_images) > 0 and len(text_blocks) < 5:
            structure['page_type'] = 'image_page'
        elif structure['has_captions']:
            structure['page_type'] = 'diagram_page'

        return structure

    def _save_extracted_image(self, extracted_image: ExtractedImage):
        """保存提取的图像"""
        try:
            # 使用建议的文件名，确保唯一性
            filename = extracted_image.suggested_filename
            base_path = self.output_dir / "images" / filename

            # 如果文件已存在，添加序号
            counter = 1
            final_path = base_path
            while final_path.exists():
                name_parts = filename.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                else:
                    new_filename = f"{filename}_{counter}"

                final_path = self.output_dir / "images" / new_filename
                counter += 1

            # 保存图像
            cv2.imwrite(str(final_path), extracted_image.raw_image)

            # 更新文件名
            extracted_image.suggested_filename = final_path.name

            logger.info(f"保存图像: {final_path.name}")

        except Exception as e:
            logger.error(f"保存图像失败: {e}")

    def _save_page_result(self, page_idx: int, result: Dict):
        """保存单页处理结果"""
        try:
            page_num = page_idx + 1

            # 保存JSON格式结果
            json_path = self.output_dir / "texts" / f"page_{page_num}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                # 转换对象为可序列化格式
                serializable_result = self._make_serializable(result)
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)

            # 保存纯文本
            txt_path = self.output_dir / "texts" / f"page_{page_num}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"第{page_num}页\n")
                f.write("="*20 + "\n\n")

                # 文本内容
                text_content = result.get('text_content', {})
                if text_content.get('full_text'):
                    f.write("识别文本:\n")
                    f.write(text_content['full_text'])
                    f.write("\n\n")

                # 图像信息
                images = result.get('extracted_images', [])
                if images:
                    f.write(f"提取图像 ({len(images)} 张):\n")
                    for i, img in enumerate(images):
                        f.write(f"{i+1}. {img.suggested_filename} ({img.image_type}, 置信度: {img.confidence:.3f})\n")

                # 页面结构
                structure = result.get('page_structure', {})
                f.write(f"\n页面类型: {structure.get('page_type', 'unknown')}\n")

        except Exception as e:
            logger.error(f"保存第{page_num}页结果失败: {e}")

    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, (ExtractedImage, TextBlock)):
            result = vars(obj).copy()
            # 处理numpy数组
            if 'raw_image' in result:
                result['raw_image_shape'] = result['raw_image'].shape if result['raw_image'] is not None else None
                del result['raw_image']
            return result
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _generate_final_result(self) -> Dict:
        """生成最终结果"""
        result = {
            'metadata': {
                'pdf_path': self.pdf_path,
                'total_pages': len(self.processing_results),
                'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'ocr_engine': 'PaddleOCR' if PADDLEOCR_AVAILABLE else 'None',
                'processor_version': 'unified_v1'
            },
            'pages': self.processing_results,
            'statistics': self._calculate_statistics(),
            'extracted_images': self._get_all_extracted_images()
        }

        # 保存完整结果
        result_path = self.output_dir / "complete_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        # 生成Markdown文档
        self._generate_markdown_document(result)

        return result

    def _calculate_statistics(self) -> Dict:
        """计算统计信息"""
        stats = {
            'total_text_pages': 0,
            'total_images': 0,
            'portrait_images': 0,
            'medical_images': 0,
            'chart_images': 0,
            'general_images': 0,
            'pages_with_titles': 0,
            'pages_with_captions': 0,
            'average_text_confidence': 0.0,
            'total_confidence_score': 0.0
        }

        text_confidence_sum = 0.0
        text_confidence_count = 0

        for page in self.processing_results:
            # 文本统计
            text_content = page.get('text_content', {})
            if text_content.get('full_text'):
                stats['total_text_pages'] += 1
                if text_content.get('confidence', 0) > 0:
                    text_confidence_sum += text_content['confidence']
                    text_confidence_count += 1

            # 文本块分析
            text_blocks = text_content.get('text_blocks', [])
            for block in text_blocks:
                if block.get('is_title'):
                    stats['pages_with_titles'] += 1
                if block.get('is_caption'):
                    stats['pages_with_captions'] += 1

            # 图像统计
            images = page.get('extracted_images', [])
            stats['total_images'] += len(images)

            for img in images:
                stats['total_confidence_score'] += img.confidence

                if img.image_type == 'portrait':
                    stats['portrait_images'] += 1
                elif img.image_type == 'medical_diagram':
                    stats['medical_images'] += 1
                elif img.image_type == 'chart':
                    stats['chart_images'] += 1
                else:
                    stats['general_images'] += 1

        if text_confidence_count > 0:
            stats['average_text_confidence'] = text_confidence_sum / text_confidence_count

        return stats

    def _get_all_extracted_images(self) -> List[Dict]:
        """获取所有提取的图像信息"""
        all_images = []

        for page in self.processing_results:
            for img in page.get('extracted_images', []):
                all_images.append({
                    'page_num': img.page_num,
                    'filename': img.suggested_filename,
                    'image_type': img.image_type,
                    'confidence': img.confidence,
                    'context_text': img.context_text,
                    'bbox': img.bbox
                })

        return all_images

    def _generate_markdown_document(self, result: Dict):
        """生成Markdown文档"""
        try:
            pdf_name = Path(self.pdf_path).stem
            md_content = f"""# {pdf_name}

## 文档信息

- **总页数**: {result['metadata']['total_pages']}
- **处理时间**: {result['metadata']['processing_time']}
- **OCR引擎**: {result['metadata']['ocr_engine']}
- **处理器版本**: {result['metadata']['processor_version']}

## 统计信息

### 文本统计
- **包含文本的页面**: {result['statistics']['total_text_pages']}
- **包含标题的页面**: {result['statistics']['pages_with_titles']}
- **包含图注的页面**: {result['statistics']['pages_with_captions']}
- **平均文本置信度**: {result['statistics']['average_text_confidence']:.3f}

### 图像统计
- **总共提取图像**: {result['statistics']['total_images']}
- **人物头像**: {result['statistics']['portrait_images']}
- **医学图像**: {result['statistics']['medical_images']}
- **图表**: {result['statistics']['chart_images']}
- **普通图像**: {result['statistics']['general_images']}
- **总置信度评分**: {result['statistics']['total_confidence_score']:.3f}

---

"""

            # 按页面添加详细内容
            current_chapter = None

            for page in result['pages']:
                page_num = page['page_num']
                text_content = page.get('text_content', {})
                extracted_images = page.get('extracted_images', [])
                page_structure = page.get('page_structure', {})

                # 章节标题检测
                text_blocks = text_content.get('text_blocks', [])
                chapter_title = None
                for block in text_blocks:
                    if block.get('is_title', False):
                        chapter_title = block.get('text', '')
                        break

                if chapter_title and chapter_title != current_chapter:
                    md_content += f"## {chapter_title}\n\n"
                    current_chapter = chapter_title

                # 页面标题
                md_content += f"### 第{page_num}页"

                page_type = page_structure.get('page_type', 'content')
                if page_type != 'content':
                    md_content += f" ({self._get_page_type_name(page_type)})"

                md_content += "\n\n"

                # 文本内容
                if text_content.get('full_text'):
                    confidence = text_content.get('confidence', 0)
                    md_content += f"**识别文本** (置信度: {confidence:.3f})\n\n"
                    md_content += f"```\n{text_content['full_text']}\n```\n\n"

                # 提取的图像
                if extracted_images:
                    md_content += "**提取图像:**\n\n"

                    for i, img in enumerate(extracted_images):
                        # 图像类型图标
                        type_icon = self._get_image_type_icon(img.image_type)
                        md_content += f"{i+1}. {type_icon} **{img.suggested_filename}**\n"
                        md_content += f"   - 类型: {self._get_image_type_name(img.image_type)}\n"
                        md_content += f"   - 置信度: {img.confidence:.3f}\n"

                        if img.context_text:
                            md_content += f"   - 上下文: {img.context_text[:100]}...\n"

                        # 图像链接
                        image_path = f"images/{img.suggested_filename}"
                        md_content += f"   ![{img.suggested_filename}]({image_path})\n\n"

                md_content += "---\n\n"

            # 图像列表附录
            md_content += """## 附录：图像列表

| 文件名 | 类型 | 页码 | 置信度 | 上下文 |
|--------|------|------|--------|--------|
"""

            for img_info in result['extracted_images']:
                filename = img_info['filename']
                img_type = self._get_image_type_name(img_info['image_type'])
                page_num = img_info['page_num']
                confidence = img_info['confidence']
                context = img_info['context_text'][:30] + "..." if img_info['context_text'] else ""

                md_content += f"| {filename} | {img_type} | {page_num} | {confidence:.3f} | {context} |\n"

            # 保存Markdown文件
            md_path = self.output_dir / "markdown" / f"{pdf_name}_unified.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            logger.info(f"Markdown文档已生成: {md_path}")

        except Exception as e:
            logger.error(f"生成Markdown文档失败: {e}")

    def _get_page_type_name(self, page_type: str) -> str:
        """获取页面类型名称"""
        type_names = {
            'title_page': '标题页',
            'image_page': '图像页',
            'diagram_page': '图表页',
            'content': '内容页'
        }
        return type_names.get(page_type, '未知类型')

    def _get_image_type_name(self, image_type: str) -> str:
        """获取图像类型名称"""
        type_names = {
            'portrait': '人物头像',
            'medical_diagram': '医学图像',
            'chart': '图表',
            'general_image': '普通图像'
        }
        return type_names.get(image_type, '未知类型')

    def _get_image_type_icon(self, image_type: str) -> str:
        """获取图像类型图标"""
        icons = {
            'portrait': '👤',
            'medical_diagram': '🔬',
            'chart': '📊',
            'general_image': '🖼️'
        }
        return icons.get(image_type, '❓')

def main():
    """主函数"""
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"

    if not os.path.exists(pdf_path):
        logger.error(f"PDF文件不存在: {pdf_path}")
        return

    # 创建处理器
    processor = UnifiedPDFProcessor(pdf_path)

    try:
        # 处理PDF
        result = processor.process_pdf()

        logger.info("PDF统一处理完成！")

        # 打印统计信息
        print(f"\n=== 处理结果统计 ===")
        print(f"总页数: {result['metadata']['total_pages']}")
        print(f"包含文本页面: {result['statistics']['total_text_pages']}")
        print(f"提取图像总数: {result['statistics']['total_images']}")
        print(f"人物头像: {result['statistics']['portrait_images']}")
        print(f"医学图像: {result['statistics']['medical_images']}")
        print(f"图表: {result['statistics']['chart_images']}")
        print(f"普通图像: {result['statistics']['general_images']}")
        print(f"平均文本置信度: {result['statistics']['average_text_confidence']:.3f}")

        # 显示提取的图像
        if result['extracted_images']:
            print(f"\n=== 提取的图像 ===")
            for img in result['extracted_images'][:10]:  # 显示前10个
                print(f"第{img['page_num']}页: {img['filename']} ({img['image_type']}, 置信度: {img['confidence']:.3f})")

            if len(result['extracted_images']) > 10:
                print(f"... 还有 {len(result['extracted_images']) - 10} 个图像")

        print(f"\n输出目录: {processor.output_dir}")
        print(f"Markdown文档: {processor.output_dir}/markdown/{Path(pdf_path).stem}_unified.md")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        raise

if __name__ == "__main__":
    main()