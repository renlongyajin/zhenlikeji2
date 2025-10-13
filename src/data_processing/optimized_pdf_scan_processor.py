#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的扫描版PDF处理器 - 专门针对医学图像识别优化
使用先进的图像处理和机器学习技术提高医学图像识别准确率
"""

import fitz  # PyMuPDF
import os
import json
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import time
import re
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage import measure, filters, morphology
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR未安装，将使用备用OCR方法")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedPDFScanProcessor:
    """优化的扫描版PDF处理器 - 专门针对医学图像识别"""

    def __init__(self, pdf_path: str, output_dir: str = "data/optimized_processed"):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建图片输出目录
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "texts").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "debug").mkdir(parents=True, exist_ok=True)

        self.doc = None
        self.ocr_engine = None
        self.processed_results = []

        # 初始化OCR引擎
        self._init_ocr()

    def _init_ocr(self):
        """初始化OCR引擎 - 支持GPU加速"""
        if PADDLEOCR_AVAILABLE:
            try:
                # 检测GPU可用性
                use_gpu = self._check_gpu_available()

                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='ch',
                    use_gpu=use_gpu,
                    det_limit_side_len=1280,  # 提高分辨率
                    drop_score=0.2,  # 降低阈值
                    det_db_thresh=0.2,  # 降低检测阈值
                    det_db_box_thresh=0.3,
                    det_db_unclip_ratio=1.8  # 增加框扩展
                )
                logger.info(f"PaddleOCR优化初始化成功 - GPU: {use_gpu}")
            except Exception as e:
                logger.error(f"PaddleOCR初始化失败: {e}")
                self.ocr_engine = None
        else:
            logger.warning("PaddleOCR不可用")

    def _check_gpu_available(self) -> bool:
        """检测GPU是否可用"""
        try:
            import paddle
            paddle.utils.run_check()
            return paddle.is_compiled_with_cuda()
        except:
            return False

    def process_pdf(self, max_pages: int = None, target_pages: List[int] = None) -> Dict:
        """处理PDF文件"""
        logger.info(f"开始优化处理PDF文件: {self.pdf_path}")

        try:
            self.doc = fitz.open(self.pdf_path)
            total_pages = len(self.doc)
            logger.info(f"PDF总页数: {total_pages}")

            if target_pages:
                # 处理指定页面
                pages_to_process = [p-1 for p in target_pages if p <= total_pages]
            elif max_pages:
                pages_to_process = range(min(total_pages, max_pages))
            else:
                pages_to_process = range(total_pages)

            # 处理每一页
            for page_num in pages_to_process:
                logger.info(f"优化处理第 {page_num + 1}/{total_pages} 页")

                try:
                    page_result = self._process_page_advanced(page_num)
                    if page_result:
                        self.processed_results.append(page_result)
                        self._save_page_result(page_num, page_result)
                except Exception as e:
                    logger.error(f"处理第{page_num + 1}页失败: {e}")
                    continue

            # 生成最终结果
            result = self._generate_final_result()

            logger.info("PDF优化处理完成")
            return result

        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            raise
        finally:
            if self.doc:
                self.doc.close()

    def _process_page_advanced(self, page_num: int) -> Optional[Dict]:
        """高级页面处理"""
        page = self.doc.load_page(page_num)

        # 获取页面图片 - 使用更高的DPI
        pix = page.get_pixmap(dpi=400)  # 提高DPI到400
        img_data = pix.tobytes("png")
        pix = None

        # 转换为numpy数组
        nparr = np.frombuffer(img_data, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # OCR识别
        ocr_result = self._perform_ocr(img_cv, page_num)

        # 使用高级图像检测
        detected_images = self._detect_medical_images_advanced(img_cv, page_num)

        # 识别章节标题
        chapter_info = self._extract_chapter_info(ocr_result.get('text', ''))

        page_result = {
            'page_num': page_num + 1,
            'ocr_text': ocr_result.get('text', ''),
            'ocr_confidence': ocr_result.get('confidence', 0.0),
            'detected_images': detected_images,
            'chapter_info': chapter_info,
            'has_medical_content': self._detect_medical_content(ocr_result.get('text', ''))
        }

        return page_result

    def _perform_ocr(self, image: np.ndarray, page_num: int) -> Dict:
        """执行OCR识别"""
        if not self.ocr_engine:
            logger.warning("OCR引擎不可用")
            return {'text': '', 'confidence': 0.0}

        try:
            # 图像预处理以提高OCR准确率
            processed_img = self._preprocess_for_ocr(image)

            # 执行OCR
            result = self.ocr_engine.ocr(processed_img, cls=True)

            if not result or not result[0]:
                return {'text': '', 'confidence': 0.0}

            # 解析OCR结果
            full_text = ""
            total_confidence = 0.0
            count = 0

            for line in result[0]:
                if line and len(line) >= 2:
                    bbox, (text, confidence) = line[0], line[1]
                    if text and confidence > 0.2:  # 降低置信度阈值
                        full_text += text + "\n"
                        total_confidence += confidence
                        count += 1

            average_confidence = total_confidence / count if count > 0 else 0.0

            return {
                'text': full_text.strip(),
                'confidence': average_confidence
            }

        except Exception as e:
            logger.error(f"OCR识别失败 (第{page_num + 1}页): {e}")
            return {'text': '', 'confidence': 0.0}

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """OCR预处理"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # 降噪
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # 锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            return sharpened

        except Exception as e:
            logger.error(f"OCR预处理失败: {e}")
            return image

    def _detect_medical_images_advanced(self, image: np.ndarray, page_num: int) -> List[Dict]:
        """高级医学图像检测 - 重新设计避免过度分割"""
        detected_images = []

        try:
            # 步骤1: 智能候选区域检测 - 每页只找最重要的1-3个区域
            candidate_regions = self._detect_smart_candidate_regions(image)

            logger.info(f"第{page_num + 1}页检测到{len(candidate_regions)}个候选区域")

            # 步骤2: 对候选区域进行质量评估和排序
            scored_regions = []
            for i, region_info in enumerate(candidate_regions):
                x, y, w, h = region_info['bbox']
                roi = image[y:y+h, x:x+w]

                # 详细内容分析
                content_analysis = self._analyze_region_content(roi)

                if content_analysis['is_meaningful'] and content_analysis['confidence'] > 0.4:
                    # 分类医学图像
                    image_type = self._classify_medical_image_advanced(roi, content_analysis)

                    scored_regions.append({
                        'index': i,
                        'bbox': (x, y, w, h),
                        'roi': roi,
                        'content_analysis': content_analysis,
                        'image_type': image_type,
                        'confidence': content_analysis['confidence'],
                        'area': w * h
                    })

            # 步骤3: 智能筛选 - 每页最多保留2个最重要的区域
            if scored_regions:
                # 按综合评分排序
                scored_regions.sort(key=lambda x: x['confidence'], reverse=True)

                # 选择最重要的区域（每页最多2个）
                top_regions = scored_regions[:2]

                # 如果有多个区域，确保它们不是重叠的
                if len(top_regions) > 1:
                    top_regions = self._remove_overlapping_regions_smart(top_regions)

                # 保存最终选择的区域
                for i, region in enumerate(top_regions):
                    x, y, w, h = region['bbox']
                    roi = image[y:y+h, x:x+w]

                    # 完整性验证 - 确保提取的是完整矩形
                    if self._validate_region_completeness(roi, region):
                        # 保存图片
                        img_filename = f"page_{page_num + 1}_region_{i}_{region['image_type']}.png"
                        img_path = self.output_dir / "images" / img_filename
                        cv2.imwrite(str(img_path), roi)

                        detected_images.append({
                            'page_num': page_num + 1,
                            'region_name': f'smart_region_{i}',
                            'image_type': region['image_type'],
                            'bbox': region['bbox'],
                            'file_path': str(img_path),
                            'confidence': region['confidence'],
                            'analysis': region['content_analysis'],
                            'area_pixels': region['area'],
                            'area_ratio': region['area'] / (image.shape[0] * image.shape[1]),
                            'completeness_validated': True
                        })

            return detected_images

        except Exception as e:
            logger.error(f"高级医学图片检测失败 (第{page_num + 1}页): {e}")
            return []

    def _detect_smart_candidate_regions(self, image: np.ndarray) -> List[Dict]:
        """智能候选区域检测 - 基于医学文献布局特点"""
        try:
            height, width = image.shape[:2]
            page_area = height * width

            # 1. 首先检测页面中的主要视觉区域
            # 使用多种方法结合，避免过度分割
            candidate_regions = []

            # 方法1: 基于医学文献常见布局的预设区域
            layout_regions = self._get_medical_document_layout_regions(width, height)
            candidate_regions.extend(layout_regions)

            # 方法2: 高级矩形边框检测 - 关键改进
            rectangular_regions = self._detect_rectangular_borders_advanced(image)
            candidate_regions.extend(rectangular_regions)

            # 方法3: 基于视觉显著性的区域检测
            salient_regions = self._detect_salient_regions(image)
            candidate_regions.extend(salient_regions)

            # 方法4: 基于颜色特征的特殊区域检测（如淡蓝色边框）
            special_regions = self._detect_special_color_regions(image)
            candidate_regions.extend(special_regions)

            # 2. 区域融合和筛选
            # 移除重叠区域，保留最佳区域
            merged_regions = self._merge_overlapping_regions(candidate_regions)

            # 3. 质量预筛选
            quality_regions = []
            for region in merged_regions:
                x, y, w, h = region['bbox']
                area = w * h
                area_ratio = area / page_area
                aspect_ratio = w / h if h > 0 else 1.0

                # 严格的质量筛选
                if (area > 15000 and  # 较大的区域（比之前更严格）
                    area_ratio > 0.02 and  # 占页面比例足够大
                    0.4 <= aspect_ratio <= 2.5 and  # 合理的宽高比
                    not self._is_likely_text_block(region, image)):

                    quality_regions.append(region)

            # 4. 按重要性排序，返回前3个最重要的区域
            quality_regions.sort(key=lambda x: x['area'], reverse=True)

            logger.info(f"智能检测到{len(quality_regions)}个高质量候选区域")
            return quality_regions[:3]  # 最多返回3个区域

        except Exception as e:
            logger.error(f"智能候选区域检测失败: {e}")
            return []

    def _get_medical_document_layout_regions(self, width: int, height: int) -> List[Dict]:
        """基于医学文献布局的预设区域"""
        regions = []

        # 医学文献中图片的常见位置
        common_positions = [
            # 主要图片区域（页面中心偏上）
            {
                'x': int(width * 0.1), 'y': int(height * 0.15),
                'w': int(width * 0.8), 'h': int(height * 0.5),
                'type': 'main_image', 'priority': 1
            },
            # 人物头像区域（页面右上角）
            {
                'x': int(width * 0.65), 'y': int(height * 0.05),
                'w': int(width * 0.3), 'h': int(height * 0.25),
                'type': 'portrait_area', 'priority': 2
            },
            # 底部图表区域
            {
                'x': int(width * 0.1), 'y': int(height * 0.7),
                'w': int(width * 0.8), 'h': int(height * 0.25),
                'type': 'chart_area', 'priority': 3
            },
            # 左侧图片区域
            {
                'x': int(width * 0.05), 'y': int(height * 0.2),
                'w': int(width * 0.4), 'h': int(height * 0.6),
                'type': 'left_image', 'priority': 4
            },
            # 右侧图片区域
            {
                'x': int(width * 0.55), 'y': int(height * 0.2),
                'w': int(width * 0.4), 'h': int(height * 0.6),
                'type': 'right_image', 'priority': 5
            }
        ]

        for pos in common_positions:
            regions.append({
                'bbox': (pos['x'], pos['y'], pos['w'], pos['h']),
                'area': pos['w'] * pos['h'],
                'aspect_ratio': pos['w'] / pos['h'],
                'region_type': pos['type'],
                'priority': pos['priority']
            })

        return regions

    def _detect_salient_regions(self, image: np.ndarray) -> List[Dict]:
        """基于视觉显著性的区域检测"""
        try:
            height, width = image.shape[:2]

            # 转换为LAB颜色空间进行显著性检测
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # 计算颜色对比度
            l, a, b = cv2.split(lab)

            # 使用高斯差分检测显著区域
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

            salient_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # 筛选显著区域
                if area > 10000 and 0.3 <= w/h <= 3.0:  # 面积和比例要求
                    salient_regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': w / h,
                        'region_type': 'salient'
                    })

            return salient_regions

        except Exception as e:
            logger.error(f"显著性区域检测失败: {e}")
            return []

    def _detect_rectangular_borders_advanced(self, image: np.ndarray) -> List[Dict]:
        """高级矩形边框检测 - 专门针对医学图像的方框特征"""
        try:
            height, width = image.shape[:2]

            # 1. 多尺度边缘检测
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 使用不同参数进行边缘检测
            edges1 = cv2.Canny(gray, 30, 100)  # 低阈值检测弱边缘
            edges2 = cv2.Canny(gray, 50, 150)  # 中等阈值
            edges3 = cv2.Canny(gray, 80, 200)  # 高阈值检测强边缘

            # 合并边缘检测结果
            combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))

            # 2. 形态学操作增强矩形特征
            # 使用矩形核增强直线和矩形结构
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            morphed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, rect_kernel)

            # 3. 查找轮廓
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rectangular_regions = []

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                # 基础筛选
                if area < 10000:  # 面积太小
                    continue

                aspect_ratio = w / h if h > 0 else 1.0
                if not (0.4 <= aspect_ratio <= 2.5):  # 宽高比不合适
                    continue

                # 4. 矩形度检测 - 关键改进
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)

                if box_area > 0:
                    rectangularity = area / box_area
                else:
                    rectangularity = 0

                if rectangularity < 0.7:  # 矩形度不够
                    continue

                # 5. 边框颜色分析
                roi = image[y:y+h, x:x+w]
                border_score = self._analyze_border_color_score(roi)

                # 6. 内部内容分析 - 避免文字块
                content_score = self._analyze_rectangular_content(roi)

                # 综合评分
                total_score = (
                    rectangularity * 0.4 +  # 矩形度权重
                    border_score * 0.3 +     # 边框特征权重
                    content_score * 0.3      # 内容特征权重
                )

                if total_score > 0.5:  # 综合阈值
                    rectangular_regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'rectangularity': rectangularity,
                        'border_score': border_score,
                        'content_score': content_score,
                        'total_score': total_score,
                        'region_type': 'rectangular_border'
                    })

            # 7. 基于霍夫变换的直线检测 - 增强矩形检测
            lines = cv2.HoughLinesP(combined_edges, 1, np.pi/180, 100,
                                    minLineLength=50, maxLineGap=10)

            if lines is not None:
                rectangular_regions.extend(
                    self._detect_rectangles_from_lines(lines, image, width, height)
                )

            logger.info(f"检测到 {len(rectangular_regions)} 个矩形边框区域")
            return rectangular_regions

        except Exception as e:
            logger.error(f"矩形边框检测失败: {e}")
            return []

    def _analyze_border_color_score(self, roi: np.ndarray) -> float:
        """分析边框颜色特征得分"""
        try:
            height, width = roi.shape[:2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 提取边框区域 (5% 边界)
            border_width = int(min(width, height) * 0.05)

            # 分析四个边框
            borders = [
                roi[0:border_width, :],                    # 上边框
                roi[height-border_width:height, :],        # 下边框
                roi[:, 0:border_width],                    # 左边框
                roi[:, width-border_width:width]           # 右边框
            ]

            border_scores = []

            for border in borders:
                border_hsv = cv2.cvtColor(border, cv2.COLOR_BGR2HSV)

                # 检测蓝色边框 (医学图像常见)
                lower_blue = np.array([100, 30, 50])
                upper_blue = np.array([130, 200, 200])
                blue_mask = cv2.inRange(border_hsv, lower_blue, upper_blue)
                blue_ratio = np.count_nonzero(blue_mask) / (border.shape[0] * border.shape[1])

                # 检测灰色边框 (文档常见)
                lower_gray = np.array([0, 0, 100])
                upper_gray = np.array([180, 50, 220])
                gray_mask = cv2.inRange(border_hsv, lower_gray, upper_gray)
                gray_ratio = np.count_nonzero(gray_mask) / (border.shape[0] * border.shape[1])

                # 边框得分
                border_score = max(blue_ratio, gray_ratio * 0.8)  # 蓝色权重更高
                border_scores.append(border_score)

            # 返回平均边框得分
            return np.mean(border_scores)

        except Exception as e:
            logger.error(f"边框颜色分析失败: {e}")
            return 0.0

    def _analyze_rectangular_content(self, roi: np.ndarray) -> float:
        """分析矩形内部内容特征"""
        try:
            height, width = roi.shape[:2]

            # 提取内部区域 (去掉边框)
            margin = int(min(width, height) * 0.1)
            inner_roi = roi[margin:height-margin, margin:width-margin]

            if inner_roi.size == 0:
                return 0.0

            gray = cv2.cvtColor(inner_roi, cv2.COLOR_BGR2GRAY)

            # 1. 方差分析 - 医学图像应该有适中的复杂度
            variance = np.var(gray)
            variance_score = min(variance / 2000, 1.0)  # 归一化

            # 2. 边缘密度分析
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (inner_roi.shape[0] * inner_roi.shape[1])
            edge_score = min(edge_density / 0.1, 1.0)  # 归一化

            # 3. 文字检测 - 医学图像不应该有大量文字
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            small_components = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if 10 < area < 200:  # 小组件可能是文字
                    small_components += 1

            text_ratio = small_components / num_labels if num_labels > 1 else 0
            non_text_score = 1.0 - text_ratio  # 非文字得分

            # 综合内容得分
            content_score = (
                variance_score * 0.4 +
                edge_score * 0.3 +
                non_text_score * 0.3
            )

            return content_score

        except Exception as e:
            logger.error(f"矩形内容分析失败: {e}")
            return 0.5  # 默认中等得分

    def _detect_rectangles_from_lines(self, lines: np.ndarray, image: np.ndarray,
                                    width: int, height: int) -> List[Dict]:
        """从检测到的直线中识别矩形"""
        try:
            rectangles = []

            # 分析直线的角度和位置
            horizontal_lines = []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                if length < 30:  # 线段太短
                    continue

                if angle < 15 or angle > 165:  # 接近水平
                    horizontal_lines.append((x1, y1, x2, y2))
                elif 75 < angle < 105:  # 接近垂直
                    vertical_lines.append((x1, y1, x2, y2))

            # 查找可能的矩形
            for h_line1 in horizontal_lines:
                for h_line2 in horizontal_lines:
                    if h_line1 == h_line2:
                        continue

                    # 检查两条水平线是否平行且距离合适
                    y1, y2 = h_line1[1], h_line2[1]
                    if abs(y1 - y2) < 50:  # 距离太近
                        continue

                    for v_line1 in vertical_lines:
                        for v_line2 in vertical_lines:
                            if v_line1 == v_line2:
                                continue

                            # 检查两条垂直线是否平行且距离合适
                            x1, x2 = v_line1[0], v_line2[0]
                            if abs(x1 - x2) < 50:  # 距离太近
                                continue

                            # 检查是否形成矩形
                            if self._lines_form_rectangle(h_line1, h_line2, v_line1, v_line2):
                                x_min = min(x1, x2)
                                x_max = max(x1, x2)
                                y_min = min(y1, y2)
                                y_max = max(y1, y2)

                                w = x_max - x_min
                                h = y_max - y_min

                                # 验证矩形参数
                                if (100 < w < width * 0.9 and
                                    100 < h < height * 0.9 and
                                    0.4 <= w/h <= 2.5):

                                    rectangles.append({
                                        'bbox': (x_min, y_min, w, h),
                                        'area': w * h,
                                        'aspect_ratio': w / h,
                                        'detection_type': 'hough_rectangle',
                                        'confidence': 0.6
                                    })

            return rectangles

        except Exception as e:
            logger.error(f"霍夫矩形检测失败: {e}")
            return []

    def _lines_form_rectangle(self, h1, h2, v1, v2) -> bool:
        """检查四条线是否形成矩形"""
        try:
            # 简化的矩形检查
            h1_y1, h1_y2 = h1[1], h1[3]
            h2_y1, h2_y2 = h2[1], h2[3]

            v1_x1, v1_x2 = v1[0], v1[2]
            v2_x1, v2_x2 = v2[0], v2[2]

            # 检查水平线是否连接垂直线
            h1_connected = (min(abs(h1[0] - v1_x1), abs(h1[2] - v1_x1)) < 20 or
                           min(abs(h1[0] - v2_x1), abs(h1[2] - v2_x1)) < 20)

            h2_connected = (min(abs(h2[0] - v1_x1), abs(h2[2] - v1_x1)) < 20 or
                           min(abs(h2[0] - v2_x1), abs(h2[2] - v2_x1)) < 20)

            return h1_connected and h2_connected

        except Exception:
            return False

    def _validate_region_completeness(self, roi: np.ndarray, region_info: Dict) -> bool:
        """验证区域完整性 - 确保提取的是完整矩形"""
        try:
            height, width = roi.shape[:2]

            # 1. 检查边界完整性
            if width < 50 or height < 50:  # 太小
                return False

            # 2. 检查矩形度
            expected_rectangularity = region_info.get('rectangularity', 0.8)
            if expected_rectangularity < 0.6:
                return False

            # 3. 检查边框特征（如果是矩形边框检测）
            if region_info.get('region_type') == 'rectangular_border':
                border_score = self._analyze_border_color_score(roi)
                if border_score < 0.1:  # 边框特征太弱
                    return False

            # 4. 检查内容完整性
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray)

            if variance < 50:  # 内容太平淡
                return False

            # 5. 检查边缘分布
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (width * height)

            if edge_density < 0.01:  # 边缘太少
                return False

            # 6. 检查边界清晰度
            # 分析四个边界的边缘密度
            border_width = max(5, int(min(width, height) * 0.05))

            borders = [
                edges[0:border_width, :],                    # 上边界
                edges[height-border_width:height, :],        # 下边界
                edges[:, 0:border_width],                    # 左边界
                edges[:, width-border_width:width]           # 右边界
            ]

            border_edge_scores = []
            for border in borders:
                if border.size > 0:
                    border_edge_score = np.count_nonzero(border) / border.size
                    border_edge_scores.append(border_edge_score)

            if border_edge_scores:
                avg_border_edge_score = np.mean(border_edge_scores)
                if avg_border_edge_score < 0.005:  # 边界太模糊
                    return False

            return True

        except Exception as e:
            logger.error(f"区域完整性验证失败: {e}")
            return False  # 默认不通过验证

    def _detect_special_color_regions(self, image: np.ndarray) -> List[Dict]:
        """检测特殊颜色区域（如淡蓝色边框）"""
        try:
            height, width = image.shape[:2]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 淡蓝色检测（医学图像常见的边框颜色）
            lower_blue = np.array([100, 30, 50])
            upper_blue = np.array([130, 150, 200])

            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            blue_morphed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(blue_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            blue_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 8000:  # 较大的蓝色区域
                    x, y, w, h = cv2.boundingRect(contour)
                    blue_regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': w / h,
                        'region_type': 'blue_border'
                    })

            return blue_regions

        except Exception as e:
            logger.error(f"特殊颜色区域检测失败: {e}")
            return []

    def _is_likely_text_block(self, region: Dict, image: np.ndarray) -> bool:
        """判断是否为文字块"""
        try:
            x, y, w, h = region['bbox']
            roi = image[y:y+h, x:x+w]

            # 转换为灰度图
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 文字块检测
            # 1. 高边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (w * h)

            # 2. 大量小组件
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            small_components = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if 10 < area < 200:  # 小组件
                    small_components += 1

            # 3. 规律性纹理
            small_component_ratio = small_components / num_labels if num_labels > 1 else 0

            # 综合判断
            is_text_block = (
                edge_density > 0.05 and  # 边缘密度高
                small_component_ratio > 0.6 and  # 大量小组件
                w > 100 and h > 50  # 合理的尺寸
            )

            return is_text_block

        except Exception as e:
            logger.error(f"文字块检测失败: {e}")
            return False

    def _merge_overlapping_regions(self, regions: List[Dict]) -> List[Dict]:
        """智能合并重叠区域 - 优化医学图像完整性"""
        if not regions:
            return []

        # 按综合评分排序（优先保留高质量区域）
        def get_region_score(region):
            # 计算区域综合评分
            score = region.get('total_score', 0.5)  # 基础得分
            score += region.get('rectangularity', 0) * 0.3  # 矩形度加分
            score += min(region.get('border_score', 0), 0.5)  # 边框特征加分
            score += region.get('confidence', 0.5) * 0.2  # 置信度加分
            return score

        regions.sort(key=get_region_score, reverse=True)

        merged = []
        for region in regions:
            x1, y1, w1, h1 = region['bbox']
            area1 = w1 * h1

            # 检查是否与已合并的区域重叠
            overlap_too_much = False
            merged_with = None

            for i, kept in enumerate(merged):
                x2, y2, w2, h2 = kept['bbox']

                # 计算重叠面积
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                # 智能重叠判断
                if overlap_area > 0:
                    overlap_ratio1 = overlap_area / area1
                    overlap_ratio2 = overlap_area / (w2 * h2)

                    # 如果重叠面积较大，考虑合并策略
                    if overlap_ratio1 > 0.6 or overlap_ratio2 > 0.6:
                        # 选择更好的区域或合并区域
                        if get_region_score(region) > get_region_score(kept):
                            # 新区域更好，替换旧区域
                            merged[i] = region
                            overlap_too_much = True
                            break
                        else:
                            # 旧区域更好，保留旧区域
                            overlap_too_much = True
                            break
                    elif overlap_ratio1 > 0.3 or overlap_ratio2 > 0.3:
                        # 中等重叠，考虑合并边界
                        merged_with = i
                        break

            if not overlap_too_much:
                if merged_with is not None:
                    # 合并两个区域的外边界
                    kept = merged[merged_with]
                    x2, y2, w2, h2 = kept['bbox']

                    # 计算合并后的边界
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y

                    # 更新合并后的区域
                    merged[merged_with]['bbox'] = (new_x, new_y, new_w, new_h)
                    merged[merged_with]['area'] = new_w * new_h

                    # 更新其他特征（取更好的特征）
                    if region.get('total_score', 0) > kept.get('total_score', 0):
                        merged[merged_with]['total_score'] = region.get('total_score', 0)
                    if region.get('rectangularity', 0) > kept.get('rectangularity', 0):
                        merged[merged_with]['rectangularity'] = region.get('rectangularity', 0)
                    if region.get('border_score', 0) > kept.get('border_score', 0):
                        merged[merged_with]['border_score'] = region.get('border_score', 0)
                else:
                    # 无重叠，直接添加
                    merged.append(region)

        return merged

    def _remove_overlapping_regions_smart(self, regions: List[Dict]) -> List[Dict]:
        """智能移除重叠区域"""
        if len(regions) <= 1:
            return regions

        # 按置信度排序
        regions.sort(key=lambda x: x['confidence'], reverse=True)

        filtered = [regions[0]]  # 保留置信度最高的

        for region in regions[1:]:
            x1, y1, w1, h1 = region['bbox']
            area1 = w1 * h1

            # 检查与已保留区域的重叠
            overlap_too_much = False
            for kept in filtered:
                x2, y2, w2, h2 = kept['bbox']

                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                if overlap_area > 0.6 * area1:  # 重叠面积超过60%
                    overlap_too_much = True
                    break

            if not overlap_too_much:
                filtered.append(region)

        return filtered

    def _analyze_region_content(self, roi: np.ndarray) -> Dict:
        """分析区域内容 - 针对医学图像优化"""
        try:
            height, width = roi.shape[:2]

            # 1. 基础统计
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray)

            # 2. 边缘密度 - 医学图像通常有适中的边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (height * width)

            # 3. 纹理复杂度 - 医学图像通常有复杂的纹理
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            texture_complexity = np.std(lbp)

            # 4. 文字块检测 - 医学图像中不应有太多文字
            text_score = self._detect_text_characteristics(gray, roi)

            # 5. 医学颜色特征分析
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 红色（血液、组织）
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_ratio = (np.count_nonzero(red_mask1) + np.count_nonzero(red_mask2)) / (width * height)

            # 蓝色（医学染色、边框）
            lower_blue = np.array([100, 30, 50])
            upper_blue = np.array([130, 150, 200])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            blue_ratio = np.count_nonzero(blue_mask) / (width * height)

            # 紫色（某些医学染色）
            lower_purple = np.array([140, 30, 50])
            upper_purple = np.array([160, 255, 255])
            purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
            purple_ratio = np.count_nonzero(purple_mask) / (width * height)

            # 6. 形状分析 - 医学图像通常有明确的形状特征
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

            shape_score = 0
            if num_labels > 1:
                # 找到最大的连通组件
                largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                area = stats[largest_idx, cv2.CC_STAT_AREA]

                if area > 0:
                    # 计算形状复杂度
                    component_mask = (labels == largest_idx).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        perimeter = cv2.arcLength(contours[0], True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter ** 2)
                            shape_score = min(circularity, 1.0)

            # 医学图像质量评分 - 针对医学图像特点优化
            medical_score = 0

            # 如果检测到明显文字，直接降低评分，但提高阈值避免误杀
            if text_score > 0.8:  # 只有非常明显的文字才大幅降分
                medical_score = 0.1
            else:
                # 医学图像的综合评分
                medical_score = (
                    min(variance / 800, 1.0) * 0.15 +  # 方差适中
                    min(edge_density / 0.08, 1.0) * 0.20 +  # 适中的边缘密度
                    min(texture_complexity / 40, 1.0) * 0.15 +  # 纹理复杂度
                    min((red_ratio + blue_ratio + purple_ratio) / 0.15, 1.0) * 0.30 +  # 医学颜色特征（提高权重）
                    shape_score * 0.10 +  # 形状特征
                    (1.0 - text_score) * 0.10  # 非文字特征
                )

            is_meaningful = medical_score > 0.35  # 略微降低阈值，避免漏检重要的医学图像

            return {
                'is_meaningful': bool(is_meaningful),
                'confidence': float(medical_score),
                'variance': float(variance),
                'edge_density': float(edge_density),
                'texture_complexity': float(texture_complexity),
                'red_ratio': float(red_ratio),
                'blue_ratio': float(blue_ratio),
                'purple_ratio': float(purple_ratio),
                'shape_score': float(shape_score),
                'is_text_block': text_score > 0.8,
                'text_score': float(text_score)
            }

        except Exception as e:
            logger.error(f"区域内容分析失败: {e}")
            return {'is_meaningful': False, 'confidence': 0.0, 'is_text_block': False}

            # 1. 基础统计
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray)

            # 2. 边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (height * width)

            # 3. 纹理复杂度
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            texture_complexity = np.std(lbp)

            # 4. 文字块检测 - 关键改进
            text_score = self._detect_text_characteristics(gray, roi)

            # 5. 颜色分析
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 医学图像通常有特定的颜色特征
            # 红色（血液、组织）
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_ratio = (np.count_nonzero(red_mask1) + np.count_nonzero(red_mask2)) / (width * height)

            # 蓝色（医学染色）
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            blue_ratio = np.count_nonzero(blue_mask) / (width * height)

            # 6. 形状分析
            # 使用阈值分割找到主要对象
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 查找连通组件
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

            shape_score = 0
            if num_labels > 1:  # 有多个组件
                # 计算最大组件的圆形度
                largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                area = stats[largest_component_idx, cv2.CC_STAT_AREA]
                perimeter = cv2.arcLength(cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True)

                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    shape_score = min(circularity, 1.0)

            # 如果检测到是文字，直接返回无意义 - 但提高阈值避免误杀医学图像
            if text_score > 0.75:  # 提高阈值到0.75，只有非常明显的文字才过滤
                return {
                    'is_meaningful': False,
                    'confidence': 0.0,
                    'variance': float(variance),
                    'edge_density': float(edge_density),
                    'texture_complexity': float(texture_complexity),
                    'red_ratio': float(red_ratio),
                    'blue_ratio': float(blue_ratio),
                    'shape_score': float(shape_score),
                    'is_text_block': True,
                    'text_score': float(text_score)
                }

            # 综合评分
            meaningful_score = (
                min(variance / 1000, 1.0) * 0.2 +  # 方差
                min(edge_density / 0.1, 1.0) * 0.3 +  # 边缘密度
                min(texture_complexity / 50, 1.0) * 0.2 +  # 纹理复杂度
                (red_ratio + blue_ratio) * 0.2 +  # 医学颜色特征
                shape_score * 0.1
            )

            is_meaningful = meaningful_score > 0.3

            return {
                'is_meaningful': bool(is_meaningful),
                'confidence': float(meaningful_score),
                'variance': float(variance),
                'edge_density': float(edge_density),
                'texture_complexity': float(texture_complexity),
                'red_ratio': float(red_ratio),
                'blue_ratio': float(blue_ratio),
                'shape_score': float(shape_score),
                'is_text_block': False,
                'text_score': float(text_score)
            }

        except Exception as e:
            logger.error(f"区域内容分析失败: {e}")
            return {'is_meaningful': False, 'confidence': 0.0, 'is_text_block': False}

    def _detect_text_characteristics(self, gray: np.ndarray, roi: np.ndarray) -> float:
        """检测文字特征 - 返回文字可能性评分"""
        try:
            # 1. 基于边缘密度的文字检测
            # 文字通常有规律的边缘分布
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])

            # 2. 基于纹理的文字检测
            # 文字区域通常有特定的纹理模式
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            texture_std = np.std(lbp)

            # 3. 基于二值化的文字检测
            # 文字通常是黑白分明的
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            black_ratio = np.count_nonzero(binary == 0) / (gray.shape[0] * gray.shape[1])

            # 4. 基于连通组件的文字检测
            # 文字有很多小的连通组件
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if num_labels > 1:
                # 计算小组件的比例（文字通常是很多小组件）
                small_components = 0
                total_components = 0
                for i in range(1, num_labels):  # 跳过背景
                    area = stats[i, cv2.CC_STAT_AREA]
                    if 10 < area < 500:  # 小组件范围
                        small_components += 1
                    total_components += 1

                small_component_ratio = small_components / total_components if total_components > 0 else 0
            else:
                small_component_ratio = 0

            # 5. 基于颜色一致性的文字检测
            # 文字区域通常是单色的
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            color_variance = np.var(hsv[:,:,1])  # 饱和度通道的方差

            # 综合文字评分 - 调整参数以更合理检测文字，避免误杀医学图像
            text_score = (
                (1.0 if edge_density > 0.03 and edge_density < 0.25 else 0.0) * 0.25 +  # 放宽边缘密度范围
                (1.0 if texture_std > 1.5 and texture_std < 6.0 else 0.0) * 0.15 +  # 放宽纹理范围
                (1.0 if black_ratio > 0.08 and black_ratio < 0.5 else 0.0) * 0.15 +  # 放宽黑白比例
                (small_component_ratio * 0.7) * 0.3 +  # 降低小组件权重，避免医学图像误判
                (1.0 if color_variance < 1500 else 0.0) * 0.15  # 放宽颜色一致性要求
            )

            return text_score

        except Exception as e:
            logger.error(f"文字特征检测失败: {e}")
            return 0.0

    def _classify_medical_image_advanced(self, roi: np.ndarray, content_analysis: Dict) -> str:
        """高级医学图像分类"""
        try:
            height, width = roi.shape[:2]
            aspect_ratio = width / height if height > 0 else 1.0

            # 1. 人物头像检测
            if self._is_portrait(roi, aspect_ratio, content_analysis):
                return 'portrait'

            # 2. 肿瘤/组织图像检测
            elif self._is_tumor_or_tissue(roi, content_analysis):
                return 'tumor_or_organ'

            # 3. 医学图表检测
            elif self._is_medical_chart(roi, content_analysis):
                return 'medical_diagram'

            # 4. 普通图表检测
            elif self._is_chart(roi, content_analysis):
                return 'chart'

            else:
                return 'medical_diagram'  # 默认为医学图表

        except Exception as e:
            logger.error(f"高级医学图像分类失败: {e}")
            return 'unknown'

    def _is_portrait(self, roi: np.ndarray, aspect_ratio: float, content_analysis: Dict) -> bool:
        """检测是否为人物头像"""
        try:
            # 基于位置和比例
            is_square_like = 0.7 <= aspect_ratio <= 1.3

            # 基于颜色分析（肤色检测）
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 肤色范围
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])

            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.count_nonzero(skin_mask) / (roi.shape[0] * roi.shape[1])

            # 基于边缘分析（面部特征）
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            has_face = len(faces) > 0

            # 综合判断
            portrait_score = (
                (1.0 if is_square_like else 0.0) * 0.3 +
                min(skin_ratio / 0.3, 1.0) * 0.4 +
                (1.0 if has_face else 0.0) * 0.3
            )

            return portrait_score > 0.5

        except Exception as e:
            logger.error(f"人物头像检测失败: {e}")
            return False

    def _is_tumor_or_tissue(self, roi: np.ndarray, content_analysis: Dict) -> bool:
        """检测是否为肿瘤或组织图像"""
        try:
            # 基于颜色特征（医学染色）
            red_ratio = content_analysis.get('red_ratio', 0)

            # 基于纹理特征
            texture_complexity = content_analysis.get('texture_complexity', 0)

            # 基于形状复杂度
            edge_density = content_analysis.get('edge_density', 0)

            # 综合评分
            tumor_score = (
                min(red_ratio / 0.1, 1.0) * 0.4 +  # 红色特征
                min(texture_complexity / 30, 1.0) * 0.3 +  # 纹理复杂度
                min(edge_density / 0.08, 1.0) * 0.3  # 边缘密度
            )

            return tumor_score > 0.4

        except Exception as e:
            logger.error(f"肿瘤/组织检测失败: {e}")
            return False

    def _is_medical_chart(self, roi: np.ndarray, content_analysis: Dict) -> bool:
        """检测是否为医学图表"""
        try:
            # 基于几何特征
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 检测直线（图表中的网格线）
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

            has_grid = lines is not None and len(lines) > 5

            # 基于颜色分布
            variance = content_analysis.get('variance', 0)

            # 综合判断
            chart_score = (
                (1.0 if has_grid else 0.0) * 0.6 +
                min(variance / 500, 1.0) * 0.4
            )

            return chart_score > 0.5

        except Exception as e:
            logger.error(f"医学图表检测失败: {e}")
            return False

    def _is_chart(self, roi: np.ndarray, content_analysis: Dict) -> bool:
        """检测是否为普通图表"""
        try:
            # 基于边缘分布
            edge_density = content_analysis.get('edge_density', 0)

            # 基于纹理规律
            texture_complexity = content_analysis.get('texture_complexity', 0)

            # 图表通常有规律的边缘和纹理
            chart_score = (
                min(edge_density / 0.05, 1.0) * 0.6 +
                (1.0 - min(texture_complexity / 20, 1.0)) * 0.4  # 纹理不能太复杂
            )

            return chart_score > 0.5

        except Exception as e:
            logger.error(f"图表检测失败: {e}")
            return False

    def _extract_chapter_info(self, text: str) -> Dict:
        """提取章节信息（保持原有实现）"""
        chapter_info = {
            'is_chapter_start': False,
            'chapter_title': '',
            'chapter_number': None
        }

        if not text:
            return chapter_info

        # 常见的章节标题模式
        patterns = [
            r'第([一二三四五六七八九十\d]+)章',  # 第一章
            r'第([一二三四五六七八九十\d]+)节',  # 第一节
            r'([一二三四五六七八九十\d]+)、',    # 一、
            r'([一二三四五六七八九十\d]+)\.',   # 1.
        ]

        lines = text.split('\n')
        for line in lines[:5]:  # 只检查前5行
            line = line.strip()
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    chapter_info['is_chapter_start'] = True
                    chapter_info['chapter_title'] = line
                    chapter_info['chapter_number'] = match.group(1)
                    return chapter_info

        return chapter_info

    def _detect_medical_content(self, text: str) -> bool:
        """检测是否包含医学内容（保持原有实现）"""
        if not text:
            return False

        medical_keywords = [
            '肺', '脏', '疾病', '肿瘤', '癌症', '细胞', '病理', '诊断',
            '治疗', '手术', '药物', '症状', '体征', '影像', 'CT', 'MRI'
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in medical_keywords)

    def _save_page_result(self, page_num: int, result: Dict):
        """保存单页处理结果"""
        try:
            # 保存JSON
            json_path = self.output_dir / "texts" / f"page_{page_num + 1}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # 保存文本内容
            text_path = self.output_dir / "texts" / f"page_{page_num + 1}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"第{page_num + 1}页\n")
                f.write(f"OCR文本:\n{result['ocr_text']}\n")
                f.write(f"章节信息: {result['chapter_info']}\n")
                f.write(f"检测到图片: {len(result['detected_images'])}\n")

                for i, img in enumerate(result['detected_images']):
                    f.write(f"  图片{i+1}: {img['image_type']} (置信度: {img['confidence']:.3f})\n")

        except Exception as e:
            logger.error(f"保存第{page_num + 1}页结果失败: {e}")

    def _generate_final_result(self) -> Dict:
        """生成最终结果"""
        result = {
            'metadata': {
                'pdf_path': self.pdf_path,
                'total_pages': len(self.processed_results),
                'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'ocr_engine': 'PaddleOCR' if PADDLEOCR_AVAILABLE else 'None',
                'processor_version': 'optimized_v1'
            },
            'pages': self.processed_results,
            'statistics': self._calculate_advanced_statistics()
        }

        # 保存完整结果
        result_path = self.output_dir / "complete_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 生成Markdown文档
        self._generate_advanced_markdown(result)

        return result

    def _calculate_advanced_statistics(self) -> Dict:
        """计算高级统计信息"""
        stats = {
            'total_text_pages': 0,
            'total_images': 0,
            'portrait_images': 0,
            'tumor_images': 0,
            'medical_diagrams': 0,
            'charts': 0,
            'medical_content_pages': 0,
            'chapter_pages': 0,
            'average_confidence': 0.0,
            'high_confidence_images': 0,  # 置信度>0.7
            'total_confidence_score': 0.0
        }

        total_confidence = 0.0
        confidence_count = 0

        for page in self.processed_results:
            if page['ocr_text']:
                stats['total_text_pages'] += 1

            if page['detected_images']:
                stats['total_images'] += len(page['detected_images'])

                for img in page['detected_images']:
                    confidence = img.get('confidence', 0)
                    stats['total_confidence_score'] += confidence

                    if confidence > 0.7:
                        stats['high_confidence_images'] += 1

                    if img['image_type'] == 'portrait':
                        stats['portrait_images'] += 1
                    elif img['image_type'] == 'tumor_or_organ':
                        stats['tumor_images'] += 1
                    elif img['image_type'] == 'medical_diagram':
                        stats['medical_diagrams'] += 1
                    elif img['image_type'] == 'chart':
                        stats['charts'] += 1

            if page['has_medical_content']:
                stats['medical_content_pages'] += 1

            if page['chapter_info']['is_chapter_start']:
                stats['chapter_pages'] += 1

            if page['ocr_confidence'] > 0:
                total_confidence += page['ocr_confidence']
                confidence_count += 1

        if confidence_count > 0:
            stats['average_confidence'] = total_confidence / confidence_count

        return stats

    def _generate_advanced_markdown(self, result: Dict):
        """生成高级Markdown文档"""
        md_content = f"""# {Path(self.pdf_path).stem}

## 文档信息

- 总页数: {result['metadata']['total_pages']}
- 处理时间: {result['metadata']['processing_time']}
- OCR引擎: {result['metadata']['ocr_engine']}
- 处理器版本: {result['metadata']['processor_version']}

## 统计信息

- 包含文本的页面: {result['statistics']['total_text_pages']}
- 检测到图片: {result['statistics']['total_images']}
  - 人物头像: {result['statistics']['portrait_images']}
  - 肿瘤/器官图像: {result['statistics']['tumor_images']}
  - 医学图表: {result['statistics']['medical_diagrams']}
  - 普通图表: {result['statistics']['charts']}
- 高置信度图片: {result['statistics']['high_confidence_images']}
- 医学内容页面: {result['statistics']['medical_content_pages']}
- 章节页面: {result['statistics']['chapter_pages']}
- 平均OCR置信度: {result['statistics']['average_confidence']:.3f}
- 总置信度评分: {result['statistics']['total_confidence_score']:.3f}

---

"""

        # 按页面添加内容
        for page in result['pages']:
            page_num = page['page_num']

            # 章节标题
            if page['chapter_info']['is_chapter_start']:
                chapter_title = page['chapter_info']['chapter_title']
                if chapter_title:
                    md_content += f"## {chapter_title}\n\n"

            # 页面内容
            md_content += f"### 第{page_num}页\n\n"

            # OCR文本
            if page['ocr_text']:
                md_content += f"**识别文本** (置信度: {page['ocr_confidence']:.3f})\n\n"
                md_content += f"```\n{page['ocr_text']}\n```\n\n"

            # 检测到的图片
            if page['detected_images']:
                md_content += "**检测到的图片:**\n\n"
                for img in page['detected_images']:
                    img_type_map = {
                        'portrait': '👤 人物头像',
                        'tumor_or_organ': '🔬 肿瘤/器官图像',
                        'medical_diagram': '📊 医学图表',
                        'chart': '📈 图表',
                        'unknown': '❓ 未知类型'
                    }

                    img_desc = img_type_map.get(img['image_type'], img['image_type'])
                    md_content += f"- {img_desc} (置信度: {img['confidence']:.3f})\n"

                    if img.get('analysis'):
                        analysis = img['analysis']
                        md_content += f"  - 方差: {analysis.get('variance', 0):.1f}\n"
                        md_content += f"  - 边缘密度: {analysis.get('edge_density', 0):.3f}\n"
                        md_content += f"  - 纹理复杂度: {analysis.get('texture_complexity', 0):.1f}\n"
                        md_content += f"  - 红色比例: {analysis.get('red_ratio', 0):.3f}\n"

                    if img.get('file_path') and Path(img['file_path']).exists():
                        rel_path = os.path.relpath(img['file_path'], self.output_dir)
                        md_content += f"  ![{img['image_type']}]({rel_path})\n\n"

                md_content += "\n"

            md_content += "---\n\n"

        # 保存Markdown文件
        md_path = self.output_dir / f"{Path(self.pdf_path).stem}_optimized.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"优化版Markdown文档已生成: {md_path}")

def main():
    """主函数"""
    pdf_path = "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf"

    if not os.path.exists(pdf_path):
        logger.error(f"PDF文件不存在: {pdf_path}")
        return

    processor = OptimizedPDFScanProcessor(pdf_path)

    try:
        # 测试特定页面 - 包括第5、6、16、17页
        result = processor.process_pdf(target_pages=[5, 6, 16, 17])

        logger.info("PDF优化处理完成！")

        # 打印统计信息
        print(f"\n=== 优化处理结果统计 ===")
        print(f"总页数: {result['metadata']['total_pages']}")
        print(f"包含文本页面: {result['statistics']['total_text_pages']}")
        print(f"检测到图片: {result['statistics']['total_images']}")
        print(f"人物头像: {result['statistics']['portrait_images']}")
        print(f"肿瘤图像: {result['statistics']['tumor_images']}")
        print(f"医学图表: {result['statistics']['medical_diagrams']}")
        print(f"普通图表: {result['statistics']['charts']}")
        print(f"高置信度图片: {result['statistics']['high_confidence_images']}")
        print(f"医学内容页面: {result['statistics']['medical_content_pages']}")
        print(f"章节页面: {result['statistics']['chapter_pages']}")
        print(f"平均OCR置信度: {result['statistics']['average_confidence']:.3f}")
        print(f"总置信度评分: {result['statistics']['total_confidence_score']:.3f}")

        # 显示详细结果
        print(f"\n=== 详细处理结果 ===")
        for page in result['pages']:
            print(f"第{page['page_num']}页: 文本长度={len(page['ocr_text'])}, 图片数={len(page['detected_images'])}, OCR置信度={page['ocr_confidence']:.3f}")
            if page['detected_images']:
                for i, img in enumerate(page['detected_images']):
                    print(f"  图片{i+1}: {img['image_type']} (置信度: {img['confidence']:.3f})")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        raise

if __name__ == "__main__":
    main()