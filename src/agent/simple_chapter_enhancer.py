#!/usr/bin/env python3
"""
极简章节增强器
提供轻量级的章节识别和搜索增强功能
"""

from typing import List, Dict, Any
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleChapterEnhancer:
    """极简章节增强器 - 只保留核心功能"""

    def __init__(self):
        """初始化简化的章节增强器"""
        # 只保留最常见的医学实体
        self.key_medical_entities = {
            '腺癌', '鳞癌', '小细胞癌', '大细胞癌', '肺腺癌', '肺鳞癌',
            '黏液腺癌', '粘液腺癌', '印戒细胞癌', '神经内分泌癌',
            '类癌', '肉瘤样癌', '腺样囊性癌', '黏液表皮样癌',
            'ROSE', '细胞学', '病理'
        }

        # 章节模式 - 简化版
        self.chapter_patterns = [
            r'第[一二三四五六七八九十]+节',
            r'第\d+节',
            r'第[一二三四五六七八九十]+章',
            r'第\d+章'
        ]

        logger.info("✅ 极简章节增强器初始化完成")

    def enhance_search_queries(self, query: str) -> List[str]:
        """
        生成增强的搜索查询 - 最多3个变种

        Args:
            query: 原始查询

        Returns:
            增强查询列表
        """
        queries = [query]  # 始终包含原始查询

        # 检查是否包含关键医学实体
        matched_entities = [entity for entity in self.key_medical_entities if entity in query]

        if matched_entities:
            # 策略1: 章节模式搜索
            queries.append(f"第.*节 {matched_entities[0]}")

            # 策略2: 专业特征搜索
            if "图像" in query or "特征" in query:
                queries.append(f"{matched_entities[0]} 细胞形态 结构特征")
            elif "病理" in query or "诊断" in query:
                queries.append(f"{matched_entities[0]} 病理特征 诊断要点")
            else:
                queries.append(f"{matched_entities[0]} 图像特征 细胞学表现")

        # 限制查询数量，避免过度搜索
        return queries[:3]

    def boost_result_scores(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        简单的分数提升 - 基于章节匹配和实体相关性

        Args:
            results: 搜索结果列表
            query: 原始查询

        Returns:
            分数提升后的结果列表
        """
        if not results:
            return results

        # 提取查询中的医学实体
        query_entities = [entity for entity in self.key_medical_entities if entity in query]

        for result in results:
            bonus_score = 0.0
            content = result.get('content', '')
            chapter_title = result.get('chapter_title', '')
            section_title = result.get('section_title', '')

            # 信号1: 章节结构存在
            if self._has_chapter_structure(content + chapter_title + section_title):
                bonus_score += 0.2
                logger.debug(f"📚 章节结构加分: +0.2")

            # 信号2: 实体在章节标题中
            if query_entities:
                for entity in query_entities:
                    if entity in chapter_title or entity in section_title:
                        bonus_score += 0.3
                        logger.debug(f"🎯 实体在标题中 '{entity}': +0.3")
                        break

            # 信号3: 内容质量指标
            if self._is_high_quality_content(content):
                bonus_score += 0.1
                logger.debug(f"✅ 高质量内容: +0.1")

            # 应用分数提升，设置上限
            original_score = result.get('score', 0.0)
            result['score'] = min(original_score + bonus_score, 1.0)
            result['chapter_boost_score'] = bonus_score

            logger.debug(f"📊 分数提升: {original_score:.3f} -> {result['score']:.3f} (提升: {bonus_score:.3f})")

        # 按提升后的分数重新排序
        return sorted(results, key=lambda x: x.get('score', 0.0), reverse=True)

    def _has_chapter_structure(self, text: str) -> bool:
        """检查是否有章节结构"""
        if not text:
            return False

        # 简单的章节检测
        return any(re.search(pattern, text) for pattern in self.chapter_patterns)

    def _is_high_quality_content(self, content: str) -> bool:
        """简单的内容质量评估"""
        if not content or len(content) < 50:
            return False

        # 检查是否有医学描述特征
        medical_indicators = ['细胞', '组织', '病理', '诊断', '呈', '可见', '显示']
        punctuation_indicators = ['。', '；', '：']

        # 至少有2个医学指标词和1个标点符号
        medical_score = sum(1 for indicator in medical_indicators if indicator in content)
        punct_score = sum(1 for punct in punctuation_indicators if punct in content)

        return medical_score >= 2 and punct_score >= 1

    def extract_chapter_info(self, content: str) -> Dict[str, str]:
        """
        极简的章节信息提取

        Args:
            content: 文档内容

        Returns:
            章节信息字典
        """
        chapter_info = {
            'chapter_title': '',
            'section_title': ''
        }

        if not content:
            return chapter_info

        # 只查找第一个匹配的章节标题
        lines = content.split('\n')[:20]  # 只看前20行

        for line in lines:
            line = line.strip()
            if not line or len(line) > 100:  # 跳过空行和过长行
                continue

            # 查找章节标题
            chapter_match = re.search(r'第([一二三四五六七八九十]+|\d+)章\s*([^\n\r]+)', line)
            if chapter_match and not chapter_info['chapter_title']:
                chapter_info['chapter_title'] = chapter_match.group(2).strip()
                continue

            # 查找节标题
            section_match = re.search(r'第([一二三四五六七八九十]+|\d+)节\s*([^\n\r]+)', line)
            if section_match and not chapter_info['section_title']:
                chapter_info['section_title'] = section_match.group(2).strip()
                # 清理标题（移除页码等）
                chapter_info['section_title'] = re.sub(r'\d+\.?\s*$', '', chapter_info['section_title']).strip()

        return chapter_info

# 创建全局实例
simple_chapter_enhancer = SimpleChapterEnhancer()

def get_chapter_enhancer() -> SimpleChapterEnhancer:
    """获取章节增强器实例"""
    return simple_chapter_enhancer