#!/usr/bin/env python3
"""
增强版ReAct智能代理
实现真正的多步推理和联合搜索
解决腺癌vs鳞癌比较类问题的局限性
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode  # 新版本不再提供ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
import json
import asyncio
import re
from datetime import datetime
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChapterIntelligence:
    """章节智能匹配模块"""

    def __init__(self, es_base_url: str, es_index: str):
        """初始化章节智能模块"""
        self.es_base_url = es_base_url
        self.es_index = es_index
        logger.info("✅ 章节智能模块初始化完成")

    def query_chapter_info(self, entity: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        查询Elasticsearch获取实体相关的章节信息（从内容中提取章节信息）

        关键策略：
        1. 首先查找包含章节标题的文档（如"第一节腺癌"）
        2. 过滤掉目录页（包含"录"或"日"）
        3. 对匹配的章节标题页给予极高权重

        Args:
            entity: 医学实体名称
            top_k: 返回结果数量

        Returns:
            章节信息列表
        """
        try:
            logger.info(f"🔍 查询章节信息: '{entity}'")

            # 构建节号列表（中文和数字）
            section_numbers = [
                "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
                "十一", "十二", "十三", "十四", "十五"
            ]

            # 构建should子句 - 重点匹配章节标题模式
            should_clauses = []

            # 1. 精确匹配"第X节{entity}"模式（极高权重）
            for num in section_numbers:
                should_clauses.append({
                    "match_phrase": {
                        "content": {
                            "query": f"第{num}节{entity}",
                            "boost": 1000.0
                        }
                    }
                })

            # 2. 匹配"第X节 {entity}"（有空格）
            for num in section_numbers:
                should_clauses.append({
                    "match_phrase": {
                        "content": {
                            "query": f"第{num}节 {entity}",
                            "boost": 800.0
                        }
                    }
                })

            # 3. 匹配"##第X节 {entity}"（Markdown格式）
            for num in section_numbers:
                should_clauses.append({
                    "match_phrase": {
                        "content": {
                            "query": f"##第{num}节 {entity}",
                            "boost": 1200.0
                        }
                    }
                })

            # 4. 基础实体匹配（低权重）
            should_clauses.append({
                "match": {
                    "content": {
                        "query": entity,
                        "boost": 1.0
                    }
                }
            })

            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "content": entity
                                }
                            }
                        ],
                        "should": should_clauses
                        # 移除must_not过滤器，改为应用层智能过滤
                    }
                },
                "size": top_k * 10,  # 多取一些候选
                "_source": ["page_number", "content", "metadata"]
            }

            response = requests.post(
                f"{self.es_base_url}/{self.es_index}/_search",
                headers={"Content-Type": "application/json"},
                data=json.dumps(search_body),
                timeout=10
            )

            if response.status_code == 200:
                results = response.json()
                hits = results['hits']['hits']

                chapter_info = []
                for hit in hits:
                    source = hit['_source']
                    content = source.get('content', '')
                    score = hit['_score']
                    page_number = source.get('page_number', 0)

                    # 智能过滤：如果是目录页但包含高权重的章节标题模式，则保留
                    is_table_of_contents = "录" in content[:50] or "日" in content[:50]
                    has_high_score_chapter_pattern = score > 1000  # 高权重匹配

                    if is_table_of_contents and not has_high_score_chapter_pattern:
                        logger.info(f"  ⏭️  跳过普通目录页: 第{page_number}页")
                        continue
                    elif is_table_of_contents and has_high_score_chapter_pattern:
                        logger.info(f"  ✅ 保留高权重目录页: 第{page_number}页 (得分: {score:.2f})")

                    # 从内容中提取章节信息（传入实体进行实体感知提取）
                    logger.debug(f"提取章节信息 - 实体: '{entity}', 内容长度: {len(content)}, 得分: {score}")
                    extracted_chapters = self._extract_chapter_info_from_content(content, entity)
                    logger.debug(f"提取结果: {extracted_chapters}")

                    # 如果成功提取到章节信息，添加到结果中
                    if extracted_chapters.get('section_title'):
                        chapter_info.append({
                            'chapter_title': extracted_chapters.get('chapter_title', ''),
                            'section_title': extracted_chapters.get('section_title', ''),
                            'page_number': page_number,
                            'chapter_path': '',
                            'score': score,
                            'content_preview': content[:200]  # 添加内容预览用于调试
                        })

                        logger.info(f"  ✅ 找到章节: '{extracted_chapters.get('chapter_title', '')}' - '{extracted_chapters.get('section_title', '')}' (第{page_number}页, 得分: {score:.2f})")

                        if len(chapter_info) >= top_k:
                            break

                logger.info(f"✅ 找到 {len(chapter_info)} 个相关章节")
                return chapter_info
            else:
                logger.warning(f"⚠️ 章节查询失败: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"❌ 章节查询异常: {e}")
            import traceback
            traceback.print_exc()
            return []

    def extract_chapter_keywords(self, question: str) -> List[str]:
        """
        从用户问题中提取章节关键词

        Args:
            question: 用户问题

        Returns:
            章节关键词列表
        """
        keywords = []

        # 章节模式匹配
        chapter_patterns = [
            r'第[一二三四五六七八九十]+章',
            r'第[一二三四五六七八九十]+节',
            r'第\d+章',
            r'第\d+节'
        ]

        for pattern in chapter_patterns:
            matches = re.findall(pattern, question)
            keywords.extend(matches)

        # 章节标题关键词
        title_keywords = [
            'ROSE', '细胞学特点', '细胞组学分型', '恶性肿瘤',
            '肺部实体', '罕见病', '快速现场评价'
        ]

        for keyword in title_keywords:
            if keyword in question:
                keywords.append(keyword)

        logger.info(f"📋 提取章节关键词: {keywords}")
        return keywords

    def build_chapter_aware_query(self, entity: str, original_question: str,
                                  chapter_info: List[Dict[str, Any]]) -> List[str]:
        """
        构建章节感知的增强查询

        Args:
            entity: 医学实体
            original_question: 原始问题
            chapter_info: 章节信息列表

        Returns:
            增强查询列表
        """
        enhanced_queries = []

        # 1. 基础实体查询
        enhanced_queries.append(entity)

        # 2. 添加章节标题模式查询（关键：直接匹配章节标题）
        enhanced_queries.append(f"第一节 {entity}")
        enhanced_queries.append(f"第二节 {entity}")
        enhanced_queries.append(f"第.*节 {entity}")
        enhanced_queries.append(f"第.*章.*{entity}")

        # 3. 如果找到章节信息，构建章节路径查询
        if chapter_info:
            top_chapter = chapter_info[0]  # 最相关的章节

            chapter_title = top_chapter.get('chapter_title', '')
            section_title = top_chapter.get('section_title', '')

            # 构建多层次查询
            if chapter_title and section_title:
                # 完整路径查询
                enhanced_queries.append(f"{chapter_title} {section_title} {entity}")
                # 章节标题查询
                enhanced_queries.append(f"{chapter_title} {entity}")
                # 小节标题查询
                enhanced_queries.append(f"{section_title} {entity}")
            elif section_title:
                enhanced_queries.append(f"{section_title} {entity}")
            elif chapter_title:
                enhanced_queries.append(f"{chapter_title} {entity}")

        # # 3. 基于问题类型的专业查询
        # if "图像特征" in original_question or "影像学" in original_question:
        #     enhanced_queries.append(f"{entity} 图像特征 细胞形态")
        # elif "病理" in original_question or "细胞" in original_question:
        #     enhanced_queries.append(f"{entity} 病理特征 细胞学表现")
        # else:
        #     enhanced_queries.append(f"{entity} 细胞形态 诊断要点")

        # logger.info(f"🔧 构建增强查询: {enhanced_queries}")
        return enhanced_queries

    def calculate_chapter_matching_score(self, query: str, result_chapter: str,
                                        result_section: str, chapter_info: List[Dict[str, Any]]) -> float:
        """
        计算章节匹配得分

        Args:
            query: 查询文本
            result_chapter: 结果的章节标题
            result_section: 结果的小节标题
            chapter_info: 预查询的章节信息

        Returns:
            章节匹配得分 (0-100)
        """
        score = 0.0

        query_lower = query.lower()
        result_chapter_lower = result_chapter.lower()
        result_section_lower = result_section.lower()

        # 1. 直接匹配查询中的章节关键词
        chapter_keywords = self.extract_chapter_keywords(query)
        for keyword in chapter_keywords:
            if keyword in result_chapter or keyword in result_section:
                score += 50.0  # 章节关键词直接匹配，高分
                logger.info(f"✅ 章节关键词匹配: '{keyword}' -> +50分")

        # 2. 匹配预查询的章节信息
        if chapter_info:
            for info in chapter_info[:3]:  # 只看前3个最相关章节
                info_chapter = info.get('chapter_title', '').lower()
                info_section = info.get('section_title', '').lower()

                # 完全匹配
                if result_chapter_lower == info_chapter and result_section_lower == info_section:
                    score += 100.0  # 完全匹配，极高分
                    logger.info(f"🎯 章节完全匹配: '{result_chapter} - {result_section}' -> +100分")
                    break
                # 章节匹配
                elif result_chapter_lower == info_chapter:
                    score += 60.0
                    logger.info(f"📖 章节匹配: '{result_chapter}' -> +60分")
                # 小节匹配
                elif result_section_lower == info_section:
                    score += 70.0
                    logger.info(f"📄 小节匹配: '{result_section}' -> +70分")

        # 3. 实体名称在标题中的匹配
        # 提取查询中的医学实体
        medical_entities = self._extract_medical_entities_from_query(query)
        for entity in medical_entities:
            entity_lower = entity.lower()
            if entity_lower in result_chapter_lower:
                score += 40.0
                logger.info(f"🔬 实体在章节标题中: '{entity}' -> +40分")
            if entity_lower in result_section_lower:
                score += 50.0
                logger.info(f"🔬 实体在小节标题中: '{entity}' -> +50分")

        return min(score, 150.0)  # 封顶150分

    def _extract_medical_entities_from_query(self, query: str) -> List[str]:
        """从查询中提取医学实体"""
        medical_entities = [
            '腺癌', '鳞癌', '小细胞癌', '大细胞癌', '肺腺癌', '肺鳞癌',
            '黏液腺癌', '粘液腺癌', '印戒细胞癌', '神经内分泌癌',
            '类癌', '肉瘤样癌', '腺样囊性癌', '黏液表皮样癌',
            'ROSE', '细胞学', '病理'
        ]

        entities = []
        for entity in medical_entities:
            if entity in query:
                entities.append(entity)

        return entities

    def _extract_chapter_info_from_content(self, content: str, entity: str = None) -> Dict[str, Any]:
        """
        从文档内容中提取章节信息

        Args:
            content: 文档内容
            entity: 可选，要查找的实体名称

        Returns:
            提取的章节信息字典
        """
        chapter_info = {
            'chapter_title': '',
            'section_title': '',
            'chapter_number': '',
            'section_number': ''
        }

        # 如果指定了实体，优先查找包含该实体的章节标题
        if entity:
            logger.debug(f"查找包含实体 '{entity}' 的章节标题")

            # 构建实体特定的节标题模式
            entity_section_patterns = [
                rf'第([一二三四五六七八九十]+)节\s*{re.escape(entity)}',    # 第X节 实体
                rf'第(\d+)节\s*{re.escape(entity)}',                      # 第1节 实体
                rf'##\s*第([一二三四五六七八九十]+)节\s*{re.escape(entity)}',  # ## 第X节 实体
                rf'##\s*第(\d+)节\s*{re.escape(entity)}',                  # ## 第1节 实体
                rf'第([一二三四五六七八九十]+)节{re.escape(entity)}',       # 第X节实体 (无空格)
                rf'第(\d+)节{re.escape(entity)}',                          # 第1节实体 (无空格)
            ]

            # 优先查找包含实体的节标题
            for pattern in entity_section_patterns:
                match = re.search(pattern, content)
                if match:
                    section_title = entity  # 节标题就是实体名称
                    section_number = match.group(1)

                    chapter_info['section_title'] = section_title
                    chapter_info['section_number'] = section_number

                    logger.debug(f"找到包含实体的节标题: 第{section_number}节 {section_title}")

                    # 同时查找对应的章节标题
                    # 查找该节之前的章节标题
                    content_before = content[:match.start()]
                    # 章节标题模式
                    chapter_patterns = [
                        r'第[一二三四五六七八九十]+章\s*([^\n\r]+)',  # 第X章 标题
                        r'第\d+章\s*([^\n\r]+)',                      # 第1章 标题
                        r'#\s*第[一二三四五六七八九十]+章\s*([^\n\r]+)',  # # 第X章 标题
                        r'#\s*第\d+章\s*([^\n\r]+)'                      # # 第1章 标题
                    ]
                    for chapter_pattern in chapter_patterns:
                        chapter_match = re.findall(chapter_pattern, content_before)
                        if chapter_match:
                            # 取最后一个匹配的章节标题
                            chapter_info['chapter_title'] = chapter_match[-1].strip()
                            # 提取章节号
                            chapter_num_match = re.search(r'第([一二三四五六七八九十\d]+)章', content_before)
                            if chapter_num_match:
                                chapter_info['chapter_number'] = chapter_num_match.group(1)
                            logger.debug(f"找到对应章节标题: {chapter_info['chapter_title']}")
                            break

                    return chapter_info

        # 如果没有指定实体或没找到包含实体的标题，使用通用提取
        logger.debug("使用通用章节提取")

        # 章节标题模式
        chapter_patterns = [
            r'第[一二三四五六七八九十]+章\s*([^\n\r]+)',  # 第X章 标题
            r'第\d+章\s*([^\n\r]+)',                      # 第1章 标题
            r'#\s*第[一二三四五六七八九十]+章\s*([^\n\r]+)',  # # 第X章 标题
            r'#\s*第\d+章\s*([^\n\r]+)'                      # # 第1章 标题
        ]

        # 节标题模式 - 支持多种格式
        section_patterns = [
            r'第[一二三四五六七八九十]+节\s*([^\n\r]+)',    # 第X节 标题 (有空格)
            r'第\d+节\s*([^\n\r]+)',                      # 第1节 标题 (有空格)
            r'##\s*第[一二三四五六七八九十]+节\s*([^\n\r]+)',  # ## 第X节 标题 (有空格)
            r'##\s*第\d+节\s*([^\n\r]+)',                  # ## 第1节 标题 (有空格)
            r'第[一二三四五六七八九十]+节([^\n\r]*)',       # 第X节标题 (无空格)
            r'第\d+节([^\n\r]*)',                          # 第1节标题 (无空格)
            r'##\s*第[一二三四五六七八九十]+节([^\n\r]*)',   # ## 第X节标题 (无空格)
            r'##\s*第\d+节([^\n\r]*)'                      # ## 第1节标题 (无空格)
        ]

        # 提取章节标题
        for pattern in chapter_patterns:
            match = re.search(pattern, content)
            if match:
                chapter_info['chapter_title'] = match.group(1).strip()
                # 提取章节号
                chapter_num_match = re.search(r'第([一二三四五六七八九十\d]+)章', match.group(0))
                if chapter_num_match:
                    chapter_info['chapter_number'] = chapter_num_match.group(1)
                logger.debug(f"找到章节标题: {chapter_info['chapter_title']}")
                break

        # 提取节标题
        for pattern in section_patterns:
            match = re.search(pattern, content)
            if match:
                section_title = match.group(1).strip()
                # 清理节标题（移除页码等无关信息）
                section_title = re.sub(r'\d+\.?\s*$', '', section_title)  # 移除结尾的页码
                section_title = re.sub(r'\.+$', '', section_title)      # 移除结尾的省略号
                section_title = section_title.strip()

                if section_title:  # 确保标题不为空
                    chapter_info['section_title'] = section_title
                    # 提取节号
                    section_num_match = re.search(r'第([一二三四五六七八九十\d]+)节', match.group(0))
                    if section_num_match:
                        chapter_info['section_number'] = section_num_match.group(1)
                    logger.debug(f"找到节标题: {chapter_info['section_title']}")
                break

        # 如果没有找到明确的章节标题，尝试从内容开头提取可能的标题
        if not chapter_info['chapter_title'] and not chapter_info['section_title']:
            lines = content.strip().split('\n')
            for line in lines[:10]:  # 检查前10行
                line = line.strip()
                # 查找可能的章节标题（包含特定关键词）
                if any(keyword in line for keyword in ['章', '节', '部分', '篇']):
                    if len(line) < 50 and not line.startswith('图'):  # 避免图像说明
                        if not chapter_info['chapter_title'] and '章' in line:
                            chapter_info['chapter_title'] = line.strip('#').strip()
                            logger.debug(f"从开头找到可能的章节标题: {chapter_info['chapter_title']}")
                        elif not chapter_info['section_title'] and '节' in line:
                            chapter_info['section_title'] = line.strip('#').strip()
                            logger.debug(f"从开头找到可能的节标题: {chapter_info['section_title']}")

        return chapter_info

class EnhancedAgentState(TypedDict):
    """增强版代理状态定义"""
    messages: List[BaseMessage]
    question: str
    original_question: str
    query_type: str  # 查询类型：comparison, single, multi_entity
    entities: List[str]  # 识别的实体列表
    current_entity_index: int
    context: List[Dict[str, Any]]
    retrieved_docs: List[Dict[str, Any]]
    search_queries: List[str]
    current_step: str
    reasoning_steps: List[Dict[str, Any]]
    final_answer: Optional[str]
    confidence: float
    tool_calls: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    search_results_map: Dict[str, List[Dict[str, Any]]]  # 按实体存储搜索结果

class EnhancedMedicalReActAgent:
    """增强版医学ReAct智能代理"""

    def __init__(self,
                 llm_manager=None,
                 retrieval_manager=None,
                 embedding_manager=None,
                 es_host: str = "elasticsearch",
                 es_port: int = 9200):
        """初始化增强版ReAct代理"""
        self.llm_manager = llm_manager
        self.retrieval_manager = retrieval_manager
        self.embedding_manager = embedding_manager

        # Elasticsearch连接配置
        self.es_base_url = f"http://{es_host}:{es_port}"
        self.es_index = "medical_documents_fixed"

        # 初始化章节智能模块
        self.chapter_intelligence = ChapterIntelligence(self.es_base_url, self.es_index)

        # 初始化工具
        self.tools = self._initialize_tools()

        # 构建图
        self._build_graph()

        logger.info("✅ 增强版ReAct代理初始化完成")

    def _initialize_tools(self):
        """初始化工具"""

        @tool
        def search_medical_documents(query: str, search_type: str = "hybrid", top_k: int = 5) -> Dict[str, Any]:
            """搜索医学文档

            Args:
                query: 搜索查询
                search_type: 搜索类型 (keyword, semantic, hybrid)
                top_k: 返回结果数量

            Returns:
                搜索结果字典
            """
            try:
                logger.info(f"🔍 搜索医学文档: '{query}' (类型: {search_type})")

                if self.retrieval_manager:
                    results = self.retrieval_manager.search(
                        query=query,
                        search_type=search_type,
                        top_k=top_k
                    )

                    # 格式化结果
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            'content': result.content,
                            'page_number': result.page_number,
                            'chapter_title': result.chapter_title,
                            'section_title': result.section_title,
                            'score': result.score,
                            'search_type': result.search_type
                        })

                    logger.info(f"✅ 搜索完成，找到 {len(formatted_results)} 个结果")
                    return {
                        'success': True,
                        'query': query,
                        'search_type': search_type,
                        'results': formatted_results,
                        'count': len(formatted_results)
                    }
                else:
                    return {
                        'success': False,
                        'error': '检索管理器未初始化'
                    }

            except Exception as e:
                logger.error(f"❌ 搜索失败: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }

        @tool
        def analyze_medical_content(content: str, analysis_type: str = "general") -> Dict[str, Any]:
            """分析医学内容

            Args:
                content: 要分析的医学内容
                analysis_type: 分析类型 (general, pathology, diagnosis, comparison)

            Returns:
                分析结果字典
            """
            try:
                logger.info(f"🔬 执行医学内容分析 (类型: {analysis_type})")

                # 根据分析类型执行不同的分析逻辑
                if analysis_type == "comparison":
                    analysis_result = self._perform_comparison_analysis(content)
                else:
                    analysis_result = self._perform_general_analysis(content)

                logger.info(f"✅ 内容分析完成")
                return {
                    'success': True,
                    'analysis': analysis_result,
                    'analysis_type': analysis_type
                }

            except Exception as e:
                logger.error(f"❌ 内容分析失败: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }

        @tool
        def extract_medical_entities(text: str) -> Dict[str, Any]:
            """提取医学实体

            Args:
                text: 输入文本

            Returns:
                提取的实体列表
            """
            try:
                logger.info(f"🏷️ 提取医学实体: '{text[:50]}...'")

                # 定义医学实体模式
                entity_patterns = {
                    'cancer_types': [
                        r'腺癌', r'鳞癌', r'小细胞癌', r'大细胞癌', r'肺腺癌',
                        r'肺鳞癌', r'乳腺癌', r'胃癌', r'肝癌', r'食道癌'
                    ],
                    'medical_terms': [
                        r'ROSE', r'细胞核', r'细胞质', r'分化', r'恶性',
                        r'良性', r'肿瘤', r'癌症', r'病理', r'诊断'
                    ],
                    'anatomy': [
                        r'肺部', r'肺脏', r'支气管', r'肺泡', r'胸膜',
                        r'纵隔', r'淋巴结', r'上皮', r'腺体'
                    ]
                }

                entities = {
                    'cancer_types': [],
                    'medical_terms': [],
                    'anatomy': [],
                    'all_entities': []
                }

                # 提取实体
                for category, patterns in entity_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        if matches:
                            entities[category].extend(matches)
                            entities['all_entities'].extend(matches)

                # 去重
                for category in entities:
                    if category != 'all_entities':
                        entities[category] = list(set(entities[category]))
                entities['all_entities'] = list(set(entities['all_entities']))

                logger.info(f"✅ 实体提取完成，找到 {len(entities['all_entities'])} 个实体")
                return {
                    'success': True,
                    'entities': entities,
                    'count': len(entities['all_entities'])
                }

            except Exception as e:
                logger.error(f"❌ 实体提取失败: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }

        return [search_medical_documents, analyze_medical_content, extract_medical_entities]

    def _perform_comparison_analysis(self, content: str) -> Dict[str, Any]:
        """执行对比分析"""
        logger.info("🔍 执行对比分析")

        # 提取对比要点
        key_points = {
            'similarities': [],
            'differences': [],
            'advantages': [],
            'disadvantages': []
        }

        # 简单的对比分析逻辑
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if any(word in line for word in ['相同', '相似', '一样']):
                key_points['similarities'].append(line)
            elif any(word in line for word in ['不同', '区别', '差异']):
                key_points['differences'].append(line)
            elif any(word in line for word in ['优势', '优点', '好处']):
                key_points['advantages'].append(line)
            elif any(word in line for word in ['劣势', '缺点', '不足']):
                key_points['disadvantages'].append(line)

        return {
            'key_points': key_points,
            'content_length': len(content),
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _perform_general_analysis(self, content: str) -> Dict[str, Any]:
        """执行通用分析"""
        return {
            'content_length': len(content),
            'key_concepts': self._extract_key_concepts(content),
            'medical_terms': self._extract_medical_terms(content),
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _extract_key_concepts(self, content: str) -> List[str]:
        """提取关键概念"""
        # 简单的关键词提取
        medical_keywords = ['细胞', '组织', '器官', '病理', '生理', '诊断', '治疗', '预后']
        concepts = []
        content_lower = content.lower()
        for keyword in medical_keywords:
            if keyword in content_lower:
                concepts.append(keyword)
        return concepts[:5]

    def _extract_medical_terms(self, content: str) -> List[str]:
        """提取医学术语"""
        terms = []
        if "腺癌" in content:
            terms.append("腺癌")
        if "鳞癌" in content:
            terms.append("鳞癌")
        if "ROSE" in content:
            terms.append("ROSE")
        return terms[:5]

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去重搜索结果

        Args:
            results: 搜索结果列表

        Returns:
            去重后的结果列表
        """
        seen_ids = set()
        unique_results = []

        for result in results:
            # 使用doc_id去重
            doc_id = result.get('doc_id', '')
            if doc_id and doc_id in seen_ids:
                continue

            # 如果没有doc_id，使用content的前100个字符作为标识
            if not doc_id:
                content_key = result.get('content', '')[:100]
                if content_key in seen_ids:
                    continue
                seen_ids.add(content_key)
            else:
                seen_ids.add(doc_id)

            unique_results.append(result)

        logger.info(f"🔄 去重: {len(results)} -> {len(unique_results)}")
        return unique_results

    def _build_graph(self):
        """构建增强版LangGraph图"""
        workflow = StateGraph(EnhancedAgentState)

        # 定义节点
        workflow.add_node("intent_analyzer", self._intent_analysis_node)
        workflow.add_node("query_decomposer", self._query_decomposition_node)
        workflow.add_node("entity_searcher", self._entity_search_node)
        workflow.add_node("content_analyzer", self._content_analysis_node)
        workflow.add_node("answer_generator", self._answer_generation_node)

        # 定义条件边
        workflow.add_conditional_edges(
            "intent_analyzer",
            self._should_decompose_query,
            {
                "decompose": "query_decomposer",
                "direct_search": "entity_searcher"
            }
        )

        workflow.add_edge("query_decomposer", "entity_searcher")
        workflow.add_conditional_edges(
            "entity_searcher",
            self._should_continue_search,
            {
                "continue": "entity_searcher",
                "analyze": "content_analyzer"
            }
        )
        workflow.add_edge("content_analyzer", "answer_generator")
        workflow.add_edge("answer_generator", END)

        # 设置入口点
        workflow.set_entry_point("intent_analyzer")

        self.graph = workflow.compile()
        logger.info("✅ 增强版ReAct代理图构建完成")

    def _intent_analysis_node(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """意图分析节点"""
        logger.info("🧠 执行意图分析节点...")

        question = state["question"]

        # 分析查询类型
        query_type = self._analyze_query_type(question)

        # 提取医学实体
        entities = self._extract_entities_from_question(question)

        logger.info(f"✅ 意图分析完成 - 类型: {query_type}, 实体数: {len(entities)}")

        reasoning_step = {
            "step": "intent_analysis",
            "thought": f"分析用户查询意图: {question}",
            "action": "analyze_query_intent",
            "action_input": {"question": question},
            "observation": f"查询类型: {query_type}, 识别实体: {entities}",
            "timestamp": datetime.now().isoformat()
        }

        return {
            "question": question,
            "query_type": query_type,
            "entities": entities,
            "current_entity_index": 0,
            "reasoning_steps": state.get("reasoning_steps", []) + [reasoning_step],
            "search_results_map": {}
        }

    def _query_decomposition_node(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """查询分解节点 - 使用增强的搜索查询生成"""
        logger.info("📝 执行查询分解节点...")

        entities = state["entities"]
        original_question = state.get("original_question", "")

        # 为每个实体生成增强的搜索查询
        search_queries = []
        for entity in entities:
            # 只对真正的医学实体生成增强查询，跳过通用术语
            medical_entities_only = {
                '腺癌', '鳞癌', '小细胞癌', '大细胞癌', '肺腺癌', '肺鳞癌',
                '肺小细胞癌', '肺大细胞癌', '粘液腺癌', '黏液腺癌', '印戒细胞癌',
                '粘液癌', '乳腺癌', '胃癌', '肝癌', '食道癌', '结肠癌', '直肠癌',
                '胰腺癌', '胆管癌', '胆囊癌', '甲状腺癌', '前列腺癌'
            }

            if entity in medical_entities_only:
                # 生成增强的搜索查询
                enhanced_queries = self._generate_enhanced_search_queries(entity, original_question)
                if enhanced_queries:
                    # 选择最具体的查询
                    search_queries.append(enhanced_queries[-1])
                else:
                    # 回退到基础查询
                    search_queries.append(f"{entity} 图像特征")
            else:
                # 通用术语使用基础查询
                search_queries.append(f"{entity} 图像特征")

        logger.info(f"✅ 查询分解完成，生成 {len(search_queries)} 个增强搜索查询")

        reasoning_step = {
            "step": "query_decomposition",
            "thought": f"将复合查询分解为多个单实体查询，使用增强搜索策略",
            "action": "decompose_query",
            "action_input": {"entities": entities},
            "observation": f"生成增强搜索查询: {search_queries}",
            "timestamp": datetime.now().isoformat()
        }

        return {
            "search_queries": search_queries,
            "reasoning_steps": state.get("reasoning_steps", []) + [reasoning_step]
        }

    def _entity_search_node(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """实体搜索节点"""
        logger.info("🔍 执行实体搜索节点...")

        entities = state["entities"]
        current_index = state.get("current_entity_index", 0)
        search_results_map = state.get("search_results_map", {})
        # 获取预生成的搜索查询
        search_queries = state.get("search_queries", [])

        if current_index < len(entities):
            current_entity = entities[current_index]

            # 定义需要搜索的真正医学实体（排除通用术语）
            medical_entities_only = {
                '腺癌', '鳞癌', '小细胞癌', '大细胞癌', '肺腺癌', '肺鳞癌',
                '肺小细胞癌', '肺大细胞癌', '粘液腺癌', '黏液腺癌', '印戒细胞癌',
                '粘液癌', '乳腺癌', '胃癌', '肝癌', '食道癌', '结肠癌', '直肠癌',
                '胰腺癌', '胆管癌', '胆囊癌', '甲状腺癌', '前列腺癌'
            }

            # 只搜索真正的医学实体，跳过通用术语
            if current_entity not in medical_entities_only:
                logger.info(f"跳过非医学实体: {current_entity}")
                return {
                    "current_entity_index": current_index + 1,
                    "search_results_map": search_results_map,
                    "reasoning_steps": state.get("reasoning_steps", [])
                }

            # 如果有预生成的搜索查询，使用最相关的那个
            if search_queries and current_index < len(search_queries):
                search_query = search_queries[current_index]
                logger.info(f"使用预生成的搜索查询: {search_query}")
            else:
                # 生成更智能的搜索查询，基于原始问题上下文和章节结构
                original_question = state.get("original_question", "")

                # 构建多层次的搜索查询
                search_queries = []

                # 1. 基础实体搜索
                search_queries.append(f"{current_entity}")

                # 2. 章节结构搜索（特别针对黏液腺癌）
                if current_entity == "黏液腺癌":
                    search_queries.append(f"第九节 {current_entity}")
                    search_queries.append(f"第二章 肺部实体恶性肿瘤 {current_entity}")
                    search_queries.append(f"{current_entity} 黏液湖 癌细胞")
                    search_queries.append(f"{current_entity} 柱状 立方形 核膜")

                # 3. 基于问题类型的专业搜索
                if "图像特征" in original_question or "影像学" in original_question:
                    search_queries.append(f"{current_entity} 图像特征 细胞形态")
                    search_queries.append(f"{current_entity} 细胞学表现 形态特征")
                    search_queries.append(f"{current_entity} 病理描述 细胞形态")
                elif "病理" in original_question or "细胞" in original_question:
                    search_queries.append(f"{current_entity} 病理特征 细胞学表现")
                    search_queries.append(f"{current_entity} 组织学特征 病理表现")
                    search_queries.append(f"{current_entity} 显微镜下表现")
                else:
                    # 对于一般查询，构建更具体的搜索词
                    search_queries.append(f"{current_entity} 细胞形态 结构特征")
                    search_queries.append(f"{current_entity} 病理描述 细胞学")
                    search_queries.append(f"{current_entity} 诊断要点 细胞特征")

                # 4. 特别针对描述性内容的搜索
                if "描述" in original_question or "特征" in original_question:
                    search_queries.append(f"{current_entity} 详细描述 细胞形态")
                    search_queries.append(f"{current_entity} 结构特征 组织学")

                # 选择最相关的搜索查询（优先选择包含具体医学术语的）
                if len(search_queries) > 1:
                    # 优先选择包含具体医学术语的查询
                    priority_terms = ['黏液湖', '柱状', '立方形', '核膜', '紧密聚集', '分泌泡']
                    best_query = search_queries[-1]  # 默认使用最后一个

                    # 查找包含最多医学术语的查询
                    max_terms_count = 0
                    for query in search_queries:
                        terms_count = sum(1 for term in priority_terms if term in query)
                        if terms_count > max_terms_count:
                            max_terms_count = terms_count
                            best_query = query

                    search_query = best_query
                else:
                    search_query = search_queries[0]

                logger.info(f"生成的搜索查询选项: {search_queries}")
                logger.info(f"使用搜索查询: {search_query}")

            logger.info(f"正在搜索实体 {current_index + 1}/{len(entities)}: {current_entity}")

    def _generate_enhanced_search_queries(self, entity: str, original_question: str) -> List[str]:
        """生成增强的搜索查询"""
        search_queries = []

        # 1. 基础实体搜索
        search_queries.append(f"{entity}")

        # 2. 章节结构搜索（特别针对黏液腺癌）
        # if entity == "黏液腺癌":
        #     search_queries.append(f"第九节 {entity}")
        #     search_queries.append(f"第二章 肺部实体恶性肿瘤 {entity}")
        #     search_queries.append(f"{entity} 黏液湖 癌细胞")
        #     search_queries.append(f"{entity} 柱状 立方形 核膜")

        # 3. 基于问题类型的专业搜索
        if "图像特征" in original_question or "影像学" in original_question:
            search_queries.append(f"{entity} 图像特征 细胞形态")
            search_queries.append(f"{entity} 细胞学表现 形态特征")
            search_queries.append(f"{entity} 病理描述 细胞形态")
        elif "病理" in original_question or "细胞" in original_question:
            search_queries.append(f"{entity} 病理特征 细胞学表现")
            search_queries.append(f"{entity} 组织学特征 病理表现")
            search_queries.append(f"{entity} 显微镜下表现")
        else:
            # 对于一般查询，构建更具体的搜索词
            search_queries.append(f"{entity} 细胞形态 结构特征")
            search_queries.append(f"{entity} 病理描述 细胞学")
            search_queries.append(f"{entity} 诊断要点 细胞特征")

        # 4. 特别针对描述性内容的搜索
        if "描述" in original_question or "特征" in original_question:
            search_queries.append(f"{entity} 详细描述 细胞形态")
            search_queries.append(f"{entity} 结构特征 组织学")

        return search_queries if len(search_queries) > 1 else []

    def _entity_search_node(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """实体搜索节点 - 集成章节智能匹配"""
        logger.info("🔍 执行实体搜索节点...")

        entities = state["entities"]
        current_index = state.get("current_entity_index", 0)
        search_results_map = state.get("search_results_map", {})
        original_question = state.get("original_question", "")
        # 获取预生成的搜索查询
        search_queries = state.get("search_queries", [])

        if current_index < len(entities):
            current_entity = entities[current_index]

            # 定义需要搜索的真正医学实体（排除通用术语）
            medical_entities_only = {
                '腺癌', '鳞癌', '小细胞癌', '大细胞癌', '肺腺癌', '肺鳞癌',
                '肺小细胞癌', '肺大细胞癌', '粘液腺癌', '黏液腺癌', '印戒细胞癌',
                '粘液癌', '乳腺癌', '胃癌', '肝癌', '食道癌', '结肠癌', '直肠癌',
                '胰腺癌', '胆管癌', '胆囊癌', '甲状腺癌', '前列腺癌'
            }

            # 只搜索真正的医学实体，跳过通用术语
            if current_entity not in medical_entities_only:
                logger.info(f"跳过非医学实体: {current_entity}")
                return {
                    "current_entity_index": current_index + 1,
                    "search_results_map": search_results_map,
                    "reasoning_steps": state.get("reasoning_steps", [])
                }

            # 使用章节智能模块构建增强查询
            logger.info(f"🧠 使用章节智能模块分析实体: {current_entity}")

            # 1. 查询该实体相关的章节信息
            chapter_info = self.chapter_intelligence.query_chapter_info(current_entity, top_k=3)

            # 2. 构建章节感知的增强查询
            if chapter_info:
                enhanced_queries = self.chapter_intelligence.build_chapter_aware_query(
                    current_entity, original_question, chapter_info
                )
                logger.info(f"📚 章节增强查询: {enhanced_queries}")
            else:
                # 如果没有章节信息，使用基础查询
                enhanced_queries = [current_entity]

            # 3. 执行多个增强查询，合并结果
            all_results = []
            for query in enhanced_queries:
                logger.info(f"🔍 执行增强查询: '{query}'")
                if self.retrieval_manager:
                    search_results = self.retrieval_manager.search(
                        query=query,
                        search_type="hybrid",
                        top_k=5
                    )
                    all_results.extend(search_results)

            # 4. 去重并应用章节匹配评分
            unique_results = self._deduplicate_results(all_results)

            # 5. 应用章节匹配评分（从内容中提取章节信息）
            if chapter_info:
                for result in unique_results:
                    # 从结果内容中提取章节信息
                    content_chapter_info = self.chapter_intelligence._extract_chapter_info_from_content(
                        result.get('content', '')
                    )

                    chapter_score = self.chapter_intelligence.calculate_chapter_matching_score(
                        original_question,
                        content_chapter_info.get('chapter_title', ''),
                        content_chapter_info.get('section_title', ''),
                        chapter_info
                    )
                    # 将章节评分叠加到原始评分上
                    original_score = result.get('score', 0.0)
                    result['score'] = original_score + chapter_score
                    result['chapter_matching_score'] = chapter_score
                    result['chapter_title'] = content_chapter_info.get('chapter_title', '')
                    result['section_title'] = content_chapter_info.get('section_title', '')
                    logger.info(f"📊 章节评分: {content_chapter_info.get('chapter_title', '')} - {content_chapter_info.get('section_title', '')} = {chapter_score}")

            # 6. 按最终评分重新排序
            unique_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)

            # 7. 格式化结果
            formatted_results = []
            for result in unique_results[:5]:  # 只取前5个结果
                formatted_results.append({
                    'content': result.get('content', ''),
                    'page_number': result.get('page_number', 0),
                    'chapter_title': result.get('chapter_title', ''),
                    'section_title': result.get('section_title', ''),
                    'score': result.get('score', 0.0),
                    'chapter_matching_score': result.get('chapter_matching_score', 0.0),
                    'search_type': result.get('search_type', 'hybrid'),
                    'entity': current_entity
                })

            # 存储结果
            search_results_map = state.get("search_results_map", {})
            search_results_map[current_entity] = formatted_results

            logger.info(f"✅ 实体 {current_entity} 搜索完成，找到 {len(formatted_results)} 个结果")

            reasoning_step = {
                "step": f"entity_search_{current_entity}",
                "thought": f"使用章节智能搜索实体 '{current_entity}' 的相关信息",
                "action": "search_medical_documents_with_chapter_intelligence",
                "action_input": {
                    "entity": current_entity,
                    "chapter_info_found": len(chapter_info) > 0,
                    "enhanced_queries": enhanced_queries
                },
                "observation": f"找到 {len(formatted_results)} 个结果，章节匹配增强完成",
                "timestamp": datetime.now().isoformat()
            }

            return {
                "current_entity_index": current_index + 1,
                "search_results_map": search_results_map,
                "reasoning_steps": state.get("reasoning_steps", []) + [reasoning_step]
            }
        else:
            logger.info("✅ 所有实体搜索完成")
            return {
                "current_entity_index": current_index
            }

    def _content_analysis_node(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """内容分析节点"""
        logger.info("🔬 执行内容分析节点...")

        search_results_map = state.get("search_results_map", {})
        query_type = state.get("query_type", "general")

        # 合并所有搜索结果
        all_docs = []
        for entity, docs in search_results_map.items():
            all_docs.extend(docs)

        # 根据查询类型进行分析
        if query_type == "comparison" and len(search_results_map) > 1:
            # 对比分析
            analysis_content = self._prepare_comparison_content(search_results_map)
            analysis_result = self._perform_comparison_analysis(analysis_content)
        else:
            # 通用分析
            combined_content = " ".join([doc['content'] for doc in all_docs])
            analysis_result = self._perform_general_analysis(combined_content)

        logger.info(f"✅ 内容分析完成，分析了 {len(all_docs)} 个文档")

        reasoning_step = {
            "step": "content_analysis",
            "thought": f"分析检索到的医学内容 (类型: {query_type})",
            "action": "analyze_medical_content",
            "action_input": {"content_type": query_type, "doc_count": len(all_docs)},
            "observation": f"分析完成，涉及 {len(all_docs)} 个文档",
            "timestamp": datetime.now().isoformat()
        }

        return {
            "retrieved_docs": all_docs,
            "reasoning_steps": state.get("reasoning_steps", []) + [reasoning_step]
        }

    def _answer_generation_node(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """答案生成节点"""
        logger.info("📝 执行答案生成节点...")

        question = state["question"]
        retrieved_docs = state.get("retrieved_docs", [])
        reasoning_steps = state.get("reasoning_steps", [])
        query_type = state.get("query_type", "general")
        search_results_map = state.get("search_results_map", {})

        # 调试日志
        logger.info(f"📋 答案生成节点接收到的文档数量: {len(retrieved_docs)}")
        if retrieved_docs:
            logger.info(f"📋 第一个文档示例: {retrieved_docs[0]}")

        # 生成答案
        if self.llm_manager:
            # 构建基于实体的回答
            if query_type == "comparison" and search_results_map:
                answer = self._generate_comparison_answer(question, search_results_map, reasoning_steps)
            else:
                answer = self._generate_general_answer(question, retrieved_docs, reasoning_steps)

            logger.info("✅ 答案生成完成")

            final_reasoning_step = {
                "step": "answer_generation",
                "thought": f"基于所有搜索结果生成最终答案",
                "action": "generate_answer",
                "action_input": {"query_type": query_type, "doc_count": len(retrieved_docs)},
                "observation": f"生成答案完成，答案长度: {len(answer)} 字符",
                "timestamp": datetime.now().isoformat()
            }

            return {
                "final_answer": answer,
                "reasoning_steps": reasoning_steps + [final_reasoning_step],
                "confidence": 0.9,
                "retrieved_docs": retrieved_docs  # 添加检索到的文档
            }
        else:
            logger.error("❌ LLM管理器未初始化")
            return {
                "final_answer": "抱歉，系统暂时无法生成答案。",
                "reasoning_steps": reasoning_steps,
                "confidence": 0.0,
                "retrieved_docs": retrieved_docs  # 添加检索到的文档
            }

    def _analyze_query_type(self, question: str) -> str:
        """分析查询类型"""
        question_lower = question.lower()

        # 对比类查询
        comparison_keywords = ['区别', '差异', '不同', '比较', 'vs', 'versus', '对比']
        if any(keyword in question_lower for keyword in comparison_keywords):
            return "comparison"

        # 多实体查询
        entity_count = len(self._extract_entities_from_question(question))
        if entity_count > 1:
            return "multi_entity"

        return "single"

    def _extract_entities_from_question(self, question: str) -> List[str]:
        """从问题中提取实体"""
        # 定义更全面的医学实体，包括各种亚型和详细分类
        medical_entities = {
            # 基础癌症类型
            '腺癌', '鳞癌', '小细胞癌', '大细胞癌',
            # 肺癌具体类型
            '肺腺癌', '肺鳞癌', '肺小细胞癌', '肺大细胞癌',
            # 粘液相关癌症
            '粘液腺癌', '黏液腺癌', '印戒细胞癌', '粘液癌',
            # 其他常见癌症
            '乳腺癌', '胃癌', '肝癌', '食道癌', '结肠癌', '直肠癌',
            '胰腺癌', '胆管癌', '胆囊癌', '甲状腺癌', '前列腺癌',
            # 医学术语
            '图像特征', '影像学', '病理学', '细胞学', '组织学',
            '分化程度', '恶性程度', '转移', '浸润', '预后'
        }

        entities = []
        # 首先检查更长的实体（避免部分匹配问题）
        sorted_entities = sorted(medical_entities, key=len, reverse=True)

        temp_question = question
        for entity in sorted_entities:
            if entity in temp_question:
                entities.append(entity)
                # 移除已匹配的实体，避免重复匹配
                temp_question = temp_question.replace(entity, '')

        return entities

    def _should_decompose_query(self, state: EnhancedAgentState) -> str:
        """判断是否需要分解查询"""
        query_type = state.get("query_type", "single")
        entities = state.get("entities", [])

        if query_type == "comparison" or len(entities) > 1:
            return "decompose"
        else:
            return "direct_search"

    def _should_continue_search(self, state: EnhancedAgentState) -> str:
        """判断是否需要继续搜索"""
        current_index = state.get("current_entity_index", 0)
        entities = state.get("entities", [])

        if current_index < len(entities):
            return "continue"
        else:
            return "analyze"

    def _prepare_comparison_content(self, search_results_map: Dict[str, List[Dict[str, Any]]]) -> str:
        """准备对比分析内容"""
        content_parts = []

        for entity, docs in search_results_map.items():
            entity_content = f"【{entity}相关信息】\n"
            for i, doc in enumerate(docs[:3]):  # 取前3个文档
                entity_content += f"文档{i+1}: {doc['content'][:200]}...\n"
            content_parts.append(entity_content)

        return "\n\n".join(content_parts)

    def _generate_comparison_answer(self, question: str, search_results_map: Dict[str, List[Dict[str, Any]]], reasoning_steps: List[Dict[str, Any]]) -> str:
        """生成对比答案"""
        # 构建对比答案的提示
        comparison_prompt = f"""
基于以下医学文献内容，回答用户关于'{question}'的问题：

{self._prepare_comparison_content(search_results_map)}

请提供详细的对比分析，包括：
1. 每种疾病的主要特征
2. 它们之间的主要区别
3. 临床意义和诊断要点
4. 基于提供文献的局限性说明

回答要求：
- 必须基于提供的医学文献内容
- 使用专业但易于理解的医学术语
- 明确指出信息来源的局限性
- 建议咨询专业医疗人员获取完整信息
"""

        if self.llm_manager:
            messages = [
                {"role": "system", "content": "你是一位专业的医学AI助手，专门进行医学内容的对比分析。"},
                {"role": "user", "content": comparison_prompt}
            ]

            try:
                response = self.llm_manager.generate_response(messages, model="deepseek-reasoner")
                logger.info(f"📊 对比分析 - LLM响应类型: {type(response)}")
                logger.info(f"📊 对比分析 - LLM响应内容: {str(response)[:200]}...")

                # 确保返回的是字符串而不是协程
                if hasattr(response, '__await__'):
                    # 如果是协程，需要运行它（在同步上下文中）
                    import asyncio
                    response = asyncio.run(response)
                    logger.info(f"📊 对比分析 - 协程执行后的响应类型: {type(response)}")

                # 如果响应是LLMResponse对象，提取content字段
                if hasattr(response, 'content'):
                    content = response.content
                    logger.info(f"📊 对比分析 - 提取到的content: {content[:100]}...")
                    return content
                else:
                    result = str(response) if response else "抱歉，无法生成对比分析。"
                    logger.info(f"📊 对比分析 - 转换后的字符串: {result[:100]}...")
                    return result
            except Exception as e:
                logger.error(f"❌ 对比答案生成失败: {e}")
                return f"抱歉，生成对比分析时出错: {str(e)}"
        else:
            return "抱歉，无法生成对比分析。"

    def _generate_general_answer(self, question: str, retrieved_docs: List[Dict[str, Any]], reasoning_steps: List[Dict[str, Any]]) -> str:
        """生成通用答案"""
        # 构建通用答案的提示
        general_prompt = f"""
基于以下医学文献内容，回答用户关于'{question}'的问题：

{chr(10).join([doc.get('content', '') for doc in retrieved_docs[:5]])}

回答要求：
- 必须基于提供的医学文献内容
- 使用专业但易于理解的医学术语
- 提供准确、可靠的医学信息
- 如果不确定，要明确说明
- 建议咨询专业医疗人员获取个性化建议
"""

        if self.llm_manager:
            messages = [
                {"role": "system", "content": "你是一位专业的医学AI助手，基于医学文献提供准确的医学信息。"},
                {"role": "user", "content": general_prompt}
            ]

            try:
                response = self.llm_manager.generate_response(messages, model="deepseek-reasoner")
                logger.info(f"📊 LLM响应类型: {type(response)}")
                logger.info(f"📊 LLM响应内容: {str(response)[:200]}...")

                # 确保返回的是字符串而不是协程
                if hasattr(response, '__await__'):
                    # 如果是协程，需要运行它（在同步上下文中）
                    import asyncio
                    response = asyncio.run(response)
                    logger.info(f"📊 协程执行后的响应类型: {type(response)}")

                # 如果响应是LLMResponse对象，提取content字段
                if hasattr(response, 'content'):
                    content = response.content
                    logger.info(f"📊 提取到的content: {content[:100]}...")
                    return content
                else:
                    result = str(response) if response else "抱歉，无法生成答案。"
                    logger.info(f"📊 转换后的字符串: {result[:100]}...")
                    return result
            except Exception as e:
                logger.error(f"❌ 通用答案生成失败: {e}")
                return f"抱歉，生成答案时出错: {str(e)}"
        else:
            return "抱歉，无法生成答案。"

    def process_question_sync(self, question: str) -> Dict[str, Any]:
        """同步处理问题（兼容原版API）"""
        return self.process_query(question, user_id="default", search_config={})

    def process_query(self, question: str, user_id: str = "default", search_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理用户查询"""
        logger.info(f"🚀 开始处理增强查询: '{question}'")

        start_time = datetime.now()  # 开始计时

        try:
            # 初始化状态
            initial_state = {
                "question": question,
                "original_question": question,
                "messages": [HumanMessage(content=question)],
                "reasoning_steps": [],
                "search_results_map": {},
                "current_entity_index": 0,
                "metadata": {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "search_config": search_config or {}
                }
            }

            # 执行图
            result = self.graph.invoke(initial_state)

            # 计算响应时间
            response_time = (datetime.now() - start_time).total_seconds()

            # 调试日志 - 检查图执行结果
            logger.info(f"📊 图执行结果包含的字段: {list(result.keys())}")
            logger.info(f"📊 final_answer 类型: {type(result.get('final_answer', 'None'))}")
            logger.info(f"📊 retrieved_docs 数量: {len(result.get('retrieved_docs', []))}")

            # 构建响应
            response = {
                "query_id": f"enhanced_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(question) % 10000}",
                "question": question,
                "answer": result.get("final_answer", "无法生成答案"),
                "confidence": result.get("confidence", 0.0),
                "reasoning_steps": result.get("reasoning_steps", []),
                "retrieved_documents": result.get("retrieved_docs", []),
                "response_time": response_time,
                "model_used": "enhanced_react",
                "metadata": {
                    "query_type": result.get("query_type", "unknown"),
                    "entities": result.get("entities", []),
                    "search_results_count": len(result.get("retrieved_docs", [])),
                    "reasoning_steps_count": len(result.get("reasoning_steps", []))
                }
            }

            logger.info(f"✅ 增强查询处理完成，推理步骤: {len(response['reasoning_steps'])}")
            return response

        except Exception as e:
            logger.error(f"❌ 增强查询处理失败: {e}")
            return {
                "query_id": f"error_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "question": question,
                "answer": f"处理查询时出错: {str(e)}",
                "confidence": 0.0,
                "reasoning_steps": [],
                "retrieved_documents": [],
                "response_time": 0,
                "model_used": "enhanced_react",
                "metadata": {"error": str(e)}
            }