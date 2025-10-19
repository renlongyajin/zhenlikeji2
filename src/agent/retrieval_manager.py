#!/usr/bin/env python3
"""
检索管理器
集成Elasticsearch和Milvus的混合检索系统
"""

import requests
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果数据结构"""
    content: str
    score: float
    source: str
    page_number: int
    chapter_title: str
    section_title: str
    doc_id: str
    search_type: str

class MedicalRetrievalManager:
    """医学检索管理器"""

    def __init__(self,
                 es_host: str = "elasticsearch",
                 es_port: int = 9200,
                 milvus_host: str = "milvus",
                 milvus_port: int = 19530,
                 embedding_manager=None):
        """初始化检索管理器"""
        self.es_base_url = f"http://{es_host}:{es_port}"
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.embedding_manager = embedding_manager
        self.es_index = "medical_documents"
        self.milvus_collection = "medical_vectors"
        self.milvus_connection_alias = "medical_retrieval"

        # 测试连接并建立持久连接
        self._test_connections()
        self._setup_milvus_connection()

    def _setup_milvus_connection(self):
        """建立持久的Milvus连接"""
        try:
            from pymilvus import connections
            # 断开之前的连接
            try:
                connections.disconnect(self.milvus_connection_alias)
            except:
                pass

            # 建立新连接
            connections.connect(
                alias=self.milvus_connection_alias,
                host=self.milvus_host,
                port=str(self.milvus_port),
                timeout=30
            )
            logger.info(f"✅ Milvus持久连接建立成功: {self.milvus_connection_alias}")
        except Exception as e:
            logger.error(f"❌ Milvus连接建立失败: {e}")

    def _test_connections(self):
        """测试数据库连接"""
        # 测试Elasticsearch
        try:
            response = requests.get(f"{self.es_base_url}/_cluster/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Elasticsearch连接成功")
            else:
                logger.warning(f"⚠️ Elasticsearch连接异常: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Elasticsearch连接失败: {e}")

        # 测试Milvus
        try:
            from pymilvus import connections, utility
            connections.connect(alias="retrieval_test", host=self.milvus_host, port=str(self.milvus_port))
            collections = utility.list_collections()
            logger.info(f"✅ Milvus连接成功，现有集合: {collections}")
        except Exception as e:
            logger.error(f"❌ Milvus连接失败: {e}")

    def keyword_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """关键词搜索 - 增强版支持标题优先级"""
        logger.info(f"🔍 执行关键词搜索: '{query}'")

        try:
            # 智能标题权重配置
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["chapter_title^10.0", "section_title^8.0"],  # 标题高权重
                                    "type": "best_fields",
                                    "boost": 2.0
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content^2.0"],  # 内容基础权重
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            },
                            {
                                "match": {
                                    "chapter_title": {
                                        "query": query,
                                        "boost": 15.0  # 完全匹配章节标题超高权重
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": top_k,
                "highlight": {
                    "fields": {
                        "content": {"fragment_size": 150, "number_of_fragments": 3},
                        "chapter_title": {},
                        "section_title": {}
                    }
                }
            }

            response = requests.post(
                f"{self.es_base_url}/{self.es_index}/_search",
                headers={"Content-Type": "application/json"},
                data=json.dumps(search_body),
                timeout=30
            )

            if response.status_code == 200:
                results = response.json()
                hits = results['hits']['hits']

                search_results = []
                for hit in hits:
                    source = hit['_source']

                    # 计算增强分数
                    base_score = hit['_score']

                    # 应用标题优先级算法
                    title_priority_score = self._calculate_title_priority_score(
                        query,
                        source.get('chapter_title', ''),
                        source.get('section_title', '')
                    )

                    # 内容类型分析加分
                    content_analysis = self._analyze_content_type(
                        source.get('content', ''),
                        source.get('chapter_title', '') + ' ' + source.get('section_title', '')
                    )

                    # 基础概念识别
                    concept_analysis = self._identify_basic_concepts(
                        query, source.get('content', '')
                    )

                    # 综合评分计算
                    enhanced_score = (
                        base_score * 0.6 +  # 基础搜索分数占60%
                        title_priority_score * 0.3 +  # 标题优先级占30%
                        (content_analysis['confidence'] * 5.0) * 0.05 +  # 内容类型占5%
                        (concept_analysis['relevance_score'] * 0.5) * 0.05  # 概念识别占5%
                    )

                    search_results.append(SearchResult(
                        content=source.get('content', ''),
                        score=enhanced_score,
                        source='elasticsearch',
                        page_number=source.get('page_number', 0),
                        chapter_title=source.get('chapter_title', ''),
                        section_title=source.get('section_title', ''),
                        doc_id=hit['_id'],
                        search_type='keyword_enhanced'
                    ))

                # 按增强分数重新排序
                search_results.sort(key=lambda x: x.score, reverse=True)

                logger.info(f"✅ 增强关键词搜索完成，找到 {len(search_results)} 个结果")
                return search_results
            else:
                logger.error(f"❌ 关键词搜索失败: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"❌ 关键词搜索失败: {e}")
            return []

    def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """语义搜索"""
        logger.info(f"🔍 执行语义搜索: '{query}'")

        try:
            # 生成查询向量
            if self.embedding_manager:
                query_vector = self.embedding_manager.encode_texts([query])[0]
            else:
                # 模拟向量
                np.random.seed(hash(query) % 1000)
                query_vector = np.random.random(768).tolist()

            # Milvus向量搜索 - 使用已建立的持久连接
            from pymilvus import Collection
            collection = Collection(self.milvus_collection, using=self.milvus_connection_alias)
            collection.load()

            search_params = {
                "metric_type": "COSINE",  # 修复：使用与集合索引一致的度量类型
                "params": {"nprobe": 16}
            }

            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "page_number"],
                using=self.milvus_connection_alias
            )

            # 获取文档详细信息
            search_results = []
            for result in results[0]:
                doc_id = result.id

                # 从Elasticsearch获取文档详细信息
                doc_response = requests.get(
                    f"{self.es_base_url}/{self.es_index}/_doc/{doc_id}",
                    timeout=10
                )

                if doc_response.status_code == 200:
                    doc_data = doc_response.json()['_source']
                    search_results.append(SearchResult(
                        content=doc_data.get('content', ''),
                        score=1.0 / (1.0 + result.distance),  # 转换距离为相似度分数
                        source='milvus',
                        page_number=getattr(result, 'page_number', 0),
                        chapter_title=doc_data.get('chapter_title', ''),
                        section_title=doc_data.get('section_title', ''),
                        doc_id=doc_id,
                        search_type='semantic'
                    ))

            collection.release()

            logger.info(f"✅ 语义搜索完成，找到 {len(search_results)} 个结果")
            return search_results

        except Exception as e:
            logger.error(f"❌ 语义搜索失败: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 10, keyword_weight: float = 0.5) -> List[SearchResult]:
        """混合搜索"""
        logger.info(f"🔍 执行混合搜索: '{query}' (权重: {keyword_weight})")

        try:
            # 执行关键词搜索
            keyword_results = self.keyword_search(query, top_k=top_k)

            # 执行语义搜索
            semantic_results = self.semantic_search(query, top_k=top_k)

            # 合并和重排序结果
            combined_results = self._merge_and_rerank(
                keyword_results,
                semantic_results,
                keyword_weight=keyword_weight
            )

            logger.info(f"✅ 混合搜索完成，返回 {len(combined_results)} 个结果")
            return combined_results[:top_k]

        except Exception as e:
            logger.error(f"❌ 混合搜索失败: {e}")
            return []

    def _merge_and_rerank(self, keyword_results: List[SearchResult],
                         semantic_results: List[SearchResult],
                         keyword_weight: float = 0.5) -> List[SearchResult]:
        """合并和重排序搜索结果"""

        # 创建文档ID到结果的映射
        doc_results = {}

        # 处理关键词搜索结果
        for result in keyword_results:
            doc_id = result.doc_id
            if doc_id not in doc_results:
                doc_results[doc_id] = result
            else:
                # 更新分数（加权平均）
                existing = doc_results[doc_id]
                existing.score = existing.score * keyword_weight + result.score * (1 - keyword_weight)
                existing.search_type = 'hybrid'

        # 处理语义搜索结果
        for result in semantic_results:
            doc_id = result.doc_id
            if doc_id not in doc_results:
                doc_results[doc_id] = result
            else:
                # 更新分数（加权平均）
                existing = doc_results[doc_id]
                existing.score = existing.score * keyword_weight + result.score * (1 - keyword_weight)
                existing.search_type = 'hybrid'

        # 按分数排序
        combined_results = list(doc_results.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)

        return combined_results

    def advanced_search(self, query: str, search_config: Dict[str, Any]) -> List[SearchResult]:
        """高级搜索"""
        logger.info(f"🔍 执行高级搜索: '{query}'")

        search_type = search_config.get('search_type', 'hybrid')
        top_k = search_config.get('top_k', 10)
        filters = search_config.get('filters', {})

        try:
            if search_type == 'keyword':
                results = self.keyword_search(query, top_k)
            elif search_type == 'semantic':
                results = self.semantic_search(query, top_k)
            elif search_type == 'hybrid':
                keyword_weight = search_config.get('keyword_weight', 0.5)
                results = self.hybrid_search(query, top_k, keyword_weight)
            else:
                logger.error(f"❌ 未知的搜索类型: {search_type}")
                return []

            # 应用过滤器
            if filters:
                results = self._apply_filters(results, filters)

            return results

        except Exception as e:
            logger.error(f"❌ 高级搜索失败: {e}")
            return []

    def _analyze_content_type(self, content: str, title: str = "") -> Dict[str, Any]:
        """分析内容类型和医学文档分类"""
        content_lower = content.lower()
        title_lower = title.lower()

        # ROSE相关关键词
        rose_keywords = ['rose', '快速现场评价', 'rapid on-site evaluation', '细胞学', '细胞诊断']
        # 诊断相关关键词
        diagnostic_keywords = ['诊断标准', '诊断要点', '鉴别诊断', '诊断依据', '确诊']
        # 治疗相关关键词
        treatment_keywords = ['治疗方案', '治疗方法', '治疗原则', '用药', '手术', '化疗', '放疗']
        # 病例相关关键词
        case_keywords = ['病例报告', '病例分析', '个案', '临床病例', '典型病例']

        content_type = 'general'
        confidence = 0.0

        # 标题优先分析
        if any(keyword in title_lower for keyword in rose_keywords):
            content_type = 'rose_technology'
            confidence = 0.9
        elif any(keyword in title_lower for keyword in diagnostic_keywords):
            content_type = 'diagnostic_criteria'
            confidence = 0.8
        elif any(keyword in title_lower for keyword in treatment_keywords):
            content_type = 'treatment_guidelines'
            confidence = 0.8
        elif any(keyword in title_lower for keyword in case_keywords):
            content_type = 'case_study'
            confidence = 0.7
        else:
            # 内容分析
            rose_score = sum(1 for keyword in rose_keywords if keyword in content_lower)
            diagnostic_score = sum(1 for keyword in diagnostic_keywords if keyword in content_lower)
            treatment_score = sum(1 for keyword in treatment_keywords if keyword in content_lower)
            case_score = sum(1 for keyword in case_keywords if keyword in content_lower)

            max_score = max(rose_score, diagnostic_score, treatment_score, case_score)

            if max_score > 0:
                if rose_score == max_score and rose_score >= 2:
                    content_type = 'rose_technology'
                    confidence = min(0.8, rose_score * 0.2)
                elif diagnostic_score == max_score and diagnostic_score >= 2:
                    content_type = 'diagnostic_criteria'
                    confidence = min(0.7, diagnostic_score * 0.15)
                elif treatment_score == max_score and treatment_score >= 2:
                    content_type = 'treatment_guidelines'
                    confidence = min(0.7, treatment_score * 0.15)
                elif case_score == max_score and case_score >= 2:
                    content_type = 'case_study'
                    confidence = min(0.6, case_score * 0.15)

        return {
            'type': content_type,
            'confidence': confidence,
            'is_rose_related': content_type == 'rose_technology'
        }

    def _identify_basic_concepts(self, query: str, content: str) -> Dict[str, Any]:
        """识别基础概念vs具体亚型"""
        query_lower = query.lower()
        content_lower = content.lower()

        # 基础概念关键词
        basic_concepts = [
            '肺部恶性肿瘤', '肺癌', '肿瘤', '癌症', '细胞学', '病理学',
            '诊断', '治疗', '症状', '预后', '分期', '分级'
        ]

        # ROSE基础概念
        rose_basic = [
            'rose技术', '快速现场评价', '细胞学诊断', '细胞采集',
            '制片技术', '染色方法', '显微镜检查'
        ]

        # 具体亚型和专业术语
        specific_subtypes = [
            '腺癌', '鳞癌', '小细胞癌', '大细胞癌', '神经内分泌癌',
            '细支气管肺泡癌', '肺类癌', '肺肉瘤样癌', '胸膜肺母细胞瘤'
        ]

        # ROSE专业术语
        rose_specific = [
            '细胞核增大', '核仁明显', '核分裂象', '细胞异型性',
            '核浆比失调', '染色质分布不均', '细胞排列紊乱',
            '印戒细胞', '砂粒体', '坏死碎片'
        ]

        query_is_basic = any(concept in query_lower for concept in basic_concepts + rose_basic)
        query_is_specific = any(term in query_lower for term in specific_subtypes + rose_specific)

        content_matches_basic = sum(1 for concept in basic_concepts + rose_basic if concept in content_lower)
        content_matches_specific = sum(1 for term in specific_subtypes + rose_specific if term in content_lower)

        if query_is_specific:
            # 查询包含专业术语，优先匹配具体亚型内容
            relevance_score = content_matches_specific * 2 + content_matches_basic * 0.5
            content_level = 'specific'
        elif query_is_basic:
            # 查询包含基础概念，平衡匹配基础和专业内容
            relevance_score = content_matches_basic * 1.5 + content_matches_specific * 1.0
            content_level = 'mixed'
        else:
            # 一般查询，按内容匹配度评分
            relevance_score = content_matches_basic + content_matches_specific
            content_level = 'general'

        return {
            'content_level': content_level,
            'relevance_score': relevance_score,
            'matches_basic': content_matches_basic,
            'matches_specific': content_matches_specific,
            'query_type': 'specific' if query_is_specific else ('basic' if query_is_basic else 'general')
        }

    def _calculate_title_priority_score(self, query: str, chapter_title: str, section_title: str) -> float:
        """计算标题优先级分数 - 整合三个脚本的最优权重"""
        query_lower = query.lower()
        chapter_lower = chapter_title.lower()
        section_lower = section_title.lower()

        # 标题权重配置（采用最高效的权重组合）
        chapter_weight = 10.0  # 章标题最高权重
        section_weight = 8.0   # 节标题高权重
        subsection_weight = 5.0 # 小节标题中等权重
        paragraph_weight = 3.0  # 段落标题基础权重

        title_score = 0.0

        # 章节标题匹配（最高优先级）
        if chapter_lower and query_lower in chapter_lower:
            title_score += chapter_weight
        elif chapter_lower and any(word in chapter_lower for word in query_lower.split()):
            word_matches = sum(1 for word in query_lower.split() if word in chapter_lower)
            title_score += chapter_weight * (word_matches / len(query_lower.split()))

        # 节标题匹配（高优先级）
        if section_lower and query_lower in section_lower:
            title_score += section_weight
        elif section_lower and any(word in section_lower for word in query_lower.split()):
            word_matches = sum(1 for word in query_lower.split() if word in section_lower)
            title_score += section_weight * (word_matches / len(query_lower.split()))

        # 医学关键词在标题中的特殊处理
        medical_keywords = [
            'rose', '肺部恶性肿瘤', '细胞学', '病理', '诊断', '治疗',
            '腺癌', '鳞癌', '细胞核', '核仁', '异型性'
        ]

        for keyword in medical_keywords:
            if keyword in chapter_lower or keyword in section_lower:
                title_score += 2.0  # 医学关键词额外加权

        return min(title_score, 25.0)  # 最高25分封顶

    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """应用过滤器"""
        filtered_results = []

        for result in results:
            include_result = True

            # 页码过滤
            if 'page_range' in filters:
                page_range = filters['page_range']
                if not (page_range[0] <= result.page_number <= page_range[1]):
                    include_result = False

            # 章节过滤
            if 'chapter_keywords' in filters:
                chapter_keywords = filters['chapter_keywords']
                if not any(keyword in result.chapter_title for keyword in chapter_keywords):
                    include_result = False

            # 分数过滤
            if 'min_score' in filters:
                if result.score < filters['min_score']:
                    include_result = False

            if include_result:
                filtered_results.append(result)

        return filtered_results

    def intelligent_search(self, query: str, search_config: Dict[str, Any] = None) -> List[SearchResult]:
        """智能搜索 - 整合所有优化算法"""
        if search_config is None:
            search_config = {}

        logger.info(f"🧠 执行智能搜索: '{query}'")

        try:
            top_k = search_config.get('top_k', 10)
            enable_title_priority = search_config.get('enable_title_priority', True)
            enable_content_analysis = search_config.get('enable_content_analysis', True)
            enable_concept_identification = search_config.get('enable_concept_identification', True)
            keyword_weight = search_config.get('keyword_weight', 0.6)

            # 执行基础混合搜索
            base_results = self.hybrid_search(query, top_k * 2, keyword_weight)

            if not base_results:
                return []

            # 应用智能优化算法
            enhanced_results = []
            for result in base_results:
                enhanced_result = result
                total_score = result.score

                # 1. 标题优先级评分
                if enable_title_priority:
                    title_score = self._calculate_title_priority_score(
                        query, result.chapter_title, result.section_title
                    )
                    total_score += title_score * 0.3  # 标题权重占30%

                # 2. 内容类型分析
                if enable_content_analysis:
                    content_analysis = self._analyze_content_type(
                        result.content, result.chapter_title + " " + result.section_title
                    )
                    if content_analysis['is_rose_related']:
                        total_score += 5.0  # ROSE相关内容加分
                    total_score += content_analysis['confidence'] * 3.0  # 类型置信度加分

                # 3. 基础概念识别
                if enable_concept_identification:
                    concept_analysis = self._identify_basic_concepts(query, result.content)

                    # 根据查询类型调整分数
                    if concept_analysis['query_type'] == 'specific':
                        total_score += concept_analysis['relevance_score'] * 2.0
                    elif concept_analysis['query_type'] == 'basic':
                        total_score += concept_analysis['relevance_score'] * 1.5
                    else:
                        total_score += concept_analysis['relevance_score'] * 1.0

                enhanced_result.score = total_score
                enhanced_results.append(enhanced_result)

            # 重新排序
            enhanced_results.sort(key=lambda x: x.score, reverse=True)

            # 应用过滤器
            if search_config.get('filters'):
                enhanced_results = self._apply_filters(enhanced_results, search_config['filters'])

            logger.info(f"✅ 智能搜索完成，返回 {len(enhanced_results[:top_k])} 个优化结果")
            return enhanced_results[:top_k]

        except Exception as e:
            logger.error(f"❌ 智能搜索失败: {e}")
            # 回退到普通混合搜索
            return self.hybrid_search(query, search_config.get('top_k', 10), search_config.get('keyword_weight', 0.6))

    def search(self, query: str, search_type: str = "hybrid", **kwargs) -> List[Dict[str, Any]]:
        """统一搜索接口 - 增强版支持智能搜索"""
        logger.info(f"🔍 统一搜索接口: '{query}' (类型: {search_type})")

        try:
            if search_type == "keyword":
                results = self.keyword_search(query, kwargs.get('top_k', 10))
            elif search_type == "semantic":
                results = self.semantic_search(query, kwargs.get('top_k', 10))
            elif search_type == "hybrid":
                top_k = kwargs.get('top_k', 10)
                keyword_weight = kwargs.get('keyword_weight', 0.5)
                results = self.hybrid_search(query, top_k, keyword_weight)
            elif search_type == "intelligent":
                # 智能搜索模式
                search_config = kwargs.get('search_config', {})
                results = self.intelligent_search(query, search_config)
            else:
                logger.error(f"❌ 未知的搜索类型: {search_type}")
                return []

            # 转换为字典格式
            dict_results = []
            for result in results:
                dict_results.append({
                    'content': result.content,
                    'score': result.score,
                    'source': result.source,
                    'page_number': result.page_number,
                    'chapter_title': result.chapter_title,
                    'section_title': result.section_title,
                    'doc_id': result.doc_id,
                    'search_type': result.search_type
                })

            return dict_results

        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            # Elasticsearch统计
            es_response = requests.get(f"{self.es_base_url}/{self.es_index}/_stats", timeout=10)
            es_stats = es_response.json() if es_response.status_code == 200 else {}

            # Milvus统计 - 使用持久连接
            from pymilvus import Collection
            collection = Collection(self.milvus_collection, using=self.milvus_connection_alias)
            collection.load()
            milvus_count = collection.num_entities
            collection.release()

            return {
                "elasticsearch": {
                    "document_count": es_stats.get('indices', {}).get(self.es_index, {}).get('primaries', {}).get('docs', {}).get('count', 0),
                    "size_mb": es_stats.get('indices', {}).get(self.es_index, {}).get('primaries', {}).get('store', {}).get('size_in_bytes', 0) / 1024 / 1024
                },
                "milvus": {
                    "vector_count": milvus_count,
                    "collection_name": self.milvus_collection
                },
                "status": "healthy"
            }

        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return {"status": "error", "error": str(e)}

def create_retrieval_manager(es_host: str = "elasticsearch",
                           es_port: int = 9200,
                           milvus_host: str = "milvus",
                           milvus_port: int = 19530,
                           embedding_manager=None) -> MedicalRetrievalManager:
    """创建检索管理器"""
    return MedicalRetrievalManager(
        es_host=es_host,
        es_port=es_port,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        embedding_manager=embedding_manager
    )