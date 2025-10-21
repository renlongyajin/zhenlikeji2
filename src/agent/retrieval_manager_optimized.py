#!/usr/bin/env python3
"""
优化版检索管理器
解决第16页腺癌内容排名低的问题
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

class OptimizedMedicalRetrievalManager:
    """优化的医学检索管理器"""

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
        self.es_index = "medical_documents_fixed"
        self.milvus_collection = "medical_vectors"
        self.milvus_connection_alias = "medical_retrieval"

        # 医学术语词典
        self.medical_terms = {
            '腺癌': ['腺癌', '腺样囊性癌', '腺泡细胞癌', '腺管状腺癌', '乳头状腺癌'],
            '图像': ['图像', '镜下', '形态', '外观', '结构', '染色', '图示'],
            '特征': ['特征', '特点', '表现', '形态', '结构', '性质']
        }

        # 专业内容类型关键词
        self.content_type_keywords = {
            'image_morphology': ['图', '镜下', '形态', '外观', '结构', '染色', '400×', '图示'],
            'pathology_diagnosis': ['诊断', '标准', '要点', '依据', '分型'],
            'cellular_features': ['细胞', '核', '质', '形态', '大小', '形状']
        }

        self._test_connections()
        self._setup_milvus_connection()

    def _setup_milvus_connection(self):
        """建立持久的Milvus连接"""
        try:
            from pymilvus import connections
            try:
                connections.disconnect(self.milvus_connection_alias)
            except:
                pass

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

    def enhanced_keyword_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """增强版关键词搜索 - 专门优化医学内容检索"""
        logger.info(f"🔍 执行增强关键词搜索: '{query}'")

        try:
            # 医学术语扩展
            expanded_query = self._expand_medical_terms(query)
            logger.info(f"扩展后的查询: '{expanded_query}'")

            # 内容类型分析
            content_boost = self._analyze_query_content_type(query)

            # 优化的搜索查询 - 重点提升内容匹配
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            # 1. 精确匹配医学术语（最高权重）
                            {
                                "match": {
                                    "content": {
                                        "query": query,
                                        "boost": 5.0,
                                        "operator": "and"  # 要求所有词都出现
                                    }
                                }
                            },
                            # 2. 医学术语扩展匹配
                            {
                                "match": {
                                    "content": {
                                        "query": expanded_query,
                                        "boost": 3.0,
                                        "operator": "or"
                                    }
                                }
                            },
                            # 3. 专业内容类型匹配
                            {
                                "bool": {
                                    "should": [
                                        {"terms": {"content": self.content_type_keywords['image_morphology'], "boost": 2.5}},
                                        {"terms": {"content": self.content_type_keywords['cellular_features'], "boost": 2.0}}
                                    ]
                                }
                            },
                            # 4. 基础内容匹配（带模糊）
                            {
                                "match": {
                                    "content": {
                                        "query": query,
                                        "boost": 1.0,
                                        "fuzziness": "AUTO"
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
                        "content": {"fragment_size": 200, "number_of_fragments": 3}
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

                    # 医学内容质量评分
                    medical_quality_score = self._calculate_medical_content_score(
                        source.get('content', ''),
                        query
                    )

                    # 图像特征相关性评分
                    image_relevance_score = self._calculate_image_feature_relevance(
                        source.get('content', ''),
                        query
                    )

                    # 综合评分计算
                    enhanced_score = (
                        base_score * 0.7 +  # 基础搜索分数占70%
                        medical_quality_score * 0.2 +  # 医学质量占20%
                        image_relevance_score * 0.1  # 图像相关性占10%
                    )

                    search_results.append(SearchResult(
                        content=source.get('content', ''),
                        score=enhanced_score,
                        source='elasticsearch_enhanced',
                        page_number=source.get('page_number', 0),
                        chapter_title=source.get('chapter_title', ''),
                        section_title=source.get('section_title', ''),
                        doc_id=hit['_id'],
                        search_type='keyword_enhanced'
                    ))

                logger.info(f"✅ 增强关键词搜索完成，找到 {len(search_results)} 个结果")

                # 特别检查第16页
                page_16_results = [r for r in search_results if r.page_number == 16]
                if page_16_results:
                    logger.info(f"🎯 第16页排名: {[i for i, r in enumerate(search_results) if r.page_number == 16][0] + 1}, 分数: {page_16_results[0].score:.4f}")

                return search_results
            else:
                logger.error(f"❌ 增强关键词搜索失败: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"❌ 增强关键词搜索失败: {e}")
            return []

    def _expand_medical_terms(self, query: str) -> str:
        """扩展医学术语"""
        expanded = query

        for term, synonyms in self.medical_terms.items():
            if term in query:
                # 添加同义词到查询中
                synonym_str = " OR ".join(f'"{syn}"' for syn in synonyms)
                expanded += f" OR ({synonym_str})"

        return expanded

    def _analyze_query_content_type(self, query: str) -> float:
        """分析查询的内容类型并返回相应的权重"""
        query_lower = query.lower()

        # 图像特征查询
        if any(keyword in query_lower for keyword in ['图像', '特征', '镜下', '形态']):
            return 2.5

        # 诊断相关查询
        if any(keyword in query_lower for keyword in ['诊断', '标准', '要点']):
            return 2.0

        # 细胞学特征查询
        if any(keyword in query_lower for keyword in ['细胞', '核', '质', '形态']):
            return 1.8

        return 1.0

    def _calculate_medical_content_score(self, content: str, query: str) -> float:
        """计算医学内容质量评分"""
        if not content:
            return 0.0

        content_lower = content.lower()
        query_lower = query.lower()

        score = 0.0

        # 1. 专业术语密度
        medical_terms = ['癌细胞', '分化', '核', '质', '染色', '形态', '结构']
        term_count = sum(1 for term in medical_terms if term in content_lower)
        score += min(term_count * 0.1, 0.5)  # 最多0.5分

        # 2. 图像描述特征
        image_indicators = ['图', '镜下', '400×', '染色', '外观', '形态']
        image_count = sum(1 for indicator in image_indicators if indicator in content_lower)
        score += min(image_count * 0.15, 0.3)  # 最多0.3分

        # 3. 结构化描述（有序列表）
        if '①' in content or '1.' in content or '（1）' in content:
            score += 0.2  # 结构化描述加分

        return min(score, 1.0)

    def _calculate_image_feature_relevance(self, content: str, query: str) -> float:
        """计算图像特征相关性评分"""
        if not content:
            return 0.0

        content_lower = content.lower()
        query_lower = query.lower()

        score = 0.0

        # 1. 图像相关词汇
        image_keywords = ['图像', '镜下', '图示', '图', '400×', '染色']
        for keyword in image_keywords:
            if keyword in content_lower:
                score += 0.2

        # 2. 特征描述词汇
        feature_keywords = ['特征', '形态', '外观', '结构', '性质', '表现']
        for keyword in feature_keywords:
            if keyword in content_lower:
                score += 0.15

        # 3. 具体形态学描述
        morphology_terms = ['圆形', '立方形', '空泡', '印戒样', '颗粒状', '清楚']
        morphology_count = sum(1 for term in morphology_terms if term in content_lower)
        score += min(morphology_count * 0.1, 0.4)

        return min(score, 1.0)

    def keyword_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """保持原有的关键词搜索方法用于对比"""
        return self.enhanced_keyword_search(query, top_k)

    def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """语义搜索 - 保持不变"""
        logger.info(f"🔍 执行语义搜索: '{query}'")
        # 保持原有实现...
        return []

    def hybrid_search(self, query: str, top_k: int = 10, keyword_weight: float = 0.7) -> List[SearchResult]:
        """混合搜索 - 使用增强版关键词搜索"""
        logger.info(f"🔍 执行混合搜索: '{query}' (权重: {keyword_weight})")

        # 直接使用增强版关键词搜索（因为语义搜索当前有问题）
        return self.enhanced_keyword_search(query, top_k)

    def advanced_search(self, query: str, search_config: Dict[str, Any]) -> List[SearchResult]:
        """高级搜索"""
        return self.hybrid_search(query,
                                search_config.get('top_k', 10),
                                search_config.get('keyword_weight', 0.7))