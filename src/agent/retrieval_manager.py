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
                 es_host: str = "localhost",
                 es_port: int = 9200,
                 milvus_host: str = "localhost",
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
        """关键词搜索"""
        logger.info(f"🔍 执行关键词搜索: '{query}'")

        try:
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "chapter_title^1.5", "section_title^1.2"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "size": top_k,
                "highlight": {
                    "fields": {
                        "content": {"fragment_size": 150, "number_of_fragments": 3}
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
                    search_results.append(SearchResult(
                        content=source.get('content', ''),
                        score=hit['_score'],
                        source='elasticsearch',
                        page_number=source.get('page_number', 0),
                        chapter_title=source.get('chapter_title', ''),
                        section_title=source.get('section_title', ''),
                        doc_id=hit['_id'],
                        search_type='keyword'
                    ))

                logger.info(f"✅ 关键词搜索完成，找到 {len(search_results)} 个结果")
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

    def search(self, query: str, search_type: str = "hybrid", **kwargs) -> List[Dict[str, Any]]:
        """统一搜索接口"""
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

def create_retrieval_manager(es_host: str = "localhost",
                           es_port: int = 9200,
                           milvus_host: str = "localhost",
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