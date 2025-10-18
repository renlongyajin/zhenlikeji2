#!/usr/bin/env python3
"""
数据库检索和测试脚本
测试Elasticsearch和Milvus的检索功能
"""

import json
import requests
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseTester:
    """数据库测试器"""

    def __init__(self,
                 es_host: str = "localhost",
                 es_port: int = 9200,
                 milvus_host: str = "localhost",
                 milvus_port: int = 19530):
        """初始化测试器"""
        self.es_base_url = f"http://{es_host}:{es_port}"
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.es_index = "medical_documents"
        self.milvus_collection = "medical_vectors"

    def test_elasticsearch_connection(self):
        """测试Elasticsearch连接"""
        logger.info("🧪 测试Elasticsearch连接...")

        try:
            response = requests.get(f"{self.es_base_url}/_cluster/health", timeout=10)
            if response.status_code == 200:
                health = response.json()
                logger.info(f"✅ Elasticsearch健康状态: {health['status']}")
                logger.info(f"📊 节点数量: {health['number_of_nodes']}")
                logger.info(f"📊 数据节点数量: {health['number_of_data_nodes']}")
                return True
            else:
                logger.error(f"❌ Elasticsearch连接失败: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Elasticsearch连接失败: {e}")
            return False

    def test_elasticsearch_index_stats(self):
        """测试Elasticsearch索引统计"""
        logger.info(f"📈 检查Elasticsearch索引统计...")

        try:
            response = requests.get(f"{self.es_base_url}/{self.es_index}/_stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()
                doc_count = stats['indices'][self.es_index]['primaries']['docs']['count']
                size = stats['indices'][self.es_index]['primaries']['store']['size_in_bytes']

                logger.info(f"📊 索引文档数量: {doc_count}")
                logger.info(f"📊 索引大小: {size / 1024 / 1024:.2f} MB")
                return True
            else:
                logger.error(f"❌ 获取索引统计失败: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ 获取索引统计失败: {e}")
            return False

    def test_elasticsearch_search(self, query: str, size: int = 10):
        """测试Elasticsearch搜索功能"""
        logger.info(f"🔍 测试Elasticsearch搜索: '{query}'")

        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "chapter_title", "section_title"]
                }
            },
            "size": size,
            "highlight": {
                "fields": {
                    "content": {}
                }
            }
        }

        try:
            response = requests.post(
                f"{self.es_base_url}/{self.es_index}/_search",
                headers={"Content-Type": "application/json"},
                data=json.dumps(search_body),
                timeout=30
            )

            if response.status_code == 200:
                results = response.json()
                hits = results['hits']['hits']

                logger.info(f"✅ 搜索完成，找到 {len(hits)} 个结果")

                for i, hit in enumerate(hits[:3]):  # 只显示前3个结果
                    score = hit['_score']
                    source = hit['_source']
                    content = source.get('content', '')[:200] + "..." if len(source.get('content', '')) > 200 else source.get('content', '')

                    logger.info(f"\n📄 结果 {i+1} (分数: {score:.3f}):")
                    logger.info(f"   内容: {content}")
                    logger.info(f"   页面: {source.get('page_number', '未知')}")
                    logger.info(f"   章节: {source.get('chapter_title', '未知')}")
                    logger.info(f"   小节: {source.get('section_title', '未知')}")

                return hits
            else:
                logger.error(f"❌ 搜索失败: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return []

    def test_milvus_connection(self):
        """测试Milvus连接"""
        logger.info("🧪 测试Milvus连接...")

        try:
            from pymilvus import connections, utility
            connections.connect(alias="default", host=self.milvus_host, port=str(self.milvus_port))

            collections = utility.list_collections()
            logger.info(f"✅ Milvus连接成功")
            logger.info(f"📋 现有集合: {collections}")

            return True
        except Exception as e:
            logger.error(f"❌ Milvus连接失败: {e}")
            return False

    def test_milvus_collection_stats(self):
        """测试Milvus集合统计"""
        logger.info(f"📈 检查Milvus集合统计...")

        try:
            from pymilvus import connections, Collection
            connections.connect(alias="default", host=self.milvus_host, port=str(self.milvus_port))

            collection = Collection(self.milvus_collection)
            collection.load()

            count = collection.num_entities
            logger.info(f"📊 集合向量数量: {count}")

            return count
        except Exception as e:
            logger.error(f"❌ 获取集合统计失败: {e}")
            return 0

    def test_milvus_vector_search(self, query_vector: List[float], top_k: int = 10):
        """测试Milvus向量搜索"""
        logger.info(f"🔍 测试Milvus向量搜索，返回前 {top_k} 个结果")

        try:
            from pymilvus import connections, Collection
            connections.connect(alias="default", host=self.milvus_host, port=str(self.milvus_port))

            collection = Collection(self.milvus_collection)
            collection.load()

            # 构建搜索参数 - 使用与索引一致的度量类型
            search_params = {
                "metric_type": "COSINE",  # 修复：与集合索引保持一致
                "params": {"nprobe": 16}
            }

            # 执行搜索
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "page_number"]
            )

            logger.info(f"✅ 向量搜索完成，找到 {len(results[0])} 个结果")

            for i, result in enumerate(results[0]):
                logger.info(f"\n🎯 结果 {i+1} (距离: {result.distance:.4f}):")
                logger.info(f"   ID: {result.id}")
                # 修复pymilvus 2.6+ API兼容性
                try:
                    page_number = result.entity.get('page_number') if hasattr(result, 'entity') else '未知'
                except:
                    page_number = '未知'
                logger.info(f"   页面: {page_number}")

            return results

        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return None

    def test_hybrid_retrieval(self, query: str, top_k: int = 10):
        """测试混合检索"""
        logger.info(f"🔍 测试混合检索: '{query}'")

        # 1. Elasticsearch关键词搜索
        es_results = self.test_elasticsearch_search(query, top_k)

        # 2. Milvus向量搜索（使用模拟向量）
        query_vector = self.generate_query_embedding(query)
        milvus_results = self.test_milvus_vector_search(query_vector, top_k)

        logger.info("\n📊 混合检索结果汇总:")
        logger.info(f"   Elasticsearch结果数量: {len(es_results)}")
        if milvus_results:
            logger.info(f"   Milvus结果数量: {len(milvus_results[0])}")

        return es_results, milvus_results

    def intelligent_hybrid_retrieval(self, query: str, top_k: int = 10):
        """智能混合检索 - 最佳实践版"""
        logger.info(f"🧠 智能混合检索: '{query}'")

        # 1. 并行搜索（获取更多结果用于融合）
        es_results = self.test_elasticsearch_search(query, top_k * 2)
        query_vector = self.generate_query_embedding(query)
        milvus_results = self.test_milvus_vector_search(query_vector, top_k * 2)

        # 2. 智能融合
        if not es_results and not milvus_results:
            logger.warning("⚠️  两种搜索均无结果")
            return []
        elif not es_results:
            logger.info("ℹ️  仅使用Milvus向量搜索结果")
            return self._extract_milvus_docs(milvus_results)[:top_k]
        elif not milvus_results or not milvus_results[0]:
            logger.info("ℹ️  仅使用Elasticsearch搜索结果")
            return self._extract_es_docs(es_results)[:top_k]

        # 3. 根据查询特征选择融合策略
        if len(query) <= 3:  # 短查询
            logger.info("📏 短查询，使用RRF融合")
            fused_results = self.reciprocal_rank_fusion(es_results, milvus_results)
        elif self.has_precise_medical_terms(query):  # 精确医学术语
            logger.info("🏥 检测到精确医学术语，强化ES结果")
            fused_results = self.fuse_search_results(es_results, milvus_results, 0.75, 0.25)
        else:  # 一般查询
            logger.info("🔍 一般查询，平衡融合")
            fused_results = self.fuse_search_results(es_results, milvus_results, 0.6, 0.4)

        # 4. 返回最佳结果
        logger.info(f"✅ 融合完成，返回 {min(len(fused_results), top_k)} 个结果")
        return fused_results[:top_k]

    def reciprocal_rank_fusion(self, es_results, milvus_results, k=60):
        """RRF (Reciprocal Rank Fusion) 算法 - 无需调参的融合方法"""
        logger.info(f"🔄 使用RRF算法融合结果 (k={k})")

        rrf_scores = {}

        # 处理ES结果
        for rank, hit in enumerate(es_results):
            doc_id = hit['_id']
            # RRF公式：1 / (k + rank)
            score = 1.0 / (k + rank + 1)  # +1避免除零

            rrf_scores[doc_id] = {
                'id': doc_id,
                'score': score,
                'content': hit['_source'].get('content', ''),
                'page_number': hit['_source'].get('page_number', '未知'),
                'es_rank': rank,
                'milvus_rank': -1,  # 初始为-1表示未找到
                'source': 'elasticsearch'
            }

        # 处理Milvus结果
        if milvus_results and milvus_results[0]:
            for rank, hit in enumerate(milvus_results[0]):
                doc_id = hit.id
                score = 1.0 / (k + rank + 1)

                if doc_id in rrf_scores:
                    # 已存在，累加分数
                    rrf_scores[doc_id]['score'] += score
                    rrf_scores[doc_id]['milvus_rank'] = rank
                    rrf_scores[doc_id]['source'] = 'both'
                else:
                    # 新文档
                    rrf_scores[doc_id] = {
                        'id': doc_id,
                        'score': score,
                        'content': f'[向量搜索结果] {doc_id}',
                        'page_number': getattr(hit, 'page_number', '未知'),
                        'es_rank': -1,
                        'milvus_rank': rank,
                        'source': 'milvus'
                    }

        # 按RRF分数排序
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        logger.info(f"✅ RRF融合完成，共 {len(sorted_results)} 个结果")
        return sorted_results

    def fuse_search_results(self, es_results, milvus_results, es_weight=0.6, milvus_weight=0.4):
        """加权融合搜索结果"""
        logger.info(f"⚖️ 加权融合 (ES: {es_weight}, Milvus: {milvus_weight})")

        # 分数归一化
        def normalize_scores(results, is_es=True):
            if not results:
                return []

            if is_es:
                # ES结果：使用_score字段
                max_score = max([r['_score'] for r in results]) if results else 1
                for result in results:
                    result['normalized_score'] = result['_score'] / max_score if max_score > 0 else 0
            else:
                # Milvus结果：距离转相似度
                if results and results[0]:
                    distances = [r.distance for r in results[0]]
                    max_dist = max(distances) if distances else 1
                    min_dist = min(distances) if distances else 0
                    dist_range = max_dist - min_dist

                    for result in results[0]:
                        if dist_range > 0:
                            result.normalized_score = 1 - (result.distance - min_dist) / dist_range
                        else:
                            result.normalized_score = 1.0

            return results

        # 归一化分数
        normalized_es = normalize_scores(es_results, is_es=True)
        normalized_milvus = normalize_scores(milvus_results, is_es=False)

        # 创建统一的文档映射
        fused_results = {}

        # 处理ES结果
        for result in normalized_es:
            doc_id = result['_id']
            es_norm_score = result.get('normalized_score', 0)

            fused_results[doc_id] = {
                'id': doc_id,
                'content': result['_source'].get('content', ''),
                'page_number': result['_source'].get('page_number', '未知'),
                'es_score': es_norm_score,
                'milvus_score': 0,
                'fused_score': es_weight * es_norm_score,  # 初始融合分数
                'source': 'elasticsearch'
            }

        # 处理Milvus结果
        if normalized_milvus and normalized_milvus[0]:
            for result in normalized_milvus[0]:
                doc_id = result.id
                milvus_norm_score = getattr(result, 'normalized_score', 0)

                if doc_id in fused_results:
                    # 已存在，更新Milvus分数和融合分数
                    fused_results[doc_id]['milvus_score'] = milvus_norm_score
                    fused_results[doc_id]['fused_score'] = (
                        es_weight * fused_results[doc_id]['es_score'] +
                        milvus_weight * milvus_norm_score
                    )
                    fused_results[doc_id]['source'] = 'both'
                else:
                    # 新文档
                    fused_results[doc_id] = {
                        'id': doc_id,
                        'content': f'[向量搜索结果] {doc_id}',
                        'page_number': getattr(result, 'page_number', '未知'),
                        'es_score': 0,
                        'milvus_score': milvus_norm_score,
                        'fused_score': milvus_weight * milvus_norm_score,
                        'source': 'milvus'
                    }

        # 按融合分数排序
        sorted_results = sorted(
            fused_results.values(),
            key=lambda x: x['fused_score'],
            reverse=True
        )

        logger.info(f"✅ 加权融合完成，共 {len(sorted_results)} 个结果")
        return sorted_results

    def has_precise_medical_terms(self, query: str) -> bool:
        """检测精确医学术语"""
        # 医学术语词典
        precise_medical_terms = [
            "ROSE", "N/C", "核质比", "m/N", "核仁长径",
            "腺泡状", "乳头状", "桑甚状", "异倍体", "多倍体",
            "肉瘤样癌", "腺泡细胞癌", "小细胞癌", "透明细胞癌"
        ]

        # 检查是否包含精确医学术语
        for term in precise_medical_terms:
            if term in query.upper() or term in query:
                return True

        return False

    def _extract_es_docs(self, es_results):
        """提取ES文档信息"""
        extracted = []
        for hit in es_results:
            extracted.append({
                'id': hit['_id'],
                'content': hit['_source'].get('content', ''),
                'page_number': hit['_source'].get('page_number', '未知'),
                'score': hit['_score'],
                'source': 'elasticsearch'
            })
        return extracted

    def _extract_milvus_docs(self, milvus_results):
        """提取Milvus文档信息"""
        extracted = []
        if milvus_results and milvus_results[0]:
            for result in milvus_results[0]:
                extracted.append({
                    'id': result.id,
                    'content': f'[向量搜索结果] {result.id}',
                    'page_number': getattr(result, 'page_number', '未知'),
                    'distance': result.distance,
                    'source': 'milvus'
                })
        return extracted

    def generate_query_embedding(self, query: str) -> List[float]:
        """生成查询向量（模拟）"""
        np.random.seed(hash(query) % 1000)
        return np.random.random(768).tolist()

    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始运行所有数据库测试...")
        logger.info("=" * 60)

        # 测试Elasticsearch
        logger.info("\n📋 Elasticsearch测试")
        logger.info("-" * 30)
        self.test_elasticsearch_connection()
        self.test_elasticsearch_index_stats()

        # 测试Milvus
        logger.info("\n📋 Milvus测试")
        logger.info("-" * 30)
        self.test_milvus_connection()
        self.test_milvus_collection_stats()

        # 传统混合检索测试
        logger.info("\n🔍 传统混合检索测试")
        logger.info("-" * 30)

        # 医学相关的测试查询
        test_queries = [
            "肺部恶性肿瘤",
            "ROSE细胞学",
            "腺癌特征",
            "细胞核增大",
            "快速现场评价"
        ]

        for query in test_queries:
            logger.info(f"\n🧪 传统混合检索测试: '{query}'")
            self.test_hybrid_retrieval(query, top_k=3)
            logger.info("-" * 50)

        # 智能混合检索测试
        logger.info("\n🧠 智能混合检索测试")
        logger.info("-" * 30)

        # 不同类型的测试查询
        intelligent_test_queries = [
            "肺部恶性肿瘤",           # 一般医学查询
            "ROSE",                  # 精确医学术语
            "腺癌",                  # 短查询
            "细胞核增大特征",        # 详细描述
            "N/C比值"                # 精确术语
        ]

        for query in intelligent_test_queries:
            logger.info(f"\n🧪 智能混合检索测试: '{query}'")
            intelligent_results = self.intelligent_hybrid_retrieval(query, top_k=3)

            logger.info(f"📊 返回 {len(intelligent_results)} 个融合结果:")
            for i, result in enumerate(intelligent_results):
                logger.info(f"\n  🎯 结果 {i+1} (融合分数: {result.get('fused_score', result.get('score', 0)):.4f}):")
                logger.info(f"     ID: {result['id']}")
                logger.info(f"     页面: {result['page_number']}")
                logger.info(f"     来源: {result['source']}")
                content = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                logger.info(f"     内容: {content}")
            logger.info("-" * 50)

        logger.info("\n🎉 所有测试完成！")
        logger.info("=" * 60)

def main():
    """主函数"""
    tester = DatabaseTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()