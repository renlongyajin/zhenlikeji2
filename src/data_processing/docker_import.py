#!/usr/bin/env python3
"""
Docker环境数据导入脚本
在Docker容器中运行，连接到Docker服务
"""

import json
import os
import sys
import time
import requests
from typing import List, Dict, Any
from pathlib import Path
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.insert(0, '/app')

# 导入嵌入模型
try:
    from src.embedding.embedding_models import JinaEmbeddingModel
except ImportError:
    logger.error("无法导入嵌入模型，请确保JinaEmbeddingModel可用")
    sys.exit(1)

class DockerImporter:
    """Docker环境数据导入器"""

    def __init__(self):
        """初始化导入器 - 使用Docker服务名"""
        # Docker环境中的服务名
        self.es_host = os.environ.get('ELASTICSEARCH_HOST', 'elasticsearch')
        self.es_port = int(os.environ.get('ELASTICSEARCH_PORT', '9200'))
        self.es_base_url = f"http://{self.es_host}:{self.es_port}"
        self.es_index = "medical_documents_fixed"

        self.milvus_host = os.environ.get('MILVUS_HOST', 'milvus')
        self.milvus_port = int(os.environ.get('MILVUS_PORT', '19530'))
        self.milvus_collection = "medical_vectors_simple_import"
        self.milvus_connection_alias = "docker_import"

        # 初始化嵌入模型
        try:
            self.embedding_model = JinaEmbeddingModel()
            self.embedding_dimension = 1024
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            # 使用备用嵌入维度
            self.embedding_dimension = 768
            logger.warning("使用备用嵌入维度: 768")

        logger.info(f"初始化Docker导入器:")
        logger.info(f"  Elasticsearch: {self.es_host}:{self.es_port}")
        logger.info(f"  Milvus: {self.milvus_host}:{self.milvus_port}")
        logger.info(f"  嵌入维度: {self.embedding_dimension}")

    def test_connections(self) -> bool:
        """测试数据库连接"""
        logger.info("测试数据库连接...")

        # 测试Elasticsearch
        try:
            response = requests.get(f"{self.es_base_url}/_cluster/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Elasticsearch连接成功")
            else:
                logger.warning(f"⚠️ Elasticsearch状态异常: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Elasticsearch连接失败: {e}")
            return False

        # 测试Milvus
        try:
            from pymilvus import connections, utility

            # 使用简单的连接测试方式
            connections.connect(
                alias="docker_import_test",
                host=self.milvus_host,
                port=str(self.milvus_port),
                timeout=30
            )

            # 测试连接 - 使用指定的连接别名
            collections = utility.list_collections(using="docker_import_test")
            logger.info(f"✅ Milvus连接成功，现有集合: {collections}")

            return True

        except Exception as e:
            logger.error(f"❌ Milvus连接失败: {e}")
            return False

    def create_indices_and_collections(self) -> bool:
        """创建索引和集合"""
        logger.info("创建索引和集合...")

        # 创建Elasticsearch索引
        try:
            index_mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "ik_max_word"},
                        "chapter_title": {"type": "keyword"},
                        "section_title": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "chunk_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "content_length": {"type": "integer"},
                        "sub_chunk_index": {"type": "integer"},
                        "timestamp": {"type": "date"}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "ik_max_word": {
                                "type": "ik_max_word"
                            }
                        }
                    }
                }
            }

            # 检查索引是否存在
            response = requests.head(f"{self.es_base_url}/{self.es_index}", timeout=10)
            if response.status_code == 200:
                logger.warning(f"⚠️ 索引 {self.es_index} 已存在")
                # Docker环境下自动使用现有索引，不询问
                logger.info("Docker环境：使用现有索引")
                # 注意：不要在这里返回，继续处理Milvus集合
            else:
                # 创建索引
                response = requests.put(
                    f"{self.es_base_url}/{self.es_index}",
                    json=index_mapping,
                    timeout=30
                )

                if response.status_code in [200, 201]:
                    logger.info(f"✅ Elasticsearch索引 {self.es_index} 创建成功")
                else:
                    logger.error(f"❌ Elasticsearch索引创建失败: {response.status_code} - {response.text}")
                    return False

            # 创建索引
            response = requests.put(
                f"{self.es_base_url}/{self.es_index}",
                json=index_mapping,
                timeout=30
            )

            if response.status_code in [200, 201]:
                logger.info(f"✅ Elasticsearch索引 {self.es_index} 创建成功")
            else:
                logger.error(f"❌ Elasticsearch索引创建失败: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Elasticsearch索引创建失败: {e}")
            return False

        # 创建Milvus集合
        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

            # 确保连接已建立
            connections.connect(
                alias=self.milvus_connection_alias,
                host=self.milvus_host,
                port=str(self.milvus_port),
                timeout=30
            )

            # 检查集合是否存在 - 使用连接别名
            collection_exists = utility.has_collection(self.milvus_collection, using=self.milvus_connection_alias)
            logger.info(f"检查集合 {self.milvus_collection} 存在状态: {collection_exists}")

            if collection_exists:
                logger.warning(f"⚠️ 集合 {self.milvus_collection} 已存在")
                # Docker环境下自动使用现有集合，不询问
                logger.info("Docker环境：使用现有集合")
                return True

            # 定义字段 - 简化模式以匹配现有集合
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dimension),
                FieldSchema(name="chapter_title", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64)
            ]

            # 创建集合模式
            schema = CollectionSchema(
                fields=fields,
                description="医学文档向量集合",
                enable_dynamic_field=False
            )

            # 创建集合
            collection = Collection(name=self.milvus_collection, schema=schema)

            # 创建索引
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

            collection.create_index("embedding", index_params)
            logger.info(f"✅ Milvus集合 {self.milvus_collection} 创建成功")

            return True

        except Exception as e:
            logger.error(f"❌ Milvus集合创建失败: {e}")
            return False

    def load_chunks(self, json_file: str) -> List[Dict[str, Any]]:
        """加载切块数据"""
        logger.info(f"加载切块数据: {json_file}")

        try:
            # 支持相对路径和绝对路径
            if not os.path.isabs(json_file):
                json_file = os.path.join('/app/data', json_file)

            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            logger.info(f"✅ 成功加载 {len(chunks)} 个切块")
            return chunks

        except Exception as e:
            logger.error(f"❌ 加载切块数据失败: {e}")
            return []

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """生成嵌入向量"""
        logger.info("生成嵌入向量...")

        embeddings = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                logger.info(f"进度: {i+1}/{total_chunks}")

            try:
                # 生成嵌入向量
                embedding = self.embedding_model.encode(chunk['content'])
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"生成嵌入失败 for chunk {chunk.get('chunk_id', i)}: {e}")
                # 使用零向量作为后备
                embeddings.append([0.0] * self.embedding_dimension)

        logger.info(f"✅ 嵌入向量生成完成: {len(embeddings)} 个")
        return embeddings

    def import_to_elasticsearch(self, chunks: List[Dict[str, Any]]) -> bool:
        """导入到Elasticsearch"""
        logger.info("导入到Elasticsearch...")

        try:
            total_chunks = len(chunks)
            batch_size = 100

            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i+batch_size]
                bulk_data = []

                for chunk in batch:
                    doc = {
                        "content": chunk['content'],
                        "chapter_title": chunk.get('chapter_title', ''),
                        "section_title": chunk.get('section_title', ''),
                        "page_number": chunk.get('page_number', 1),
                        "chunk_id": chunk['chunk_id'],
                        "chunk_index": chunk['chunk_index'],
                        "content_length": chunk.get('content_length', len(chunk['content'])),
                        "sub_chunk_index": chunk.get('sub_chunk_index', 0),
                        "timestamp": datetime.now().isoformat()
                    }

                    # 添加bulk操作头
                    bulk_data.append({"index": {"_index": self.es_index, "_id": chunk['chunk_id']}})
                    bulk_data.append(doc)

                # 执行bulk导入
                if bulk_data:
                    response = requests.post(
                        f"{self.es_base_url}/{self.es_index}/_bulk",
                        headers={"Content-Type": "application/x-ndjson"},
                        data='\n'.join(json.dumps(line) for line in bulk_data) + '\n',
                        timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if not result.get('errors', False):
                            logger.info(f"✅ 批量导入成功: {i+1}-{min(i+batch_size, total_chunks)}/{total_chunks}")
                        else:
                            logger.error(f"❌ 批量导入有错误: {result}")
                            return False
                    else:
                        logger.error(f"❌ 批量导入失败: {response.status_code} - {response.text}")
                        return False

            logger.info(f"✅ Elasticsearch导入完成: {total_chunks} 个文档")
            return True

        except Exception as e:
            logger.error(f"❌ Elasticsearch导入失败: {e}")
            return False

    def import_to_milvus(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """导入到Milvus"""
        logger.info("导入到Milvus...")

        try:
            from pymilvus import Collection, connections

            # 确保连接已建立
            connections.connect(
                alias=self.milvus_connection_alias,
                host=self.milvus_host,
                port=str(self.milvus_port),
                timeout=30
            )

            # 获取集合
            collection = Collection(self.milvus_collection, using=self.milvus_connection_alias)

            total_chunks = len(chunks)
            batch_size = 100

            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]

                # 准备数据 - 按字段组织成列格式
                ids = []
                vectors = []
                chapter_titles = []
                section_titles = []
                page_numbers = []
                chunk_indices = []

                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    ids.append(chunk['chunk_id'])
                    vectors.append(embedding)
                    chapter_titles.append(chunk.get('chapter_title', ''))
                    section_titles.append(chunk.get('section_title', ''))
                    page_numbers.append(chunk.get('page_number', 1))
                    chunk_indices.append(chunk['chunk_index'])

                # 插入数据 - Milvus需要列式格式
                data = [ids, vectors, chapter_titles, section_titles, page_numbers, chunk_indices]
                if any(data):
                    collection.insert(data)
                    logger.info(f"✅ 批量插入成功: {i+1}-{min(i+batch_size, total_chunks)}/{total_chunks}")

            # 加载数据到内存
            collection.load()
            logger.info(f"✅ Milvus导入完成: {total_chunks} 个向量")
            return True

        except Exception as e:
            logger.error(f"❌ Milvus导入失败: {e}")
            return False

    def run_import(self, json_file: str) -> bool:
        """运行导入流程"""
        logger.info(f"开始导入流程: {json_file}")

        # 测试连接
        if not self.test_connections():
            logger.error("数据库连接测试失败")
            return False

        # 创建索引和集合
        if not self.create_indices_and_collections():
            logger.error("索引和集合创建失败")
            return False

        # 加载数据
        chunks = self.load_chunks(json_file)
        if not chunks:
            logger.error("数据加载失败")
            return False

        # 生成嵌入向量
        embeddings = self.generate_embeddings(chunks)
        if not embeddings:
            logger.error("嵌入向量生成失败")
            return False

        # 导入到Elasticsearch
        if not self.import_to_elasticsearch(chunks):
            logger.error("Elasticsearch导入失败")
            return False

        # 导入到Milvus
        if not self.import_to_milvus(chunks, embeddings):
            logger.error("Milvus导入失败")
            return False

        logger.info("🎉 Docker导入流程完成！")
        return True

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Docker环境数据导入工具")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="simple_chunks.json",
        help="输入的JSON文件路径 (默认: simple_chunks.json)"
    )
    parser.add_argument(
        "--test-only", "-t",
        action="store_true",
        help="仅测试连接，不执行导入"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    json_file = args.input if os.path.isabs(args.input) else f"/app/data/{args.input}"
    if not os.path.exists(json_file):
        logger.error(f"输入文件不存在: {json_file}")
        return 1

    # 创建导入器
    importer = DockerImporter()

    # 测试连接
    if args.test_only:
        logger.info("仅测试连接模式")
        success = importer.test_connections()
        return 0 if success else 1

    # 运行导入
    success = importer.run_import(args.input)
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)