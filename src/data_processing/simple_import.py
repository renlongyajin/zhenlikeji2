#!/usr/bin/env python3
"""
Simple Chunks数据导入脚本
将simple_chunks.json导入到Elasticsearch和Milvus数据库
使用项目标准配置：medical_documents_fixed 和 medical_vectors_fixed
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
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入嵌入模型
try:
    from src.embedding.embedding_models import JinaEmbeddingModel
except ImportError as e:
    logger.error(f"无法导入嵌入模型: {e}")
    sys.exit(1)

class SimpleImporter:
    """简单数据导入器 - 使用项目标准配置"""

    def __init__(self):
        """初始化导入器 - 使用项目标准配置"""
        # 从环境变量获取连接配置，使用项目标准配置
        # Docker环境使用不同的环境变量名
        self.es_host = os.environ.get('ELASTICSEARCH_HOST', 'elasticsearch')  # Docker服务名
        self.es_port = int(os.environ.get('ELASTICSEARCH_PORT', '9200'))
        self.es_base_url = f"http://{self.es_host}:{self.es_port}"
        self.es_index = "medical_documents_fixed"  # 项目标准ES索引

        self.milvus_host = os.environ.get('MILVUS_HOST', 'milvus')  # Docker服务名
        self.milvus_port = int(os.environ.get('MILVUS_PORT', '19530'))
        self.milvus_collection = "medical_vectors_fixed"  # 项目标准Milvus集合
        self.milvus_connection_alias = "simple_import"

        # 初始化嵌入模型
        try:
            self.embedding_model = JinaEmbeddingModel()
            # 根据现有集合的维度调整，medical_vectors_fixed使用768维
            self.embedding_dimension = 768
            logger.info("✅ Jina嵌入模型初始化成功")
            logger.info(f"使用适配的嵌入维度: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            # 使用备用配置
            self.embedding_dimension = 768
            logger.warning(f"使用备用嵌入维度: {self.embedding_dimension}")

        logger.info(f"初始化导入器:")
        logger.info(f"  Elasticsearch: {self.es_host}:{self.es_port} -> {self.es_index}")
        logger.info(f"  Milvus: {self.milvus_host}:{self.milvus_port} -> {self.milvus_collection}")
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

            # 建立连接
            connections.connect(
                alias=self.milvus_connection_alias,
                host=self.milvus_host,
                port=str(self.milvus_port),
                timeout=30
            )

            # 测试连接
            collections = utility.list_collections(using=self.milvus_connection_alias)
            logger.info(f"✅ Milvus连接成功，现有集合: {len(collections)}个")
            if self.milvus_collection in collections:
                logger.info(f"  目标集合 {self.milvus_collection} 已存在")
            else:
                logger.info(f"  目标集合 {self.milvus_collection} 不存在，将创建新集合")

            return True

        except Exception as e:
            logger.error(f"❌ Milvus连接失败: {e}")
            return False

    def load_chunks(self, json_file: str) -> List[Dict[str, Any]]:
        """加载切块数据"""
        logger.info(f"加载切块数据: {json_file}")

        try:
            # 支持相对路径和绝对路径
            if not os.path.isabs(json_file):
                json_file = os.path.join('data', json_file)

            if not os.path.exists(json_file):
                logger.error(f"文件不存在: {json_file}")
                return []

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
                # 生成嵌入向量 - 使用正确的encode方法
                embedding = self.embedding_model.encode(chunk['content']).tolist()
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"生成嵌入失败 for chunk {chunk.get('chunk_id', i)}: {e}")
                # 使用零向量作为后备
                embeddings.append([0.0] * self.embedding_dimension)

        logger.info(f"✅ 嵌入向量生成完成: {len(embeddings)} 个")
        return embeddings

    def import_to_elasticsearch(self, chunks: List[Dict[str, Any]]) -> bool:
        """导入到Elasticsearch"""
        logger.info(f"导入到Elasticsearch索引: {self.es_index}")

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
                            logger.info(f"✅ ES批量导入成功: {i+1}-{min(i+batch_size, total_chunks)}/{total_chunks}")
                        else:
                            logger.error(f"❌ ES批量导入有错误: {result}")
                            return False
                    else:
                        logger.error(f"❌ ES批量导入失败: {response.status_code} - {response.text}")
                        return False

            logger.info(f"✅ Elasticsearch导入完成: {total_chunks} 个文档")
            return True

        except Exception as e:
            logger.error(f"❌ Elasticsearch导入失败: {e}")
            return False

    def import_to_milvus(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """导入到Milvus"""
        logger.info(f"导入到Milvus集合: {self.milvus_collection}")

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

                # 准备数据 - 按字段组织成列格式（Milvus要求）
                ids = []
                vectors = []
                chapter_titles = []
                section_titles = []
                page_numbers = []
                chunk_indices = []

                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    ids.append(chunk['chunk_id'])
                    # 确保向量是numpy数组格式
                    vectors.append(np.array(embedding, dtype=np.float32))
                    chapter_titles.append(chunk.get('chapter_title', ''))
                    section_titles.append(chunk.get('section_title', ''))
                    page_numbers.append(chunk.get('page_number', 1))
                    chunk_indices.append(chunk['chunk_index'])

                # 插入数据 - Milvus需要列式格式
                data = [ids, vectors, chapter_titles, section_titles, page_numbers, chunk_indices]
                if any(ids):  # 确保有数据
                    collection.insert(data)
                    logger.info(f"✅ Milvus批量插入成功: {i+1}-{min(i+batch_size, total_chunks)}/{total_chunks}")

            # 加载数据到内存
            collection.load()
            logger.info(f"✅ Milvus导入完成: {total_chunks} 个向量")
            return True

        except Exception as e:
            logger.error(f"❌ Milvus导入失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def verify_import(self) -> bool:
        """验证导入结果"""
        logger.info("验证导入结果...")

        try:
            # 检查Elasticsearch文档数量
            es_response = requests.get(f"{self.es_base_url}/{self.es_index}/_count", timeout=10)
            if es_response.status_code == 200:
                es_count = es_response.json().get('count', 0)
                logger.info(f"✅ Elasticsearch索引 {self.es_index} 文档数量: {es_count}")
            else:
                logger.warning(f"⚠️ 无法验证ES文档数量: {es_response.status_code}")

            # 检查Milvus向量数量
            from pymilvus import Collection, connections
            connections.connect(
                alias="verify_connection",
                host=self.milvus_host,
                port=str(self.milvus_port),
                timeout=30
            )
            collection = Collection(self.milvus_collection, using="verify_connection")
            milvus_count = collection.num_entities
            logger.info(f"✅ Milvus集合 {self.milvus_collection} 向量数量: {milvus_count}")

            return True

        except Exception as e:
            logger.error(f"❌ 导入验证失败: {e}")
            return False

    def run_import(self, json_file: str) -> bool:
        """运行导入流程"""
        logger.info(f"开始导入流程: {json_file}")

        # 测试连接
        if not self.test_connections():
            logger.error("数据库连接测试失败")
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

        # 验证导入结果
        self.verify_import()

        logger.info("🎉 导入流程完成！")
        return True

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Chunks数据导入工具")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/simple_chunks.json",
        help="输入的JSON文件路径 (默认: data/simple_chunks.json)"
    )
    parser.add_argument(
        "--test-only", "-t",
        action="store_true",
        help="仅测试连接，不执行导入"
    )

    args = parser.parse_args()

    # 检查文件是否存在（仅导入模式）
    if not args.test_only and not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return 1

    # 创建导入器
    importer = SimpleImporter()

    # 测试连接
    if args.test_only:
        logger.info("仅测试连接模式")
        success = importer.test_connections()
        return 0 if success else 1

    # 运行导入
    success = importer.run_import(args.input)
    return 0 if success else 1

if __name__ == "__main__":
    # 加载环境变量
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()

    exit_code = main()
    sys.exit(exit_code)