#!/usr/bin/env python3
"""
Simple Chunks数据导入脚本
将simple_chunks.json导入到Elasticsearch和Milvus数据库
使用项目标准配置：medical_documents_fixed 和 medical_vectors_fixed
增加了 --clean 选项，可在导入前自动清理旧数据。
"""

import json
import os
import sys
import requests
from typing import List, Dict, Any
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入相关模块
try:
    from src.embedding.embedding_models import JinaEmbeddingModel
    from pymilvus import (
        connections, utility, Collection, CollectionSchema, FieldSchema, DataType
    )
except ImportError as e:
    logger.error(f"无法导入所需模块: {e}")
    sys.exit(1)

class SimpleImporter:
    """简单数据导入器 - 使用项目标准配置"""

    def __init__(self, embedding_model: str = None):
        """初始化导入器"""
        self.es_host = os.environ.get('ELASTICSEARCH_HOST', 'localhost')
        self.es_port = int(os.environ.get('ELASTICSEARCH_PORT', '9200'))
        self.es_base_url = f"http://{self.es_host}:{self.es_port}"
        self.es_index = "medical_documents_fixed"

        self.milvus_host = os.environ.get('MILVUS_HOST', 'localhost')
        self.milvus_port = int(os.environ.get('MILVUS_PORT', '19530'))
        self.milvus_collection = "medical_vectors_fixed"
        self.milvus_connection_alias = "simple_import"

        self.embedding_dimension = 768
        self.embedding_model = self._init_embedding_model(embedding_model)

        logger.info(f"初始化导入器:")
        logger.info(f"   Elasticsearch: {self.es_host}:{self.es_port} -> {self.es_index}")
        logger.info(f"   Milvus: {self.milvus_host}:{self.milvus_port} -> {self.milvus_collection}")
        logger.info(f"   嵌入维度: {self.embedding_dimension}")
        logger.info(f"   嵌入模型: {embedding_model or '默认 (Jina)'}")


    def _init_embedding_model(self, model_name: str):
        """初始化嵌入模型"""
        try:
            if model_name == "qwen3-0.6b":
                from src.embedding.embedding_models import Qwen3EmbeddingModel
                logger.info("✅ 使用指定的千问3-0.6B嵌入模型")
                return Qwen3EmbeddingModel()
            logger.info("✅ 使用默认的Jina嵌入模型")
            return JinaEmbeddingModel()
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            sys.exit(1)

    def clean_databases(self):
        """【新功能】清空并重建数据库、集合和索引"""
        logger.info("--- 正在执行清理操作 ---")

        # 1. 清理Elasticsearch
        try:
            url = f"{self.es_base_url}/{self.es_index}"
            response = requests.head(url)
            if response.status_code == 200:
                logger.info(f"正在删除Elasticsearch索引: {self.es_index}...")
                requests.delete(url)
                logger.info(f"✅ 索引 {self.es_index} 删除成功。")

            logger.info(f"正在重建Elasticsearch索引: {self.es_index}...")
            requests.put(url, json={"settings": {"number_of_shards": 1, "number_of_replicas": 0}})
            logger.info(f"✅ 索引 {self.es_index} 重建成功。")
        except Exception as e:
            logger.error(f"❌ 清理Elasticsearch时出错: {e}")
            return False

        # 2. 清理Milvus
        try:
            self._connect_milvus()
            if utility.has_collection(self.milvus_collection, using=self.milvus_connection_alias):
                logger.info(f"正在删除Milvus集合: {self.milvus_collection}...")
                utility.drop_collection(self.milvus_collection, using=self.milvus_connection_alias)
                logger.info(f"✅ 集合 {self.milvus_collection} 删除成功。")

            logger.info(f"正在重建Milvus集合: {self.milvus_collection}...")
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=255),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dimension),
                FieldSchema(name="chapter_title", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64)
            ]
            schema = CollectionSchema(fields, description="Medical document vectors")
            collection = Collection(name=self.milvus_collection, schema=schema, using=self.milvus_connection_alias)

            logger.info("正在为Milvus集合创建索引...")
            index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
            collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"✅ 集合 {self.milvus_collection} 和索引重建成功。")

        except Exception as e:
            logger.error(f"❌ 清理Milvus时出错: {e}")
            return False
        
        logger.info("--- 清理操作完成 ---")
        return True

    def _connect_milvus(self):
        """建立或复用Milvus连接"""
        if self.milvus_connection_alias not in connections.list_connections():
            connections.connect(
                alias=self.milvus_connection_alias,
                host=self.milvus_host,
                port=str(self.milvus_port),
                timeout=30
            )

    def test_connections(self) -> bool:
        """测试数据库连接"""
        logger.info("测试数据库连接...")
        try:
            response = requests.get(f"{self.es_base_url}/_cluster/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Elasticsearch连接成功")
            else:
                logger.warning(f"⚠️ Elasticsearch状态异常: {response.status_code}")
                return False
            
            self._connect_milvus()
            collections = utility.list_collections(using=self.milvus_connection_alias)
            logger.info(f"✅ Milvus连接成功，现有集合: {len(collections)}个")
            return True
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            return False

    def load_chunks(self, json_file: str) -> List[Dict[str, Any]]:
        """加载切块数据"""
        logger.info(f"加载切块数据: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 加载切块数据失败: {e}")
            return []

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """生成嵌入向量"""
        logger.info("生成嵌入向量...")
        embeddings = []
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                logger.info(f"进度: {i+1}/{len(chunks)}")
            try:
                embedding = self.embedding_model.encode(chunk['content'])[0].tolist()
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"生成嵌入失败 for chunk {chunk.get('chunk_id', i)}: {e}")
                embeddings.append([0.0] * self.embedding_dimension)
        logger.info(f"✅ 嵌入向量生成完成: {len(embeddings)} 个")
        return embeddings

    def import_to_elasticsearch(self, chunks: List[Dict[str, Any]]) -> bool:
        """导入到Elasticsearch"""
        logger.info(f"导入到Elasticsearch索引: {self.es_index}")
        try:
            bulk_data = []
            for chunk in chunks:
                doc = {
                    "content": chunk['content'],
                    "chapter_title": chunk.get('chapter_title', ''),
                    "section_title": chunk.get('section_title', ''),
                    "page_number": chunk.get('page_number', 1),
                    "chunk_id": chunk['chunk_id'],
                    "chunk_index": chunk['chunk_index'],
                    "timestamp": datetime.now().isoformat()
                }
                bulk_data.append({"index": {"_index": self.es_index, "_id": chunk['chunk_id']}})
                bulk_data.append(doc)

            response = requests.post(
                f"{self.es_base_url}/_bulk",
                headers={"Content-Type": "application/x-ndjson"},
                data='\n'.join(json.dumps(line) for line in bulk_data) + '\n',
                timeout=60
            )
            if response.status_code == 200 and not response.json().get('errors'):
                logger.info(f"✅ ES批量导入成功: {len(chunks)}/{len(chunks)}")
                return True
            else:
                logger.error(f"❌ ES批量导入失败: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Elasticsearch导入失败: {e}")
            return False

    def import_to_milvus(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """导入到Milvus"""
        logger.info(f"导入到Milvus集合: {self.milvus_collection}")
        try:
            self._connect_milvus()
            collection = Collection(self.milvus_collection, using=self.milvus_connection_alias)
            
            data = [
                [chunk['chunk_id'] for chunk in chunks],
                embeddings,
                [chunk.get('chapter_title', '') for chunk in chunks],
                [chunk.get('section_title', '') for chunk in chunks],
                [chunk.get('page_number', 1) for chunk in chunks],
                [chunk['chunk_index'] for chunk in chunks]
            ]
            
            collection.insert(data)
            logger.info(f"✅ Milvus批量插入成功: {len(chunks)}/{len(chunks)}")

            logger.info("正在刷新Milvus集合以固化数据...")
            collection.flush()
            logger.info("✅ Milvus集合刷新完成。")

            logger.info("正在加载Milvus集合到内存...")
            collection.load()
            logger.info(f"✅ Milvus导入完成: {len(chunks)} 个向量已加载")
            return True
        except Exception as e:
            logger.error(f"❌ Milvus导入失败: {e}", exc_info=True)
            return False

    def verify_import(self) -> bool:
        """验证导入结果"""
        logger.info("验证导入结果...")
        try:
            es_count = requests.get(f"{self.es_base_url}/{self.es_index}/_count").json().get('count', 0)
            logger.info(f"✅ Elasticsearch索引文档数量: {es_count}")

            self._connect_milvus()
            collection = Collection(self.milvus_collection, using=self.milvus_connection_alias)
            collection.flush()
            milvus_count = collection.num_entities
            logger.info(f"✅ Milvus集合向量数量: {milvus_count}")
            return True
        except Exception as e:
            logger.error(f"❌ 导入验证失败: {e}")
            return False

    def run_import(self, json_file: str, clean: bool = False) -> bool:
        """运行导入流程"""
        logger.info(f"开始导入流程: {json_file}")

        if clean:
            if not self.clean_databases():
                logger.error("数据库清理失败，导入中止。")
                return False

        if not self.test_connections():
            return False

        chunks = self.load_chunks(json_file)
        if not chunks:
            return False

        embeddings = self.generate_embeddings(chunks)
        if not embeddings or len(chunks) != len(embeddings):
            return False

        if not self.import_to_elasticsearch(chunks):
            return False

        if not self.import_to_milvus(chunks, embeddings):
            return False

        self.verify_import()
        logger.info("🎉 导入流程完成！")
        return True

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="Simple Chunks数据导入工具")
    parser.add_argument(
        "--input", "-i", type=str, default="data/simple_chunks.json",
        help="输入的JSON文件路径 (默认: data/simple_chunks.json)"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="【新功能】在导入前清空并重建Elasticsearch索引和Milvus集合"
    )
    parser.add_argument(
        "--test-only", "-t", action="store_true",
        help="仅测试连接，不执行导入"
    )
    parser.add_argument(
        "--embedding-model", "-e", type=str, default=None,
        help="指定嵌入模型 (jina 或 qwen3-0.6b)，默认使用Jina"
    )
    args = parser.parse_args()

    if not args.test_only and not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return 1

    importer = SimpleImporter(embedding_model=args.embedding_model)
    if args.test_only:
        return 0 if importer.test_connections() else 1
    
    success = importer.run_import(args.input, clean=args.clean)
    return 0 if success else 1

if __name__ == "__main__":
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
    sys.exit(main())


# ```

# ### 如何使用

# 现在你的操作流程变得非常简单：

# 1.  **首次导入或希望覆盖全部数据时**，使用 `--clean` 参数：
#     ```bash
#     python your_script_name.py --clean
#     ```
#     脚本会先删除旧的索引和集合，然后创建一个全新的、干净的环境，最后导入数据。

# 2.  **增量添加数据时**（注意：当前脚本逻辑会覆盖同ID数据），不加 `--clean` 参数：
#     ```bash
#     python your_script_name.py
