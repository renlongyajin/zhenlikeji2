#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG系统数据库导入器（基于小节分块）
将处理好的Markdown小节数据块导入Elasticsearch和Milvus
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# --- 数据库客户端 ---
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

# --- 嵌入模型 ---
# 根据您的项目结构调整路径
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.embedding.embedding_models import get_embedding_manager

# --- 配置日志 ---
log_dir = Path(__file__).parent.parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'section_import_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)


class SectionBasedImporter:
    """
    基于小节分块的数据库导入器
    """

    def __init__(self,
                 es_host: str = "localhost",
                 es_port: int = 9200,
                 milvus_host: str = "localhost",
                 milvus_port: int = 19530,
                 embedding_model: str = "jina"):
        """
        初始化数据库连接和嵌入模型

        Args:
            es_host (str): Elasticsearch主机地址
            es_port (int): Elasticsearch端口
            milvus_host (str): Milvus主机地址
            milvus_port (int): Milvus端口
            embedding_model (str): 嵌入模型类型 (例如: "jina", "openai")
        """
        # 索引和集合名称
        self.es_index = "medical_sections"
        self.milvus_collection = "medical_section_vectors"
        self.es = None
        self.collection = None

        try:
            # 连接Elasticsearch
            self.es = Elasticsearch(f"http://{es_host}:{es_port}", request_timeout=30)
            if self.es.ping():
                logger.info("✅ 成功连接到Elasticsearch")
            else:
                raise ConnectionError("连接Elasticsearch失败")

            # 连接Milvus
            connections.connect(alias="default", host=milvus_host, port=str(milvus_port))
            logger.info("✅ 成功连接到Milvus")

            # 初始化嵌入模型
            self.embedding_manager = get_embedding_manager(model_type=embedding_model)

            # [修正] 动态检测嵌入向量的维度，而不是调用不存在的方法
            logger.info("动态检测嵌入向量维度...")
            sample_embedding = self.embedding_manager.encode_texts(["test"])[0]
            self.vector_dim = len(sample_embedding)
            logger.info(f"✅ 初始化嵌入模型: {embedding_model} (检测到维度: {self.vector_dim})")

            # 确保索引和集合存在
            self._create_elasticsearch_index()
            self._create_milvus_collection()

        except Exception as e:
            logger.error(f"❌ 初始化导入器失败: {e}")
            raise

    def _create_elasticsearch_index(self):
        """如果索引不存在，则创建Elasticsearch索引"""
        index_mapping = {
            "properties": {
                "title": {"type": "text", "analyzer": "standard"},
                "path": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "standard"},
                "created_at": {"type": "date"}
            }
        }
        try:
            if not self.es.indices.exists(index=self.es_index):
                self.es.indices.create(index=self.es_index, mappings=index_mapping)
                logger.info(f"✅ 创建Elasticsearch索引: {self.es_index}")
            else:
                logger.info(f"ℹ️ Elasticsearch索引已存在: {self.es_index}")
        except Exception as e:
            logger.error(f"⚠️ Elasticsearch索引检查/创建失败: {e}", exc_info=True)
            raise

    def _create_milvus_collection(self):
        """如果集合不存在，则创建Milvus集合"""
        if utility.has_collection(self.milvus_collection):
            logger.info(f"ℹ️ Milvus集合已存在: {self.milvus_collection}")
            self.collection = Collection(name=self.milvus_collection)
            # 确保集合已加载以备插入
            self.collection.load()
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        schema = CollectionSchema(fields, "Medical sections vector collection")
        self.collection = Collection(name=self.milvus_collection, schema=schema)
        
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index("vector", index_params)
        logger.info(f"✅ 创建Milvus集合: {self.milvus_collection}")
        # 新创建的集合也需要加载
        self.collection.load()

    def load_sections_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """从JSON文件加载小节数据"""
        logger.info(f"📖 开始从JSON文件加载数据: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chunks = data.get('chunks', [])
            logger.info(f"✅ 成功加载 {len(chunks)} 个数据块")
            return chunks
        except FileNotFoundError:
            logger.error(f"❌ 文件未找到: {json_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"❌ JSON文件格式错误: {json_path}")
            return []

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """为文本列表生成嵌入向量"""
        if not texts:
            return []
        logger.info(f"🧮 开始生成嵌入向量，共 {len(texts)} 个文本...")
        try:
            embeddings_array = self.embedding_manager.encode_texts(texts)
            embeddings = embeddings_array.tolist()
            logger.info(f"✅ 嵌入向量生成完成")
            return embeddings
        except Exception as e:
            logger.error(f"❌ 嵌入向量生成失败: {e}", exc_info=True)
            raise

    def import_to_elasticsearch(self, sections: List[Dict[str, Any]]):
        """批量导入数据到Elasticsearch"""
        if not sections:
            return
        logger.info(f"📤 开始向Elasticsearch导入数据，共 {len(sections)} 个文档...")
        
        actions = [
            {
                "_index": self.es_index,
                "_id": section['id'],
                "_source": {
                    "title": section['title'],
                    "path": section['path'],
                    "content": section['content'],
                    "created_at": datetime.now().isoformat()
                }
            }
            for section in sections
        ]
        
        try:
            success, failed = bulk(self.es, actions, raise_on_error=False)
            logger.info(f"✅ Elasticsearch导入完成: 成功 {success} 个, 失败 {len(failed)} 个")
            if failed:
                logger.warning(f"部分文档导入失败详情: {failed[:5]}") # 只显示前5个错误
            self.es.indices.refresh(index=self.es_index)
        except Exception as e:
            logger.error(f"❌ Elasticsearch批量导入失败: {e}", exc_info=True)
            raise

    def import_to_milvus(self, sections: List[Dict[str, Any]], embeddings: List[List[float]]):
        """导入数据到Milvus"""
        if not sections or not embeddings:
            return
        if len(sections) != len(embeddings):
            raise ValueError("数据块数量和嵌入向量数量不匹配")

        logger.info(f"📤 开始向Milvus导入数据，共 {len(sections)} 个向量...")
        
        entities = [
            [section['id'] for section in sections],  # ids
            embeddings  # vectors
        ]

        try:
            self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"✅ Milvus数据导入完成")
            logger.info(f"📈 Milvus集合向量总数: {self.collection.num_entities}")
        except Exception as e:
            logger.error(f"❌ Milvus数据导入失败: {e}", exc_info=True)
            raise

    def close(self):
        """关闭数据库连接"""
        if self.es:
            self.es.close()
            logger.info("🔌 关闭Elasticsearch连接")
        connections.disconnect("default")
        logger.info("🔌 关闭Milvus连接")

def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(
        description="将分块后的JSON数据导入Elasticsearch和Milvus。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置导入
  python3 %(prog)s data/processed_chunks.json

  # 指定嵌入模型和数据库地址
  python3 %(prog)s data/clean_data_chunks.json \\
    --embedding jina \\
    --es-host localhost \\
    --milvus-host localhost
"""
    )
    parser.add_argument(
        "json_file",
        help="输入的已分块JSON文件路径"
    )
    parser.add_argument(
        "--embedding",
        choices=["jina", "openai"],
        default="jina",
        help="嵌入模型类型 (默认: jina)"
    )
    parser.add_argument("--es-host", default="localhost", help="Elasticsearch主机")
    parser.add_argument("--es-port", type=int, default=9200, help="Elasticsearch端口")
    parser.add_argument("--milvus-host", default="localhost", help="Milvus主机")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus端口")

    args = parser.parse_args()

    importer = None
    try:
        # 初始化导入器
        importer = SectionBasedImporter(
            es_host=args.es_host,
            es_port=args.es_port,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            embedding_model=args.embedding
        )
        
        # 1. 加载数据
        sections = importer.load_sections_from_json(args.json_file)
        if not sections:
            logger.warning("⚠️ 未加载到任何数据，程序退出。")
            return

        # 2. 导入到Elasticsearch
        importer.import_to_elasticsearch(sections)
        
        # 3. 生成嵌入向量
        contents = [section['content'] for section in sections]
        embeddings = importer.generate_embeddings(contents)
        
        # 4. 导入到Milvus
        importer.import_to_milvus(sections, embeddings)

        logger.info("\n🎉 所有数据导入流程成功完成！")

    except Exception as e:
        logger.error(f"\n❌ 导入流程发生严重错误: {e}", exc_info=True)
    finally:
        if importer:
            importer.close()

if __name__ == "__main__":
    main()

