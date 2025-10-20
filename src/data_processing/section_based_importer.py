#!/usr/bin/env python3
"""
基于小节的数据库导入器
将完整的医学小节导入Elasticsearch和Milvus
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import sys

# 添加项目路径
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

# 数据库客户端
from elasticsearch import Elasticsearch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 嵌入模型
from embedding.embedding_models import get_embedding_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SectionBasedImporter:
    """基于小节的数据库导入器"""

    def __init__(self,
                 es_host: str = "localhost",
                 es_port: int = 9200,
                 milvus_host: str = "localhost",
                 milvus_port: int = 19530,
                 embedding_model: str = "jina"):
        """
        初始化数据库连接

        Args:
            es_host: Elasticsearch主机地址
            es_port: Elasticsearch端口
            milvus_host: Milvus主机地址
            milvus_port: Milvus端口
            embedding_model: 嵌入模型类型
        """
        # 连接Elasticsearch
        self.es = Elasticsearch(f"http://{es_host}:{es_port}")

        # 连接Milvus
        try:
            connections.connect(alias="default", host=milvus_host, port=str(milvus_port))
            logger.info("✅ 成功连接到Milvus")
        except Exception as e:
            logger.error(f"❌ 连接Milvus失败: {e}")
            raise

        # 初始化嵌入模型
        try:
            self.embedding_manager = get_embedding_manager(model_type=embedding_model)
            logger.info(f"✅ 初始化嵌入模型: {embedding_model}")
        except Exception as e:
            logger.error(f"❌ 初始化嵌入模型失败: {e}")
            raise

        # 使用新的索引和集合名称（不影响旧数据）
        self.es_index = "medical_sections"
        self.milvus_collection = "medical_section_vectors"

        # 确保索引和集合存在
        self._create_elasticsearch_index()
        self._create_milvus_collection()

    def _create_elasticsearch_index(self):
        """创建Elasticsearch索引"""
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "section_id": {"type": "keyword"},
                    "chapter_number": {"type": "integer"},
                    "chapter_title": {"type": "text"},
                    "section_number": {"type": "integer"},
                    "section_title": {"type": "text"},
                    "disease_name": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "page_range": {"type": "integer"},
                    "content_length": {"type": "integer"},
                    "page_count": {"type": "integer"},
                    "metadata": {"type": "object"},
                    "created_at": {"type": "date"}
                }
            }
        }

        try:
            if not self.es.indices.exists(index=self.es_index):
                self.es.indices.create(index=self.es_index, body=index_settings)
                logger.info(f"✅ 创建Elasticsearch索引: {self.es_index}")
            else:
                logger.info(f"ℹ️  Elasticsearch索引已存在: {self.es_index}")
        except Exception as e:
            logger.warning(f"⚠️  Elasticsearch索引检查/创建失败: {e}")

    def _create_milvus_collection(self):
        """创建Milvus集合"""
        # 检查集合是否已存在
        if utility.has_collection(self.milvus_collection):
            utility.drop_collection(self.milvus_collection)
            logger.info(f"🗑️  删除已存在的Milvus集合: {self.milvus_collection}")

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="chapter_number", dtype=DataType.INT32),
            FieldSchema(name="section_number", dtype=DataType.INT32),
            FieldSchema(name="page_start", dtype=DataType.INT32),
            FieldSchema(name="page_end", dtype=DataType.INT32),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64)
        ]

        # 创建schema
        schema = CollectionSchema(fields, f"Medical section vectors for {self.milvus_collection}")

        # 创建集合
        self.collection = Collection(name=self.milvus_collection, schema=schema)

        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index("vector", index_params)

        logger.info(f"✅ 创建Milvus集合: {self.milvus_collection}")

    def load_sections_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        从JSON文件加载小节数据

        Args:
            json_path: JSON文件路径

        Returns:
            小节列表
        """
        logger.info(f"📖 开始加载小节数据: {json_path}")

        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sections = data.get('sections', [])
        logger.info(f"✅ 加载完成，共 {len(sections)} 个小节")

        return sections

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        logger.info(f"🧮 开始生成嵌入向量，共 {len(texts)} 个文本")

        try:
            # 使用嵌入管理器生成向量
            embeddings_array = self.embedding_manager.encode_texts(texts)
            embeddings = embeddings_array.tolist()

            logger.info(f"✅ 嵌入向量生成完成，向量维度: {len(embeddings[0])}")
            return embeddings

        except Exception as e:
            logger.error(f"❌ 嵌入向量生成失败: {e}")
            raise

    def import_to_elasticsearch(self, sections: List[Dict[str, Any]]):
        """
        导入数据到Elasticsearch

        Args:
            sections: 小节数据列表
        """
        logger.info(f"📤 开始向Elasticsearch导入数据，共 {len(sections)} 个小节")

        success_count = 0
        error_count = 0

        for section in sections:
            try:
                # 准备文档数据
                doc = {
                    "section_id": section['id'],
                    "chapter_number": section['chapter_number'],
                    "chapter_title": section['chapter_title'],
                    "section_number": section['section_number'],
                    "section_title": section['section_title'],
                    "disease_name": section['disease_name'],
                    "content": section['content'],
                    "page_range": section['page_range'],
                    "content_length": section['metadata'].get('content_length', 0),
                    "page_count": section['metadata'].get('page_count', 0),
                    "metadata": section['metadata'],
                    "created_at": datetime.now()
                }

                # 索引文档
                self.es.index(index=self.es_index, id=section['id'], body=doc)
                success_count += 1

                if success_count % 10 == 0:
                    logger.info(f"📊 已导入 {success_count} 个小节")

            except Exception as e:
                logger.error(f"❌ 导入小节 {section['id']} 失败: {e}")
                error_count += 1

        logger.info(f"✅ Elasticsearch导入完成: 成功 {success_count}, 失败 {error_count}")

        # 刷新索引
        self.es.indices.refresh(index=self.es_index)

        # 显示索引统计
        stats = self.es.indices.stats(index=self.es_index)
        doc_count = stats['indices'][self.es_index]['primaries']['docs']['count']
        logger.info(f"📈 Elasticsearch索引文档总数: {doc_count}")

    def import_to_milvus(self, sections: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        导入数据到Milvus

        Args:
            sections: 小节数据列表
            embeddings: 对应的嵌入向量列表
        """
        logger.info(f"📤 开始向Milvus导入数据，共 {len(sections)} 个向量")

        if len(sections) != len(embeddings):
            raise ValueError("小节数量和嵌入向量数量不匹配")

        # 准备数据
        ids = [section['id'] for section in sections]
        chapter_numbers = [section['chapter_number'] for section in sections]
        section_numbers = [section['section_number'] for section in sections]
        page_starts = [section['page_range'][0] for section in sections]
        page_ends = [section['page_range'][1] for section in sections]

        # 生成内容哈希
        content_hashes = []
        for section in sections:
            content = section['content']
            hash_obj = hashlib.md5(content.encode('utf-8'))
            content_hashes.append(hash_obj.hexdigest())

        # 插入数据
        entities = [
            ids,
            embeddings,
            chapter_numbers,
            section_numbers,
            page_starts,
            page_ends,
            content_hashes
        ]

        try:
            self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"✅ Milvus数据导入完成")

            # 显示集合统计
            self.collection.load()
            stats = self.collection.num_entities
            logger.info(f"📈 Milvus集合向量总数: {stats}")

        except Exception as e:
            logger.error(f"❌ Milvus数据导入失败: {e}")
            raise

    def close(self):
        """关闭连接"""
        if self.es:
            self.es.close()
            logger.info("✅ 关闭Elasticsearch连接")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="基于小节的数据库导入器")
    parser.add_argument("json_file", help="小节JSON文件路径")
    parser.add_argument("--embedding", choices=["jina", "openai"], default="jina",
                       help="嵌入模型类型")
    parser.add_argument("--es-host", default="localhost", help="Elasticsearch主机")
    parser.add_argument("--es-port", type=int, default=9200, help="Elasticsearch端口")
    parser.add_argument("--milvus-host", default="localhost", help="Milvus主机")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus端口")

    args = parser.parse_args()

    # 创建导入器
    importer = SectionBasedImporter(
        es_host=args.es_host,
        es_port=args.es_port,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        embedding_model=args.embedding
    )

    try:
        # 加载小节数据
        sections = importer.load_sections_from_json(args.json_file)

        if not sections:
            logger.error("❌ 没有找到任何小节数据")
            return 1

        logger.info(f"📈 总共加载 {len(sections)} 个小节")

        # 导入到Elasticsearch
        importer.import_to_elasticsearch(sections)

        # 生成嵌入向量
        texts = [section['content'] for section in sections]
        embeddings = importer.generate_embeddings(texts)

        # 导入到Milvus
        importer.import_to_milvus(sections, embeddings)

        logger.info("🎉 数据导入完成！")
        return 0

    except Exception as e:
        logger.error(f"❌ 数据导入过程中出现错误: {e}")
        return 1

    finally:
        importer.close()

if __name__ == "__main__":
    exit(main())
