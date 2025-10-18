#!/usr/bin/env python3
"""
简化版本的数据库导入器
使用requests库直接操作Elasticsearch
"""

import json
import re
import os
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDatabaseImporter:
    """简化版本的数据库导入器"""

    def __init__(self,
                 es_host: str = "localhost",
                 es_port: int = 9200,
                 milvus_host: str = "localhost",
                 milvus_port: int = 19530):
        """初始化导入器"""
        self.es_base_url = f"http://{es_host}:{es_port}"
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.es_index = "medical_documents"
        self.milvus_collection = "medical_vectors"

        # 测试连接
        self._test_connections()

    def _test_connections(self):
        """测试数据库连接"""
        # 测试Elasticsearch
        try:
            response = requests.get(f"{self.es_base_url}/_cluster/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Elasticsearch连接成功")
            else:
                raise Exception(f"Elasticsearch连接失败: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Elasticsearch连接失败: {e}")
            raise

        # 测试Milvus
        try:
            from pymilvus import connections
            connections.connect(alias="default", host=self.milvus_host, port=str(self.milvus_port))
            logger.info("✅ Milvus连接成功")
        except Exception as e:
            logger.error(f"❌ Milvus连接失败: {e}")
            raise

    def create_elasticsearch_index(self):
        """创建Elasticsearch索引"""
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "chapter_title": {"type": "keyword"},
                    "section_title": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "created_at": {"type": "date"}
                }
            }
        }

        try:
            response = requests.put(
                f"{self.es_base_url}/{self.es_index}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(index_settings)
            )

            if response.status_code in [200, 201]:
                logger.info(f"✅ 创建Elasticsearch索引: {self.es_index}")
            elif response.status_code == 400:
                logger.info(f"ℹ️  Elasticsearch索引已存在: {self.es_index}")
            else:
                logger.warning(f"⚠️  Elasticsearch索引创建返回: {response.status_code}")

        except Exception as e:
            logger.warning(f"⚠️  Elasticsearch索引创建失败: {e}")

    def parse_text_file(self, text_path: str, chunk_size: int = 500) -> List[Dict[str, Any]]:
        """解析文本文件"""
        logger.info(f"📖 开始解析文本文件: {text_path}")

        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        documents = []
        doc_id = 0

        # 按页面分割
        page_pattern = r'#### 第(\d+)页'
        pages = re.split(page_pattern, text)

        for i in range(1, len(pages), 2):
            if i + 1 < len(pages):
                page_num = int(pages[i])
                page_content = pages[i + 1]

                # 清理内容
                page_content = re.sub(r'\n+', '\n', page_content.strip())

                if len(page_content) > 50:  # 过滤短内容
                    doc = {
                        'id': f"doc_{doc_id}",
                        'page_number': page_num,
                        'content': page_content,
                        'chapter_title': '',
                        'section_title': '',
                        'created_at': datetime.now().isoformat()
                    }
                    documents.append(doc)
                    doc_id += 1

        logger.info(f"✅ 文本解析完成，共获得 {len(documents)} 个文档片段")
        return documents

    def parse_structured_json(self, json_path: str) -> List[Dict[str, Any]]:
        """解析结构化JSON文件"""
        logger.info(f"📖 开始解析JSON文件: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        doc_id = 0

        # 解析章节结构
        for chapter in data.get('content_structure', {}).get('hierarchy', []):
            chapter_title = chapter.get('title', '')

            for section in chapter.get('sections', []):
                section_title = section.get('title', '')

                doc = {
                    'id': f"doc_{doc_id}",
                    'chapter_title': chapter_title,
                    'section_title': section_title,
                    'page_number': section.get('page', 0),
                    'content': f"{chapter_title} - {section_title}",
                    'created_at': datetime.now().isoformat()
                }
                documents.append(doc)
                doc_id += 1

        logger.info(f"✅ 解析完成，共获得 {len(documents)} 个文档片段")
        return documents

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成模拟嵌入向量"""
        logger.info(f"🧮 开始生成嵌入向量，共 {len(texts)} 个文本")

        np.random.seed(42)
        embeddings = []

        for text in texts:
            # 生成768维向量
            vector = np.random.random(768).tolist()
            embeddings.append(vector)

        logger.info("✅ 嵌入向量生成完成")
        return embeddings

    def import_to_elasticsearch(self, documents: List[Dict[str, Any]]):
        """导入数据到Elasticsearch"""
        logger.info(f"📤 开始向Elasticsearch导入数据，共 {len(documents)} 个文档")

        success_count = 0
        error_count = 0

        for doc in documents:
            try:
                response = requests.put(
                    f"{self.es_base_url}/{self.es_index}/_doc/{doc['id']}",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(doc),
                    timeout=30
                )

                if response.status_code in [200, 201]:
                    success_count += 1
                else:
                    logger.error(f"❌ 导入文档 {doc['id']} 失败: {response.status_code} - {response.text}")
                    error_count += 1

                if success_count % 50 == 0:
                    logger.info(f"📊 已导入 {success_count} 个文档")

            except Exception as e:
                logger.error(f"❌ 导入文档 {doc['id']} 失败: {e}")
                error_count += 1

        logger.info(f"✅ Elasticsearch导入完成: 成功 {success_count}, 失败 {error_count}")

        # 刷新索引
        try:
            requests.post(f"{self.es_base_url}/{self.es_index}/_refresh")
        except Exception as e:
            logger.warning(f"⚠️  刷新索引失败: {e}")

    def import_to_milvus(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """导入数据到Milvus"""
        logger.info(f"📤 开始向Milvus导入数据，共 {len(documents)} 个向量")

        try:
            from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

            # 连接Milvus
            connections.connect(alias="default", host=self.milvus_host, port=str(self.milvus_port))

            # 删除已存在的集合
            if utility.has_collection(self.milvus_collection):
                utility.drop_collection(self.milvus_collection)

            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema(name="page_number", dtype=DataType.INT32),
            ]

            # 创建schema
            schema = CollectionSchema(fields, f"Medical document vectors")

            # 创建集合
            collection = Collection(name=self.milvus_collection, schema=schema)

            # 创建索引
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("vector", index_params)

            # 准备数据
            ids = [doc['id'] for doc in documents]
            page_numbers = [doc.get('page_number', 0) for doc in documents]

            # 插入数据
            entities = [ids, embeddings, page_numbers]
            collection.insert(entities)
            collection.flush()

            logger.info(f"✅ Milvus数据导入完成")

            # 显示集合统计
            collection.load()
            stats = collection.num_entities
            logger.info(f"📈 Milvus集合向量总数: {stats}")

        except Exception as e:
            logger.error(f"❌ Milvus数据导入失败: {e}")
            raise

def main():
    """主函数"""
    # 文件路径
    json_path = "/home/ubuntu/myproject/zhenlikeji2/data/extracted/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224_structured.json"
    text_path = "/home/ubuntu/myproject/zhenlikeji2/data/extracted/text_stable/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224_stable_extracted.txt"

    # 创建导入器
    importer = SimpleDatabaseImporter()

    try:
        # 创建索引
        importer.create_elasticsearch_index()

        # 解析数据
        documents = []

        # 从JSON文件解析
        if os.path.exists(json_path):
            json_docs = importer.parse_structured_json(json_path)
            documents.extend(json_docs)
            logger.info(f"📊 从JSON文件获得 {len(json_docs)} 个文档")

        # 从文本文件解析
        if os.path.exists(text_path):
            text_docs = importer.parse_text_file(text_path)
            documents.extend(text_docs)
            logger.info(f"📊 从文本文件获得 {len(text_docs)} 个文档")

        if not documents:
            logger.error("❌ 没有找到任何文档数据")
            return

        logger.info(f"📈 总共获得 {len(documents)} 个文档片段")

        # 导入到Elasticsearch
        importer.import_to_elasticsearch(documents)

        # 生成嵌入向量
        texts = [doc['content'] for doc in documents]
        embeddings = importer.generate_embeddings(texts)

        # 导入到Milvus
        importer.import_to_milvus(documents, embeddings)

        logger.info("🎉 数据导入完成！")

    except Exception as e:
        logger.error(f"❌ 数据导入过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()