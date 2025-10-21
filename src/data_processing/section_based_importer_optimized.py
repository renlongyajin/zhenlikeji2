#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG系统数据库导入器（优化版）
改进包括：
1. 数据清洗（去除Markdown标记）
2. 支持中文分词器（IK/SmartCN）
3. 多字段索引（原始内容+清洗内容）
4. 更好的错误处理和性能优化
"""

import json
import logging
import argparse
import re
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
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.embedding.embedding_models import get_embedding_manager

# --- 配置日志 ---
log_dir = Path(__file__).parent.parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'section_import_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)


class ContentCleaner:
    """内容清洗器"""

    @staticmethod
    def clean_markdown(content: str) -> str:
        """
        清洗Markdown格式标记

        Args:
            content: 原始内容

        Returns:
            清洗后的纯文本内容
        """
        if not content:
            return content

        # 去除标题标记 (# ## ###)
        content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*#+\s+', '', content, flags=re.MULTILINE)

        # 去除粗体和斜体标记
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^*]+)\*', r'\1', content)

        # 去除链接和图片标记
        content = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', content)
        content = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', content)

        # 去除图片引用 (图2-1 腺癌（分化较高）（1）)
        content = re.sub(r'图\d+-\d+[^)]*\)', '', content)
        content = re.sub(r'\(图[^)]*\)', '', content)

        # 去除多余空白字符
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)

        return content.strip()

    @staticmethod
    def extract_keywords(content: str) -> List[str]:
        """
        提取关键词（用于增强检索）

        Args:
            content: 内容

        Returns:
            关键词列表
        """
        # 简单的医学关键词提取
        medical_terms = [
            '腺癌', '鳞癌', '小细胞癌', '大细胞癌', '类癌', '肉瘤样癌',
            '肺泡', '支气管', '细胞', '肿瘤', '恶性', '良性',
            '分化', '核分裂', '核仁', '染色质', '细胞质', '细胞核'
        ]

        found_keywords = []
        content_lower = content.lower()
        for term in medical_terms:
            if term in content_lower:
                found_keywords.append(term)

        return found_keywords


class OptimizedSectionImporter:
    """
    优化的基于小节分块的数据库导入器
    """

    def __init__(self,
                 es_host: str = "localhost",
                 es_port: int = 9200,
                 milvus_host: str = "localhost",
                 milvus_port: int = 19530,
                 embedding_model: str = "jina",
                 chinese_analyzer: str = "ik_max_word"):
        """
        初始化数据库连接和嵌入模型

        Args:
            es_host (str): Elasticsearch主机地址
            es_port (int): Elasticsearch端口
            milvus_host (str): Milvus主机地址
            milvus_port (int): Milvus端口
            embedding_model (str): 嵌入模型类型 (例如: "jina", "openai")
            chinese_analyzer (str): 中文分词器类型 (例如: "ik_max_word", "smartcn")
        """
        # 索引和集合名称
        self.es_index = "medical_sections_optimized"
        self.milvus_collection = "medical_section_vectors_optimized"
        self.es = None
        self.collection = None
        self.content_cleaner = ContentCleaner()
        self.chinese_analyzer = chinese_analyzer

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

            # 动态检测嵌入向量的维度
            logger.info("动态检测嵌入向量维度...")
            sample_embedding = self.embedding_manager.encode_texts(["test"])[0]
            self.vector_dim = len(sample_embedding)
            logger.info(f"✅ 初始化嵌入模型: {embedding_model} (维度: {self.vector_dim})")

            # 确保索引和集合存在
            self._create_elasticsearch_index()
            self._create_milvus_collection()

        except Exception as e:
            logger.error(f"❌ 初始化导入器失败: {e}")
            raise

    def _create_elasticsearch_index(self):
        """创建优化的Elasticsearch索引"""
        index_mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "chinese_analyzer": {
                            "type": "custom",
                            "tokenizer": self.chinese_analyzer,
                            "filter": ["lowercase", "stop"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "chinese_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "path": {
                        "type": "keyword"
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "chinese_analyzer"
                    },
                    "content_clean": {
                        "type": "text",
                        "analyzer": "chinese_analyzer"
                    },
                    "keywords": {
                        "type": "keyword"
                    },
                    "content_length": {
                        "type": "integer"
                    },
                    "created_at": {
                        "type": "date"
                    }
                }
            }
        }

        try:
            if not self.es.indices.exists(index=self.es_index):
                self.es.indices.create(index=self.es_index, body=index_mapping)
                logger.info(f"✅ 创建优化Elasticsearch索引: {self.es_index}")
                logger.info(f"   使用分词器: {self.chinese_analyzer}")
            else:
                logger.info(f"ℹ️ 优化Elasticsearch索引已存在: {self.es_index}")
        except Exception as e:
            logger.error(f"⚠️ Elasticsearch索引创建失败: {e}", exc_info=True)
            raise

    def _create_milvus_collection(self):
        """创建优化的Milvus集合"""
        if utility.has_collection(self.milvus_collection):
            logger.info(f"ℹ️ 优化Milvus集合已存在: {self.milvus_collection}")
            self.collection = Collection(name=self.milvus_collection)
            self.collection.load()
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        schema = CollectionSchema(fields, "Medical sections optimized vector collection")
        self.collection = Collection(name=self.milvus_collection, schema=schema)

        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index("vector", index_params)
        logger.info(f"✅ 创建优化Milvus集合: {self.milvus_collection}")
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

    def process_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理和清洗数据

        Args:
            sections: 原始数据块

        Returns:
            处理后的数据块
        """
        logger.info(f"🧹 开始处理和清洗数据...")
        processed_sections = []

        for section in sections:
            try:
                # 清洗内容
                content_clean = self.content_cleaner.clean_markdown(section['content'])

                # 提取关键词
                keywords = self.content_cleaner.extract_keywords(content_clean)

                # 计算内容长度
                content_length = len(content_clean.split())

                # 创建增强的数据结构
                enhanced_section = {
                    **section,
                    'content_clean': content_clean,
                    'keywords': keywords,
                    'content_length': content_length,
                    'has_markdown': section['content'] != content_clean
                }

                processed_sections.append(enhanced_section)

            except Exception as e:
                logger.warning(f"⚠️ 处理数据块失败 {section.get('id', 'unknown')}: {e}")
                # 使用原始数据作为后备
                processed_sections.append(section)

        # 统计清洗效果
        cleaned_count = sum(1 for s in processed_sections if s.get('has_markdown', False))
        avg_length_before = sum(len(s['content'].split()) for s in sections) / len(sections)
        avg_length_after = sum(s.get('content_length', len(s['content'].split())) for s in processed_sections) / len(processed_sections)

        logger.info(f"✅ 数据处理完成:")
        logger.info(f"   - 处理的数据块: {len(processed_sections)}")
        logger.info(f"   - 清洗的数据块: {cleaned_count}")
        logger.info(f"   - 平均长度变化: {avg_length_before:.1f} -> {avg_length_after:.1f} 词")

        return processed_sections

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
        """批量导入数据到优化后的Elasticsearch"""
        if not sections:
            return
        logger.info(f"📤 开始向优化Elasticsearch导入数据，共 {len(sections)} 个文档...")

        actions = [
            {
                "_index": self.es_index,
                "_id": section['id'],
                "_source": {
                    "title": section['title'],
                    "path": section['path'],
                    "content": section['content'],  # 原始内容
                    "content_clean": section.get('content_clean', section['content']),  # 清洗内容
                    "keywords": section.get('keywords', []),
                    "content_length": section.get('content_length', len(section['content'].split())),
                    "created_at": datetime.now().isoformat()
                }
            }
            for section in sections
        ]

        try:
            success, failed = bulk(self.es, actions, raise_on_error=False, chunk_size=100)
            logger.info(f"✅ 优化Elasticsearch导入完成: 成功 {success} 个, 失败 {len(failed)} 个")
            if failed:
                logger.warning(f"部分文档导入失败详情: {failed[:3]}")  # 只显示前3个错误
            self.es.indices.refresh(index=self.es_index)
        except Exception as e:
            logger.error(f"❌ 优化Elasticsearch批量导入失败: {e}", exc_info=True)
            raise

    def import_to_milvus(self, sections: List[Dict[str, Any]], embeddings: List[List[float]]):
        """导入数据到优化后的Milvus"""
        if not sections or not embeddings:
            return
        if len(sections) != len(embeddings):
            raise ValueError("数据块数量和嵌入向量数量不匹配")

        logger.info(f"📤 开始向优化Milvus导入数据，共 {len(sections)} 个向量...")

        entities = [
            [section['id'] for section in sections],  # ids
            embeddings  # vectors
        ]

        try:
            self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"✅ 优化Milvus数据导入完成")
            logger.info(f"📈 优化Milvus集合向量总数: {self.collection.num_entities}")
        except Exception as e:
            logger.error(f"❌ 优化Milvus数据导入失败: {e}", exc_info=True)
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
        description="将分块后的JSON数据导入优化的Elasticsearch和Milvus",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置导入
  python3 %(prog)s data/processed_chunks.json

  # 指定中文分词器
  python3 %(prog)s data/processed_chunks.json --chinese-analyzer ik_max_word

  # 指定嵌入模型和数据库地址
  python3 %(prog)s data/processed_chunks.json \
    --embedding jina \
    --chinese-analyzer smartcn \
    --es-host localhost \
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
    parser.add_argument(
        "--chinese-analyzer",
        choices=["ik_max_word", "ik_smart", "smartcn", "standard"],
        default="standard",
        help="中文分词器类型 (默认: standard, 推荐: ik_max_word)"
    )
    parser.add_argument("--es-host", default="localhost", help="Elasticsearch主机")
    parser.add_argument("--es-port", type=int, default=9200, help="Elasticsearch端口")
    parser.add_argument("--milvus-host", default="localhost", help="Milvus主机")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus端口")

    args = parser.parse_args()

    importer = None
    try:
        # 初始化导入器
        importer = OptimizedSectionImporter(
            es_host=args.es_host,
            es_port=args.es_port,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            embedding_model=args.embedding,
            chinese_analyzer=args.chinese_analyzer
        )

        # 1. 加载数据
        sections = importer.load_sections_from_json(args.json_file)
        if not sections:
            logger.warning("⚠️ 未加载到任何数据，程序退出。")
            return

        # 2. 处理和清洗数据
        processed_sections = importer.process_sections(sections)

        # 3. 导入到Elasticsearch
        importer.import_to_elasticsearch(processed_sections)

        # 4. 生成嵌入向量
        contents = [section.get('content_clean', section['content']) for section in processed_sections]
        embeddings = importer.generate_embeddings(contents)

        # 5. 导入到Milvus
        importer.import_to_milvus(processed_sections, embeddings)

        logger.info("\n🎉 所有优化数据导入流程成功完成！")
        logger.info(f"📊 总计导入: {len(processed_sections)} 个数据块")

    except Exception as e:
        logger.error(f"\n❌ 优化导入流程发生严重错误: {e}", exc_info=True)
    finally:
        if importer:
            importer.close()


if __name__ == "__main__":
    main()