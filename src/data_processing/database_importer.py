#!/usr/bin/env python3
"""
RAG系统数据库导入器
将PDF提取的文本数据导入Elasticsearch和Milvus
"""

import json
import re
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 数据库客户端
from elasticsearch import Elasticsearch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 嵌入模型
import sys
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')
from embedding.embedding_models import get_embedding_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """文本块数据结构"""
    id: str
    content: str
    page_number: int
    chapter_title: str
    section_title: str
    subsection_title: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1
    metadata: Dict[str, Any] = None

class DatabaseImporter:
    """数据库导入器类"""

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

        # 索引和集合名称
        self.es_index = "medical_documents"
        self.milvus_collection = "medical_vectors"

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
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "chapter_title": {"type": "keyword"},
                    "section_title": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
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
            # 继续执行，可能是索引已存在或其他非关键错误

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
            FieldSchema(name="page_number", dtype=DataType.INT32),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64)
        ]

        # 创建schema
        schema = CollectionSchema(fields, f"Medical document vectors for {self.milvus_collection}")

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

    def parse_structured_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        解析结构化的JSON文件

        Args:
            json_path: JSON文件路径

        Returns:
            解析后的文档数据列表
        """
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

                # 创建文档记录
                doc = {
                    'id': f"doc_{doc_id}",
                    'chapter_title': chapter_title,
                    'section_title': section_title,
                    'page_number': section.get('page', 0),
                    'content': f"{chapter_title} - {section_title}",
                    'metadata': {
                        'type': 'section',
                        'chapter': chapter_title,
                        'section': section_title,
                        'page': section.get('page', 0)
                    }
                }
                documents.append(doc)
                doc_id += 1

        # 解析页面内容
        for page in data.get('content_structure', {}).get('pages', []):
            page_number = page.get('page_number', 0)

            for block in page.get('text_blocks', []):
                text = block.get('text', '').strip()
                if text and len(text) > 10:  # 过滤短文本
                    doc = {
                        'id': f"doc_{doc_id}",
                        'page_number': page_number,
                        'content': text,
                        'metadata': {
                            'type': 'text_block',
                            'page': page_number,
                            'is_title': block.get('is_title', False),
                            'confidence': block.get('confidence', 0.0)
                        }
                    }
                    documents.append(doc)
                    doc_id += 1

        logger.info(f"✅ 解析完成，共获得 {len(documents)} 个文档片段")
        return documents

    def parse_text_file(self, text_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """
        解析文本文件并进行分块

        Args:
            text_path: 文本文件路径
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠大小

        Returns:
            分块后的文档数据列表
        """
        logger.info(f"📖 开始解析文本文件: {text_path}")

        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        documents = []
        doc_id = 0

        # 按页面分割
        page_pattern = r'#### 第(\d+)页'
        pages = re.split(page_pattern, text)

        current_page = 0
        for i in range(1, len(pages), 2):  # 跳过分割符
            if i + 1 < len(pages):
                page_num = int(pages[i])
                page_content = pages[i + 1]

                # 对页面内容进行分块 - 使用新的智能分块策略
                # 首先尝试按章节结构分割
                chapter_sections = self._split_by_chapter_structure(page_content)
                if len(chapter_sections) > 1:
                    # 如果成功按章节分割，对每个章节再进行智能分块
                    all_chunks = []
                    for section in chapter_sections:
                        if len(section) <= chunk_size:
                            all_chunks.append(section)
                        else:
                            # 对章节内部进行智能分块
                            section_chunks = self._split_section_intelligently(section, chunk_size, chunk_overlap)
                            all_chunks.extend(section_chunks)
                    chunks = all_chunks
                else:
                    # 如果无法按章节分割，使用改进的智能分块
                    chunks = self._split_section_intelligently(page_content, chunk_size, chunk_overlap)

                # 提取页面级别的章节信息
                page_chapter_info = self._extract_chapter_info_from_page(page_content)

                for chunk_idx, chunk in enumerate(chunks):
                    # 提取当前块的章节信息
                    chunk_chapter_info = self._extract_chapter_info_from_chunk(chunk, page_chapter_info)

                    doc = {
                        'id': f"doc_{doc_id}",
                        'page_number': page_num,
                        'content': chunk,
                        'chapter_title': chunk_chapter_info.get('chapter_title', ''),
                        'section_title': chunk_chapter_info.get('section_title', ''),
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'metadata': {
                            'type': 'text_chunk',
                            'page': page_num,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'chapter_path': chunk_chapter_info.get('chapter_path', ''),
                            'has_descriptive_content': chunk_chapter_info.get('has_descriptive_content', False),
                            'has_figure_numbers': chunk_chapter_info.get('has_figure_numbers', False)
                        }
                    }
                    documents.append(doc)
                    doc_id += 1

        logger.info(f"✅ 文本解析完成，共获得 {len(documents)} 个文档片段")
        return documents

    def _split_text_into_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """将文本分割成块 - 章节边界优先的动态分块策略"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # 如果这是最后一个块，直接添加
            if end >= len(text):
                chunks.append(text[start:])
                break

            # 尝试在句子边界分割
            chunk = text[start:end]

            # 查找最后一个句子结束符
            last_sentence_end = max(
                chunk.rfind('。'),
                chunk.rfind('！'),
                chunk.rfind('？'),
                chunk.rfind('\\n')
            )

            if last_sentence_end > chunk_size * 0.8:  # 如果找到合适的分割点
                actual_end = start + last_sentence_end + 1
                chunks.append(text[start:actual_end])
                start = actual_end - chunk_overlap
            else:
                chunks.append(chunk)
                start = end - chunk_overlap

        return chunks

    def _split_by_chapter_structure(self, text: str) -> List[str]:
        """按章节结构分割文本"""
        # 识别章节标题模式
        chapter_patterns = [
            r'##\s+第[一二三四五六七八九十]+节\s+[^\n]+',  # ## 第九节 黏液腺癌
            r'###\s+[^\n]+',  # ### 二级标题
            r'图\s*2-\d+\s*[^\n]+',  # 图号标题
        ]

        sections = []
        current_pos = 0

        for pattern in chapter_patterns:
            matches = list(re.finditer(pattern, text))
            if len(matches) > 1:
                # 找到多个匹配，按此模式分割
                for i, match in enumerate(matches):
                    start_pos = match.start()
                    if i < len(matches) - 1:
                        end_pos = matches[i + 1].start()
                    else:
                        end_pos = len(text)

                    section_text = text[start_pos:end_pos].strip()
                    if section_text:
                        sections.append(section_text)
                return sections

        # 如果没有找到章节结构，返回原文本
        return [text]

    def _split_section_intelligently(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """智能分块单个章节或段落"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # 如果这是最后一个块，直接添加
            if end >= len(text):
                chunks.append(text[start:])
                break

            # 查找描述性内容的分割点
            chunk = text[start:end]

            # 优先在描述段落边界分割
            # 1. 查找描述性段落的结束（分号后、句号后）
            desc_end_patterns = [
                r'；[^图]',  # 分号后不是图字
                r'。\s*[^图]',  # 句号后不是图字
                r'\n\s*',  # 换行
            ]

            best_split_pos = -1
            best_score = 0

            for pattern in desc_end_patterns:
                matches = list(re.finditer(pattern, chunk))
                for match in matches:
                    pos = match.end() - 1
                    if pos > chunk_size * 0.6:  # 在块的60%之后
                        score = pos / chunk_size  # 越靠后越好
                        if score > best_score:
                            best_score = score
                            best_split_pos = pos

            # 如果没有找到好的描述性分割点，使用句子边界
            if best_split_pos == -1:
                # 查找最后一个句子结束符
                last_sentence_end = max(
                    chunk.rfind('。'),
                    chunk.rfind('！'),
                    chunk.rfind('？'),
                    chunk.rfind('\n')
                )

                if last_sentence_end > chunk_size * 0.6:
                    best_split_pos = last_sentence_end + 1
                else:
                    # 强制在块大小处分割
                    best_split_pos = chunk_size

            actual_end = start + best_split_pos
            chunks.append(text[start:actual_end])
            start = actual_end - chunk_overlap

        return chunks

    def _extract_chapter_info_from_page(self, page_content: str) -> Dict[str, Any]:
        """从页面内容提取章节信息"""
        chapter_info = {
            'chapter_title': '',
            'section_title': '',
            'chapter_path': '',
            'has_descriptive_content': False,
            'has_figure_numbers': False
        }

        # 提取章节标题
        chapter_match = re.search(r'#\s+第[一二三四五六七八九十]+章\s+([^\n]+)', page_content)
        if chapter_match:
            chapter_info['chapter_title'] = chapter_match.group(1).strip()

        # 提取小节标题
        section_match = re.search(r'##\s+第[一二三四五六七八九十]+节\s+([^\n]+)', page_content)
        if section_match:
            chapter_info['section_title'] = section_match.group(1).strip()

        # 构建章节路径
        if chapter_info['chapter_title'] and chapter_info['section_title']:
            chapter_info['chapter_path'] = f"{chapter_info['chapter_title']} - {chapter_info['section_title']}"
        elif chapter_info['section_title']:
            chapter_info['chapter_path'] = chapter_info['section_title']

        # 检测内容类型
        chapter_info['has_descriptive_content'] = bool(re.search(r'[；。]\s*[^图]', page_content))
        chapter_info['has_figure_numbers'] = bool(re.search(r'图\s*2-\d+', page_content))

        return chapter_info

    def _extract_chapter_info_from_chunk(self, chunk: str, page_chapter_info: Dict[str, Any]) -> Dict[str, Any]:
        """从分块内容提取章节信息"""
        chunk_info = page_chapter_info.copy()

        # 重新检测当前块的具体内容类型
        chunk_info['has_descriptive_content'] = bool(re.search(r'[；。]\s*[^图]', chunk))
        chunk_info['has_figure_numbers'] = bool(re.search(r'图\s*2-\d+', chunk))

        # 如果当前块包含章节标题，更新章节信息
        section_match = re.search(r'##\s+第[一二三四五六七八九十]+节\s+([^\n]+)', chunk)
        if section_match:
            chunk_info['section_title'] = section_match.group(1).strip()
            if chunk_info['chapter_title']:
                chunk_info['chapter_path'] = f"{chunk_info['chapter_title']} - {chunk_info['section_title']}"
            else:
                chunk_info['chapter_path'] = chunk_info['section_title']

        return chunk_info

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

    def import_to_elasticsearch(self, documents: List[Dict[str, Any]]):
        """
        导入数据到Elasticsearch

        Args:
            documents: 文档数据列表
        """
        logger.info(f"📤 开始向Elasticsearch导入数据，共 {len(documents)} 个文档")

        success_count = 0
        error_count = 0

        for doc in documents:
            try:
                # 添加时间戳
                doc['created_at'] = datetime.now()

                # 索引文档
                self.es.index(index=self.es_index, id=doc['id'], body=doc)
                success_count += 1

                if success_count % 100 == 0:
                    logger.info(f"📊 已导入 {success_count} 个文档")

            except Exception as e:
                logger.error(f"❌ 导入文档 {doc['id']} 失败: {e}")
                error_count += 1

        logger.info(f"✅ Elasticsearch导入完成: 成功 {success_count}, 失败 {error_count}")

        # 刷新索引
        self.es.indices.refresh(index=self.es_index)

        # 显示索引统计
        stats = self.es.indices.stats(index=self.es_index)
        doc_count = stats['indices'][self.es_index]['primaries']['docs']['count']
        logger.info(f"📈 Elasticsearch索引文档总数: {doc_count}")

    def import_to_milvus(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        导入数据到Milvus

        Args:
            documents: 文档数据列表
            embeddings: 对应的嵌入向量列表
        """
        logger.info(f"📤 开始向Milvus导入数据，共 {len(documents)} 个向量")

        if len(documents) != len(embeddings):
            raise ValueError("文档数量和嵌入向量数量不匹配")

        # 准备数据
        ids = [doc['id'] for doc in documents]
        page_numbers = [doc.get('page_number', 0) for doc in documents]
        chunk_indices = [doc.get('chunk_index', 0) for doc in documents]

        # 生成内容哈希（简化版本）
        import hashlib
        content_hashes = []
        for doc in documents:
            content = doc.get('content', '')
            hash_obj = hashlib.md5(content.encode('utf-8'))
            content_hashes.append(hash_obj.hexdigest())

        # 插入数据
        entities = [
            ids,
            embeddings,
            page_numbers,
            chunk_indices,
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

    parser = argparse.ArgumentParser(description="导入PDF提取数据到数据库")
    parser.add_argument("--json", help="结构化JSON文件路径",
                       default="/home/ubuntu/myproject/zhenlikeji2/data/extracted/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224_final_structured.json")
    parser.add_argument("--text", help="纯文本文件路径",
                       default="/home/ubuntu/myproject/zhenlikeji2/data/extracted/text_stable/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224_extracted.txt")
    parser.add_argument("--embedding", choices=["jina", "openai"], default="jina",
                       help="嵌入模型类型")
    parser.add_argument("--use-json", action="store_true", default=True,
                       help="使用JSON文件")
    parser.add_argument("--use-text", action="store_true", default=True,
                       help="使用文本文件")

    args = parser.parse_args()

    # 创建导入器
    importer = DatabaseImporter(embedding_model=args.embedding)

    try:
        # 解析数据
        documents = []

        # 从JSON文件解析
        if args.use_json and os.path.exists(args.json):
            json_docs = importer.parse_structured_json(args.json)
            documents.extend(json_docs)
            logger.info(f"📊 从JSON文件获得 {len(json_docs)} 个文档")
        elif args.use_json:
            logger.warning(f"⚠️  JSON文件不存在: {args.json}")

        # 从文本文件解析
        if args.use_text and os.path.exists(args.text):
            text_docs = importer.parse_text_file(args.text)
            documents.extend(text_docs)
            logger.info(f"📊 从文本文件获得 {len(text_docs)} 个文档")
        elif args.use_text:
            logger.warning(f"⚠️  文本文件不存在: {args.text}")

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

    finally:
        importer.close()

if __name__ == "__main__":
    main()