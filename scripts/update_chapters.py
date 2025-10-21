#!/usr/bin/env python3
"""
章节信息更新脚本
为现有文档批量添加章节标题信息
"""

import sys
import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Any

# 添加项目路径
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')
sys.path.append('/home/ubuntu/myproject/zhenlikeji2')

try:
    from utils.chapter_extractor import chapter_extractor, extract_chapter_info
except ImportError:
    # 备用导入路径
    try:
        from src.utils.chapter_extractor import chapter_extractor, extract_chapter_info
    except ImportError:
        # 如果都失败了，创建一个简单的提取器
        class SimpleChapterExtractor:
            def extract_from_content(self, content: str, max_lines: int = 10):
                """简单的章节提取器"""
                lines = content.split('\n')[:max_lines]

                for line in lines:
                    line = line.strip()
                    # 查找章节标题
                    if re.search(r'第[一二三四五六七八九十百千万\d]+章', line):
                        return {'chapter_title': line, 'section_title': '', 'subsection_title': ''}
                    elif re.search(r'第[一二三四五六七八九十百千万\d]+节', line):
                        return {'chapter_title': '', 'section_title': line, 'subsection_title': ''}

                return {'chapter_title': '', 'section_title': '', 'subsection_title': ''}

        chapter_extractor = SimpleChapterExtractor()
        def extract_chapter_info(content: str, position: int = 0):
            return chapter_extractor.extract_from_content(content)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch
except ImportError:
    logger.error("❌ Elasticsearch客户端未安装")
    sys.exit(1)

class ChapterUpdater:
    """章节信息更新器"""

    def __init__(self, es_host: str = "elasticsearch", es_port: int = 9200):
        """初始化更新器"""
        self.es_host = es_host
        self.es_port = es_port
        self.es_index = "medical_documents_fixed"
        self.es_client = None
        self._connect_elasticsearch()

    def _connect_elasticsearch(self):
        """连接Elasticsearch"""
        try:
            self.es_client = Elasticsearch(
                hosts=[f"http://{self.es_host}:{self.es_port}"],
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )

            # 测试连接
            if self.es_client.ping():
                logger.info(f"✅ 连接到Elasticsearch: {self.es_host}:{self.es_port}")
            else:
                raise Exception("无法连接到Elasticsearch")

        except Exception as e:
            logger.error(f"❌ Elasticsearch连接失败: {e}")
            raise

    def get_documents_without_chapters(self, max_docs: int = 1000) -> List[Dict[str, Any]]:
        """获取没有章节信息的文档"""
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term": {"metadata.type": "text_chunk"}
                            }
                        ],
                        "must_not": [
                            {
                                "exists": {"field": "chapter_title"}
                            }
                        ]
                    }
                },
                "size": max_docs,
                "_source": ["id", "content", "page_number", "metadata"]
            }

            response = self.es_client.search(
                index=self.es_index,
                body=query
            )

            documents = []
            for hit in response['hits']['hits']:
                doc = {
                    'doc_id': hit['_id'],
                    'id': hit['_source'].get('id', ''),
                    'content': hit['_source'].get('content', ''),
                    'page_number': hit['_source'].get('page_number', 0),
                    'metadata': hit['_source'].get('metadata', {})
                }
                documents.append(doc)

            logger.info(f"📄 找到 {len(documents)} 个需要更新的文档")
            return documents

        except Exception as e:
            logger.error(f"❌ 查询文档失败: {e}")
            return []

    def extract_chapter_from_document(self, doc: Dict[str, Any]) -> Dict[str, str]:
        """从文档内容提取章节信息"""
        try:
            content = doc.get('content', '')

            # 使用章节提取器
            chapter_info = extract_chapter_info(content, 0)

            # 如果没有提取到章节，尝试从内容开头查找
            if not chapter_info['chapter_title'] and not chapter_info['section_title']:
                # 查找明显的章节标题
                lines = content.split('\n')[:15]  # 检查前15行
                for line in lines:
                    line = line.strip()
                    if len(line) > 5 and len(line) < 100:  # 合理长度
                        # 查找章节模式
                        if re.search(r'第[一二三四五六七八九十百千万\d]+章', line):
                            chapter_info['chapter_title'] = line
                            break
                        elif re.search(r'第[一二三四五六七八九十百千万\d]+节', line):
                            chapter_info['section_title'] = line
                            break
                        # 查找标题模式（以#开头）
                        elif line.startswith('#') and len(line.strip('#').strip()) > 0:
                            title_text = line.strip('#').strip()
                            if '章' in title_text:
                                chapter_info['chapter_title'] = title_text
                            elif '节' in title_text:
                                chapter_info['section_title'] = title_text
                            break

            return chapter_info

        except Exception as e:
            logger.error(f"❌ 提取章节信息失败: {e}")
            return {'chapter_title': '', 'section_title': '', 'subsection_title': ''}

    def update_document_chapter(self, doc_id: str, chapter_info: Dict[str, str]) -> bool:
        """更新单个文档的章节信息"""
        try:
            update_body = {
                "doc": {
                    "chapter_title": chapter_info.get('chapter_title', ''),
                    "section_title": chapter_info.get('section_title', ''),
                    "subsection_title": chapter_info.get('subsection_title', ''),
                    "metadata": {
                        "chapter_updated": datetime.now().isoformat(),
                        "has_chapter_info": bool(chapter_info.get('chapter_title') or chapter_info.get('section_title'))
                    }
                }
            }

            response = self.es_client.update(
                index=self.es_index,
                id=doc_id,
                body=update_body
            )

            if response.get('result') in ['updated', 'noop']:
                return True
            else:
                logger.warning(f"⚠️ 文档更新结果异常: {response.get('result')}")
                return False

        except Exception as e:
            logger.error(f"❌ 更新文档失败 (ID: {doc_id}): {e}")
            return False

    def update_all_documents(self, batch_size: int = 50) -> Dict[str, int]:
        """批量更新所有文档的章节信息"""
        logger.info("🚀 开始批量更新文档章节信息...")

        stats = {
            'total_processed': 0,
            'successfully_updated': 0,
            'failed': 0,
            'has_chapter_info': 0
        }

        try:
            # 获取需要更新的文档
            documents = self.get_documents_without_chapters(10000)

            if not documents:
                logger.info("📋 没有需要更新的文档")
                return stats

            logger.info(f"📝 开始处理 {len(documents)} 个文档...")

            for i, doc in enumerate(documents):
                try:
                    # 提取章节信息
                    chapter_info = self.extract_chapter_from_document(doc)

                    # 更新文档
                    if self.update_document_chapter(doc['doc_id'], chapter_info):
                        stats['successfully_updated'] += 1
                        if chapter_info.get('chapter_title') or chapter_info.get('section_title'):
                            stats['has_chapter_info'] += 1
                    else:
                        stats['failed'] += 1

                    stats['total_processed'] += 1

                    # 批量处理日志
                    if (i + 1) % batch_size == 0:
                        logger.info(f"📊 进度: {i + 1}/{len(documents)} 已处理")

                except Exception as e:
                    logger.error(f"❌ 处理文档失败 (索引: {i}): {e}")
                    stats['failed'] += 1
                    stats['total_processed'] += 1

            logger.info(f"✅ 批量更新完成!")
            logger.info(f"📊 统计信息: {stats}")
            return stats

        except Exception as e:
            logger.error(f"❌ 批量更新过程失败: {e}")
            return stats

    def test_extraction(self, sample_content: str):
        """测试章节提取功能"""
        logger.info("🧪 测试章节提取功能...")

        test_doc = {
            'content': sample_content,
            'page_number': 1,
            'metadata': {}
        }

        chapter_info = self.extract_chapter_from_document(test_doc)
        logger.info(f"📋 提取结果: {chapter_info}")
        return chapter_info

    def close(self):
        """关闭连接"""
        if self.es_client:
            self.es_client.close()
            logger.info("🔌 Elasticsearch连接已关闭")

def main():
    """主函数"""
    try:
        # 创建更新器
        updater = ChapterUpdater()

        # 测试提取功能
        test_content = """
        第四章 肺脏少见病的ROSE组学特征

        第六节 急性纤维素性机化性肺炎

        急性纤维素性机化性肺炎（AFOP）是一种罕见的急性或亚急性弥漫性肺疾病...
        """
        updater.test_extraction(test_content)

        # 批量更新所有文档
        logger.info("🚀 开始批量更新文档章节信息...")
        stats = updater.update_all_documents(batch_size=25)

        logger.info(f"🎉 任务完成! 统计信息: {stats}")

    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        sys.exit(1)

    finally:
        if 'updater' in locals():
            updater.close()

if __name__ == "__main__":
    main()