#!/usr/bin/env python3
"""
搜索包含章节标题模式的文档
"""

import sys
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

import logging
import os
import requests
import json
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_chapter_title_pages():
    """查找包含章节标题的页面"""

    logger.info("=" * 80)
    logger.info("搜索包含章节标题模式的文档")
    logger.info("=" * 80)

    try:
        # 直接查询Elasticsearch
        es_base_url = f"http://{os.getenv('ELASTICSEARCH_HOST', 'localhost')}:{os.getenv('ELASTICSEARCH_PORT', '9200')}"
        es_index = "medical_documents"

        # 搜索包含"第一节"模式的文档
        search_body = {
            "query": {
                "match_phrase": {
                    "content": "第一节"
                }
            },
            "size": 20,
            "_source": ["page_number", "content"]
        }

        response = requests.post(
            f"{es_base_url}/{es_index}/_search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_body),
            timeout=10
        )

        if response.status_code == 200:
            results = response.json()
            hits = results['hits']['hits']

            logger.info(f"找到 {len(hits)} 个包含'第一节'的文档\n")

            for i, hit in enumerate(hits, 1):
                source = hit['_source']
                content = source.get('content', '')
                page_number = source.get('page_number', 0)
                score = hit['_score']

                logger.info(f"文档 {i} (第{page_number}页, 得分: {score}):")
                logger.info(f"内容长度: {len(content)} 字符")

                # 检查是否包含腺癌、鳞癌等
                entities = ["腺癌", "鳞癌", "小细胞癌", "黏液腺癌"]
                found_entities = [e for e in entities if e in content]
                if found_entities:
                    logger.info(f"包含实体: {found_entities}")

                # 显示内容的前800字符
                logger.info(f"内容预览:")
                logger.info(content[:800])

                # 尝试匹配章节标题
                patterns = [
                    r'##\s*第一节\s*\n\n([^\n\r]+)',
                    r'第一节\s*\n\n([^\n\r]+)',
                    r'第一节\s+([^\n\r]+)',
                ]

                for pattern in patterns:
                    match = re.search(pattern, content)
                    if match:
                        logger.info(f"  ✅ 匹配章节标题 (模式: {pattern}): {match.group(1)}")

                logger.info("-" * 80)

        logger.info("\n" + "=" * 80)
        logger.info("✅ 搜索完成")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 搜索失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    find_chapter_title_pages()