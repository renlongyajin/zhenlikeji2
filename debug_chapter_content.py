#!/usr/bin/env python3
"""
调试章节内容提取
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

def debug_chapter_content():
    """调试章节内容提取"""

    logger.info("=" * 80)
    logger.info("调试章节内容提取")
    logger.info("=" * 80)

    try:
        # 直接查询Elasticsearch
        es_base_url = f"http://{os.getenv('ELASTICSEARCH_HOST', 'localhost')}:{os.getenv('ELASTICSEARCH_PORT', '9200')}"
        es_index = "medical_documents"

        # 查询腺癌相关内容
        search_body = {
            "query": {
                "match": {
                    "content": "腺癌"
                }
            },
            "size": 10,
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

            logger.info(f"找到 {len(hits)} 个包含'腺癌'的文档")

            for i, hit in enumerate(hits, 1):
                source = hit['_source']
                content = source.get('content', '')
                page_number = source.get('page_number', 0)
                score = hit['_score']

                logger.info(f"\n文档 {i} (第{page_number}页, 得分: {score}):")
                logger.info(f"内容长度: {len(content)} 字符")
                logger.info(f"内容预览 (前500字符):")
                logger.info(content[:500])

                # 手动提取章节信息
                logger.info(f"\n手动章节提取:")

                # 查看前1000字符中的章节模式
                first_part = content[:1000]

                # 查找各种章节模式
                patterns = [
                    r'##\s*第([一二三四五六七八九十]+)节\s*([^\n\r]+)',
                    r'##\s*第(\d+)节\s*([^\n\r]+)',
                    r'第([一二三四五六七八九十]+)节\s*([^\n\r]+)',
                    r'第(\d+)节\s*([^\n\r]+)',
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, first_part)
                    if matches:
                        logger.info(f"  模式 '{pattern}' 找到匹配:")
                        for match in matches:
                            logger.info(f"    第{match[0]}节: {match[1].strip()}")

                # 查找实体出现位置
                adenocarcinoma_pos = content.find("腺癌")
                if adenocarcinoma_pos != -1:
                    logger.info(f"  '腺癌' 出现在位置 {adenocarcinoma_pos}")
                    # 显示上下文
                    start = max(0, adenocarcinoma_pos - 100)
                    end = min(len(content), adenocarcinoma_pos + 100)
                    context = content[start:end]
                    logger.info(f"  上下文: {context}")

                if i >= 3:  # 只检查前3个文档
                    break

        logger.info("\n" + "=" * 80)
        logger.info("✅ 调试完成")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chapter_content()