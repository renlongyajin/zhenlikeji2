#!/usr/bin/env python3
"""
直接调试章节查询 - 查看Elasticsearch响应
"""

import sys
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

import logging
import os
import requests
import json

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_chapter_query():
    """直接调试章节查询"""

    logger.info("=" * 80)
    logger.info("直接调试章节查询")
    logger.info("=" * 80)

    try:
        # 直接查询Elasticsearch
        es_base_url = f"http://{os.getenv('ELASTICSEARCH_HOST', 'localhost')}:{os.getenv('ELASTICSEARCH_PORT', '9200')}"
        es_index = "medical_documents"

        # 测试腺癌的章节查询
        entity = "腺癌"

        # 构建查询 - 匹配"第X节{entity}"模式
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "content": entity
                            }
                        }
                    ],
                    "should": [
                        {
                            "match_phrase": {
                                "content": {
                                    "query": f"第一节{entity}",
                                    "boost": 1000.0
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "content": {
                                    "query": f"第一节 {entity}",
                                    "boost": 800.0
                                }
                            }
                        }
                    ],
                    "must_not": [
                        {
                            "match_phrase": {
                                "content": "录\n\n日"
                            }
                        }
                    ]
                }
            },
            "size": 10,
            "_source": ["page_number", "content"]
        }

        logger.info(f"查询实体: '{entity}'")
        logger.info(f"查询体: {json.dumps(search_body, ensure_ascii=False, indent=2)}")

        response = requests.post(
            f"{es_base_url}/{es_index}/_search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(search_body),
            timeout=10
        )

        if response.status_code == 200:
            results = response.json()
            hits = results['hits']['hits']

            logger.info(f"找到 {len(hits)} 个结果")

            for i, hit in enumerate(hits, 1):
                source = hit['_source']
                content = source.get('content', '')
                page_number = source.get('page_number', 0)
                score = hit['_score']

                logger.info(f"\n结果 {i} (第{page_number}页, 得分: {score}):")
                logger.info(f"内容长度: {len(content)} 字符")

                # 检查是否包含目标模式
                patterns_to_check = [
                    f"第一节{entity}",
                    f"第一节 {entity}",
                    f"第.*节{entity}",
                    f"第.*节 {entity}"
                ]

                for pattern in patterns_to_check:
                    if pattern in content:
                        logger.info(f"  ✅ 找到模式: '{pattern}'")
                        # 显示上下文
                        pos = content.find(pattern)
                        start = max(0, pos - 50)
                        end = min(len(content), pos + 50)
                        context = content[start:end]
                        logger.info(f"  上下文: {context}")

                # 显示内容预览
                logger.info(f"内容预览 (前200字符):")
                logger.info(content[:200])

        else:
            logger.error(f"查询失败: {response.status_code}")
            logger.error(response.text)

        logger.info("\n" + "=" * 80)
        logger.info("✅ 调试完成")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chapter_query()