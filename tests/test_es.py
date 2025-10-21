#!/usr/bin/env python3
"""
从Elasticsearch检索数据 (关键词搜索)
"""
import json
from elasticsearch import Elasticsearch

# --- 配置 ---
ES_HOST = "localhost"
ES_PORT = 9200
INDEX_NAME = "medical_documents_simple" # 确保这和您导入时用的索引名称一致

# 您的搜索查询
QUERY_TEXT = "黏液腺癌" # 试试您文档中的关键词

# ----------------

def search_elasticsearch(query: str):
    """
    连接到ES并执行关键词搜索
    """
    try:
        # 1. 连接Elasticsearch
        es = Elasticsearch([f"http://{ES_HOST}:{ES_PORT}"])
        if not es.ping():
            print("❌ Elasticsearch连接失败")
            return

        print(f"✅ Elasticsearch连接成功，正在索引 '{INDEX_NAME}' 中搜索...")

        # 2. 构建查询
        # 使用 "match" 查询，这是ES中最标准的全文搜索
        search_body = {
            "query": {
                "match": {
                    "content": query 
                }
            }
        }

        # 3. 执行搜索
        # "size=3" 表示我们希望返回最相关的3个结果
        response = es.search(index=INDEX_NAME, body=search_body, size=3)

        # 4. 解析并打印结果
        hits = response.get('hits', {}).get('hits', [])
        
        if not hits:
            print(f"🤷 未找到关于 '{query}' 的结果。")
            return

        print(f"🎉 找到 {len(hits)} 条相关结果 (总共 {response['hits']['total']['value']} 条匹配):")
        
        for i, hit in enumerate(hits):
            print(f"\n--- 结果 {i+1} (相关度: {hit['_score']}) ---")
            
            # 从 _source 中提取内容
            source = hit.get('_source', {})
            content = source.get('content', 'N/A')
            chapter = source.get('chapter_title', 'N/A')
            section = source.get('section_title', 'N/A')

            # 打印摘要
            print(f"  章节: {chapter}")
            print(f"  小节: {section}")
            print(f"  内容 (摘要): {content[:200]}...") # 打印前200个字符

    except Exception as e:
        print(f"❌ 搜索时发生错误: {e}")

if __name__ == "__main__":
    search_elasticsearch(QUERY_TEXT)