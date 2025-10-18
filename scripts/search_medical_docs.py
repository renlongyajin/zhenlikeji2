#!/usr/bin/env python3
"""
医学文档搜索展示工具
以友好的中文格式展示搜索结果
"""

import json
import requests
import sys
from typing import List, Dict, Any

def search_medical_documents(query: str, size: int = 5):
    """搜索医学文档并以中文格式展示结果"""

    print(f"🔍 搜索医学文档: '{query}'")
    print("=" * 60)

    try:
        response = requests.post(
            'http://localhost:9200/medical_documents/_search',
            json={
                'query': {
                    'match': {
                        'content': query
                    }
                },
                'size': size,
                '_source': ['id', 'page_number', 'content', 'chapter_title', 'section_title', 'created_at']
            },
            timeout=10
        )

        if response.status_code != 200:
            print(f"❌ 搜索失败: HTTP {response.status_code}")
            return

        results = response.json()
        total_hits = results['hits']['total']['value']
        max_score = results['hits']['max_score']
        hits = results['hits']['hits']

        print(f"📊 找到 {total_hits} 个相关文档")
        print(f"🎯 最高匹配分数: {max_score:.3f}")
        print()

        if not hits:
            print("⚠️ 未找到相关文档")
            return

        for i, hit in enumerate(hits, 1):
            score = hit['_score']
            source = hit['_source']

            print(f"📄 结果 {i} (匹配分数: {score:.3f})")
            print(f"   📖 ID: {source['id']}")
            print(f"   📄 页面: {source.get('page_number', '未知')}")
            print(f"   📝 章节: {source.get('chapter_title', '无')}")
            print(f"   📋 小节: {source.get('section_title', '无')}")

            # 显示内容预览
            content = source.get('content', '')
            if len(content) > 200:
                content_preview = content[:200] + "..."
            else:
                content_preview = content

            print(f"   📑 内容预览:")
            # 处理内容中的换行符
            for line in content_preview.split('\n'):
                if line.strip():
                    print(f"      {line.strip()}")

            print("-" * 50)

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Elasticsearch服务")
        print("💡 请确保Elasticsearch正在运行 (http://localhost:9200)")
    except Exception as e:
        print(f"❌ 搜索过程出错: {str(e)}")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("🔍 医学文档搜索工具")
        print("=" * 40)
        print("使用方法: python3 search_medical_docs.py '搜索关键词'")
        print("示例: python3 search_medical_docs.py '肺部恶性肿瘤'")
        print()

        # 提供一些示例搜索
        examples = [
            "肺部恶性肿瘤",
            "ROSE细胞学",
            "腺癌特征",
            "细胞核增大",
            "快速现场评价"
        ]

        print("📋 示例搜索词:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")

        print("\n💡 您可以直接运行: python3 search_medical_docs.py '示例词'")
        return

    query = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    search_medical_documents(query, size)

if __name__ == "__main__":
    main()