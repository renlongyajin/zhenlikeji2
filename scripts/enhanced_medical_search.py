#!/usr/bin/env python3
"""
增强版医学文档搜索工具
解决基础概念 vs 具体亚型的搜索排名问题
"""

import json
import requests
import sys
from typing import List, Dict, Any

def enhanced_medical_search(query: str, size: int = 10):
    """增强版医学搜索，优化基础概念的排名"""

    print(f"🔍 增强搜索医学文档: '{query}'")
    print("=" * 60)

    try:
        # 策略1: 多字段加权搜索
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        # 标题字段高权重 - 基础概念通常在标题中
                        {
                            "match": {
                                "content": {
                                    "query": f"# {query}",
                                    "boost": 3.0
                                }
                            }
                        },
                        # 章节标题匹配
                        {
                            "match": {
                                "content": {
                                    "query": f"## {query}",
                                    "boost": 2.0
                                }
                            }
                        },
                        # 普通内容匹配（降低权重避免噪音）
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "boost": 1.0
                                }
                            }
                        },
                        # 精确短语匹配
                        {
                            "match_phrase": {
                                "content": {
                                    "query": query,
                                    "boost": 2.5
                                }
                            }
                        }
                    ],
                    "must_not": [
                        # 排除过于具体的亚型（可选）
                        {
                            "match": {
                                "content": "腺泡细胞癌"
                            }
                        }
                    ] if query == "腺癌" else {}
                }
            },
            "size": size,
            "_source": ["id", "page_number", "content", "chapter_title", "section_title", "created_at"],
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 200,
                        "number_of_fragments": 3
                    }
                }
            }
        }

        response = requests.post(
            'http://localhost:9200/medical_documents/_search',
            json=search_body,
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
        print(f"🎯 最高匹配分数: {max_score:.3f}" if max_score else "🎯 最高匹配分数: 无")
        print()

        if not hits:
            print("⚠️ 未找到相关文档")
            return

        print("🔍 搜索结果分析:")
        print("-" * 60)

        for i, hit in enumerate(hits, 1):
            score = hit['_score']
            source = hit['_source']

            # 分析内容类型
            content = source.get('content', '')
            content_type = analyze_content_type(content, query)

            print(f"📄 结果 {i} (匹配分数: {score:.3f}) [{content_type}]")
            print(f"   📖 ID: {source['id']}")
            print(f"   📄 页面: {source.get('page_number', '未知')}")
            print(f"   🏷️  类型: {content_type}")

            # 智能内容预览
            content_preview = get_smart_preview(content, query)
            print(f"   📑 内容预览:")
            for line in content_preview:
                if line.strip():
                    print(f"      {line.strip()}")

            print("-" * 50)

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Elasticsearch服务")
        print("💡 请确保Elasticsearch正在运行 (http://localhost:9200)")
    except Exception as e:
        print(f"❌ 搜索过程出错: {str(e)}")

def analyze_content_type(content: str, query: str) -> str:
    """分析内容类型"""
    if f"# {query}" in content:
        return "🔴 基础概念章节"
    elif f"## {query}" in content:
        return "🟡 子章节标题"
    elif f"### {query}" in content:
        return "🟢 小节标题"
    elif query in content[:50]:
        return "🔵 开篇提及"
    elif f"图" in content and query in content:
        return "🟣 图示说明"
    else:
        return "⚪ 一般提及"

def get_smart_preview(content: str, query: str, max_lines: int = 5) -> List[str]:
    """智能获取内容预览"""
    lines = content.split('\n')

    # 查找查询词出现的位置
    query_lines = []
    for i, line in enumerate(lines):
        if query in line:
            # 获取上下文（前后各2行）
            start = max(0, i-2)
            end = min(len(lines), i+3)
            context = lines[start:end]
            query_lines.extend(context)
            query_lines.append("...")
            break

    if query_lines:
        return query_lines[:max_lines]
    else:
        # 如果没有找到具体行，返回开头内容
        return lines[:max_lines]

def test_search_quality():
    """测试搜索质量改进"""
    print("🧪 搜索质量对比测试")
    print("=" * 60)

    test_queries = [
        "腺癌",
        "肺部恶性肿瘤",
        "ROSE细胞学",
        "细胞核增大"
    ]

    for query in test_queries:
        print(f"\n📋 测试查询: '{query}'")
        print("-" * 40)

        # 运行增强搜索
        enhanced_medical_search(query, size=5)
        print("\n" + "="*60)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("🔍 增强版医学文档搜索工具")
        print("=" * 40)
        print("使用方法: python3 enhanced_medical_search.py '搜索关键词'")
        print("示例: python3 enhanced_medical_search.py '腺癌'")
        print()
        print("🎯 特殊功能:")
        print("  • 优化基础概念排名")
        print("  • 区分概念层次（基础vs亚型）")
        print("  • 智能内容预览")
        print("  • 内容类型标记")
        return

    if sys.argv[1] == "--test":
        test_search_quality()
    else:
        query = sys.argv[1]
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        enhanced_medical_search(query, size)

if __name__ == "__main__":
    main()