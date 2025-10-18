#!/usr/bin/env python3
"""
智能医学搜索工具 - 专门解决基础概念排名问题
"""

import json
import requests
import sys
from typing import List, Dict, Any

def smart_medical_search(query: str, size: int = 10):
    """智能医学搜索，优化基础概念排名"""

    print(f"🔍 智能搜索医学文档: '{query}'")
    print("=" * 60)

    try:
        # 策略1: 标题优先搜索
        title_priority_search = {
            "query": {
                "bool": {
                    "should": [
                        # 主标题匹配 - 最高优先级
                        {
                            "match_phrase": {
                                "content": {
                                    "query": f"# {query}",
                                    "boost": 5.0
                                }
                            }
                        },
                        # 节标题匹配
                        {
                            "match_phrase": {
                                "content": {
                                    "query": f"## {query}",
                                    "boost": 3.0
                                }
                            }
                        },
                        # 小节标题匹配
                        {
                            "match_phrase": {
                                "content": {
                                    "query": f"### {query}",
                                    "boost": 2.0
                                }
                            }
                        },
                        # 普通内容匹配
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "boost": 1.0
                                }
                            }
                        }
                    ]
                }
            },
            "size": size,
            "_source": ["id", "page_number", "content"],
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 2,
                        "pre_tags": ["***"],
                        "post_tags": ["***"]
                    }
                }
            }
        }

        response = requests.post(
            'http://localhost:9200/medical_documents/_search',
            json=title_priority_search,
            timeout=10
        )

        if response.status_code != 200:
            print(f"❌ 搜索失败: HTTP {response.status_code}")
            return

        results = response.json()
        total_hits = results['hits']['total']['value']
        hits = results['hits']['hits']

        print(f"📊 找到 {total_hits} 个相关文档")
        print()

        if not hits:
            print("⚠️ 未找到相关文档")
            return

        print("🔍 智能搜索结果分析:")
        print("-" * 60)

        for i, hit in enumerate(hits, 1):
            score = hit['_score']
            source = hit['_source']
            content = source.get('content', '')

            # 分析内容类型和重要性
            content_type = analyze_medical_content(content, query)
            importance_level = assess_content_importance(content, query)

            print(f"📄 结果 {i} (分数: {score:.3f}) [{importance_level}]")
            print(f"   📖 ID: {source['id']}")
            print(f"   📄 页面: {source.get('page_number', '未知')}")
            print(f"   🏷️  类型: {content_type}")

            # 显示高亮内容
            if 'highlight' in hit and 'content' in hit['highlight']:
                highlights = hit['highlight']['content']
                print(f"   ✨ 匹配内容:")
                for highlight in highlights:
                    print(f"      {highlight}")
            else:
                # 智能内容提取
                smart_preview = extract_relevant_content(content, query)
                if smart_preview:
                    print(f"   📑 相关内容:")
                    for line in smart_preview:
                        print(f"      {line}")

            print("-" * 50)

def analyze_medical_content(content: str, query: str) -> str:
    """分析医学内容类型"""
    lines = content.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith(f'# {query}'):
            return "🔴 基础概念主标题"
        elif line.startswith(f'## {query}'):
            return "🟡 重要子章节"
        elif line.startswith(f'### {query}'):
            return "🟢 小节标题"
        elif line.startswith('## 第一节') and query in line:
            return "🔴 第一节基础介绍"
        elif line.startswith('## 第二节') and query in line:
            return "🟡 第二节详细说明"

    # 检查是否是开篇介绍
    first_500_chars = content[:500]
    if query in first_500_chars and ('介绍' in first_500_chars or '定义' in first_500_chars):
        return "🔵 开篇定义"

    # 检查是否是具体亚型
    if '细胞癌' in content and query in content and len(query) < 4:
        return "🟣 具体亚型"

    return "⚪ 一般提及"

def assess_content_importance(content: str, query: str) -> str:
    """评估内容重要性级别"""
    lines = content.split('\n')

    # 检查标题层次
    for line in lines:
        if line.strip().startswith(f'# {query}'):
            return "⭐⭐⭐ 高重要"
        elif line.strip().startswith(f'## {query}'):
            return "⭐⭐ 中重要"
        elif line.strip().startswith(f'### {query}'):
            return "⭐ 一般重要"

    # 检查是否是第一节
    for line in lines:
        if '第一节' in line and query in line:
            return "⭐⭐⭐ 高重要"
        elif '第二节' in line and query in line:
            return "⭐⭐ 中重要"

    # 检查内容位置（开头更重要）
    first_1000 = content[:1000]
    if query in first_1000:
        return "⭐⭐ 中重要"

    return "⭐ 一般重要"

def extract_relevant_content(content: str, query: str, max_lines: int = 4) -> List[str]:
    """智能提取相关内容"""
    lines = content.split('\n')

    # 查找查询词出现的位置
    for i, line in enumerate(lines):
        if query in line and line.strip():
            # 获取上下文（前后各1-2行）
            start = max(0, i-1)
            end = min(len(lines), i+2)
            context = lines[start:end]

            # 清理和格式化
            formatted_lines = []
            for ctx_line in context:
                ctx_line = ctx_line.strip()
                if ctx_line:
                    # 标记查询词
                    if query in ctx_line:
                        ctx_line = ctx_line.replace(query, f"***{query}***")
                    formatted_lines.append(ctx_line)

            return formatted_lines[:max_lines]

    # 如果没找到，返回开头内容
    relevant_lines = []
    for line in lines[:max_lines]:
        line = line.strip()
        if line and not line.startswith('#'):
            relevant_lines.append(line)
        elif line.startswith('#'):
            relevant_lines.append(line)
            break

    return relevant_lines

def compare_search_approaches(query: str):
    """对比不同搜索方法的效果"""
    print(f"\n🔍 搜索方法对比测试: '{query}'")
    print("=" * 70)

    # 方法1: 基础搜索
    print("\n📋 方法1: 基础match搜索")
    print("-" * 40)

    basic_search = {
        "query": {"match": {"content": query}},
        "size": 3
    }

    try:
        response = requests.post(
            'http://localhost:9200/medical_documents/_search',
            json=basic_search,
            timeout=10
        )

        results = response.json()
        for i, hit in enumerate(results['hits']['hits'][:3], 1):
            content = hit['_source']['content'][:100]
            page = hit['_source']['page_number']
            score = hit['_score']
            print(f"{i}. 页面 {page} (分数: {score:.3f}): {content}...")

    except Exception as e:
        print(f"基础搜索错误: {e}")

    # 方法2: 智能标题优先搜索
    print("\n🧠 方法2: 标题优先搜索")
    print("-" * 40)

    smart_search = {
        "query": {
            "bool": {
                "should": [
                    {"match_phrase": {"content": {"query": f"# {query}", "boost": 5}}},
                    {"match_phrase": {"content": {"query": f"## {query}", "boost": 3}}},
                    {"match_phrase": {"content": {"query": f"### {query}", "boost": 2}}},
                    {"match": {"content": {"query": query, "boost": 1}}}
                ]
            }
        },
        "size": 3
    }

    try:
        response = requests.post(
            'http://localhost:9200/medical_documents/_search',
            json=smart_search,
            timeout=10
        )

        results = response.json()
        for i, hit in enumerate(results['hits']['hits'][:3], 1):
            content = hit['_source']['content'][:100]
            page = hit['_source']['page_number']
            score = hit['_score']
            content_type = analyze_medical_content(hit['_source']['content'], query)
            print(f"{i}. 页面 {page} (分数: {score:.3f}) [{content_type}]: {content}...")

    except Exception as e:
        print(f"智能搜索错误: {e}")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("🔍 智能医学搜索工具")
        print("=" * 40)
        print("使用方法: python3 smart_medical_search.py '搜索关键词'")
        print("示例: python3 smart_medical_search.py '腺癌'")
        print()
        print("🔧 特殊功能:")
        print("  • 标题优先排名")
        print("  • 基础概念识别")
        print("  • 内容重要性评估")
        print("  • 智能内容预览")
        print()
        print("📊 对比测试:")
        print("  python3 smart_medical_search.py --compare '腺癌'")
        return

    if sys.argv[1] == "--compare":
        if len(sys.argv) > 2:
            compare_search_approaches(sys.argv[2])
        else:
            print("请提供对比测试的关键词")
    else:
        query = sys.argv[1]
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        smart_medical_search(query, size)

if __name__ == "__main__":
    main()