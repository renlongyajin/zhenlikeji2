#!/usr/bin/env python3
"""
智能医学搜索工具 - 解决基础概念vs具体亚型排名问题
"""

import requests
import sys

def search_with_title_priority(query: str, size: int = 8):
    """标题优先的医学文档搜索"""

    print(f"🔍 标题优先搜索: '{query}'")
    print("=" * 60)

    # 构建标题优先的搜索查询
    search_query = {
        "query": {
            "bool": {
                "should": [
                    # 主标题匹配 - 最高权重
                    {
                        "match_phrase": {
                            "content": {
                                "query": f"# {query}",
                                "boost": 10.0
                            }
                        }
                    },
                    # 节标题匹配
                    {
                        "match_phrase": {
                            "content": {
                                "query": f"## {query}",
                                "boost": 5.0
                            }
                        }
                    },
                    # 小节标题匹配
                    {
                        "match_phrase": {
                            "content": {
                                "query": f"### {query}",
                                "boost": 3.0
                            }
                        }
                    },
                    # 第一节匹配（特别重要）
                    {
                        "match_phrase": {
                            "content": {
                                "query": f"## 第一节 {query}",
                                "boost": 8.0
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
                    "fragment_size": 200,
                    "number_of_fragments": 2,
                    "pre_tags": ["***"],
                    "post_tags": ["***"]
                }
            }
        }
    }

    try:
        response = requests.post(
            'http://localhost:9200/medical_documents/_search',
            json=search_query,
            timeout=10
        )

        results = response.json()
        total_hits = results['hits']['total']['value']
        hits = results['hits']['hits']

        print(f"📊 找到 {total_hits} 个相关文档")
        print()

        if not hits:
            print("⚠️ 未找到相关文档")
            return

        print("🔍 标题优先搜索结果:")
        print("-" * 60)

        for i, hit in enumerate(hits, 1):
            score = hit['_score']
            source = hit['_source']
            content = source.get('content', '')

            # 分析内容类型
            content_type = analyze_content_structure(content, query)

            print(f"📄 结果 {i} (分数: {score:.3f}) [{content_type}]")
            print(f"   📖 ID: {source['id']}")
            print(f"   📄 页面: {source.get('page_number', '未知')}")

            # 显示匹配内容
            if 'highlight' in hit and 'content' in hit['highlight']:
                print(f"   ✨ 匹配内容:")
                for highlight in hit['highlight']['content']:
                    print(f"      {highlight}")
            else:
                # 智能内容提取
                relevant_lines = extract_relevant_lines(content, query)
                if relevant_lines:
                    print(f"   📑 相关内容:")
                    for line in relevant_lines:
                        print(f"      {line}")

            print("-" * 50)

    except Exception as e:
        print(f"❌ 搜索错误: {str(e)}")

def analyze_content_structure(content: str, query: str) -> str:
    """分析内容结构类型"""
    lines = content.split('\n')

    for line in lines:
        line = line.strip()
        # 检查是否是标题格式
        if line.startswith(f'# {query}'):
            return "🔴 主标题"
        elif line.startswith(f'## {query}'):
            return "🟡 节标题"
        elif line.startswith(f'## 第一节') and query in line:
            return "🔴 第一节基础介绍"
        elif line.startswith(f'## 第二节') and query in line:
            return "🟡 第二节详细说明"
        elif line.startswith(f'### {query}'):
            return "🟢 小节标题"

    # 检查是否是具体亚型
    if '细胞癌' in content and query in content and len(query) <= 3:
        return "🟣 亚型描述"

    return "⚪ 一般提及"

def extract_relevant_lines(content: str, query: str, max_lines: int = 3) -> list:
    """提取相关内容的行"""
    lines = content.split('\n')

    # 查找查询词出现的位置
    for i, line in enumerate(lines):
        if query in line and line.strip():
            # 获取上下文（前后各1行）
            start = max(0, i-1)
            end = min(len(lines), i+2)
            context = lines[start:end]

            # 格式化和标记
            formatted_lines = []
            for ctx_line in context:
                ctx_line = ctx_line.strip()
                if ctx_line:
                    # 高亮查询词
                    if query in ctx_line:
                        ctx_line = ctx_line.replace(query, f"***{query}***")
                    formatted_lines.append(ctx_line)

            return formatted_lines[:max_lines]

    # 如果没找到，返回开头几行
    preview_lines = []
    for line in lines[:5]:
        line = line.strip()
        if line:
            preview_lines.append(line)
            if len(preview_lines) >= 3:
                break

    return preview_lines

def compare_search_methods(query: str):
    """对比基础搜索vs标题优先搜索"""
    print(f"\n🔍 搜索方法对比: '{query}'")
    print("=" * 70)

    # 方法1: 基础搜索
    print("\n📋 方法1: 基础match搜索")
    print("-" * 40)

    basic_query = {
        "query": {"match": {"content": query}},
        "size": 3
    }

    try:
        response = requests.post(
            'http://localhost:9200/medical_documents/_search',
            json=basic_query,
            timeout=10
        )

        results = response.json()
        for i, hit in enumerate(results['hits']['hits'][:3], 1):
            content = hit['_source']['content'][:100]
            page = hit['_source']['page_number']
            score = hit['_score']
            content_type = analyze_content_structure(hit['_source']['content'], query)
            print(f"{i}. 页面 {page} (分数: {score:.3f}) [{content_type}]: {content}...")

    except Exception as e:
        print(f"基础搜索错误: {e}")

    # 方法2: 标题优先搜索
    print("\n🧠 方法2: 标题优先搜索")
    print("-" * 40)

    # 使用相同的搜索逻辑，但只显示结果
    title_priority_query = {
        "query": {
            "bool": {
                "should": [
                    {"match_phrase": {"content": {"query": f"# {query}", "boost": 10}}},
                    {"match_phrase": {"content": {"query": f"## {query}", "boost": 5}}},
                    {"match_phrase": {"content": {"query": f"## 第一节 {query}", "boost": 8}}},
                    {"match": {"content": {"query": query, "boost": 1}}}
                ]
            }
        },
        "size": 3
    }

    try:
        response = requests.post(
            'http://localhost:9200/medical_documents/_search',
            json=title_priority_query,
            timeout=10
        )

        results = response.json()
        for i, hit in enumerate(results['hits']['hits'][:3], 1):
            content = hit['_source']['content'][:100]
            page = hit['_source']['page_number']
            score = hit['_score']
            content_type = analyze_content_structure(hit['_source']['content'], query)
            print(f"{i}. 页面 {page} (分数: {score:.3f}) [{content_type}]: {content}...")

    except Exception as e:
        print(f"标题优先搜索错误: {e}")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("🔍 智能医学搜索工具")
        print("=" * 40)
        print("使用方法: python3 intelligent_medical_search.py '搜索关键词'")
        print("示例: python3 intelligent_medical_search.py '腺癌'")
        print()
        print("🔧 特殊功能:")
        print("  • 标题优先排名")
        print("  • 基础概念识别")
        print("  • 内容结构分析")
        print("  • 搜索方法对比")
        print()
        print("📊 对比测试:")
        print("  python3 intelligent_medical_search.py --compare '腺癌'")
        return

    if sys.argv[1] == "--compare":
        if len(sys.argv) > 2:
            compare_search_methods(sys.argv[2])
        else:
            print("请提供对比测试的关键词")
    else:
        query = sys.argv[1]
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        search_with_title_priority(query, size)

if __name__ == "__main__":
    main()