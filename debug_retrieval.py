#!/usr/bin/env python3
"""
直接调试检索管理器的脚本
绕过API层，直接调用检索函数进行调试
"""

import sys
import os
import json
import pdb  # Python调试器

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 现在可以导入模块了
from agent.enhanced_retrieval_manager import EnhancedMedicalRetrievalManager, create_enhanced_retrieval_manager

def debug_retrieval():
    """直接调试检索功能"""
    print("🚀 启动检索管理器调试...")

    # 创建检索管理器实例
    print("📦 创建增强版检索管理器...")
    retrieval_manager = create_enhanced_retrieval_manager(
        es_host="localhost",
        es_port=9200,
        milvus_host="localhost",
        milvus_port=19530,
        embedding_manager=None  # 使用模拟嵌入
    )

    # 设置断点 - 程序会在这里暂停
    print("⏰ 设置断点，准备进入调试模式...")
    pdb.set_trace()  # ← 程序会在这里暂停，您可以单步调试

    # 测试查询
    test_query = "鳞癌的图像特征是什么？"
    print(f"🔍 执行查询: {test_query}")

    # 构建搜索配置
    search_config = {
        'search_type': 'keyword',
        'top_k': 5,
        'title_priority': True,
        'title_priority_config': {
            'chapter_title_weight': 25.0,
            'section_title_weight': 20.0,
            'subsection_title_weight': 15.0,
            'exact_match_boost': 3.0,
            'medical_term_bonus': 5.0,
            'descriptive_content_boost': 2.0,
            'min_description_length': 150
        }
    }

    print(f"🔧 搜索配置: {json.dumps(search_config, indent=2, ensure_ascii=False)}")

    # 执行搜索 - 这里会调用到requests.post
    print("🔄 执行增强版搜索...")
    results = retrieval_manager.enhanced_search(test_query, search_config)

    print(f"✅ 搜索完成，找到 {len(results)} 个结果")

    # 打印结果摘要
    for i, result in enumerate(results[:3], 1):
        print(f"\n📄 结果 {i}:")
        print(f"   章节: {result.get('chapter_title', 'N/A')}")
        print(f"   小节: {result.get('section_title', 'N/A')}")
        print(f"   页码: 第{result.get('page_number', 'N/A')}页")
        print(f"   分数: {result.get('score', 0):.3f}")
        print(f"   搜索类型: {result.get('search_type', 'unknown')}")
        print(f"   内容预览: {result.get('content', '')[:100]}...")

if __name__ == "__main__":
    debug_retrieval()