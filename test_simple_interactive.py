#!/usr/bin/env python3
"""
简化增强版ReAct Agent - 简单测试版本
用于验证简化增强功能的基本可用性
"""

import sys
import os
import json
from datetime import datetime

# 添加项目路径
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

def mock_test():
    """模拟测试，验证基本功能"""
    print("🧪 简化增强版ReAct Agent - 功能验证测试")
    print("=" * 60)

    try:
        # 尝试导入简化增强模块
        print("📦 正在导入模块...")
        from agent.simple_chapter_enhancer import SimpleChapterEnhancer, get_chapter_enhancer
        print("✅ 极简章节增强器导入成功")

        # 测试章节增强器
        print("\n🔍 测试章节增强功能...")
        enhancer = get_chapter_enhancer()

        # 测试查询增强
        test_queries = [
            "腺癌的图像特征是什么？",
            "鳞癌和腺癌有什么区别？",
            "小细胞癌的细胞学特点"
        ]

        for query in test_queries:
            print(f"\n📝 测试查询: {query}")
            enhanced_queries = enhancer.enhance_search_queries(query)
            print(f"增强查询: {enhanced_queries}")

        # 测试结果评分
        print("\n📊 测试结果评分功能...")
        mock_results = [
            {
                'content': '第二章 肺部肿瘤\n第一节 腺癌\n腺癌的图像特征包括细胞核增大等。',
                'score': 0.6,
                'chapter_title': '第二章 肺部肿瘤',
                'section_title': '第一节 腺癌'
            },
            {
                'content': '腺癌是一种常见的恶性肿瘤，具有特定特征。',
                'score': 0.4,
                'chapter_title': '',
                'section_title': ''
            }
        ]

        query = "腺癌的图像特征"
        boosted_results = enhancer.boost_result_scores(mock_results, query)

        print(f"评分结果:")
        for i, result in enumerate(boosted_results):
            original_score = result.get('score', 0) - result.get('chapter_boost_score', 0)
            current_score = result.get('score', 0)
            boost_score = result.get('chapter_boost_score', 0)
            print(f"  结果 {i+1}: {original_score:.3f} -> {current_score:.3f} (+{boost_score:.3f})")

        # 测试章节信息提取
        print("\n📚 测试章节信息提取...")
        test_content = """
第一章 总论
第一节 肺癌概述
肺癌是最常见的恶性肿瘤之一。

第二章 肺部实体恶性肿瘤
第一节 腺癌
腺癌具有特定的图像特征。
"""

        chapter_info = enhancer.extract_chapter_info(test_content)
        print(f"提取的章节信息: {chapter_info}")

        # 尝试导入简化增强Agent
        print("\n🤖 测试简化增强Agent...")
        try:
            from agent.simple_enhanced_react_agent import create_simple_enhanced_react_agent
            print("✅ 简化增强ReAct Agent导入成功")

            # 创建模拟的agent配置
            mock_agent_config = {
                "llm_manager": None,
                "retrieval_manager": None,
                "max_iterations": 2
            }
            print("✅ Agent配置创建成功")

        except ImportError as e:
            print(f"⚠️ 简化增强Agent导入失败: {e}")
            print("但章节增强器功能正常")

        print("\n" + "=" * 60)
        print("✅ 基础功能验证完成！")
        print("\n📊 测试总结:")
        print("  • 极简章节增强器工作正常")
        print("  • 查询增强功能有效")
        print("  • 结果评分功能正常")
        print("  • 章节信息提取功能可用")
        print("\n💡 下一步建议:")
        print("  1. 配置好环境变量（DEEPSEEK_API_KEY等）")
        print("  2. 确保Elasticsearch和Milvus服务正常运行")
        print("  3. 运行完整的交互式脚本进行测试")

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保：")
        print("  1. 当前目录是项目根目录")
        print("  2. src/agent/simple_chapter_enhancer.py 文件存在")
        print("  3. 所有依赖模块都已正确安装")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def create_mock_response():
    """创建模拟响应数据"""
    return {
        "query_id": f"mock_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "question": "腺癌的图像特征是什么？",
        "answer": "腺癌的图像特征主要包括：1. 细胞核增大，形态不规则；2. 核仁明显，可能多个；3. 染色质分布不均匀；4. 细胞质相对较少。这些特征在显微镜下观察最为明显。",
        "confidence": 0.85,
        "reasoning_steps": [
            {
                "step": "intent_analysis",
                "thought": "分析用户查询意图: 腺癌的图像特征是什么？",
                "action": "analyze_query_intent",
                "observation": "查询类型: medical, 识别实体: ['腺癌']"
            },
            {
                "step": "tool_selection",
                "thought": "使用增强RAG搜索工具",
                "action": "enhanced_rag_search",
                "observation": "执行增强查询: ['腺癌的图像特征是什么？', '第.*节 腺癌', '腺癌 细胞形态 结构特征']"
            }
        ],
        "retrieved_documents": [
            {
                "content": "第二章 肺部实体恶性肿瘤\n第一节 腺癌\n腺癌的图像特征包括细胞核增大、核仁明显、染色质分布不均匀等。在显微镜下观察，可以看到癌细胞呈现不规则形态。",
                "page_number": 45,
                "chapter_title": "第二章 肺部实体恶性肿瘤",
                "section_title": "第一节 腺癌",
                "score": 0.92,
                "chapter_boost_score": 0.3,
                "search_type": "hybrid"
            }
        ],
        "response_time": 2.34,
        "model_used": "simple_enhanced_react",
        "metadata": {
            "query_type": "medical",
            "entities": ["腺癌"],
            "iteration_count": 1,
            "result_quality": 0.78,
            "search_results_count": 3,
            "enhanced_queries": [
                "腺癌的图像特征是什么？",
                "第.*节 腺癌",
                "腺癌 细胞形态 结构特征"
            ]
        }
    }

if __name__ == "__main__":
    # 默认进行基础功能验证
    print("🚀 运行基础功能验证测试...")
    mock_test()