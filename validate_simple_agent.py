#!/usr/bin/env python3
"""
直接验证简化增强ReAct Agent的可用性
绕过交互式输入，快速测试核心功能
"""

import sys
import os
import json
import time
from datetime import datetime

# 添加项目路径
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

def test_agent_components():
    """测试Agent各个组件的可用性"""
    print("🔍 测试简化增强ReAct Agent组件")
    print("=" * 60)

    results = {
        "import_success": False,
        "chapter_enhancer": False,
        "agent_creation": False,
        "basic_functionality": False,
        "search_capability": False,
        "overall_status": "FAILED"
    }

    try:
        # 1. 测试导入
        print("\n1. 测试模块导入...")
        from agent.simple_enhanced_react_agent import SimpleEnhancedReActAgent, create_simple_enhanced_react_agent
        from agent.llm_manager import LLMManager
        from agent.enhanced_retrieval_manager import create_enhanced_retrieval_manager
        from embedding.embedding_models import get_embedding_manager
        from agent.simple_chapter_enhancer import get_chapter_enhancer
        print("✅ 所有模块导入成功")
        results["import_success"] = True

        # 2. 测试章节增强器
        print("\n2. 测试章节增强器...")
        enhancer = get_chapter_enhancer()
        test_query = "腺癌的图像特征"
        enhanced_queries = enhancer.enhance_search_queries(test_query)
        print(f"✅ 章节增强器工作正常，生成 {len(enhanced_queries)} 个增强查询")
        print(f"   增强查询: {enhanced_queries}")
        results["chapter_enhancer"] = True

        # 3. 测试Agent创建（不依赖外部服务）
        print("\n3. 测试Agent创建...")

        # 创建模拟的LLM管理器
        llm_config = {
            'default_provider': 'deepseek',
            'deepseek': {
                'api_key': 'test_key',
                'base_url': 'https://api.deepseek.com',
                'model': 'deepseek-reasoner'
            }
        }

        llm_manager = LLMManager(config=llm_config)

        # 创建模拟的检索管理器（最小化依赖）
        try:
            embedding_manager = get_embedding_manager(model_type="jina")
            retrieval_manager = create_enhanced_retrieval_manager(
                es_host='localhost',
                es_port=9200,
                milvus_host='localhost',
                milvus_port=19530,
                embedding_manager=embedding_manager
            )
            print("✅ 检索管理器创建成功")
        except Exception as e:
            print(f"⚠️ 检索管理器创建失败: {e}")
            print("   将使用模拟模式继续测试")
            retrieval_manager = None

        # 创建Agent
        agent = create_simple_enhanced_react_agent(
            llm_manager=llm_manager,
            enhanced_retrieval_manager=retrieval_manager,
            max_iterations=1
        )
        print("✅ Agent创建成功")
        results["agent_creation"] = True

        # 4. 测试基本功能
        print("\n4. 测试基本功能...")

        # 测试意图分析
        question = "腺癌的图像特征是什么？"
        entities = agent._extract_entities(question)
        query_type = agent._determine_query_type(question)
        print(f"✅ 意图分析功能正常")
        print(f"   问题: {question}")
        print(f"   提取实体: {entities}")
        print(f"   查询类型: {query_type}")

        # 测试查询增强
        enhanced_queries = agent.chapter_enhancer.enhance_search_queries(question)
        print(f"✅ 查询增强功能正常: {enhanced_queries}")
        results["basic_functionality"] = True

        # 5. 测试搜索能力（如果可能）
        print("\n5. 测试搜索能力...")
        if retrieval_manager:
            try:
                # 直接测试增强搜索工具
                from agent.simple_enhanced_react_agent import enhanced_rag_search

                # 获取工具函数
                tools = agent.tools
                if 'enhanced_rag_search' in tools:
                    search_tool = tools['enhanced_rag_search']
                    result = search_tool.invoke({
                        "query": "腺癌",
                        "search_type": "keyword",
                        "max_results": 2,
                        "use_chapter_boost": False
                    })

                    if result.get('success') and result.get('results'):
                        print(f"✅ 搜索功能正常，找到 {len(result['results'])} 个结果")
                        results["search_capability"] = True
                    else:
                        print(f"⚠️ 搜索功能异常: {result.get('error', '未知错误')}")
                else:
                    print("⚠️ 未找到增强搜索工具")
            except Exception as e:
                print(f"⚠️ 搜索测试失败: {e}")
        else:
            print("⚠️ 无检索管理器，跳过搜索测试")

        # 6. 整体评估
        print("\n6. 整体评估...")
        success_count = sum([results["import_success"], results["chapter_enhancer"],
                           results["agent_creation"], results["basic_functionality"]])

        if success_count >= 4:
            results["overall_status"] = "SUCCESS"
            print("✅ 简化增强ReAct Agent基本可用")
        elif success_count >= 3:
            results["overall_status"] = "PARTIAL"
            print("⚠️ 简化增强ReAct Agent部分可用")
        else:
            results["overall_status"] = "FAILED"
            print("❌ 简化增强ReAct Agent不可用")

        return results

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return results

def test_direct_search():
    """直接测试搜索功能"""
    print("\n🔍 直接测试搜索功能")
    print("=" * 60)

    try:
        # 使用Agent实例来测试搜索功能
        from agent.llm_manager import LLMManager
        from agent.enhanced_retrieval_manager import create_enhanced_retrieval_manager
        from embedding.embedding_models import get_embedding_manager

        # 初始化组件
        print("初始化搜索组件...")
        embedding_manager = get_embedding_manager(model_type="jina")
        retrieval_manager = create_enhanced_retrieval_manager(
            es_host='localhost',
            es_port=9200,
            milvus_host='localhost',
            milvus_port=19530,
            embedding_manager=embedding_manager
        )

        # 创建LLM管理器
        llm_manager = LLMManager(config={
            'default_provider': 'deepseek',
            'deepseek': {'api_key': 'test_key', 'base_url': 'https://api.deepseek.com'}
        })

        # 创建Agent实例
        from agent.simple_enhanced_react_agent import create_simple_enhanced_react_agent
        agent = create_simple_enhanced_react_agent(
            llm_manager=llm_manager,
            enhanced_retrieval_manager=retrieval_manager,
            max_iterations=1
        )

        # 获取搜索工具
        tools = agent.tools
        search_tool = None
        for tool in tools:
            if hasattr(tool, 'name') and tool.name == 'enhanced_rag_search':
                search_tool = tool
                break

        if not search_tool:
            print("⚠️ 未找到增强搜索工具，尝试直接搜索...")
            # 直接测试检索管理器
            if hasattr(retrieval_manager, 'search'):
                results = retrieval_manager.search("腺癌", search_type="keyword", top_k=3)
                print(f"✅ 直接检索成功，找到 {len(results)} 个结果")
                return len(results) > 0
            else:
                print("❌ 无法找到可用的搜索方法")
                return False

        # 测试搜索
        print("测试关键词搜索...")
        result = search_tool.invoke({
            "query": "腺癌",
            "search_type": "keyword",
            "max_results": 3,
            "use_chapter_boost": True
        })

        print(f"搜索结果: {result.get('success', False)}")
        if result.get('success'):
            print(f"找到 {len(result.get('results', []))} 个结果")
            for i, doc in enumerate(result.get('results', [])[:2]):
                print(f"  {i+1}. 第{doc.get('page_number', '未知')}页 - 分数:{doc.get('score', 0):.3f}")
        else:
            print(f"搜索失败: {result.get('error', '未知错误')}")

        return result.get('success', False) and len(result.get('results', [])) > 0

    except Exception as e:
        print(f"直接搜索测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 简化增强ReAct Agent - 可用性验证")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 运行组件测试
    component_results = test_agent_components()

    # 运行直接搜索测试
    search_success = test_direct_search()

    # 生成最终报告
    print(f"\n{'='*60}")
    print("📊 最终验证报告")
    print(f"{'='*60}")

    print(f"组件测试结果: {component_results['overall_status']}")
    print(f"搜索测试结果: {'SUCCESS' if search_success else 'FAILED'}")

    print(f"\n详细结果:")
    for key, value in component_results.items():
        if key != 'overall_status':
            status_icon = "✅" if value else "❌"
            print(f"  {status_icon} {key.replace('_', ' ').title()}: {value}")

    # 整体评估
    overall_success = (component_results['overall_status'] == 'SUCCESS' and search_success)

    print(f"\n🎯 整体评估:")
    if overall_success:
        print("✅ 简化增强ReAct Agent完全可用")
        print("💡 建议: 可以开始交互式测试")
    elif component_results['overall_status'] in ['SUCCESS', 'PARTIAL']:
        print("⚠️ 简化增强ReAct Agent部分可用")
        print("💡 建议: 检查外部服务连接")
    else:
        print("❌ 简化增强ReAct Agent不可用")
        print("💡 建议: 检查代码实现和依赖")

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)