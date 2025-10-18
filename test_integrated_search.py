#!/usr/bin/env python3
"""
集成搜索算法测试脚本
测试整合后的智能搜索功能
"""

import sys
import os
import time
import json
from typing import List, Dict, Any

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from agent.retrieval_manager import MedicalRetrievalManager, create_retrieval_manager
    from embedding.embedding_models import get_embedding_manager
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保在正确的项目目录中运行")
    sys.exit(1)

def print_search_results(results: List[Dict[str, Any]], query: str, search_type: str, execution_time: float):
    """打印搜索结果"""
    print(f"\n{'='*80}")
    print(f"🔍 {search_type.upper()} 搜索结果")
    print(f"查询: {query}")
    print(f"执行时间: {execution_time:.3f}秒")
    print(f"结果数量: {len(results)}")
    print(f"{'='*80}")

    for i, result in enumerate(results, 1):
        print(f"\n📄 结果 {i} (分数: {result['score']:.3f})")
        print(f"来源: {result['source']} | 页面: {result['page_number']}")
        print(f"章节: {result['chapter_title']}")
        print(f"小节: {result['section_title']}")
        print(f"搜索类型: {result['search_type']}")
        print(f"内容预览: {result['content'][:200]}...")

def test_basic_functionality():
    """测试基础功能"""
    print("🧪 开始基础功能测试...")

    try:
        # 创建检索管理器
        print("📦 创建检索管理器...")
        embedding_manager = get_embedding_manager("jina")
        retrieval_manager = create_retrieval_manager(
            es_host="localhost",
            es_port=9200,
            milvus_host="localhost",
            milvus_port=19530,
            embedding_manager=embedding_manager
        )
        print("✅ 检索管理器创建成功")

        # 测试查询
        test_queries = [
            "什么是肺部恶性肿瘤的ROSE细胞学特征？",
            "ROSE技术的诊断标准是什么？",
            "腺癌的细胞学特点",
            "细胞核增大和核仁明显"
        ]

        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"📝 测试查询: {query}")
            print(f"{'='*80}")

            # 普通混合搜索
            start_time = time.time()
            hybrid_results = retrieval_manager.search(query, search_type="hybrid", top_k=3)
            hybrid_time = time.time() - start_time
            print_search_results(hybrid_results, query, "混合搜索", hybrid_time)

            # 智能搜索
            search_config = {
                'top_k': 3,
                'enable_title_priority': True,
                'enable_content_analysis': True,
                'enable_concept_identification': True,
                'keyword_weight': 0.6
            }
            start_time = time.time()
            intelligent_results = retrieval_manager.search(query, search_type="intelligent", search_config=search_config)
            intelligent_time = time.time() - start_time
            print_search_results(intelligent_results, query, "智能搜索", intelligent_time)

            # 对比分析
            if hybrid_results and intelligent_results:
                print(f"\n📊 对比分析:")
                print(f"混合搜索最高分: {hybrid_results[0]['score']:.3f}")
                print(f"智能搜索最高分: {intelligent_results[0]['score']:.3f}")

                # 检查是否有更好的标题匹配
                hybrid_first_title = hybrid_results[0]['chapter_title'] + ' ' + hybrid_results[0]['section_title']
                intelligent_first_title = intelligent_results[0]['chapter_title'] + ' ' + intelligent_results[0]['section_title']

                query_in_hybrid_title = query.lower() in hybrid_first_title.lower()
                query_in_intelligent_title = query.lower() in intelligent_first_title.lower()

                if query_in_intelligent_title and not query_in_hybrid_title:
                    print("🎯 智能搜索在标题匹配方面有改进")
                elif intelligent_results and 'is_rose_related' in str(intelligent_results[0]):
                    print("🎯 智能搜索识别到ROSE相关内容")

            time.sleep(1)  # 避免过快请求

        return True

    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return False

def test_algorithm_components():
    """测试算法组件"""
    print(f"\n{'='*80}")
    print("🔬 测试算法组件...")
    print(f"{'='*80}")

    try:
        # 创建检索管理器
        retrieval_manager = create_retrieval_manager()

        # 测试标题优先级算法
        test_cases = [
            ("ROSE技术", "ROSE细胞学诊断技术", "基本原理"),
            ("肺部恶性肿瘤", "肺部恶性肿瘤诊断", "细胞学特征"),
            ("细胞核增大", "细胞形态学变化", "核结构异常")
        ]

        print("\n📏 标题优先级算法测试:")
        for query, chapter_title, section_title in test_cases:
            score = retrieval_manager._calculate_title_priority_score(query, chapter_title, section_title)
            print(f"查询: '{query}' | 章节: '{chapter_title}' | 小节: '{section_title}'")
            print(f"优先级分数: {score:.3f}")

        # 测试内容类型分析
        test_content = """
        ROSE（快速现场评价）技术是一种在介入性操作过程中，由细胞病理学家对采集的标本
        进行实时评估的技术。该技术能够提高诊断准确率，减少不必要的重复操作。
        诊断标准包括：细胞核增大、核仁明显、核浆比失调等特征。
        """

        print("\n🔍 内容类型分析测试:")
        analysis = retrieval_manager._analyze_content_type(test_content, "ROSE技术诊断标准")
        print(f"内容类型: {analysis['type']}")
        print(f"置信度: {analysis['confidence']:.3f}")
        print(f"ROSE相关: {analysis['is_rose_related']}")

        # 测试基础概念识别
        print("\n🧠 基础概念识别测试:")
        concept_analysis = retrieval_manager._identify_basic_concepts(
            "什么是肺部恶性肿瘤的ROSE细胞学特征？",
            test_content
        )
        print(f"内容级别: {concept_analysis['content_level']}")
        print(f"相关度分数: {concept_analysis['relevance_score']:.3f}")
        print(f"查询类型: {concept_analysis['query_type']}")
        print(f"基础概念匹配: {concept_analysis['matches_basic']}")
        print(f"专业术语匹配: {concept_analysis['matches_specific']}")

        return True

    except Exception as e:
        print(f"❌ 算法组件测试失败: {e}")
        return False

def test_performance_comparison():
    """性能对比测试"""
    print(f"\n{'='*80}")
    print("⚡ 性能对比测试...")
    print(f"{'='*80}")

    try:
        retrieval_manager = create_retrieval_manager()
        query = "肺部恶性肿瘤ROSE细胞学特征"

        # 多次测试取平均值
        num_tests = 5
        hybrid_times = []
        intelligent_times = []

        for i in range(num_tests):
            print(f"\n第{i+1}轮测试...")

            # 混合搜索
            start_time = time.time()
            retrieval_manager.search(query, search_type="hybrid", top_k=5)
            hybrid_time = time.time() - start_time
            hybrid_times.append(hybrid_time)

            # 智能搜索
            search_config = {'top_k': 5, 'keyword_weight': 0.6}
            start_time = time.time()
            retrieval_manager.search(query, search_type="intelligent", search_config=search_config)
            intelligent_time = time.time() - start_time
            intelligent_times.append(intelligent_time)

        # 计算平均值
        avg_hybrid = sum(hybrid_times) / len(hybrid_times)
        avg_intelligent = sum(intelligent_times) / len(intelligent_times)

        print(f"\n📊 性能对比结果:")
        print(f"混合搜索平均时间: {avg_hybrid:.3f}秒")
        print(f"智能搜索平均时间: {avg_intelligent:.3f}秒")
        print(f"性能开销: {((avg_intelligent - avg_hybrid) / avg_hybrid * 100):.1f}%")

        return True

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("="*80)
    print("🏥 医学RAG系统 - 集成搜索算法测试")
    print("="*80)

    success_count = 0
    total_tests = 3

    # 测试1: 基础功能
    if test_basic_functionality():
        success_count += 1
        print("\n✅ 基础功能测试通过")
    else:
        print("\n❌ 基础功能测试失败")

    # 测试2: 算法组件
    if test_algorithm_components():
        success_count += 1
        print("\n✅ 算法组件测试通过")
    else:
        print("\n❌ 算法组件测试失败")

    # 测试3: 性能对比
    if test_performance_comparison():
        success_count += 1
        print("\n✅ 性能对比测试通过")
    else:
        print("\n❌ 性能对比测试失败")

    print(f"\n{'='*80}")
    print(f"📋 测试总结: {success_count}/{total_tests} 项测试通过")

    if success_count == total_tests:
        print("🎉 所有测试通过！集成搜索算法工作正常")
        return 0
    else:
        print("⚠️  部分测试失败，请检查系统配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())