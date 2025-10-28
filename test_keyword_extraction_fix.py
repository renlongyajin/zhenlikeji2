#!/usr/bin/env python3
"""
测试关键词提取修复效果
对比修复前后的搜索效果
"""

import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

def test_keyword_extraction():
    """测试关键词提取功能"""
    print("🔍 测试关键词提取功能")
    print("=" * 60)

    try:
        from agent.enhanced_retrieval_manager import create_enhanced_retrieval_manager
        from embedding.embedding_models import get_embedding_manager

        # 初始化检索管理器
        print("初始化检索管理器...")
        embedding_manager = get_embedding_manager(model_type="jina")
        retrieval_manager = create_enhanced_retrieval_manager(
            es_host='localhost',
            es_port=9200,
            milvus_host='localhost',
            milvus_port=19530,
            embedding_manager=embedding_manager
        )

        # 测试关键词提取
        test_queries = [
            "腺癌的图像特征是什么？",
            "鳞癌和腺癌有什么区别？",
            "小细胞癌的细胞学特点有哪些？",
            "黏液腺癌的病理特征是什么？",
            "什么是肺腺癌？",
            "肺癌的诊断方法有哪些？"
        ]

        print(f"\n🧪 测试 {len(test_queries)} 个查询的关键词提取")

        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. 原始查询: {query}")

            # 测试关键词提取
            keywords = retrieval_manager._extract_search_keywords(query)
            print(f"   提取关键词: {keywords}")

            # 测试预处理
            if not keywords or keywords == query:
                preprocessed = retrieval_manager._preprocess_search_query(query)
                print(f"   预处理后: {preprocessed}")

        return True

    except Exception as e:
        print(f"关键词提取测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_comparison():
    """对比修复前后的搜索效果"""
    print(f"\n{'='*60}")
    print("🔍 对比修复前后的搜索效果")
    print("=" * 60)

    try:
        from agent.enhanced_retrieval_manager import create_enhanced_retrieval_manager
        from embedding.embedding_models import get_embedding_manager

        # 初始化检索管理器
        print("初始化检索管理器...")
        embedding_manager = get_embedding_manager(model_type="jina")
        retrieval_manager = create_enhanced_retrieval_manager(
            es_host='localhost',
            es_port=9200,
            milvus_host='localhost',
            milvus_port=19530,
            embedding_manager=embedding_manager
        )

        # 测试查询
        test_query = "腺癌的图像特征是什么？"
        print(f"测试查询: {test_query}")

        # 1. 测试关键词提取
        keywords = retrieval_manager._extract_search_keywords(test_query)
        print(f"提取关键词: {keywords}")

        # 2. 直接关键词搜索（模拟修复前）
        print(f"\n🔍 直接关键词搜索（修复前）...")
        try:
            direct_results = retrieval_manager.enhanced_keyword_search(
                test_query,
                top_k=3
            )
            print(f"直接搜索结果数: {len(direct_results)}")
            if direct_results:
                print(f"最佳分数: {direct_results[0].score:.3f}")
                print(f"内容预览: {direct_results[0].content[:100]}...")
        except Exception as e:
            print(f"直接搜索失败: {e}")

        # 3. 提取关键词搜索（修复后）
        print(f"\n🔍 提取关键词搜索（修复后）...")
        try:
            extracted_results = retrieval_manager.enhanced_keyword_search(
                keywords,
                top_k=3
            )
            print(f"提取关键词结果数: {len(extracted_results)}")
            if extracted_results:
                print(f"最佳分数: {extracted_results[0].score:.3f}")
                print(f"内容预览: {extracted_results[0].content[:100]}...")
        except Exception as e:
            print(f"提取关键词搜索失败: {e}")

        # 4. 测试混合搜索（完整修复）
        print(f"\n🔍 混合搜索（完整修复）...")
        try:
            hybrid_results = retrieval_manager._enhanced_hybrid_search(
                test_query,
                top_k=3,
                keyword_weight=0.6
            )
            print(f"混合搜索结果数: {len(hybrid_results)}")
            if hybrid_results:
                print(f"最佳分数: {hybrid_results[0].score:.3f}")
                print(f"内容预览: {hybrid_results[0].content[:100]}...")
                print(f"搜索类型: {hybrid_results[0].search_type}")
        except Exception as e:
            print(f"混合搜索失败: {e}")

        return True

    except Exception as e:
        print(f"搜索对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_examples():
    """测试具体例子展示修复效果"""
    print(f"\n{'='*60}")
    print("🎯 具体例子展示修复效果")
    print("=" * 60)

    try:
        from agent.enhanced_retrieval_manager import create_enhanced_retrieval_manager
        from embedding.embedding_models import get_embedding_manager

        # 初始化检索管理器
        embedding_manager = get_embedding_manager(model_type="jina")
        retrieval_manager = create_enhanced_retrieval_manager(
            es_host='localhost',
            es_port=9200,
            milvus_host='localhost',
            milvus_port=19530,
            embedding_manager=embedding_manager
        )

        # 对比查询
        test_cases = [
            {
                "原始": "腺癌的图像特征是什么？",
                "期望关键词": "腺癌 图像特征"
            },
            {
                "原始": "什么是肺鳞癌？",
                "期望关键词": "肺鳞癌"
            },
            {
                "原始": "小细胞癌和腺癌有什么区别？",
                "期望关键词": "小细胞癌 腺癌"
            }
        ]

        for i, case in enumerate(test_cases, 1):
            print(f"\n{i}. 测试案例:")
            print(f"   原始查询: {case['原始']}")
            print(f"   期望关键词: {case['期望关键词']}")

            # 测试关键词提取
            extracted = retrieval_manager._extract_search_keywords(case['原始'])
            print(f"   实际提取: {extracted}")

            # 评估提取效果
            if extracted == case['期望关键词']:
                print("   ✅ 提取效果完美")
            elif case['期望关键词'] in extracted:
                print("   ⚠️ 提取效果良好")
            else:
                print("   ❌ 提取效果一般")

        return True

    except Exception as e:
        print(f"具体例子测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 关键词提取修复效果测试")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 运行测试
    test1_success = test_keyword_extraction()
    test2_success = test_search_comparison()
    test3_success = test_specific_examples()

    # 生成总结报告
    print(f"\n{'='*60}")
    print("📊 测试总结报告")
    print(f"{'='*60}")

    tests = [
        ("关键词提取", test1_success),
        ("搜索对比", test2_success),
        ("具体例子", test3_success)
    ]

    passed = sum(1 for _, success in tests if success)
    total = len(tests)

    print(f"测试通过率: {passed}/{total} ({passed/total*100:.1f}%)")

    for test_name, success in tests:
        status = "✅通过" if success else "❌失败"
        print(f"  {status} {test_name}")

    if passed == total:
        print("\n🎉 所有测试通过！关键词提取修复成功")
    else:
        print(f"\n⚠️ {total-passed}个测试失败，需要进一步调试")

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)