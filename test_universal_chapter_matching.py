#!/usr/bin/env python3
"""
测试通用章节匹配算法 - 验证多种癌症类型的匹配效果
"""

import sys
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

import logging
import os
from dotenv import load_dotenv
from src.agent.enhanced_react_agent import EnhancedMedicalReActAgent
from src.agent.llm_manager import LLMManager
from src.agent.retrieval_manager import MedicalRetrievalManager
from src.embedding.embedding_models import get_embedding_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_universal_chapter_matching():
    """测试通用章节匹配算法"""

    logger.info("=" * 80)
    logger.info("测试通用章节匹配算法 - 多种癌症类型")
    logger.info("=" * 80)

    try:
        # 加载.env文件
        load_dotenv()

        # 初始化组件（使用模拟提供者避免API调用）
        logger.info("📦 初始化组件...")

        llm_config = {
            'default_provider': 'mock',
            'mock': {'enabled': True}
        }

        llm_manager = LLMManager(config=llm_config)
        embedding_manager = get_embedding_manager(model_type="jina")

        retrieval_manager = MedicalRetrievalManager(
            es_host=os.getenv('ELASTICSEARCH_HOST', 'localhost'),
            es_port=int(os.getenv('ELASTICSEARCH_PORT', '9200')),
            milvus_host=os.getenv('MILVUS_HOST', 'localhost'),
            milvus_port=int(os.getenv('MILVUS_PORT', '19530')),
            embedding_manager=embedding_manager
        )

        # 增强版ReAct代理（带章节智能）
        agent = EnhancedMedicalReActAgent(
            llm_manager=llm_manager,
            retrieval_manager=retrieval_manager,
            embedding_manager=embedding_manager,
            es_host=os.getenv('ELASTICSEARCH_HOST', 'localhost'),
            es_port=int(os.getenv('ELASTICSEARCH_PORT', '9200'))
        )

        logger.info("✅ 组件初始化完成\n")

        # 测试多种癌症类型
        test_cases = [
            {
                "query": "腺癌的图像特征是什么？",
                "expected_section": "腺癌",
                "expected_chapter": "第一节",
                "expected_page": 16
            },
            {
                "query": "鳞癌的图像特征是什么？",
                "expected_section": "鳞癌",
                "expected_chapter": "第二节",
                "expected_page": 25
            },
            {
                "query": "小细胞癌的图像特征是什么？",
                "expected_section": "小细胞癌",
                "expected_chapter": "第三节",
                "expected_page": None  # 未知具体页码
            },
            {
                "query": "黏液腺癌的图像特征是什么？",
                "expected_section": "黏液腺癌",
                "expected_chapter": "第九节",
                "expected_page": None  # 未知具体页码
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            logger.info("=" * 80)
            logger.info(f"测试案例 {i}: {test_case['query']}")
            logger.info(f"期望结果: {test_case['expected_chapter']} {test_case['expected_section']}")
            logger.info("=" * 80)

            # 执行查询
            result = agent.process_query(test_case['query'])

            # 分析结果
            logger.info(f"\n📊 查询结果分析:")
            logger.info(f"  - 查询类型: {result['metadata'].get('query_type', 'unknown')}")
            logger.info(f"  - 识别实体: {result['metadata'].get('entities', [])}")
            logger.info(f"  - 检索文档数: {result['metadata'].get('search_results_count', 0)}")
            logger.info(f"  - 置信度: {result['confidence']:.2f}")

            # 检查前3个结果
            success = False
            found_pages = []

            if result.get('retrieved_documents'):
                logger.info(f"\n📚 前3个检索结果分析:")
                for j, doc in enumerate(result['retrieved_documents'][:3], 1):
                    page = doc.get('page_number', 0)
                    chapter = doc.get('chapter_title', '未知')
                    section = doc.get('section_title', '未知')
                    score = doc.get('score', 0.0)
                    chapter_score = doc.get('chapter_matching_score', 0.0)

                    found_pages.append(page)

                    logger.info(f"  文档 {j} (第{page}页):")
                    logger.info(f"    - 章节: '{chapter}' - '{section}'")
                    logger.info(f"    - 总分: {score:.2f} (章节匹配分: {chapter_score:.2f})")

                    # 检查是否匹配期望
                    if (test_case['expected_section'] in section and
                        test_case['expected_chapter'] in chapter):
                        if j <= 3:  # 在前3个结果中
                            success = True
                            logger.info(f"    ✅ 找到目标章节！")
                        else:
                            logger.info(f"    ⚠️  找到目标章节但排名较后")

            # 总结结果
            if success:
                logger.info(f"\n🎉 测试成功！{test_case['expected_chapter']} {test_case['expected_section']} 在前3个结果中")
            else:
                logger.info(f"\n⚠️  测试未完全成功")
                if test_case['expected_page']:
                    logger.info(f"期望页码 {test_case['expected_page']}，实际找到页码: {found_pages}")

            logger.info("-" * 40)

        logger.info("=" * 80)
        logger.info("✅ 通用章节匹配测试完成")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_universal_chapter_matching()