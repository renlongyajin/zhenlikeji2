#!/usr/bin/env python3
"""
测试腺癌查询是否优先返回第一节内容（用户反馈的问题）
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

def test_adenocarcinoma_priority():
    """测试腺癌查询优先级"""

    logger.info("=" * 80)
    logger.info("测试腺癌查询优先级 - 应该优先返回第一节内容")
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

        # 测试查询
        test_query = "腺癌的图像特征是什么？"

        logger.info(f"测试查询: {test_query}")
        logger.info("期望结果: 应该优先返回第16页（第一节 腺癌）的内容")
        logger.info("-" * 80)

        # 执行查询
        result = agent.process_query(test_query)

        # 分析结果
        logger.info(f"\n📊 查询结果分析:")
        logger.info(f"  - 查询类型: {result['metadata'].get('query_type', 'unknown')}")
        logger.info(f"  - 识别实体: {result['metadata'].get('entities', [])}")
        logger.info(f"  - 检索文档数: {result['metadata'].get('search_results_count', 0)}")
        logger.info(f"  - 推理步骤数: {result['metadata'].get('reasoning_steps_count', 0)}")
        logger.info(f"  - 置信度: {result['confidence']:.2f}")
        logger.info(f"  - 响应时间: {result['response_time']:.2f}秒")

        # 详细分析检索到的文档
        if result.get('retrieved_documents'):
            logger.info(f"\n📚 检索到的文档详细分析:")
            for j, doc in enumerate(result['retrieved_documents'], 1):
                page = doc.get('page_number', 0)
                chapter = doc.get('chapter_title', '未知')
                section = doc.get('section_title', '未知')
                score = doc.get('score', 0.0)
                chapter_score = doc.get('chapter_matching_score', 0.0)

                logger.info(f"  文档 {j} (第{page}页):")
                logger.info(f"    - 章节: '{chapter}' - '{section}'")
                logger.info(f"    - 总分: {score:.2f} (章节匹配分: {chapter_score:.2f})")

                # 检查是否是我们期望的页面16（第一节 腺癌）
                if page == 16 and section == "腺癌":
                    logger.info(f"    ✅ 找到目标页面！这是第16页（第一节 腺癌）")
                    if j <= 3:  # 如果在前3个结果中
                        logger.info(f"    ✅✅✅ 成功！目标页面在前3个结果中")
                    else:
                        logger.info(f"    ⚠️  目标页面排名较后，需要优化")

                logger.info(f"    - 内容预览: {doc.get('content', '')[:100]}...")

        # 检查是否成功
        success = False
        for j, doc in enumerate(result['retrieved_documents'], 1):
            if doc.get('page_number') == 16 and doc.get('section_title') == "腺癌":
                if j <= 3:
                    success = True
                break

        if success:
            logger.info(f"\n🎉 测试成功！腺癌第一节内容被正确优先返回")
        else:
            logger.info(f"\n⚠️  测试未完全成功，需要进一步优化章节匹配逻辑")

        logger.info("=" * 80)
        logger.info("✅ 测试完成")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_adenocarcinoma_priority()