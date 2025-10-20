#!/usr/bin/env python3
"""
直接测试章节智能模块 - 强制调试输出
"""

import sys
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

import logging
import os
import json

# 强制设置调试级别
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 也设置章节智能模块的日志级别
chapter_logger = logging.getLogger('src.agent.enhanced_react_agent')
chapter_logger.setLevel(logging.DEBUG)

from src.agent.enhanced_react_agent import ChapterIntelligence

def test_chapter_intelligence_direct():
    """直接测试章节智能模块"""

    logger.info("=" * 80)
    logger.info("直接测试章节智能模块")
    logger.info("=" * 80)

    try:
        # 初始化章节智能模块
        chapter_intelligence = ChapterIntelligence(
            es_base_url=f"http://{os.getenv('ELASTICSEARCH_HOST', 'localhost')}:{os.getenv('ELASTICSEARCH_PORT', '9200')}",
            es_index="medical_documents"
        )

        # 强制设置模块日志级别
        chapter_intelligence_logger = logging.getLogger('src.agent.enhanced_react_agent.ChapterIntelligence')
        chapter_intelligence_logger.setLevel(logging.DEBUG)

        # 测试腺癌
        entity = "腺癌"
        logger.info(f"\n🔍 测试实体: '{entity}'")
        logger.info("-" * 40)

        # 手动调用查询方法
        chapter_info = chapter_intelligence.query_chapter_info(entity, top_k=3)

        logger.info(f"\n最终结果:")
        if chapter_info:
            logger.info(f"✅ 找到 {len(chapter_info)} 个相关章节:")
            for i, info in enumerate(chapter_info, 1):
                logger.info(f"  {i}. 第{info['page_number']}页: '{info['chapter_title']}' - '{info['section_title']}' (得分: {info['score']:.2f})")
        else:
            logger.info(f"⚠️  未找到相关章节信息")

        logger.info("\n" + "=" * 80)
        logger.info("✅ 直接测试完成")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chapter_intelligence_direct()