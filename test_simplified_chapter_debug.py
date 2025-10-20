#!/usr/bin/env python3
"""
调试简化的章节匹配逻辑 - 启用详细日志
"""

import sys
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

import logging
import os
from src.agent.enhanced_react_agent import ChapterIntelligence

# 配置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simplified_chapter_matching_debug():
    """测试简化的章节匹配逻辑 - 启用详细日志"""

    logger.info("=" * 80)
    logger.info("测试简化的章节匹配逻辑 - 详细调试")
    logger.info("=" * 80)

    try:
        # 初始化章节智能模块
        chapter_intelligence = ChapterIntelligence(
            es_base_url=f"http://{os.getenv('ELASTICSEARCH_HOST', 'localhost')}:{os.getenv('ELASTICSEARCH_PORT', '9200')}",
            es_index="medical_documents"
        )

        # 测试腺癌
        logger.info(f"\n🔍 测试实体: '腺癌'")
        logger.info("-" * 40)

        chapter_info = chapter_intelligence.query_chapter_info("腺癌", top_k=3)

        if chapter_info:
            logger.info(f"✅ 找到 {len(chapter_info)} 个相关章节:")
            for i, info in enumerate(chapter_info, 1):
                logger.info(f"  {i}. 第{info['page_number']}页: '{info['chapter_title']}' - '{info['section_title']}' (得分: {info['score']:.2f})")
        else:
            logger.info(f"⚠️  未找到相关章节信息")

        logger.info("\n" + "=" * 80)
        logger.info("✅ 详细调试测试完成")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simplified_chapter_matching_debug()