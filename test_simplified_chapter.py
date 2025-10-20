#!/usr/bin/env python3
"""
测试简化的章节匹配逻辑
"""

import sys
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

import logging
import os
from dotenv import load_dotenv
from src.agent.enhanced_react_agent import ChapterIntelligence

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simplified_chapter_matching():
    """测试简化的章节匹配逻辑"""

    logger.info("=" * 80)
    logger.info("测试简化的章节匹配逻辑")
    logger.info("=" * 80)

    try:
        # 加载.env文件
        load_dotenv()

        # 初始化章节智能模块
        chapter_intelligence = ChapterIntelligence(
            es_base_url=f"http://{os.getenv('ELASTICSEARCH_HOST', 'localhost')}:{os.getenv('ELASTICSEARCH_PORT', '9200')}",
            es_index="medical_documents"
        )

        # 测试多种癌症类型
        test_entities = ["腺癌", "鳞癌", "小细胞癌", "黏液腺癌"]

        for entity in test_entities:
            logger.info(f"\n🔍 测试实体: '{entity}'")
            logger.info("-" * 40)

            # 查询章节信息
            chapter_info = chapter_intelligence.query_chapter_info(entity, top_k=3)

            if chapter_info:
                logger.info(f"✅ 找到 {len(chapter_info)} 个相关章节:")
                for i, info in enumerate(chapter_info, 1):
                    logger.info(f"  {i}. 第{info['page_number']}页: '{info['chapter_title']}' - '{info['section_title']}' (得分: {info['score']:.2f})")
                    logger.info(f"     内容预览: {info['content_preview'][:100]}...")
            else:
                logger.info(f"⚠️  未找到相关章节信息")

        logger.info("\n" + "=" * 80)
        logger.info("✅ 简化章节匹配测试完成")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simplified_chapter_matching()