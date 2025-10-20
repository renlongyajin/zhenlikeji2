#!/usr/bin/env python3
"""
调试实体感知的章节提取
"""

import sys
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

import logging
import re

# 配置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_entity_extraction():
    """调试实体感知的章节提取"""

    logger.info("=" * 80)
    logger.info("调试实体感知的章节提取")
    logger.info("=" * 80)

    # 模拟从Elasticsearch获取的内容 - 目录页
    test_content = """

日

录

第一章肺部实体恶性肿瘤的ROSE细胞学特点..··

第一节细胞及其成分径线增加

第二节细胞及其成分成角度..

第三节细胞及其成分浓染.

第四节细胞成分增多.

2

第五节细胞核膜厚而浆膜相对菲薄.

第六节细胞及其成分拥挤层叠..

.3

第七节 细胞及其成分排列紊乱

第八节细胞背景分析.

第二章肺部实体恶性肿瘤的ROSE细胞组学分型要点..

第一节腺癌

6

第二节鳞癌.

.15

第三节小细胞癌.

..22

第四节大细胞神经内分泌癌.

.27

第五节典型类癌..

.33

第六节不典型类癌...

.41

第七节黏液表皮样癌.

.47

第八节腺样囊性癌.

.53

..61

第九节黏液腺癌.

"""

    entity = "腺癌"

    logger.info(f"测试实体: '{entity}'")
    logger.info(f"测试内容长度: {len(test_content)} 字符")
    logger.info(f"内容预览 (前500字符):")
    logger.info(test_content[:500])

    # 构建实体特定的节标题模式
    entity_section_patterns = [
        rf'第([一二三四五六七八九十]+)节\s*{re.escape(entity)}',    # 第X节 实体
        rf'第(\d+)节\s*{re.escape(entity)}',                      # 第1节 实体
        rf'##\s*第([一二三四五六七八九十]+)节\s*{re.escape(entity)}',  # ## 第X节 实体
        rf'##\s*第(\d+)节\s*{re.escape(entity)}',                  # ## 第1节 实体
        rf'第([一二三四五六七八九十]+)节{re.escape(entity)}',       # 第X节实体 (无空格)
        rf'第(\d+)节{re.escape(entity)}',                          # 第1节实体 (无空格)
    ]

    logger.info(f"\n测试实体特定模式:")
    for i, pattern in enumerate(entity_section_patterns, 1):
        logger.info(f"  模式 {i}: {pattern}")
        match = re.search(pattern, test_content)
        if match:
            section_number = match.group(1)
            logger.info(f"    ✅ 找到匹配: 第{section_number}节 {entity}")
            logger.info(f"    匹配位置: {match.start()}-{match.end()}")
            # 显示上下文
            start = max(0, match.start() - 30)
            end = min(len(test_content), match.end() + 30)
            context = test_content[start:end]
            logger.info(f"    上下文: {context}")
        else:
            logger.info(f"    ❌ 未找到匹配")

    # 测试通用提取
    logger.info(f"\n测试通用章节提取:")

    # 章节标题模式
    chapter_patterns = [
        r'第[一二三四五六七八九十]+章\s*([^\n\r]+)',  # 第X章 标题
        r'第\d+章\s*([^\n\r]+)',                      # 第1章 标题
        r'#\s*第[一二三四五六七八九十]+章\s*([^\n\r]+)',  # # 第X章 标题
        r'#\s*第\d+章\s*([^\n\r]+)'                      # # 第1章 标题
    ]

    # 节标题模式 - 支持多种格式
    section_patterns = [
        r'第[一二三四五六七八九十]+节\s*([^\n\r]+)',    # 第X节 标题 (有空格)
        r'第\d+节\s*([^\n\r]+)',                      # 第1节 标题 (有空格)
        r'##\s*第[一二三四五六七八九十]+节\s*([^\n\r]+)',  # ## 第X节 标题 (有空格)
        r'##\s*第\d+节\s*([^\n\r]+)',                  # ## 第1节 标题 (有空格)
        r'第[一二三四五六七八九十]+节([^\n\r]*)',       # 第X节标题 (无空格)
        r'第\d+节([^\n\r]*)',                          # 第1节标题 (无空格)
        r'##\s*第[一二三四五六七八九十]+节([^\n\r]*)',   # ## 第X节标题 (无空格)
        r'##\s*第\d+节([^\n\r]*)'                      # ## 第1节标题 (无空格)
    ]

    # 如果没有指定实体或没找到包含实体的标题，使用通用提取
    logger.info("使用通用章节提取")

    # 提取章节标题
    for pattern in chapter_patterns:
        match = re.search(pattern, test_content)
        if match:
            chapter_title = match.group(1).strip()
            logger.info(f"找到章节标题: '{chapter_title}'")
            # 提取章节号
            chapter_num_match = re.search(r'第([一二三四五六七八九十\d]+)章', match.group(0))
            if chapter_num_match:
                chapter_number = chapter_num_match.group(1)
                logger.info(f"章节号: {chapter_number}")
            break

    # 提取节标题
    for pattern in section_patterns:
        match = re.search(pattern, test_content)
        if match:
            section_title = match.group(1).strip()
            # 清理节标题（移除页码等无关信息）
            section_title = re.sub(r'\d+\.?\s*$', '', section_title)  # 移除结尾的页码
            section_title = re.sub(r'\.+$', '', section_title)      # 移除结尾的省略号
            section_title = section_title.strip()

            if section_title:  # 确保标题不为空
                logger.info(f"找到节标题: '{section_title}'")
                # 提取节号
                section_num_match = re.search(r'第([一二三四五六七八九十\d]+)节', match.group(0))
                if section_num_match:
                    section_number = section_num_match.group(1)
                    logger.info(f"节号: {section_number}")
            break

    logger.info("\n" + "=" * 80)
    logger.info("✅ 调试完成")
    logger.info("=" * 80)

if __name__ == "__main__":
    debug_entity_extraction()