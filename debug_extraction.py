#!/usr/bin/env python3
"""
调试章节信息提取方法
"""

import sys
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

import logging
import re

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_extraction():
    """调试章节信息提取"""

    logger.info("=" * 80)
    logger.info("调试章节信息提取方法")
    logger.info("=" * 80)

    # 模拟从Elasticsearch获取的内容
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

    logger.info("测试内容:")
    logger.info(test_content[:500])

    # 调用提取方法
    def _extract_chapter_info_from_content(content: str) -> dict:
        """从内容中提取章节信息"""
        chapter_info = {
            'chapter_title': '',
            'section_title': '',
            'chapter_number': '',
            'section_number': ''
        }

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

        logger.debug("开始提取章节信息...")

        # 提取章节标题
        for pattern in chapter_patterns:
            match = re.search(pattern, content)
            if match:
                chapter_info['chapter_title'] = match.group(1).strip()
                # 提取章节号
                chapter_num_match = re.search(r'第([一二三四五六七八九十\d]+)章', match.group(0))
                if chapter_num_match:
                    chapter_info['chapter_number'] = chapter_num_match.group(1)
                logger.debug(f"找到章节标题: {chapter_info['chapter_title']}")
                break

        # 提取节标题
        for pattern in section_patterns:
            match = re.search(pattern, content)
            if match:
                section_title = match.group(1).strip()
                # 清理节标题（移除页码等无关信息）
                section_title = re.sub(r'\d+\.?\s*$', '', section_title)  # 移除结尾的页码
                section_title = re.sub(r'\.+$', '', section_title)      # 移除结尾的省略号
                section_title = section_title.strip()

                if section_title:  # 确保标题不为空
                    chapter_info['section_title'] = section_title
                    # 提取节号
                    section_num_match = re.search(r'第([一二三四五六七八九十\d]+)节', match.group(0))
                    if section_num_match:
                        chapter_info['section_number'] = section_num_match.group(1)
                    logger.debug(f"找到节标题: {chapter_info['section_title']}")
                    break

        # 如果没有找到明确的章节标题，尝试从内容开头提取可能的标题
        if not chapter_info['chapter_title'] and not chapter_info['section_title']:
            lines = content.strip().split('\n')
            for line in lines[:10]:  # 检查前10行
                line = line.strip()
                # 查找可能的章节标题（包含特定关键词）
                if any(keyword in line for keyword in ['章', '节', '部分', '篇']):
                    if len(line) < 50 and not line.startswith('图'):  # 避免图像说明
                        if not chapter_info['chapter_title'] and '章' in line:
                            chapter_info['chapter_title'] = line.strip('#').strip()
                            logger.debug(f"从开头找到可能的章节标题: {chapter_info['chapter_title']}")
                        elif not chapter_info['section_title'] and '节' in line:
                            chapter_info['section_title'] = line.strip('#').strip()
                            logger.debug(f"从开头找到可能的节标题: {chapter_info['section_title']}")

        logger.debug(f"最终提取结果: {chapter_info}")
        return chapter_info

    # 测试提取
    result = _extract_chapter_info_from_content(test_content)

    logger.info(f"\n提取结果:")
    logger.info(f"章节标题: '{result['chapter_title']}'")
    logger.info(f"节标题: '{result['section_title']}'")
    logger.info(f"章节号: '{result['chapter_number']}'")
    logger.info(f"节号: '{result['section_number']}'")

    logger.info("\n" + "=" * 80)
    logger.info("✅ 调试完成")
    logger.info("=" * 80)

if __name__ == "__main__":
    debug_extraction()