#!/usr/bin/env python3
"""
自动化后端流程测试脚本
用于验证后端查询处理流程，无需交互式输入
"""

import requests
import json
import time
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API配置
API_BASE_URL = "http://localhost:8001"
QUERY_ENDPOINT = f"{API_BASE_URL}/query/sync"
STATUS_ENDPOINT = f"{API_BASE_URL}/status"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

def test_health_check():
    """测试健康检查"""
    logger.info("🔍 测试健康检查端点...")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        logger.info(f"✅ 健康检查成功: {response.status_code}")
        logger.info(f"响应: {response.json()}")
        return True
    except Exception as e:
        logger.error(f"❌ 健康检查失败: {e}")
        return False

def test_system_status():
    """测试系统状态"""
    logger.info("🔍 测试系统状态端点...")
    try:
        response = requests.get(STATUS_ENDPOINT, timeout=30)
        logger.info(f"✅ 系统状态检查成功: {response.status_code}")
        status_data = response.json()

        # 显示组件状态
        components = status_data.get('components', {})
        for component, comp_status in components.items():
            status = comp_status.get('status', 'unknown')
            logger.info(f"  📊 {component}: {status}")

        return True
    except Exception as e:
        logger.error(f"❌ 系统状态检查失败: {e}")
        return False

def test_simple_query():
    """测试简单查询"""
    logger.info("🚀 测试简单查询...")

    test_query = "鳞癌的图像特征是什么？"
    query_data = {
        'question': test_query,
        'user_id': 'test_user',
        'search_config': {
            'top_k': 3,
            'search_type': 'hybrid',
            'model_provider': 'mock'  # 使用模拟模型避免API调用
        }
    }

    try:
        start_time = time.time()
        response = requests.post(
            QUERY_ENDPOINT,
            json=query_data,
            headers={'Content-Type': 'application/json'},
            timeout=120
        )
        response_time = time.time() - start_time

        logger.info(f"✅ 查询成功！响应时间: {response_time:.2f}秒")
        logger.info(f"HTTP状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # 显示查询结果摘要
            logger.info(f"\n📋 查询结果摘要:")
            logger.info(f"  查询ID: {result.get('query_id', 'N/A')}")
            logger.info(f"  问题: {result.get('question', 'N/A')}")
            logger.info(f"  置信度: {result.get('confidence', 0):.2f}")
            logger.info(f"  响应时间: {result.get('response_time', 0):.2f}秒")
            logger.info(f"  使用模型: {result.get('model_used', 'N/A')}")

            # 显示搜索查询历史
            search_queries = result.get('search_queries', [])
            if search_queries:
                logger.info(f"\n🔍 搜索查询历史 ({len(search_queries)} 个):")
                for i, query in enumerate(search_queries, 1):
                    logger.info(f"  {i}. {query}")

            # 显示推理步骤
            reasoning_steps = result.get('reasoning_steps', [])
            if reasoning_steps:
                logger.info(f"\n🧠 推理步骤 ({len(reasoning_steps)} 步):")
                for i, step in enumerate(reasoning_steps, 1):
                    step_name = step.get('step', f'步骤{i}')
                    logger.info(f"  🔍 {step_name}")
                    if 'thought' in step:
                        logger.info(f"     💭 思考: {step['thought'][:100]}...")
                    if 'action' in step:
                        logger.info(f"     🛠️  动作: {step['action']}")
                    if 'observation' in step:
                        logger.info(f"     👁️  观察: {step['observation'][:100]}...")

            # 显示检索到的文档
            retrieved_docs = result.get('retrieved_documents', [])
            if retrieved_docs:
                logger.info(f"\n📚 检索到的文档 ({len(retrieved_docs)} 个):")
                for i, doc in enumerate(retrieved_docs[:3], 1):  # 显示前3个
                    logger.info(f"  📄 文档 {i}:")
                    logger.info(f"     📖 章节: {doc.get('chapter_title', 'N/A')}")
                    logger.info(f"     📑 小节: {doc.get('section_title', 'N/A')}")
                    logger.info(f"     📄 页码: 第{doc.get('page_number', 'N/A')}页")
                    logger.info(f"     ⭐ 相关度分数: {doc.get('score', 0):.3f}")
                    logger.info(f"     🔍 搜索类型: {doc.get('search_type', 'unknown')}")

            # 显示答案
            answer = result.get('answer', '无答案')
            logger.info(f"\n📝 答案: {answer[:200]}...")

            return True
        else:
            logger.error(f"❌ 查询失败: {response.status_code}")
            logger.error(f"错误信息: {response.text}")
            return False

    except requests.exceptions.Timeout:
        logger.error(f"❌ 查询超时 (120秒)")
        return False
    except Exception as e:
        logger.error(f"❌ 查询异常: {e}")
        return False

def main():
    """主函数"""
    logger.info("🚀 启动自动化后端流程测试")
    logger.info("="*80)

    # 测试健康检查
    if not test_health_check():
        logger.error("❌ 系统健康检查失败，终止测试")
        return

    # 测试系统状态
    if not test_system_status():
        logger.error("❌ 系统状态检查失败，终止测试")
        return

    logger.info("✅ 系统准备就绪，开始查询测试")

    # 测试简单查询
    if test_simple_query():
        logger.info("✅ 所有测试通过！")
    else:
        logger.error("❌ 查询测试失败")

    logger.info("🎯 测试完成")

if __name__ == "__main__":
    main()