#!/usr/bin/env python3
"""
真实LLM模型后端流程测试
用于验证使用真实LLM模型时的完整后端处理流程
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

def test_real_llm_query():
    """测试真实LLM模型查询"""
    logger.info("🚀 测试真实LLM模型查询...")

    test_query = "什么是肺腺癌的ROSE技术诊断特征？"
    query_data = {
        'question': test_query,
        'user_id': 'test_user',
        'search_config': {
            'top_k': 5,
            'search_type': 'hybrid',
            'model_provider': 'deepseek'  # 使用真实LLM模型
        }
    }

    logger.info(f"📋 测试问题: {test_query}")
    logger.info(f"🔧 搜索配置: {query_data['search_config']}")

    try:
        start_time = time.time()
        response = requests.post(
            QUERY_ENDPOINT,
            json=query_data,
            headers={'Content-Type': 'application/json'},
            timeout=300  # 5分钟超时
        )
        response_time = time.time() - start_time

        logger.info(f"✅ 查询成功！响应时间: {response_time:.2f}秒")
        logger.info(f"HTTP状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # 显示完整的后端处理流程
            logger.info(f"\n🔍 完整后端处理流程分析:")
            logger.info(f"="*80)

            # 基本信息
            logger.info(f"📋 查询基本信息:")
            logger.info(f"  查询ID: {result.get('query_id', 'N/A')}")
            logger.info(f"  问题: {result.get('question', 'N/A')}")
            logger.info(f"  置信度: {result.get('confidence', 0):.2f}")
            logger.info(f"  响应时间: {result.get('response_time', 0):.2f}秒")
            logger.info(f"  使用模型: {result.get('model_used', 'N/A')}")

            # 搜索查询历史
            search_queries = result.get('search_queries', [])
            if search_queries:
                logger.info(f"\n🔍 搜索查询历史 ({len(search_queries)} 个):")
                for i, query in enumerate(search_queries, 1):
                    logger.info(f"  {i}. {query}")

            # 推理步骤详细分析
            reasoning_steps = result.get('reasoning_steps', [])
            if reasoning_steps:
                logger.info(f"\n🧠 推理步骤详细分析 ({len(reasoning_steps)} 步):")
                for i, step in enumerate(reasoning_steps, 1):
                    step_name = step.get('step', f'步骤{i}')
                    logger.info(f"\n  🔍 {step_name}:")

                    if 'thought' in step:
                        logger.info(f"     💭 思考过程: {step['thought']}")
                    if 'action' in step:
                        logger.info(f"     🛠️  执行动作: {step['action']}")
                    if 'observation' in step:
                        logger.info(f"     👁️  观察结果: {step['observation'][:200]}...")
                    if 'timestamp' in step:
                        logger.info(f"     ⏰ 时间戳: {step['timestamp']}")

            # 检索到的文档详细分析
            retrieved_docs = result.get('retrieved_documents', [])
            if retrieved_docs:
                logger.info(f"\n📚 检索到的文档详细分析 ({len(retrieved_docs)} 个):")
                for i, doc in enumerate(retrieved_docs, 1):
                    logger.info(f"\n  📄 文档 {i}:")
                    logger.info(f"     📖 章节标题: {doc.get('chapter_title', 'N/A')}")
                    logger.info(f"     📑 小节标题: {doc.get('section_title', 'N/A')}")
                    logger.info(f"     📄 页码: 第{doc.get('page_number', 'N/A')}页")
                    logger.info(f"     ⭐ 相关度分数: {doc.get('score', 0):.3f}")
                    logger.info(f"     🔍 搜索类型: {doc.get('search_type', 'unknown')}")

                    # 增强版检索的额外信息
                    if 'title_match_score' in doc:
                        logger.info(f"     🎯 标题匹配分数: {doc.get('title_match_score', 0):.3f}")
                    if 'content_quality_score' in doc:
                        logger.info(f"     📊 内容质量分数: {doc.get('content_quality_score', 0):.3f}")
                    if 'is_descriptive' in doc:
                        logger.info(f"     📝 是否描述性内容: {doc.get('is_descriptive', False)}")
                    if 'has_medical_terms' in doc:
                        logger.info(f"     🏥 是否包含医学术语: {doc.get('has_medical_terms', False)}")

                    # 内容预览
                    content = doc.get('content', '')
                    if len(content) > 300:
                        preview = content[:300] + "..."
                    else:
                        preview = content
                    logger.info(f"     📝 内容预览: {preview}")

            # 最终答案
            answer = result.get('answer', '无答案')
            logger.info(f"\n📝 最终答案:")
            logger.info(f"{answer}")

            # 元数据
            metadata = result.get('metadata', {})
            if metadata:
                logger.info(f"\n📊 元数据:")
                for key, value in metadata.items():
                    logger.info(f"  {key}: {value}")

            return True
        else:
            logger.error(f"❌ 查询失败: {response.status_code}")
            logger.error(f"错误信息: {response.text}")
            return False

    except requests.exceptions.Timeout:
        logger.error(f"❌ 查询超时 (300秒)")
        return False
    except Exception as e:
        logger.error(f"❌ 查询异常: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False

def main():
    """主函数"""
    logger.info("🚀 启动真实LLM模型后端流程测试")
    logger.info("="*80)
    logger.info("🏥 医学RAG问答系统 - 真实LLM模型测试")
    logger.info("="*80)

    # 测试真实LLM模型查询
    if test_real_llm_query():
        logger.info("✅ 真实LLM模型测试通过！")
    else:
        logger.error("❌ 真实LLM模型测试失败")

    logger.info("🎯 测试完成")

if __name__ == "__main__":
    main()