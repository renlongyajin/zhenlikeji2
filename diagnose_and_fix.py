#!/usr/bin/env python3
"""
系统诊断和修复脚本
详细分析连接问题并提供修复方案
"""
import sys
import os
import json
import logging
from datetime import datetime
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

def test_api_connections():
    """测试各种API连接"""
    logger.info("🔍 测试API连接...")

    results = {}

    # 测试DeepSeek API
    deepseek_key = os.getenv('DEEPSEEK_API_KEY')
    if deepseek_key:
        logger.info("🧪 测试DeepSeek API连接...")
        try:
            import openai
            client = openai.OpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com"
            )
            # 测试简单的API调用
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            logger.info(f"✅ DeepSeek API连接成功，响应: {response.choices[0].message.content[:50]}...")
            results['deepseek'] = {'status': 'success', 'response': response.choices[0].message.content[:50]}
        except Exception as e:
            logger.error(f"❌ DeepSeek API连接失败: {e}")
            results['deepseek'] = {'status': 'failed', 'error': str(e)}
    else:
        logger.warning("⚠️  未配置DeepSeek API密钥")
        results['deepseek'] = {'status': 'not_configured'}

    # 测试其他API
    apis_to_test = [
        ('DashScope', os.getenv('DASHSCOPE_API_KEY'), 'https://dashscope.aliyuncs.com/api/v1'),
        ('SiliconFlow', os.getenv('SILICONFLOW_API_KEY'), 'https://api.siliconflow.cn/v1'),
    ]

    for api_name, api_key, base_url in apis_to_test:
        if api_key:
            logger.info(f"🧪 测试{api_name} API连接...")
            try:
                import openai
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                response = client.chat.completions.create(
                    model="qwen-turbo",  # 使用较稳定的模型
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                logger.info(f"✅ {api_name} API连接成功")
                results[api_name.lower()] = {'status': 'success'}
            except Exception as e:
                logger.error(f"❌ {api_name} API连接失败: {e}")
                results[api_name.lower()] = {'status': 'failed', 'error': str(e)}
        else:
            logger.warning(f"⚠️  未配置{api_name} API密钥")
            results[api_name.lower()] = {'status': 'not_configured'}

    return results

def test_database_connections():
    """测试数据库连接"""
    logger.info("\n🗄️ 测试数据库连接...")

    results = {}

    # 测试Elasticsearch
    es_host = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    es_port = int(os.getenv('ELASTICSEARCH_PORT', '9200'))

    logger.info(f"🧪 测试Elasticsearch连接 ({es_host}:{es_port})...")
    try:
        response = requests.get(f"http://{es_host}:{es_port}/_cluster/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ Elasticsearch连接成功")
            logger.info(f"  集群状态: {health_data.get('status', 'unknown')}")
            logger.info(f"  节点数量: {health_data.get('number_of_nodes', 0)}")
            logger.info(f"  文档数量: {health_data.get('active_primary_shards', 0)}")
            results['elasticsearch'] = {'status': 'success', 'details': health_data}
        else:
            logger.error(f"❌ Elasticsearch连接异常: HTTP {response.status_code}")
            results['elasticsearch'] = {'status': 'failed', 'error': f'HTTP {response.status_code}'}
    except Exception as e:
        logger.error(f"❌ Elasticsearch连接失败: {e}")
        results['elasticsearch'] = {'status': 'failed', 'error': str(e)}

    # 测试Milvus
    milvus_host = os.getenv('MILVUS_HOST', 'localhost')
    milvus_port = int(os.getenv('MILVUS_PORT', '19530'))

    logger.info(f"\n🧪 测试Milvus连接 ({milvus_host}:{milvus_port})...")
    try:
        from pymilvus import connections, utility

        # 尝试连接
        connections.connect(
            alias="test_connection",
            host=milvus_host,
            port=str(milvus_port),
            timeout=30
        )

        # 测试连接
        collections = utility.list_collections()
        logger.info(f"✅ Milvus连接成功")
        logger.info(f"  集合数量: {len(collections)}")
        logger.info(f"  集合列表: {collections}")
        results['milvus'] = {'status': 'success', 'collections': collections}

        # 断开测试连接
        connections.disconnect("test_connection")

    except Exception as e:
        logger.error(f"❌ Milvus连接失败: {e}")
        results['milvus'] = {'status': 'failed', 'error': str(e)}

    return results

def analyze_and_fix_llm_issues():
    """分析并修复LLM连接问题"""
    logger.info("\n🔧 分析LLM连接问题...")

    # 检查DeepSeek特定的proxies错误
    deepseek_key = os.getenv('DEEPSEEK_API_KEY')
    if deepseek_key:
        logger.info("🔍 检查DeepSeek客户端初始化问题...")
        try:
            import openai
            import inspect

            # 检查OpenAI客户端的构造函数参数
            sig = inspect.signature(openai.OpenAI.__init__)
            logger.info(f"OpenAI.OpenAI构造函数参数: {list(sig.parameters.keys())}")

            # 尝试用最小参数创建客户端
            client = openai.OpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com"
            )
            logger.info("✅ 基础客户端创建成功")

            # 检查是否有proxies参数问题
            if 'proxies' in sig.parameters:
                logger.warning("⚠️  发现proxies参数，可能导致兼容性问题")
            else:
                logger.info("✅ 未发现proxies参数问题")

        except Exception as e:
            logger.error(f"❌ 客户端分析失败: {e}")

    # 提供修复建议
    logger.info("\n💡 修复建议:")
    logger.info("1. 降级openai库版本: pip install openai==1.12.0")
    logger.info("2. 检查网络代理设置")
    logger.info("3. 验证API密钥有效性")
    logger.info("4. 考虑使用其他LLM提供者作为备选")

def create_fixed_llm_config():
    """创建修复后的LLM配置"""
    logger.info("\n🔧 创建修复后的LLM配置...")

    # 测试各个API的可用性
    api_results = test_api_connections()

    # 按优先级排序可用的API
    available_providers = []
    provider_configs = {}

    # 检查DeepSeek
    if api_results.get('deepseek', {}).get('status') == 'success':
        available_providers.append('deepseek')
        provider_configs['deepseek'] = {
            'api_key': os.getenv('DEEPSEEK_API_KEY'),
            'base_url': 'https://api.deepseek.com',
            'model': 'deepseek-reasoner'
        }

    # 检查DashScope
    if api_results.get('dashscope', {}).get('status') == 'success':
        available_providers.append('qwen')
        provider_configs['qwen'] = {
            'api_key': os.getenv('DASHSCOPE_API_KEY'),
            'base_url': 'https://dashscope.aliyuncs.com/api/v1',
            'model': 'qwen-max'
        }

    # 检查SiliconFlow
    if api_results.get('siliconflow', {}).get('status') == 'success':
        available_providers.append('siliconflow')
        provider_configs['siliconflow'] = {
            'api_key': os.getenv('SILICONFLOW_API_KEY'),
            'base_url': 'https://api.siliconflow.cn/v1',
            'model': 'Qwen/Qwen2.5-7B-Instruct'
        }

    # 总是添加模拟提供者作为最后的备选
    available_providers.append('mock')
    provider_configs['mock'] = {}

    config = {
        'llm': {
            'default_provider': available_providers[0] if available_providers else 'mock',
            **provider_configs
        }
    }

    logger.info(f"✅ 修复后的配置:")
    logger.info(f"  默认提供者: {config['llm']['default_provider']}")
    logger.info(f"  可用提供者: {available_providers}")

    return config

def test_with_fixed_config():
    """使用修复后的配置进行测试"""
    logger.info("\n🚀 使用修复配置进行测试...")

    # 获取修复后的配置
    llm_config = create_fixed_llm_config()
    db_results = test_database_connections()

    # 创建完整配置
    config = {
        **llm_config,
        'retrieval': {
            'es_host': os.getenv('ELASTICSEARCH_HOST', 'localhost'),
            'es_port': int(os.getenv('ELASTICSEARCH_PORT', '9200')),
            'milvus_host': os.getenv('MILVUS_HOST', 'localhost'),
            'milvus_port': int(os.getenv('MILVUS_PORT', '19530'))
        },
        'embedding': {
            'type': 'jina',
            'api_key': os.getenv('JINA_API_KEY'),
            'model': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-zh-v1.5'),
            'dimension': int(os.getenv('EMBEDDING_DIMENSION', '1024'))
        },
        'architecture': {
            'use_langgraph': True,
            'max_iterations': 3,
            'fallback_to_legacy': True
        }
    }

    logger.info("🔧 完整配置已创建，可以保存到配置文件中使用")

    # 保存配置
    with open('fixed_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info("✅ 修复配置已保存到: fixed_config.json")

    return config, db_results

def main():
    """主函数"""
    logger.info("="*60)
    logger.info("🔧 系统诊断和修复工具")
    logger.info("="*60)

    # 运行所有诊断测试
    api_results = test_api_connections()
    db_results = test_database_connections()
    analyze_and_fix_llm_issues()

    # 生成修复配置
    config, db_status = test_with_fixed_config()

    # 总结报告
    logger.info("\n" + "="*60)
    logger.info("📊 诊断总结报告")
    logger.info("="*60)

    logger.info("\n🔌 API连接状态:")
    for api, status in api_results.items():
        status_icon = "✅" if status['status'] == 'success' else "❌" if status['status'] == 'failed' else "⚠️"
        logger.info(f"  {status_icon} {api}: {status['status']}")

    logger.info("\n🗄️ 数据库连接状态:")
    for db, status in db_results.items():
        status_icon = "✅" if status['status'] == 'success' else "❌"
        logger.info(f"  {status_icon} {db}: {status['status']}")

    logger.info("\n📋 修复建议:")
    logger.info("1. 使用生成的 fixed_config.json 配置")
    logger.info("2. 检查网络连接和防火墙设置")
    logger.info("3. 验证API密钥有效性")
    logger.info("4. 考虑升级相关库版本")

    logger.info("\n🎯 下一步操作:")
    logger.info("1. 运行: python test_with_fixed_config.py")
    logger.info("2. 或使用交互式测试: python test_interactive.py")

    return 0

if __name__ == "__main__":
    sys.exit(main())

# 创建快速测试脚本
test_script_content = '''#!/usr/bin/env python3
"""使用修复配置的快速测试"""
import json
import logging
from src.agent.rag_engine import create_rag_engine, RAGQuery

logging.basicConfig(level=logging.INFO)

# 加载修复后的配置
with open('fixed_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 创建引擎
engine = create_rag_engine(config)

# 测试查询
query = engine.create_query("腺癌特征有哪些？")
response = engine.process_query_sync(query)

print(f"问题: {response.question}")
print(f"答案: {response.answer[:200]}...")
print(f"置信度: {response.confidence}")
print(f"文档数量: {len(response.retrieved_documents)}")
'''

with open('test_with_fixed_config.py', 'w', encoding='utf-8') as f:
    f.write(test_script_content)

print("\n✅ 快速测试脚本已创建: test_with_fixed_config.py")