#!/usr/bin/env python3
"""
ReAct系统测试套件
测试ReAct智能代理、RAG引擎和API服务的完整功能
"""

import sys
import os

# 解决相对导入问题：将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import asyncio
import json
import time
import requests
from typing import Dict, List, Any
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReActSystemTester:
    """ReAct系统测试器"""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """初始化测试器"""
        self.api_base_url = api_base_url
        self.test_results = []

    async def test_system_health(self) -> bool:
        """测试系统健康状态"""
        logger.info("🧪 测试系统健康状态...")

        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ 系统健康状态正常: {data['status']}")
                return True
            else:
                logger.error(f"❌ 系统健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ 系统健康检查失败: {e}")
            return False

    async def test_system_status(self) -> bool:
        """测试系统状态"""
        logger.info("🧪 测试系统状态...")

        try:
            response = requests.get(f"{self.api_base_url}/status", timeout=30)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ 系统状态正常")
                logger.info(f"📊 Elasticsearch文档数: {data['components']['retrieval']['elasticsearch']['document_count']}")
                logger.info(f"📊 Milvus向量数: {data['components']['retrieval']['milvus']['vector_count']}")
                logger.info(f"🔧 活动LLM提供者: {data['components']['llm']['active_provider']}")
                return True
            else:
                logger.error(f"❌ 系统状态检查失败: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ 系统状态检查失败: {e}")
            return False

    async def test_basic_query(self) -> bool:
        """测试基础查询"""
        logger.info("🧪 测试基础查询...")

        test_questions = [
            "什么是肺部恶性肿瘤的ROSE细胞学特征？",
            "腺癌的病理特征有哪些？",
            "ROSE技术在肺部疾病诊断中的作用是什么？",
            "细胞核增大在恶性肿瘤诊断中的意义？",
            "快速现场评价技术的实施要点？"
        ]

        success_count = 0

        for question in test_questions:
            try:
                start_time = time.time()

                # 发送查询请求
                payload = {
                    "question": question,
                    "user_id": "test_user",
                    "metadata": {"test": True}
                }

                response = requests.post(
                    f"{self.api_base_url}/query/sync",
                    json=payload,
                    timeout=60
                )

                response_time = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()

                    # 验证响应结构
                    required_fields = ['query_id', 'question', 'answer', 'confidence', 'response_time']
                    if all(field in data for field in required_fields):
                        logger.info(f"✅ 查询成功: '{question[:30]}...'")
                        logger.info(f"   响应时间: {response_time:.2f}s")
                        logger.info(f"   置信度: {data['confidence']}")
                        logger.info(f"   答案长度: {len(data['answer'])} 字符")
                        success_count += 1

                        # 保存测试结果
                        self.test_results.append({
                            'test_type': 'basic_query',
                            'question': question,
                            'success': True,
                            'response_time': response_time,
                            'confidence': data['confidence'],
                            'answer_length': len(data['answer'])
                        })
                    else:
                        logger.warning(f"⚠️ 响应结构不完整: {list(data.keys())}")
                else:
                    logger.error(f"❌ 查询失败: {response.status_code} - {response.text}")
                    self.test_results.append({
                        'test_type': 'basic_query',
                        'question': question,
                        'success': False,
                        'error': f"HTTP {response.status_code}"
                    })

            except Exception as e:
                logger.error(f"❌ 查询测试失败: {e}")
                self.test_results.append({
                    'test_type': 'basic_query',
                    'question': question,
                    'success': False,
                    'error': str(e)
                })

        success_rate = success_count / len(test_questions)
        logger.info(f"📊 基础查询测试完成，成功率: {success_rate:.1%}")
        return success_rate >= 0.8  # 80% 成功率阈值

    async def test_search_types(self) -> bool:
        """测试不同搜索类型"""
        logger.info("🧪 测试不同搜索类型...")

        search_configs = [
            {"search_type": "keyword"},
            {"search_type": "semantic"},
            {"search_type": "hybrid", "keyword_weight": 0.5},
            {"search_type": "hybrid", "keyword_weight": 0.7}
        ]

        test_question = "肺部恶性肿瘤的细胞学特征"
        success_count = 0

        for config in search_configs:
            try:
                payload = {
                    "question": test_question,
                    "search_config": config,
                    "user_id": "test_user"
                }

                response = requests.post(
                    f"{self.api_base_url}/query/sync",
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ 搜索类型 {config['search_type']} 测试成功")
                    success_count += 1

                    self.test_results.append({
                        'test_type': 'search_types',
                        'search_config': config,
                        'success': True,
                        'confidence': data['confidence']
                    })
                else:
                    logger.error(f"❌ 搜索类型 {config['search_type']} 测试失败")
                    self.test_results.append({
                        'test_type': 'search_types',
                        'search_config': config,
                        'success': False,
                        'error': f"HTTP {response.status_code}"
                    })

            except Exception as e:
                logger.error(f"❌ 搜索类型测试失败: {e}")
                self.test_results.append({
                    'test_type': 'search_types',
                    'search_config': config,
                    'success': False,
                    'error': str(e)
                })

        success_rate = success_count / len(search_configs)
        logger.info(f"📊 搜索类型测试完成，成功率: {success_rate:.1%}")
        return success_rate >= 0.75

    async def test_llm_providers(self) -> bool:
        """测试LLM提供者"""
        logger.info("🧪 测试LLM提供者...")

        try:
            # 获取可用提供者
            response = requests.get(f"{self.api_base_url}/llm/providers")
            if response.status_code != 200:
                logger.error(f"❌ 获取LLM提供者列表失败: {response.status_code}")
                return False

            providers_data = response.json()
            available_providers = providers_data.get('available_providers', [])
            logger.info(f"📋 可用LLM提供者: {available_providers}")

            test_question = "什么是ROSE技术？"
            success_count = 0

            for provider in available_providers:
                try:
                    # 切换到该提供者
                    switch_response = requests.post(f"{self.api_base_url}/llm/providers/{provider}")
                    if switch_response.status_code != 200:
                        logger.warning(f"⚠️ 切换LLM提供者 {provider} 失败")
                        continue

                    # 测试查询
                    payload = {
                        "question": test_question,
                        "user_id": "test_user"
                    }

                    query_response = requests.post(
                        f"{self.api_base_url}/query/sync",
                        json=payload,
                        timeout=60
                    )

                    if query_response.status_code == 200:
                        data = query_response.json()
                        logger.info(f"✅ LLM提供者 {provider} 测试成功")
                        success_count += 1

                        self.test_results.append({
                            'test_type': 'llm_providers',
                            'provider': provider,
                            'success': True,
                            'confidence': data['confidence']
                        })
                    else:
                        logger.error(f"❌ LLM提供者 {provider} 测试失败")
                        self.test_results.append({
                            'test_type': 'llm_providers',
                            'provider': provider,
                            'success': False,
                            'error': f"HTTP {query_response.status_code}"
                        })

                except Exception as e:
                    logger.error(f"❌ LLM提供者 {provider} 测试失败: {e}")
                    self.test_results.append({
                        'test_type': 'llm_providers',
                        'provider': provider,
                        'success': False,
                        'error': str(e)
                    })

            success_rate = success_count / max(len(available_providers), 1)
            logger.info(f"📊 LLM提供者测试完成，成功率: {success_rate:.1%}")
            return success_rate >= 0.5

        except Exception as e:
            logger.error(f"❌ LLM提供者测试失败: {e}")
            return False

    async def test_query_suggestions(self) -> bool:
        """测试查询建议"""
        logger.info("🧪 测试查询建议...")

        test_partial_queries = ["肺部", "ROSE", "细胞", "肿瘤", "诊断"]
        success_count = 0

        for partial_query in test_partial_queries:
            try:
                response = requests.get(
                    f"{self.api_base_url}/suggestions",
                    params={"q": partial_query, "max_suggestions": 3},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    suggestions = data.get('suggestions', [])
                    logger.info(f"✅ 查询建议 '{partial_query}' 成功，获得 {len(suggestions)} 个建议")
                    success_count += 1

                    self.test_results.append({
                        'test_type': 'query_suggestions',
                        'partial_query': partial_query,
                        'success': True,
                        'suggestion_count': len(suggestions)
                    })
                else:
                    logger.error(f"❌ 查询建议 '{partial_query}' 失败")
                    self.test_results.append({
                        'test_type': 'query_suggestions',
                        'partial_query': partial_query,
                        'success': False,
                        'error': f"HTTP {response.status_code}"
                    })

            except Exception as e:
                logger.error(f"❌ 查询建议测试失败: {e}")
                self.test_results.append({
                    'test_type': 'query_suggestions',
                    'partial_query': partial_query,
                    'success': False,
                    'error': str(e)
                })

        success_rate = success_count / len(test_partial_queries)
        logger.info(f"📊 查询建议测试完成，成功率: {success_rate:.1%}")
        return success_rate >= 0.8

    async def test_batch_queries(self) -> bool:
        """测试批量查询"""
        logger.info("🧪 测试批量查询...")

        batch_questions = [
            "肺部恶性肿瘤的特征是什么？",
            "ROSE技术的优势有哪些？",
            "如何进行细胞学诊断？",
            "腺癌和鳞癌的区别？"
        ]

        try:
            payload = [
                {
                    "question": question,
                    "user_id": "test_user"
                }
                for question in batch_questions
            ]

            start_time = time.time()
            response = requests.post(
                f"{self.api_base_url}/batch",
                json=payload,
                timeout=120
            )
            batch_time = time.time() - start_time

            if response.status_code == 200:
                results = response.json()
                success_count = sum(1 for r in results if r['confidence'] > 0)

                logger.info(f"✅ 批量查询测试成功")
                logger.info(f"   批量大小: {len(batch_questions)}")
                logger.info(f"   成功数量: {success_count}")
                logger.info(f"   总耗时: {batch_time:.2f}s")
                logger.info(f"   平均耗时: {batch_time/len(batch_questions):.2f}s")

                self.test_results.append({
                    'test_type': 'batch_queries',
                    'success': True,
                    'batch_size': len(batch_questions),
                    'success_count': success_count,
                    'total_time': batch_time,
                    'average_time': batch_time/len(batch_questions)
                })

                return success_count >= len(batch_questions) * 0.75
            else:
                logger.error(f"❌ 批量查询失败: {response.status_code}")
                self.test_results.append({
                    'test_type': 'batch_queries',
                    'success': False,
                    'error': f"HTTP {response.status_code}"
                })
                return False

        except Exception as e:
            logger.error(f"❌ 批量查询测试失败: {e}")
            self.test_results.append({
                'test_type': 'batch_queries',
                'success': False,
                'error': str(e)
            })
            return False

    async def test_performance(self) -> bool:
        """测试性能"""
        logger.info("🧪 测试性能...")

        test_question = "肺部恶性肿瘤的诊断方法有哪些？"
        num_tests = 5
        response_times = []

        for i in range(num_tests):
            try:
                start_time = time.time()

                payload = {
                    "question": test_question,
                    "user_id": "test_user"
                }

                response = requests.post(
                    f"{self.api_base_url}/query/sync",
                    json=payload,
                    timeout=60
                )

                response_time = time.time() - start_time

                if response.status_code == 200:
                    response_times.append(response_time)
                    logger.info(f"   第 {i+1} 次测试: {response_time:.2f}s")
                else:
                    logger.error(f"   第 {i+1} 次测试失败: {response.status_code}")

            except Exception as e:
                logger.error(f"   第 {i+1} 次测试异常: {e}")

        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)

            logger.info(f"📊 性能测试结果:")
            logger.info(f"   平均响应时间: {avg_response_time:.2f}s")
            logger.info(f"   最大响应时间: {max_response_time:.2f}s")
            logger.info(f"   最小响应时间: {min_response_time:.2f}s")

            self.test_results.append({
                'test_type': 'performance',
                'success': True,
                'average_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'min_response_time': min_response_time,
                'test_count': len(response_times)
            })

            # 性能要求：平均响应时间 < 10秒
            return avg_response_time < 10.0
        else:
            logger.error("❌ 性能测试失败，没有成功响应")
            return False

    async def test_concurrent_queries(self) -> bool:
        """测试并发查询"""
        logger.info("🧪 测试并发查询...")

        test_questions = [
            "肺部肿瘤的分类？",
            "ROSE技术的原理？",
            "细胞学检查的步骤？",
            "恶性肿瘤的特征？",
            "诊断准确率如何？"
        ]

        async def single_query(question: str):
            try:
                payload = {
                    "question": question,
                    "user_id": "test_user"
                }

                response = requests.post(
                    f"{self.api_base_url}/query/sync",
                    json=payload,
                    timeout=60
                )

                return response.status_code == 200
            except:
                return False

        # 执行并发查询
        tasks = [single_query(q) for q in test_questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        success_rate = success_count / len(test_questions)

        logger.info(f"📊 并发查询测试完成，成功率: {success_rate:.1%}")

        self.test_results.append({
            'test_type': 'concurrent_queries',
            'success': success_rate >= 0.8,
            'success_rate': success_rate,
            'concurrent_count': len(test_questions)
        })

        return success_rate >= 0.8

    def generate_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        logger.info("📋 生成测试报告...")

        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get('success', False))

        # 按测试类型分组
        test_by_type = {}
        for result in self.test_results:
            test_type = result['test_type']
            if test_type not in test_by_type:
                test_by_type[test_type] = []
            test_by_type[test_type].append(result)

        # 计算各类型成功率
        type_success_rates = {}
        for test_type, results in test_by_type.items():
            type_passed = sum(1 for r in results if r.get('success', False))
            type_success_rates[test_type] = type_passed / len(results) if results else 0

        # 性能统计
        performance_results = [r for r in self.test_results if r['test_type'] == 'performance']
        avg_response_time = performance_results[0]['average_response_time'] if performance_results else 0

        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'overall_success_rate': passed_tests / max(total_tests, 1),
                'test_timestamp': datetime.now().isoformat()
            },
            'test_by_type': test_by_type,
            'type_success_rates': type_success_rates,
            'performance_metrics': {
                'average_response_time': avg_response_time,
                'performance_tests': performance_results
            },
            'detailed_results': self.test_results
        }

        return report

    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始运行ReAct系统完整测试...")
        logger.info("=" * 60)

        start_time = time.time()
        test_results = {}

        # 1. 系统健康检查
        logger.info("\n" + "="*40)
        logger.info("1. 系统健康检查")
        logger.info("="*40)
        test_results['health_check'] = await self.test_system_health()

        # 2. 系统状态检查
        logger.info("\n" + "="*40)
        logger.info("2. 系统状态检查")
        logger.info("="*40)
        test_results['system_status'] = await self.test_system_status()

        # 3. 基础查询测试
        logger.info("\n" + "="*40)
        logger.info("3. 基础查询测试")
        logger.info("="*40)
        test_results['basic_queries'] = await self.test_basic_query()

        # 4. 搜索类型测试
        logger.info("\n" + "="*40)
        logger.info("4. 搜索类型测试")
        logger.info("="*40)
        test_results['search_types'] = await self.test_search_types()

        # 5. LLM提供者测试
        logger.info("\n" + "="*40)
        logger.info("5. LLM提供者测试")
        logger.info("="*40)
        test_results['llm_providers'] = await self.test_llm_providers()

        # 6. 查询建议测试
        logger.info("\n" + "="*40)
        logger.info("6. 查询建议测试")
        logger.info("="*40)
        test_results['query_suggestions'] = await self.test_query_suggestions()

        # 7. 批量查询测试
        logger.info("\n" + "="*40)
        logger.info("7. 批量查询测试")
        logger.info("="*40)
        test_results['batch_queries'] = await self.test_batch_queries()

        # 8. 性能测试
        logger.info("\n" + "="*40)
        logger.info("8. 性能测试")
        logger.info("="*40)
        test_results['performance'] = await self.test_performance()

        # 9. 并发测试
        logger.info("\n" + "="*40)
        logger.info("9. 并发测试")
        logger.info("="*40)
        test_results['concurrent_queries'] = await self.test_concurrent_queries()

        # 生成测试报告
        total_time = time.time() - start_time
        report = self.generate_test_report()

        # 计算总体结果
        all_tests_passed = all(test_results.values())
        passed_count = sum(test_results.values())
        total_count = len(test_results)

        logger.info("\n" + "="*60)
        logger.info("📊 测试总结")
        logger.info("="*60)
        logger.info(f"总测试时间: {total_time:.2f}s")
        logger.info(f"测试项目: {passed_count}/{total_count} 通过")
        logger.info(f"总体结果: {'✅ 通过' if all_tests_passed else '❌ 失败'}")

        # 详细结果
        logger.info("\n详细测试结果:")
        for test_name, passed in test_results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            logger.info(f"  {test_name}: {status}")

        final_report = {
            'test_execution': {
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': total_time,
                'test_results': test_results,
                'overall_passed': all_tests_passed
            },
            'detailed_report': report
        }

        return final_report

# 独立测试函数
def test_react_system_locally():
    """本地测试ReAct系统组件"""
    logger.info("🧪 开始本地ReAct系统测试...")

    # 添加详细调试信息
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info(f"Python路径前3项: {sys.path[:3]}")
    logger.info(f"项目根目录检测: {os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")

    try:
        # 测试组件导入
        try:
            logger.info("尝试绝对导入...")
            from src.agent.react_agent import MedicalReActAgent
            from src.agent.llm_manager import LLMManager, create_llm_manager
            from src.agent.retrieval_manager import MedicalRetrievalManager, create_retrieval_manager
            from src.agent.rag_engine import RAGEngine, create_rag_engine
            logger.info("✅ 所有组件导入成功")
        except ImportError as e:
            logger.warning(f"⚠️ 绝对导入失败: {e}")
            logger.info(f"错误类型: {type(e).__name__}")
            import traceback
            logger.info("完整错误追踪:")
            logger.info(traceback.format_exc())

            logger.info("尝试相对导入回退...")
            try:
                # 回退到相对导入（如果在包内运行）
                from .react_agent import MedicalReActAgent
                from .llm_manager import LLMManager, create_llm_manager
                from .retrieval_manager import MedicalRetrievalManager, create_retrieval_manager
                from .rag_engine import RAGEngine, create_rag_engine
                logger.info("✅ 相对导入成功")
            except ImportError as e2:
                logger.error(f"❌ 相对导入也失败: {e2}")
                logger.error("两种导入方式都失败，无法继续测试")
                return False

        # 测试创建默认配置
        try:
            from src.agent.llm_manager import create_default_llm_config
            from src.agent.rag_engine import create_default_rag_config
        except ImportError:
            from .llm_manager import create_default_llm_config
            from .rag_engine import create_default_rag_config

        llm_config = create_default_llm_config()
        rag_config = create_default_rag_config()

        logger.info("✅ 默认配置创建成功")

        # 测试创建各个管理器
        llm_manager = create_llm_manager(llm_config)
        retrieval_manager = create_retrieval_manager()

        logger.info("✅ 管理器创建成功")

        # 测试创建RAG引擎
        rag_engine = create_rag_engine(rag_config)

        logger.info("✅ RAG引擎创建成功")

        # 测试简单查询
        test_query = rag_engine.create_query("什么是ROSE技术？", user_id="test_user")
        response = rag_engine.process_query_sync(test_query)

        logger.info(f"✅ 简单查询测试成功")
        logger.info(f"   问题: {response.question}")
        logger.info(f"   答案长度: {len(response.answer)} 字符")
        logger.info(f"   置信度: {response.confidence}")
        logger.info(f"   响应时间: {response.response_time:.2f}s")

        return True

    except Exception as e:
        logger.error(f"❌ 本地ReAct系统测试失败: {e}")
        return False

# 主测试函数
async def main():
    """主测试函数"""
    logger.info("🚀 启动ReAct系统测试...")

    # 首先进行本地组件测试
    local_test_passed = test_react_system_locally()
    if not local_test_passed:
        logger.error("❌ 本地组件测试失败，停止API测试")
        return

    # 等待API服务启动
    logger.info("⏳ 等待API服务启动...")
    await asyncio.sleep(5)

    # 创建测试器
    tester = ReActSystemTester()

    # 运行完整测试
    test_report = await tester.run_all_tests()

    # 保存测试报告
    report_file = f"/home/ubuntu/myproject/zhenlikeji2/logs/react_system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)

    logger.info(f"📄 测试报告已保存: {report_file}")

    # 输出测试总结
    overall_passed = test_report['test_execution']['overall_passed']
    logger.info("\n" + "="*60)
    logger.info("🎯 测试完成总结")
    logger.info("="*60)
    logger.info(f"整体结果: {'🎉 通过' if overall_passed else '💥 失败'}")
    logger.info(f"测试报告: {report_file}")

    return overall_passed

if __name__ == "__main__":
    asyncio.run(main())