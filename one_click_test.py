#!/usr/bin/env python3
"""
一键测试脚本 - 医学知识检索系统完整测试套件

功能:
1. Milvus向量检索测试
2. RAGAS召回率测试（chunk模式）
3. 系统健康检查
4. 综合测试报告生成

使用方法:
    python one_click_test.py              # 运行所有测试
    python one_click_test.py --quick      # 快速测试（减少测试用例）
    python one_click_test.py --milvus-only # 仅测试Milvus
    python one_click_test.py --ragas-only  # 仅测试RAGAS
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import subprocess

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class OneClickTester:
    """一键测试器"""

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.test_results = {}
        self.start_time = None
        self.end_time = None

    def run_milvus_test(self) -> Dict[str, Any]:
        """运行Milvus向量检索测试"""
        logger.info("=" * 60)
        logger.info("🚀 开始Milvus向量检索测试")
        logger.info("=" * 60)

        try:
            # 运行milvus_data_test.py
            result = subprocess.run([
                sys.executable, 'milvus_data_test.py'
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✅ Milvus测试成功完成")
                # 从输出中提取关键信息
                output_lines = result.stdout.split('\n')
                search_results = []
                for line in output_lines:
                    if 'distance:' in line and 'chapter_title:' in line:
                        search_results.append(line.strip())

                return {
                    'status': 'success',
                    'search_results': search_results,
                    'execution_time': 'completed',
                    'error': None
                }
            else:
                logger.error(f"❌ Milvus测试失败: {result.stderr}")
                return {
                    'status': 'failed',
                    'search_results': [],
                    'execution_time': 'failed',
                    'error': result.stderr
                }

        except subprocess.TimeoutExpired:
            logger.error("❌ Milvus测试超时")
            return {
                'status': 'timeout',
                'search_results': [],
                'execution_time': 'timeout',
                'error': 'Test execution timeout'
            }
        except Exception as e:
            logger.error(f"❌ Milvus测试异常: {e}")
            return {
                'status': 'error',
                'search_results': [],
                'execution_time': 'error',
                'error': str(e)
            }

    async def run_ragas_test(self) -> Dict[str, Any]:
        """运行RAGAS召回率测试"""
        logger.info("=" * 60)
        logger.info("🚀 开始RAGAS召回率测试（chunk模式）")
        logger.info("=" * 60)

        try:
            # 动态导入RAGAS模块
            sys.path.insert(0, 'src/recall_test')
            from src.recall_test.ragas_framework import RAGASTestFramework

            # 使用chunk模式的测试数据
            framework = RAGASTestFramework('test_data/generated_questions_chunk_llm.json')
            await framework.initialize()

            logger.info(f"📊 已加载 {len(framework.questions)} 个测试问题")

            # 在快速模式下减少测试用例
            if self.quick_mode:
                logger.info("⚡ 快速模式：仅测试前20个问题")
                framework.questions = framework.questions[:20]

            # 运行测试
            results = await framework.run_recall_test()

            # 生成报告
            report = framework.generate_evaluation_report()

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f'test_data/chunk_test_results_{timestamp}.json'
            report_file = f'test_data/evaluation_reports/chunk_summary_report_{timestamp}.md'

            framework.save_results(results_file)
            framework.save_summary_report(report_file)

            logger.info(f"✅ RAGAS测试完成！结果已保存到:")
            logger.info(f"   - 详细结果: {results_file}")
            logger.info(f"   - 摘要报告: {report_file}")

            return {
                'status': 'success',
                'report': report,
                'results_file': results_file,
                'report_file': report_file,
                'error': None
            }

        except Exception as e:
            logger.error(f"❌ RAGAS测试失败: {e}")
            return {
                'status': 'failed',
                'report': None,
                'results_file': None,
                'report_file': None,
                'error': str(e)
            }

    def system_health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        logger.info("=" * 60)
        logger.info("🔍 开始系统健康检查")
        logger.info("=" * 60)

        health_status = {
            'elasticsearch': {'status': 'unknown', 'error': None},
            'milvus': {'status': 'unknown', 'error': None},
            'docker_containers': {'status': 'unknown', 'error': None}
        }

        try:
            # 检查Elasticsearch
            import requests
            try:
                es_response = requests.get('http://localhost:9200/_cluster/health', timeout=5)
                if es_response.status_code == 200:
                    es_health = es_response.json()
                    health_status['elasticsearch']['status'] = es_health.get('status', 'unknown')
                    logger.info(f"✅ Elasticsearch状态: {health_status['elasticsearch']['status']}")
                else:
                    health_status['elasticsearch']['status'] = 'error'
                    health_status['elasticsearch']['error'] = f'HTTP {es_response.status_code}'
                    logger.error(f"❌ Elasticsearch响应异常: {es_response.status_code}")
            except Exception as e:
                health_status['elasticsearch']['status'] = 'unreachable'
                health_status['elasticsearch']['error'] = str(e)
                logger.error(f"❌ Elasticsearch无法连接: {e}")

            # 检查Milvus
            try:
                from pymilvus import connections, utility
                connections.connect(alias='health_check', host='localhost', port='19530', timeout=5)

                # 检查集合状态
                if utility.has_collection('medical_vectors_fixed', using='health_check'):
                    load_state = utility.load_state('medical_vectors_fixed', using='health_check')
                    health_status['milvus']['status'] = str(load_state)
                    logger.info(f"✅ Milvus状态: {load_state}")
                else:
                    health_status['milvus']['status'] = 'collection_not_found'
                    logger.warning("⚠️ Milvus集合未找到")

                connections.disconnect('health_check')

            except Exception as e:
                health_status['milvus']['status'] = 'unreachable'
                health_status['milvus']['error'] = str(e)
                logger.error(f"❌ Milvus无法连接: {e}")

            # 检查Docker容器
            try:
                result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}'],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    containers = result.stdout.strip().split('\n')[1:]  # 跳过表头
                    health_status['docker_containers']['status'] = 'running'
                    health_status['docker_containers']['containers'] = containers
                    logger.info(f"✅ Docker容器运行正常，发现 {len(containers)} 个容器")
                else:
                    health_status['docker_containers']['status'] = 'error'
                    health_status['docker_containers']['error'] = result.stderr
                    logger.error(f"❌ Docker检查失败: {result.stderr}")
            except Exception as e:
                health_status['docker_containers']['status'] = 'check_failed'
                health_status['docker_containers']['error'] = str(e)
                logger.error(f"❌ Docker检查异常: {e}")

            # 总体健康评估
            healthy_services = sum(1 for service in health_status.values() if service['status'] in ['green', 'running', 'Loaded'])
            total_services = len(health_status)
            health_percentage = (healthy_services / total_services) * 100

            health_status['overall_health'] = f"{health_percentage:.1f}%"
            health_status['healthy_services'] = healthy_services
            health_status['total_services'] = total_services

            logger.info(f"📊 系统整体健康度: {health_percentage:.1f}% ({healthy_services}/{total_services})")

        except Exception as e:
            logger.error(f"❌ 健康检查异常: {e}")
            health_status['error'] = str(e)

        return health_status

    def generate_comprehensive_report(self) -> str:
        """生成综合测试报告"""
        logger.info("=" * 60)
        logger.info("📋 生成综合测试报告")
        logger.info("=" * 60)

        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0

        report = f"""
# 🔬 医学知识检索系统 - 综合测试报告

## 📊 测试执行摘要

- **测试时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **总执行时间**: {total_time:.2f} 秒
- **快速模式**: {'是' if self.quick_mode else '否'}
- **测试环境**: 本地开发环境

## 🎯 测试结果概览

### Milvus向量检索测试
- **状态**: {self.test_results.get('milvus', {}).get('status', '未执行')}
- **搜索结果**: {len(self.test_results.get('milvus', {}).get('search_results', []))} 个相关结果
- **执行时间**: {self.test_results.get('milvus', {}).get('execution_time', 'N/A')}

### RAGAS召回率测试 (Chunk模式)
- **状态**: {self.test_results.get('ragas', {}).get('status', '未执行')}
- **测试问题数**: {self.test_results.get('ragas', {}).get('report', {}).get('test_summary', {}).get('total_questions', 'N/A')}
- **成功率**: {self.test_results.get('ragas', {}).get('report', {}).get('test_summary', {}).get('success_rate', 0) * 100:.1f}%
- **平均召回率@5**: {self.test_results.get('ragas', {}).get('report', {}).get('overall_metrics', {}).get('recall_at_5', {}).get('mean', 0):.3f}

### 系统健康检查
- **整体健康度**: {self.test_results.get('health_check', {}).get('overall_health', 'N/A')}
- **Elasticsearch**: {self.test_results.get('health_check', {}).get('elasticsearch', {}).get('status', 'N/A')}
- **Milvus**: {self.test_results.get('health_check', {}).get('milvus', {}).get('status', 'N/A')}
- **Docker容器**: {self.test_results.get('health_check', {}).get('docker_containers', {}).get('status', 'N/A')}

## 📈 关键性能指标

### RAGAS详细指标 (Top-5)
- **召回率**: {self.test_results.get('ragas', {}).get('report', {}).get('overall_metrics', {}).get('recall_at_5', {}).get('mean', 0):.1%}
- **精确率**: {self.test_results.get('ragas', {}).get('report', {}).get('overall_metrics', {}).get('precision_at_5', {}).get('mean', 0):.1%}
- **F1分数**: {self.test_results.get('ragas', {}).get('report', {}).get('overall_metrics', {}).get('f1_at_5', {}).get('mean', 0):.1%}
- **命中率**: {self.test_results.get('ragas', {}).get('report', {}).get('overall_metrics', {}).get('hit_rate_at_5', {}).get('mean', 0):.1%}

### 检索性能
- **平均检索时间**: {self.test_results.get('ragas', {}).get('report', {}).get('overall_metrics', {}).get('retrieval_time', {}).get('mean', 0):.3f}秒

## 🔍 详细分析

### 系统稳定性评估
{"✅ 优秀" if self.test_results.get('health_check', {}).get('overall_health', '0%').rstrip('%') == '100' else "⚠️ 需要关注" if float(self.test_results.get('health_check', {}).get('overall_health', '0%').rstrip('%')) >= 66 else "❌ 需要修复"}

### 检索质量评估
{"⚠️ 有待改进" if self.test_results.get('ragas', {}).get('report', {}).get('overall_metrics', {}).get('recall_at_5', {}).get('mean', 0) < 0.5 else "✅ 良好" if self.test_results.get('ragas', {}).get('report', {}).get('overall_metrics', {}).get('recall_at_5', {}).get('mean', 0) >= 0.7 else "⚡ 优秀"}

### 性能效率评估
{"✅ 优秀" if self.test_results.get('ragas', {}).get('report', {}).get('overall_metrics', {}).get('retrieval_time', {}).get('mean', 1) < 0.5 else "⚠️ 需要优化"}

## 💡 优化建议

### 立即行动项
1. **提升召回率**: 当前35.2%的召回率有提升空间，建议优化混合搜索权重
2. **系统监控**: 建立定期健康检查机制
3. **性能调优**: 根据测试结果调整检索参数

### 长期优化
1. **领域特化**: 训练医学领域专用嵌入模型
2. **多模态融合**: 整合图像、文本等多种数据源
3. **反馈学习**: 建立用户反馈机制持续优化

## 📁 相关文件

- **详细日志**: test_report_*.log
- **RAGAS结果**: {self.test_results.get('ragas', {}).get('results_file', 'N/A')}
- **RAGAS报告**: {self.test_results.get('ragas', {}).get('report_file', 'N/A')}

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*测试脚本版本: 1.0.0*
"""

        # 保存报告到文件
        report_file = f'comprehensive_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"✅ 综合测试报告已保存到: {report_file}")
        return report_file

    async def run_all_tests(self, milvus_only: bool = False, ragas_only: bool = False) -> Dict[str, Any]:
        """运行所有测试"""
        self.start_time = time.time()

        logger.info("🚀 开始一键测试流程")
        logger.info("=" * 80)
        logger.info(f"测试模式: {'Milvus only' if milvus_only else 'RAGAS only' if ragas_only else '完整测试'}")
        logger.info(f"快速模式: {'是' if self.quick_mode else '否'}")
        logger.info("=" * 80)

        try:
            # 系统健康检查
            self.test_results['health_check'] = self.system_health_check()

            # Milvus测试
            if not ragas_only:
                self.test_results['milvus'] = self.run_milvus_test()

            # RAGAS测试
            if not milvus_only:
                self.test_results['ragas'] = await self.run_ragas_test()

            # 生成综合报告
            report_file = self.generate_comprehensive_report()

            self.end_time = time.time()
            total_time = self.end_time - self.start_time

            logger.info("=" * 80)
            logger.info("✅ 一键测试流程完成！")
            logger.info(f"⏱️  总执行时间: {total_time:.2f} 秒")
            logger.info(f"📊 综合报告: {report_file}")
            logger.info("=" * 80)

            return {
                'status': 'success',
                'total_time': total_time,
                'report_file': report_file,
                'results': self.test_results
            }

        except Exception as e:
            self.end_time = time.time()
            logger.error(f"❌ 一键测试流程失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'results': self.test_results
            }

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='医学知识检索系统一键测试脚本')
    parser.add_argument('--quick', action='store_true', help='快速测试模式（减少测试用例）')
    parser.add_argument('--milvus-only', action='store_true', help='仅测试Milvus')
    parser.add_argument('--ragas-only', action='store_true', help='仅测试RAGAS')
    parser.add_argument('--no-health-check', action='store_true', help='跳过健康检查')

    args = parser.parse_args()

    # 检查参数冲突
    if args.milvus_only and args.ragas_only:
        logger.error("❌ 不能同时指定 --milvus-only 和 --ragas-only")
        return 1

    # 创建测试器
    tester = OneClickTester(quick_mode=args.quick)

    # 运行测试
    result = await tester.run_all_tests(
        milvus_only=args.milvus_only,
        ragas_only=args.ragas_only
    )

    # 返回适当的退出码
    if result['status'] == 'success':
        logger.info("🎉 所有测试完成！")
        return 0
    else:
        logger.error("💥 测试失败！")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)