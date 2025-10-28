"""
RAGAS测试框架核心实现
集成现有检索系统，执行召回率测试
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agent.retrieval_manager import MedicalRetrievalManager
from src.agent.enhanced_retrieval_manager import EnhancedMedicalRetrievalManager
from config import TEST_CONFIG, DB_CONFIG
from question_generator import MedicalQuestion

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    scores: List[float]
    retrieval_time: float
    method: str  # 'semantic', 'keyword', 'hybrid'

@dataclass
class TestResult:
    """测试结果数据结构"""
    question_id: str
    question: str
    expected_chunk_ids: List[str]
    retrieved_chunk_ids: List[str]
    retrieval_scores: List[float]
    recall_at_k: Dict[int, float]  # {3: 0.5, 5: 0.6, 10: 0.7}
    precision_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    retrieval_time: float
    method: str
    success: bool
    error_message: Optional[str] = None

class RAGASTestFramework:
    """RAGAS测试框架"""

    def __init__(self, test_data_path: str):
        self.test_data_path = Path(test_data_path)
        self.questions = []
        self.test_results = []
        self.retrieval_manager = None
        self.embedding_model = None
        self.executor = ThreadPoolExecutor(max_workers=TEST_CONFIG["max_workers"])

    async def initialize(self):
        """初始化测试框架"""
        logger.info("初始化RAGAS测试框架...")

        # 加载测试问题
        await self._load_test_questions()

        # 初始化嵌入模型（必需用于语义搜索）
        logger.info("初始化嵌入模型...")
        try:
            from src.embedding.embedding_models import get_embedding_manager
            embedding_manager = get_embedding_manager(model_type="jina")
            logger.info("✅ 嵌入模型初始化成功")
        except Exception as e:
            logger.error(f"❌ 嵌入模型初始化失败: {e}")
            logger.info("将使用基础检索管理器（无语义搜索功能）")
            embedding_manager = None

        # 初始化检索管理器（使用正确的参数）
        logger.info("初始化检索管理器...")
        try:
            self.retrieval_manager = EnhancedMedicalRetrievalManager(
                es_host=DB_CONFIG.get('elasticsearch_host', 'localhost'),
                es_port=DB_CONFIG.get('elasticsearch_port', 9200),
                milvus_host=DB_CONFIG.get('milvus_host', 'localhost'),
                milvus_port=DB_CONFIG.get('milvus_port', 19530),
                embedding_manager=embedding_manager
            )
            logger.info("✅ 增强版检索管理器初始化成功")
        except Exception as e:
            logger.error(f"❌ 增强版检索管理器初始化失败: {e}")
            logger.info("将使用基础检索管理器")
            from src.agent.retrieval_manager import MedicalRetrievalManager
            self.retrieval_manager = MedicalRetrievalManager(
                es_host=DB_CONFIG.get('elasticsearch_host', 'localhost'),
                es_port=DB_CONFIG.get('elasticsearch_port', 9200)
            )

        logger.info(f"RAGAS测试框架初始化完成，加载了{len(self.questions)}个测试问题")

    async def _load_test_questions(self):
        """加载测试问题"""
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"测试数据文件不存在: {self.test_data_path}")

        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)

        self.questions = []
        for q_data in questions_data:
            question = MedicalQuestion(**q_data)
            self.questions.append(question)

        logger.info(f"已加载{len(self.questions)}个测试问题")

    async def run_recall_test(self, top_k_values: List[int] = None,
                            batch_size: int = None) -> List[TestResult]:
        """运行召回率测试"""
        if top_k_values is None:
            top_k_values = TEST_CONFIG["top_k_values"]

        if batch_size is None:
            batch_size = TEST_CONFIG["batch_size"]

        logger.info(f"开始召回率测试，top-k值: {top_k_values}, 批大小: {batch_size}")

        self.test_results = []
        total_questions = len(self.questions)

        # 分批处理测试问题
        for i in range(0, total_questions, batch_size):
            batch_questions = self.questions[i:i + batch_size]
            batch_results = await self._process_batch(batch_questions, top_k_values)
            self.test_results.extend(batch_results)

            logger.info(f"处理进度: {min(i + batch_size, total_questions)}/{total_questions}")

        logger.info(f"召回率测试完成，共处理{len(self.test_results)}个问题")
        return self.test_results

    async def _process_batch(self, questions: List[MedicalQuestion],
                           top_k_values: List[int]) -> List[TestResult]:
        """处理一批测试问题"""
        tasks = []
        for question in questions:
            task = self._test_single_question(question, top_k_values)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"测试问题 {questions[i].id} 失败: {result}")
                # 创建失败的测试结果
                failed_result = TestResult(
                    question_id=questions[i].id,
                    question=questions[i].question,
                    expected_chunk_ids=questions[i].expected_chunk_ids,
                    retrieved_chunk_ids=[],
                    retrieval_scores=[],
                    recall_at_k={k: 0.0 for k in top_k_values},
                    precision_at_k={k: 0.0 for k in top_k_values},
                    f1_at_k={k: 0.0 for k in top_k_values},
                    hit_rate_at_k={k: 0.0 for k in top_k_values},
                    retrieval_time=0.0,
                    method="unknown",
                    success=False,
                    error_message=str(result)
                )
                valid_results.append(failed_result)
            else:
                valid_results.append(result)

        return valid_results

    async def _test_single_question(self, question: MedicalQuestion,
                                  top_k_values: List[int]) -> TestResult:
        """测试单个问题"""
        start_time = time.time()

        try:
            # 执行检索
            retrieval_result = await self._retrieve_documents(question.question, max(top_k_values))

            # 计算各项指标
            metrics = self._calculate_metrics(
                question.expected_chunk_ids,
                retrieval_result.retrieved_docs,
                top_k_values
            )

            retrieval_time = time.time() - start_time

            test_result = TestResult(
                question_id=question.id,
                question=question.question,
                expected_chunk_ids=question.expected_chunk_ids,
                retrieved_chunk_ids=[doc.get("id", f"chunk_{i:04d}") for i, doc in enumerate(retrieval_result.retrieved_docs)],
                retrieval_scores=retrieval_result.scores,
                recall_at_k=metrics["recall_at_k"],
                precision_at_k=metrics["precision_at_k"],
                f1_at_k=metrics["f1_at_k"],
                hit_rate_at_k=metrics["hit_rate_at_k"],
                retrieval_time=retrieval_time,
                method=retrieval_result.method,
                success=True
            )

            return test_result

        except Exception as e:
            logger.error(f"测试问题 {question.id} 时出错: {e}")
            retrieval_time = time.time() - start_time

            return TestResult(
                question_id=question.id,
                question=question.question,
                expected_chunk_ids=question.expected_chunk_ids,
                retrieved_chunk_ids=[],
                retrieval_scores=[],
                recall_at_k={k: 0.0 for k in top_k_values},
                precision_at_k={k: 0.0 for k in top_k_values},
                f1_at_k={k: 0.0 for k in top_k_values},
                hit_rate_at_k={k: 0.0 for k in top_k_values},
                retrieval_time=retrieval_time,
                method="error",
                success=False,
                error_message=str(e)
            )

    async def _retrieve_documents(self, query: str, top_k: int) -> RetrievalResult:
        """检索文档 - 使用真实的增强检索管理器"""
        start_time = time.time()

        try:
            logger.info(f"🔍 开始真实检索: '{query}' (top_k={top_k})")

            # 配置搜索参数 - 使用混合搜索模式以获得最佳效果
            search_config = {
                'search_type': 'hybrid',  # 使用混合搜索
                'top_k': top_k,
                'title_priority': True,   # 启用标题优先级
                'keyword_weight': 0.6     # 关键词搜索权重60%，语义搜索40%
            }

            # 调用真实的增强检索管理器
            results = self.retrieval_manager.enhanced_search(query, search_config)

            if not results:
                logger.warning(f"⚠️ 检索未返回结果，查询: '{query}'")
                # 返回空结果而不是抛出异常，让测试可以继续
                retrieval_time = time.time() - start_time
                return RetrievalResult(
                    query=query,
                    retrieved_docs=[],
                    scores=[],
                    retrieval_time=retrieval_time,
                    method="hybrid"
                )

            logger.info(f"✅ 真实检索完成，找到 {len(results)} 个结果")

            # 转换结果格式以匹配测试框架期望的格式
            retrieved_docs = []
            scores = []

            for i, result in enumerate(results):
                # 确保所有必需的字段都存在
                doc = {
                    "id": result.get('doc_id', f"doc_{i:04d}"),
                    "content": result.get('content', ''),
                    "score": result.get('score', 0.0),
                    "title": result.get('chapter_title', '') + ' - ' + result.get('section_title', ''),
                    "chapter": result.get('chapter_title', ''),
                    "section": result.get('section_title', ''),
                    "page_number": result.get('page_number', 0),
                    "source": result.get('source', 'unknown'),
                    "search_type": result.get('search_type', 'hybrid'),
                    "title_match_score": result.get('title_match_score', 0.0),
                    "content_quality_score": result.get('content_quality_score', 0.0),
                    "is_descriptive": result.get('is_descriptive', False),
                    "has_medical_terms": result.get('has_medical_terms', False)
                }
                retrieved_docs.append(doc)
                scores.append(doc["score"])

            retrieval_time = time.time() - start_time

            logger.info(f"📊 检索统计 - 耗时: {retrieval_time:.3f}s, 最高分数: {max(scores) if scores else 0:.3f}")

            return RetrievalResult(
                query=query,
                retrieved_docs=retrieved_docs,
                scores=scores,
                retrieval_time=retrieval_time,
                method="hybrid"
            )

        except Exception as e:
            logger.error(f"❌ 真实文档检索失败: {e}")
            retrieval_time = time.time() - start_time
            # 返回错误结果而不是抛出异常，让测试可以继续
            return RetrievalResult(
                query=query,
                retrieved_docs=[],
                scores=[],
                retrieval_time=retrieval_time,
                method="error"
            )

    def _calculate_metrics(self, expected_chunk_ids: List[str],
                          retrieved_docs: List[Dict[str, Any]],
                          top_k_values: List[int]) -> Dict[str, Dict[int, float]]:
        """计算评估指标 - 直接使用chunk ID格式"""
        # 获取检索到的文档ID（已经是chunk_XXXX格式）
        retrieved_chunk_ids = [doc.get("id", f"chunk_{i:04d}") for i, doc in enumerate(retrieved_docs)]

        # 直接使用chunk ID进行匹配，无需映射
        metrics = {
            "recall_at_k": {},
            "precision_at_k": {},
            "f1_at_k": {},
            "hit_rate_at_k": {}
        }

        for k in top_k_values:
            # 计算召回率 - 直接使用chunk ID
            recall = self._calculate_recall_at_k(expected_chunk_ids, retrieved_chunk_ids, k)
            metrics["recall_at_k"][k] = recall

            # 计算精确率 - 直接使用chunk ID
            precision = self._calculate_precision_at_k(expected_chunk_ids, retrieved_chunk_ids, k)
            metrics["precision_at_k"][k] = precision

            # 计算F1分数
            f1 = self._calculate_f1_score(precision, recall)
            metrics["f1_at_k"][k] = f1

            # 计算命中率 - 直接使用chunk ID
            hit_rate = self._calculate_hit_rate_at_k(expected_chunk_ids, retrieved_chunk_ids, k)
            metrics["hit_rate_at_k"][k] = hit_rate

        return metrics

    def _map_document_ids(self, retrieved_doc_ids: List[str], expected_doc_ids: List[str],
                         retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """
        将真实文档ID(chunk_XXXX)映射到期望格式(doc_XXXX)

        策略：
        1. 如果检索到的ID已经在期望列表中，直接使用
        2. 否则，基于内容相似度和页面位置创建映射
        """
        mapped_ids = []

        for i, (doc_id, doc) in enumerate(zip(retrieved_doc_ids, retrieved_docs)):
            # 如果ID已经是期望格式，直接使用
            if doc_id in expected_doc_ids:
                mapped_ids.append(doc_id)
                continue

            # 对于chunk_XXXX格式，尝试智能映射
            if doc_id.startswith('chunk_'):
                # 提取数字部分
                try:
                    chunk_num = int(doc_id.split('_')[1])
                    # 映射到对应的doc_XXXX格式（基于chunk编号）
                    mapped_id = f"doc_{chunk_num:04d}"
                    mapped_ids.append(mapped_id)
                    logger.debug(f"映射文档ID: {doc_id} -> {mapped_id}")
                except (ValueError, IndexError):
                    # 如果无法解析，使用基于位置的映射
                    mapped_id = f"doc_{i+1:04d}"
                    mapped_ids.append(mapped_id)
                    logger.debug(f"位置映射文档ID: {doc_id} -> {mapped_id}")
            else:
                # 其他格式，使用基于位置的映射
                mapped_id = f"doc_{i+1:04d}"
                mapped_ids.append(mapped_id)
                logger.debug(f"默认映射文档ID: {doc_id} -> {mapped_id}")

        return mapped_ids

    def _calculate_recall_at_k(self, expected_doc_ids: List[str],
                              retrieved_doc_ids: List[str], k: int) -> float:
        """计算召回率@K"""
        if not expected_doc_ids:
            return 0.0

        retrieved_at_k = set(retrieved_doc_ids[:k])
        expected_set = set(expected_doc_ids)

        relevant_retrieved = len(expected_set.intersection(retrieved_at_k))
        return relevant_retrieved / len(expected_set) if expected_set else 0.0

    def _calculate_precision_at_k(self, expected_doc_ids: List[str],
                                 retrieved_doc_ids: List[str], k: int) -> float:
        """计算精确率@K"""
        if k == 0:
            return 0.0

        retrieved_at_k = set(retrieved_doc_ids[:k])
        expected_set = set(expected_doc_ids)

        relevant_retrieved = len(expected_set.intersection(retrieved_at_k))
        return relevant_retrieved / k

    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _calculate_hit_rate_at_k(self, expected_doc_ids: List[str],
                                retrieved_doc_ids: List[str], k: int) -> float:
        """计算命中率@K"""
        if not expected_doc_ids:
            return 0.0

        retrieved_at_k = set(retrieved_doc_ids[:k])
        expected_set = set(expected_doc_ids)

        # 如果至少有一个相关文档被检索到，则命中率为1
        return 1.0 if expected_set.intersection(retrieved_at_k) else 0.0

    def generate_evaluation_report(self) -> Dict[str, Any]:
        """生成评估报告"""
        if not self.test_results:
            return {"error": "没有测试结果可生成报告"}

        successful_results = [r for r in self.test_results if r.success]
        failed_results = [r for r in self.test_results if not r.success]

        # 计算整体指标
        overall_metrics = self._calculate_overall_metrics(successful_results)

        # 按难度分析
        difficulty_analysis = self._analyze_by_difficulty()

        # 按问题类型分析
        type_analysis = self._analyze_by_question_type()

        # 按疾病类别分析
        category_analysis = self._analyze_by_disease_category()

        report = {
            "test_summary": {
                "total_questions": len(self.test_results),
                "successful_tests": len(successful_results),
                "failed_tests": len(failed_results),
                "success_rate": len(successful_results) / len(self.test_results) if self.test_results else 0,
                "test_date": datetime.now().isoformat()
            },
            "overall_metrics": overall_metrics,
            "difficulty_analysis": difficulty_analysis,
            "question_type_analysis": type_analysis,
            "disease_category_analysis": category_analysis,
            "detailed_results": [self._test_result_to_dict(result) for result in self.test_results]
        }

        return report

    def _calculate_overall_metrics(self, results: List[TestResult]) -> Dict[str, Any]:
        """计算整体指标"""
        if not results:
            return {}

        metrics = {}

        # 计算各top-k的平均指标
        for k in TEST_CONFIG["top_k_values"]:
            recalls = [r.recall_at_k.get(k, 0.0) for r in results]
            precisions = [r.precision_at_k.get(k, 0.0) for r in results]
            f1_scores = [r.f1_at_k.get(k, 0.0) for r in results]
            hit_rates = [r.hit_rate_at_k.get(k, 0.0) for r in results]

            metrics[f"recall_at_{k}"] = {
                "mean": np.mean(recalls),
                "std": np.std(recalls),
                "min": np.min(recalls),
                "max": np.max(recalls),
                "median": np.median(recalls)
            }

            metrics[f"precision_at_{k}"] = {
                "mean": np.mean(precisions),
                "std": np.std(precisions),
                "min": np.min(precisions),
                "max": np.max(precisions),
                "median": np.median(precisions)
            }

            metrics[f"f1_at_{k}"] = {
                "mean": np.mean(f1_scores),
                "std": np.std(f1_scores),
                "min": np.min(f1_scores),
                "max": np.max(f1_scores),
                "median": np.median(f1_scores)
            }

            metrics[f"hit_rate_at_{k}"] = {
                "mean": np.mean(hit_rates),
                "std": np.std(hit_rates),
                "min": np.min(hit_rates),
                "max": np.max(hit_rates),
                "median": np.median(hit_rates)
            }

        # 计算平均检索时间
        retrieval_times = [r.retrieval_time for r in results]
        metrics["retrieval_time"] = {
            "mean": np.mean(retrieval_times),
            "std": np.std(retrieval_times),
            "min": np.min(retrieval_times),
            "max": np.max(retrieval_times),
            "median": np.median(retrieval_times)
        }

        return metrics

    def _analyze_by_difficulty(self) -> Dict[str, Any]:
        """按难度分析结果"""
        # 这里需要根据问题的难度信息进行分析
        # 暂时返回基础结构
        return {
            "basic": {"count": 0, "avg_recall": 0.0},
            "medium": {"count": 0, "avg_recall": 0.0},
            "hard": {"count": 0, "avg_recall": 0.0}
        }

    def _analyze_by_question_type(self) -> Dict[str, Any]:
        """按问题类型分析结果"""
        # 这里需要根据问题的类型信息进行分析
        # 暂时返回基础结构
        return {
            "concept": {"count": 0, "avg_recall": 0.0},
            "diagnosis": {"count": 0, "avg_recall": 0.0},
            "differential": {"count": 0, "avg_recall": 0.0},
            "case_analysis": {"count": 0, "avg_recall": 0.0}
        }

    def _analyze_by_disease_category(self) -> Dict[str, Any]:
        """按疾病类别分析结果"""
        # 这里需要根据问题的疾病类别信息进行分析
        # 暂时返回基础结构
        return {
            "lung_cancer": {"count": 0, "avg_recall": 0.0},
            "metastatic_tumor": {"count": 0, "avg_recall": 0.0},
            "rare_disease": {"count": 0, "avg_recall": 0.0},
            "rose_technique": {"count": 0, "avg_recall": 0.0}
        }

    def _test_result_to_dict(self, result: TestResult) -> Dict[str, Any]:
        """转换测试结果到字典"""
        return {
            "question_id": result.question_id,
            "question": result.question,
            "expected_chunk_ids": result.expected_chunk_ids,
            "retrieved_chunk_ids": result.retrieved_chunk_ids,
            "retrieval_scores": result.retrieval_scores,
            "recall_at_k": result.recall_at_k,
            "precision_at_k": result.precision_at_k,
            "f1_at_k": result.f1_at_k,
            "hit_rate_at_k": result.hit_rate_at_k,
            "retrieval_time": result.retrieval_time,
            "method": result.method,
            "success": result.success,
            "error_message": result.error_message
        }

    def save_results(self, output_path: str):
        """保存测试结果"""
        report = self.generate_evaluation_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"测试结果已保存到: {output_path}")

    def save_summary_report(self, output_path: str):
        """保存摘要报告"""
        report = self.generate_evaluation_report()

        # 生成Markdown格式的摘要报告
        summary_md = f"""# RAGAS召回率测试报告

## 测试摘要

- **测试时间**: {report['test_summary']['test_date']}
- **总问题数**: {report['test_summary']['total_questions']}
- **成功测试**: {report['test_summary']['successful_tests']}
- **失败测试**: {report['test_summary']['failed_tests']}
- **成功率**: {report['test_summary']['success_rate']:.2%}

## 整体性能指标

"""

        # 添加各top-k的指标
        for k in TEST_CONFIG["top_k_values"]:
            recall_data = report["overall_metrics"][f"recall_at_{k}"]
            precision_data = report["overall_metrics"][f"precision_at_{k}"]
            f1_data = report["overall_metrics"][f"f1_at_{k}"]

            summary_md += f"""### Top-{k} 指标

- **召回率**: {recall_data['mean']:.3f} (±{recall_data['std']:.3f})
- **精确率**: {precision_data['mean']:.3f} (±{precision_data['std']:.3f})
- **F1分数**: {f1_data['mean']:.3f} (±{f1_data['std']:.3f})
- **命中率**: {report['overall_metrics'][f'hit_rate_at_{k}']['mean']:.3f}

"""

        # 添加检索时间
        time_data = report["overall_metrics"]["retrieval_time"]
        summary_md += f"""## 检索性能

- **平均检索时间**: {time_data['mean']:.3f}秒 (±{time_data['std']:.3f})
- **最短检索时间**: {time_data['min']:.3f}秒
- **最长检索时间**: {time_data['max']:.3f}秒

## 结论与建议

基于测试结果，系统在不同top-k值下的召回率表现{'良好' if report['overall_metrics']['recall_at_5']['mean'] > 0.7 else '有待改进'}。
建议进一步优化检索算法和文档索引策略。
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_md)

        logger.info(f"摘要报告已保存到: {output_path}")

# 测试函数
async def test_ragas_framework():
    """测试RAGAS框架"""
    # 注意：这需要先生成测试问题
    framework = RAGASTestFramework("test_data/generated_questions.json")
    await framework.initialize()

    # 运行测试
    results = await framework.run_recall_test()

    # 生成报告
    report = framework.generate_evaluation_report()
    framework.save_results("test_data/test_results.json")
    framework.save_summary_report("evaluation_reports/summary_report.md")

    print("测试完成！")
    print(f"平均召回率@5: {report['overall_metrics']['recall_at_5']['mean']:.3f}")

if __name__ == "__main__":
    asyncio.run(test_ragas_framework())