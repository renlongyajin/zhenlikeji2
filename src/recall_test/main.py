"""
RAGAS召回率测试系统主程序
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_parser import MedicalDataParser
from question_generator import MedicalQuestionGenerator
from ragas_framework import RAGASTestFramework
from config import LOG_CONFIG

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"],
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_CONFIG["file"], encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

async def parse_medical_data(data_path: str):
    """解析医学数据"""
    logger.info("开始解析医学数据...")

    parser = MedicalDataParser(data_path)
    parsed_data = parser.parse()

    # 保存解析结果
    output_path = Path(__file__).parent / "data" / "parsed_medical_data.json"
    parser.save_parsed_data(str(output_path))

    # 打印统计信息
    categories = parser.get_disease_categories()
    logger.info("疾病分类统计:")
    for category, diseases in categories.items():
        logger.info(f"{category}: {len(diseases)}种疾病")

    return str(output_path)

async def generate_test_questions(parsed_data_path: str, num_questions: int = 150):
    """生成测试问题"""
    logger.info(f"开始生成{num_questions}个测试问题...")

    generator = MedicalQuestionGenerator(parsed_data_path)
    await generator.load_parsed_data()

    # 生成问题
    questions = await generator.generate_questions(num_questions)

    # 构建文档映射（使用模拟文档ID）
    mock_doc_ids = [f"doc_{i:04d}" for i in range(1, 201)]
    generator.build_expected_doc_mapping(mock_doc_ids)

    # 保存问题
    questions_path = Path(__file__).parent / "test_data" / "generated_questions.json"
    generator.save_questions(str(questions_path))

    # 生成统计
    stats = generator.generate_statistics()
    logger.info("问题生成统计:")
    logger.info(json.dumps(stats, ensure_ascii=False, indent=2))

    return str(questions_path)

async def run_recall_test(questions_path: str, top_k_values: list = None):
    """运行召回率测试"""
    logger.info("开始运行召回率测试...")

    if top_k_values is None:
        top_k_values = [3, 5, 10]

    framework = RAGASTestFramework(questions_path)
    await framework.initialize()

    # 运行测试
    results = await framework.run_recall_test(top_k_values)

    # 生成报告
    report = framework.generate_evaluation_report()

    # 保存结果
    results_path = Path(__file__).parent / "test_data" / "test_results.json"
    framework.save_results(str(results_path))

    # 保存摘要报告
    summary_path = Path(__file__).parent / "evaluation_reports" / "summary_report.md"
    framework.save_summary_report(str(summary_path))

    logger.info("召回率测试完成！")
    logger.info(f"平均召回率@5: {report['overall_metrics']['recall_at_5']['mean']:.3f}")
    logger.info(f"结果已保存到: {results_path}")
    logger.info(f"摘要报告: {summary_path}")

    return report

async def run_full_pipeline(data_path: str, num_questions: int = 150, skip_generation: bool = False):
    """运行完整测试流程"""
    logger.info("开始运行完整RAGAS测试流程...")

    try:
        # 步骤1: 解析医学数据
        if not skip_generation:
            parsed_data_path = await parse_medical_data(data_path)
        else:
            parsed_data_path = Path(__file__).parent / "data" / "parsed_medical_data.json"
            if not parsed_data_path.exists():
                logger.error("解析数据文件不存在，无法跳过生成步骤")
                return None
            parsed_data_path = str(parsed_data_path)

        # 步骤2: 生成测试问题
        if not skip_generation:
            questions_path = await generate_test_questions(parsed_data_path, num_questions)
        else:
            questions_path = Path(__file__).parent / "test_data" / "generated_questions.json"
            if not questions_path.exists():
                logger.error("测试问题文件不存在，无法跳过生成步骤")
                return None
            questions_path = str(questions_path)

        # 步骤3: 运行召回率测试
        report = await run_recall_test(questions_path)

        logger.info("完整RAGAS测试流程完成！")
        return report

    except Exception as e:
        logger.error(f"运行完整流程时出错: {e}")
        raise e

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAGAS召回率测试系统")
    parser.add_argument("--data-path", type=str, default="data/clean_data.md",
                       help="医学数据文件路径")
    parser.add_argument("--num-questions", type=int, default=150,
                       help="生成问题数量")
    parser.add_argument("--skip-generation", action="store_true",
                       help="跳过问题生成，直接运行测试")
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 10],
                       help="测试的top-k值")
    parser.add_argument("--step", type=str, choices=["parse", "generate", "test", "full"],
                       default="full", help="执行步骤")

    args = parser.parse_args()

    # 创建必要的目录
    Path(__file__).parent.joinpath("data").mkdir(exist_ok=True)
    Path(__file__).parent.joinpath("test_data").mkdir(exist_ok=True)
    Path(__file__).parent.joinpath("evaluation_reports").mkdir(exist_ok=True)
    Path(__file__).parent.joinpath("logs").mkdir(exist_ok=True)

    # 运行指定步骤
    if args.step == "parse":
        asyncio.run(parse_medical_data(args.data_path))
    elif args.step == "generate":
        parsed_data_path = Path(__file__).parent / "data" / "parsed_medical_data.json"
        if not parsed_data_path.exists():
            logger.error("请先运行解析步骤")
            return
        asyncio.run(generate_test_questions(str(parsed_data_path), args.num_questions))
    elif args.step == "test":
        questions_path = Path(__file__).parent / "test_data" / "generated_questions.json"
        if not questions_path.exists():
            logger.error("请先运行生成步骤")
            return
        asyncio.run(run_recall_test(str(questions_path), args.top_k))
    elif args.step == "full":
        asyncio.run(run_full_pipeline(args.data_path, args.num_questions, args.skip_generation))

if __name__ == "__main__":
    # 添加json模块导入
    import json
    main()