#!/usr/bin/env python3
"""
交互式测试脚本 - 详细展示查询过程和结果
"""

import sys
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

import logging
import os
from dotenv import load_dotenv
from src.agent.enhanced_react_agent import EnhancedMedicalReActAgent
from src.agent.llm_manager import LLMManager
from src.agent.retrieval_manager import MedicalRetrievalManager
from src.embedding.embedding_models import get_embedding_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_section(text):
    """打印章节"""
    print("\n" + "-" * 80)
    print(f"  {text}")
    print("-" * 80)

def interactive_test():
    """交互式测试"""

    print_header("🧠 医学RAG智能问答系统 - 交互式测试")

    try:
        # 加载环境变量
        load_dotenv()

        print("\n📦 正在初始化系统组件...")

        # 初始化LLM管理器（使用真实的DeepSeek API）
        llm_config = {
            'default_provider': 'deepseek',
            'deepseek': {
                'api_key': os.getenv('DEEPSEEK_API_KEY'),
                'base_url': 'https://api.deepseek.com',
                'model': 'deepseek-reasoner'
            }
        }

        llm_manager = LLMManager(config=llm_config)
        print("  ✅ LLM管理器初始化完成")

        # 初始化嵌入模型
        embedding_manager = get_embedding_manager(model_type="jina")
        print("  ✅ 嵌入模型初始化完成")

        # 初始化检索管理器
        retrieval_manager = MedicalRetrievalManager(
            es_host=os.getenv('ELASTICSEARCH_HOST', 'localhost'),
            es_port=int(os.getenv('ELASTICSEARCH_PORT', '9200')),
            milvus_host=os.getenv('MILVUS_HOST', 'localhost'),
            milvus_port=int(os.getenv('MILVUS_PORT', '19530')),
            embedding_manager=embedding_manager
        )
        print("  ✅ 检索管理器初始化完成")

        # 初始化增强版ReAct代理
        agent = EnhancedMedicalReActAgent(
            llm_manager=llm_manager,
            retrieval_manager=retrieval_manager,
            embedding_manager=embedding_manager,
            es_host=os.getenv('ELASTICSEARCH_HOST', 'localhost'),
            es_port=int(os.getenv('ELASTICSEARCH_PORT', '9200'))
        )
        print("  ✅ 增强版ReAct代理初始化完成")

        print_header("✅ 系统初始化完成，准备就绪！")

        # 交互式循环
        while True:
            print_section("💬 请输入您的医学问题")
            print("提示：输入 'exit' 或 'quit' 退出程序")
            print("示例问题：")
            print("  - 腺癌的图像特征是什么？")
            print("  - 腺癌和鳞癌的区别是什么？")
            print("  - 黏液腺癌的细胞学特点有哪些？")
            print()

            question = input("➤ 您的问题: ").strip()

            if not question:
                print("⚠️  问题不能为空，请重新输入")
                continue

            if question.lower() in ['exit', 'quit', '退出']:
                print("\n👋 感谢使用！再见！")
                break

            # 处理问题
            print_header(f"🔍 正在处理您的问题: {question}")

            try:
                # 调用代理处理查询
                result = agent.process_query(question)

                # 显示查询分析
                print_section("📊 查询分析结果")
                metadata = result.get('metadata', {})
                print(f"  查询类型: {metadata.get('query_type', '未知')}")
                print(f"  识别实体: {', '.join(metadata.get('entities', []))}")
                print(f"  检索文档数: {metadata.get('search_results_count', 0)}")
                print(f"  推理步骤数: {metadata.get('reasoning_steps_count', 0)}")
                print(f"  置信度: {result.get('confidence', 0.0):.2f}")
                print(f"  响应时间: {result.get('response_time', 0.0):.2f}秒")

                # 显示推理步骤
                if result.get('reasoning_steps'):
                    print_section("🧠 推理过程")
                    for i, step in enumerate(result['reasoning_steps'], 1):
                        print(f"\n  步骤 {i}: {step.get('step', '未知步骤')}")
                        print(f"    思考: {step.get('thought', '')}")
                        print(f"    动作: {step.get('action', '')}")
                        print(f"    观察: {step.get('observation', '')}")

                # 显示检索到的文档
                if result.get('retrieved_documents'):
                    print_section("📚 参考文献（检索到的相关文档）")
                    docs = result['retrieved_documents'][:5]  # 只显示前5个
                    for i, doc in enumerate(docs, 1):
                        print(f"\n  文档 {i}:")
                        print(f"    页码: 第 {doc.get('page_number', '未知')} 页")
                        print(f"    章节: {doc.get('chapter_title', '未知')}")
                        print(f"    节标题: {doc.get('section_title', '未知')}")
                        print(f"    相关度得分: {doc.get('score', 0.0):.2f}")

                        # 显示章节匹配得分（如果有）
                        if doc.get('chapter_matching_score'):
                            print(f"    章节匹配得分: {doc.get('chapter_matching_score', 0.0):.2f}")

                        # 显示内容预览
                        content = doc.get('content', '')
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"    内容预览: {preview}")

                # 显示最终答案
                print_section("💡 系统回答")
                answer = result.get('answer', '抱歉，无法生成答案')
                print(f"\n{answer}\n")

                # 显示参考引用
                if result.get('retrieved_documents'):
                    print_section("📖 参考引用")
                    docs = result['retrieved_documents'][:5]
                    print("\n本回答基于以下医学文献：")
                    for i, doc in enumerate(docs, 1):
                        page = doc.get('page_number', '未知')
                        chapter = doc.get('chapter_title', '未知章节')
                        section = doc.get('section_title', '未知小节')
                        print(f"  [{i}] 第{page}页 - {chapter} - {section}")

            except Exception as e:
                print(f"\n❌ 处理问题时出错: {e}")
                import traceback
                traceback.print_exc()
                print("\n请尝试重新输入问题")
                continue

            print("\n" + "=" * 80)
            input("\n按回车键继续...")

    except KeyboardInterrupt:
        print("\n\n👋 程序被中断，再见！")
    except Exception as e:
        print(f"\n❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    interactive_test()
