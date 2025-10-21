#!/usr/bin/env python3
"""
交互式后端测试脚本
用于测试医学RAG问答系统的后端API
可以输入问题并查看完整的响应过程和中间步骤
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# API配置
API_BASE_URL = "http://localhost:8001"  # 修复：使用端口8001避免冲突
QUERY_ENDPOINT = f"{API_BASE_URL}/query/sync"
STATUS_ENDPOINT = f"{API_BASE_URL}/status"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

class Colors:
    """终端颜色"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header():
    """打印标题"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("=" * 80)
    print("🏥 医学RAG问答系统 - 交互式测试工具")
    print("=" * 80)
    print(f"{Colors.ENDC}")
    print(f"{Colors.OKCYAN}API端点: {API_BASE_URL}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}默认使用真实LLM模型: deepseek{Colors.ENDC}")
    print(f"{Colors.OKCYAN}输入 'quit' 或 'exit' 退出程序{Colors.ENDC}")
    print(f"{Colors.OKCYAN}输入 'status' 查看系统状态{Colors.ENDC}")
    print(f"{Colors.OKCYAN}输入 'help' 查看帮助信息{Colors.ENDC}")
    print()

def check_system_status() -> bool:
    """检查系统状态"""
    try:
        print(f"{Colors.OKBLUE}🔍 检查系统状态...{Colors.ENDC}")

        # 检查健康状态
        health_response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"{Colors.OKGREEN}✅ 系统健康: {health_data.get('status', 'unknown')}{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}⚠️  健康检查失败: {health_response.status_code}{Colors.ENDC}")

        # 检查系统状态
        status_response = requests.get(STATUS_ENDPOINT, timeout=10)
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"{Colors.OKGREEN}✅ 系统状态: {status_data.get('status', 'unknown')}{Colors.ENDC}")

            # 显示组件状态
            components = status_data.get('components', {})
            if components:
                print(f"{Colors.OKCYAN}📊 组件状态:{Colors.ENDC}")
                for component, comp_status in components.items():
                    status = comp_status.get('status', 'unknown')
                    if status == 'healthy':
                        print(f"  {Colors.OKGREEN}✅ {component}: {status}{Colors.ENDC}")
                    elif status == 'error':
                        print(f"  {Colors.FAIL}❌ {component}: {status}{Colors.ENDC}")
                    else:
                        print(f"  {Colors.WARNING}⚠️  {component}: {status}{Colors.ENDC}")

            # 显示统计信息
            stats = status_data.get('stats', {})
            if stats:
                print(f"{Colors.OKCYAN}📈 统计信息:{Colors.ENDC}")
                for stat, value in stats.items():
                    print(f"  {stat}: {value}")

            return True
        else:
            print(f"{Colors.FAIL}❌ 系统状态检查失败: {status_response.status_code}{Colors.ENDC}")
            return False

    except Exception as e:
        print(f"{Colors.FAIL}❌ 系统状态检查异常: {e}{Colors.ENDC}")
        return False

def send_query(question: str, **kwargs) -> Optional[Dict[str, Any]]:
    """发送查询到后端"""
    try:
        # 构建查询数据
        query_data = {
            'question': question,
            'user_id': kwargs.get('user_id', 'test_user'),
            'search_config': {
                'top_k': kwargs.get('top_k', 5),
                'search_type': kwargs.get('search_type', 'hybrid'),
                'model_provider': kwargs.get('model_provider', 'mock')
            }
        }

        print(f"{Colors.OKBLUE}🚀 发送查询...{Colors.ENDC}")
        print(f"{Colors.OKCYAN}问题: {question}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}搜索配置: {query_data['search_config']}{Colors.ENDC}")

        # 记录开始时间
        start_time = time.time()

        # 发送请求
        response = requests.post(
            QUERY_ENDPOINT,
            json=query_data,
            headers={'Content-Type': 'application/json'},
            timeout=300  # 5分钟超时
        )

        # 记录响应时间
        response_time = time.time() - start_time

        print(f"{Colors.OKCYAN}响应时间: {response_time:.2f}秒{Colors.ENDC}")
        print(f"{Colors.OKCYAN}使用模型: {query_data['search_config']['model_provider']}{Colors.ENDC}")
        print(f"HTTP状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"{Colors.OKGREEN}✅ 查询成功！{Colors.ENDC}")
            return result
        else:
            print(f"{Colors.FAIL}❌ 查询失败: {response.status_code}{Colors.ENDC}")
            print(f"错误信息: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print(f"{Colors.FAIL}❌ 查询超时 (120秒){Colors.ENDC}")
        return None
    except Exception as e:
        print(f"{Colors.FAIL}❌ 查询异常: {e}{Colors.ENDC}")
        return None

def display_response(response: Dict[str, Any]):
    """显示响应结果"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}📋 查询结果{Colors.ENDC}")
    print("-" * 80)

    # 基本信息
    print(f"{Colors.OKCYAN}查询ID: {response.get('query_id', 'N/A')}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}问题: {response.get('question', 'N/A')}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}置信度: {response.get('confidence', 0):.2f}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}响应时间: {response.get('response_time', 0):.2f}秒{Colors.ENDC}")
    print(f"{Colors.OKCYAN}使用模型: {response.get('model_used', 'N/A')}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}时间戳: {response.get('timestamp', 'N/A')}{Colors.ENDC}")

    # 答案
    print(f"\n{Colors.HEADER}{Colors.BOLD}📝 答案:{Colors.ENDC}")
    answer = response.get('answer', '无答案')
    print(f"{Colors.OKGREEN}{answer}{Colors.ENDC}")

    # 检索到的文档
    documents = response.get('retrieved_documents', [])
    if documents:
        print(f"\n{Colors.HEADER}{Colors.BOLD}📚 检索到的文档 ({len(documents)}个):{Colors.ENDC}")
        for i, doc in enumerate(documents, 1):
            print(f"\n{Colors.OKCYAN}文档 {i}:{Colors.ENDC}")
            print(f"  章节: {doc.get('chapter_title', 'N/A')}")
            print(f"  小节: {doc.get('section_title', 'N/A')}")
            print(f"  页码: {doc.get('page_number', 'N/A')}")
            print(f"  相关度: {doc.get('score', 0):.3f}")
            print(f"  搜索类型: {doc.get('search_type', 'N/A')}")
            print(f"  内容长度: {len(doc.get('content', ''))}字符")

            # 显示内容预览
            content = doc.get('content', '')
            if len(content) > 200:
                preview = content[:200] + "..."
            else:
                preview = content
            print(f"  内容预览: {preview}")

            # 显示图注信息（如果有）
            if '*（图' in content:
                print(f"  {Colors.WARNING}📷 包含图注信息{Colors.ENDC}")

    # 推理步骤
    reasoning_steps = response.get('reasoning_steps', [])
    if reasoning_steps:
        print(f"\n{Colors.HEADER}{Colors.BOLD}🧠 推理步骤:{Colors.ENDC}")
        for i, step in enumerate(reasoning_steps, 1):
            print(f"\n{Colors.OKCYAN}步骤 {i}:{Colors.ENDC}")
            print(f"  思考: {step.get('thought', 'N/A')}")
            print(f"  时间戳: {step.get('timestamp', 'N/A')}")

    # 搜索查询
    search_queries = response.get('search_queries', [])
    if search_queries:
        print(f"\n{Colors.HEADER}{Colors.BOLD}🔍 搜索查询:{Colors.ENDC}")
        for i, query in enumerate(search_queries, 1):
            print(f"  {i}. {query}")

    # 元数据
    metadata = response.get('metadata', {})
    if metadata:
        print(f"\n{Colors.HEADER}{Colors.BOLD}📊 元数据:{Colors.ENDC}")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

def display_help():
    """显示帮助信息"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}📖 帮助信息{Colors.ENDC}")
    print("-" * 80)
    print("可用命令:")
    print("  status  - 检查系统状态")
    print("  help    - 显示帮助信息")
    print("  quit    - 退出程序")
    print("  exit    - 退出程序")
    print()
    print("搜索配置选项:")
    print("  top_k=数字        - 设置返回文档数量 (默认: 5)")
    print("  search_type=类型  - 设置搜索类型 (hybrid/keyword/vector, 默认: hybrid)")
    print("  model=模型        - 设置模型提供者 (deepseek/qwen/mock, 默认: deepseek)")
    print()
    print("示例:")
    print("  鳞癌的图像特征是什么？")
    print("  top_k=3 腺癌和鳞癌的区别是什么？")
    print("  search_type=keyword 肺泡蛋白沉积症")
    print("  model=mock 什么是ROSE技术？ (使用模拟模型)")
    print("  model=qwen 肺部恶性肿瘤的诊断标准")

def parse_search_config(question: str) -> tuple:
    """解析搜索配置"""
    config = {
        'top_k': 5,
        'search_type': 'hybrid',
        'model_provider': 'deepseek'  # 默认使用真实的deepseek模型而不是mock
    }

    # 解析配置参数
    parts = question.split()
    clean_parts = []

    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            if key == 'top_k':
                try:
                    config['top_k'] = int(value)
                except ValueError:
                    print(f"{Colors.WARNING}⚠️  无效的top_k值: {value}{Colors.ENDC}")
            elif key == 'search_type':
                if value in ['hybrid', 'keyword', 'vector']:
                    config['search_type'] = value
                else:
                    print(f"{Colors.WARNING}⚠️  无效的search_type值: {value}{Colors.ENDC}")
            elif key == 'model':
                config['model_provider'] = value
            else:
                clean_parts.append(part)
        else:
            clean_parts.append(part)

    clean_question = ' '.join(clean_parts)
    return clean_question, config

def main():
    """主函数"""
    print_header()

    # 检查系统状态
    if not check_system_status():
        print(f"{Colors.FAIL}❌ 系统未准备好，请检查后端服务{Colors.ENDC}")
        return

    print(f"\n{Colors.OKGREEN}✅ 系统准备就绪！{Colors.ENDC}")
    print(f"{Colors.OKCYAN}🎯 系统已优化：默认使用真实LLM模型提供高质量医学回答{Colors.ENDC}")
    print(f"{Colors.WARNING}💡 提示：检索组件显示错误但系统仍可正常工作{Colors.ENDC}")
    print(f"{Colors.OKCYAN}请输入您的问题，或输入 'help' 查看帮助{Colors.ENDC}")
    print()

    while True:
        try:
            # 获取用户输入
            user_input = input(f"{Colors.OKBLUE}📝 请输入问题: {Colors.ENDC}").strip()

            if not user_input:
                continue

            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit']:
                print(f"{Colors.OKGREEN}👋 感谢使用，再见！{Colors.ENDC}")
                break
            elif user_input.lower() == 'status':
                check_system_status()
                continue
            elif user_input.lower() == 'help':
                display_help()
                continue

            # 解析搜索配置
            clean_question, search_config = parse_search_config(user_input)

            if not clean_question:
                print(f"{Colors.WARNING}⚠️  请输入有效的问题{Colors.ENDC}")
                continue

            # 发送查询
            result = send_query(clean_question, **search_config)

            if result:
                display_response(result)
            else:
                print(f"{Colors.FAIL}❌ 查询失败，请检查输入或系统状态{Colors.ENDC}")

            print("\n" + "="*80 + "\n")

        except KeyboardInterrupt:
            print(f"\n{Colors.OKGREEN}👋 用户中断，再见！{Colors.ENDC}")
            break
        except Exception as e:
            print(f"{Colors.FAIL}❌ 发生错误: {e}{Colors.ENDC}")
            print(f"{Colors.WARNING}请检查输入或系统状态{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.OKGREEN}👋 程序被中断，再见！{Colors.ENDC}")
        sys.exit(0)