#!/usr/bin/env python3
"""
RAG问答系统API启动脚本
修复导入问题并正确启动FastAPI服务
"""

import sys
import os
import subprocess
import signal
import time

def setup_python_path():
    """设置正确的Python路径"""
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_root, 'src')

    # 确保src目录在Python路径中
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # 确保项目根目录也在路径中
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"📁 项目根目录: {project_root}")
    print(f"📁 源代码目录: {src_dir}")
    print(f"🐍 Python路径: {sys.path[:3]}")  # 显示前3个路径

def test_imports():
    """测试关键模块导入"""
    print("🔍 测试模块导入...")
    try:
        # 测试导入
        from agent.api_service import app
        from agent.rag_engine import RAGEngine, create_rag_engine
        from agent.llm_manager import LLMManager
        from agent.retrieval_manager import MedicalRetrievalManager
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def start_api_service():
    """启动API服务"""
    print("🚀 开始启动医学RAG问答系统API服务...")

    # 设置Python路径
    setup_python_path()

    # 测试导入
    if not test_imports():
        print("❌ 无法继续启动，模块导入失败")
        return False

    # 启动参数
    host = "0.0.0.0"
    port = 8000
    reload = True  # 开发模式，支持热重载

    print(f"🌐 服务地址: http://{host}:{port}")
    print(f"📊 健康检查: http://{host}:{port}/health")
    print(f"📋 API文档: http://{host}:{port}/docs")
    print(f"🔧 自动重载: {'启用' if reload else '禁用'}")

    try:
        # 使用正确的模块路径启动
        cmd = [
            'python3', '-m', 'uvicorn',
            'agent.api_service:app',
            '--host', host,
            '--port', str(port),
            '--reload' if reload else '--no-reload',
            '--log-level', 'info'
        ]

        print(f"🎯 启动命令: {' '.join(cmd)}")

        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = ':'.join(sys.path)

        # 启动服务
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
            preexec_fn=os.setsid  # 创建新的进程组
        )

        print(f"✅ API服务进程已启动 (PID: {process.pid})")
        print("⏳ 等待服务初始化...")

        # 等待服务启动
        time.sleep(3)

        # 测试服务是否可用
        import requests
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=5)
            if response.status_code == 200:
                print("🎉 API服务启动成功！")
                print(f"📊 健康检查响应: {response.json()}")
                return True
            else:
                print(f"⚠️  服务响应异常: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ 服务未响应，可能启动失败")
        except Exception as e:
            print(f"❌ 测试服务时出错: {e}")

        return False

    except KeyboardInterrupt:
        print("\n🛑 用户中断启动")
        return False
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🏥 医学RAG问答系统 - API服务启动器")
    print("=" * 60)

    success = start_api_service()

    if success:
        print("\n🎊 API服务已成功启动！")
        print("📖 使用说明：")
        print("  • 健康检查: curl http://localhost:8000/health")
        print("  • 系统状态: curl http://localhost:8000/status")
        print("  • 查询测试: curl -X POST http://localhost:8000/query/sync \\")
        print("            -H 'Content-Type: application/json' \\")
        print('            -d \'{"question": "什么是ROSE技术？"}\'')
        print("\n🛑 按 Ctrl+C 停止服务")

        try:
            # 保持进程运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 正在关闭服务...")
            return True
    else:
        print("\n❌ API服务启动失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)