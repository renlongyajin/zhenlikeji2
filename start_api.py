#!/usr/bin/env python3
"""
FastAPI服务启动脚本
解决模块导入问题并启动API服务
"""

import sys
import os
import subprocess

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

print(f"🚀 启动医学RAG问答系统API服务...")
print(f"📁 项目根目录: {project_root}")
print(f"📁 源代码目录: {src_path}")

# 使用uvicorn启动服务
cmd = [
    'python3', '-m', 'uvicorn',
    'src.agent.api_service:app',
    '--host', '0.0.0.0',
    '--port', '8000',
    '--reload'
]

print(f"🎯 启动命令: {' '.join(cmd)}")
print(f"🌐 API服务地址: http://localhost:8000")
print(f"📊 健康检查端点: http://localhost:8000/health")
print(f"📋 API文档: http://localhost:8000/docs")

try:
    subprocess.run(cmd, cwd=project_root, check=True)
except KeyboardInterrupt:
    print(f"\n🛑 服务已停止")
except Exception as e:
    print(f"❌ 启动失败: {e}")
    sys.exit(1)