#!/usr/bin/env python3
"""
ReAct智能代理系统初始化脚本
初始化所有组件并启动服务
"""

import sys
import os
import subprocess
import time
import requests
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ReActSystemInitializer:
    """ReAct系统初始化器"""

    def __init__(self):
        """初始化器"""
        self.project_root = project_root
        self.scripts_dir = self.project_root / "scripts"
        self.src_dir = self.project_root / "src"
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    def check_prerequisites(self) -> bool:
        """检查先决条件"""
        logger.info("🔍 检查系统先决条件...")

        try:
            # 检查Python依赖
            required_packages = [
                'langgraph', 'langchain', 'fastapi', 'uvicorn',
                'requests', 'pymilvus', 'elasticsearch'
            ]

            for package in required_packages:
                try:
                    __import__(package)
                    logger.info(f"✅ {package} 已安装")
                except ImportError:
                    logger.error(f"❌ {package} 未安装")
                    return False

            # 检查Docker服务
            docker_check = self.check_docker_services()
            if not docker_check:
                logger.error("❌ Docker服务检查失败")
                return False

            logger.info("✅ 所有先决条件检查通过")
            return True

        except Exception as e:
            logger.error(f"❌ 先决条件检查失败: {e}")
            return False

    def check_docker_services(self) -> bool:
        """检查Docker服务状态"""
        logger.info("🐳 检查Docker服务...")

        try:
            # 运行系统状态检查脚本
            result = subprocess.run(
                [sys.executable, str(self.scripts_dir / "check_system_status.py")],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                logger.info("✅ Docker服务状态正常")
                return True
            else:
                logger.error(f"❌ Docker服务检查失败: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Docker服务检查异常: {e}")
            return False

    def initialize_components(self) -> bool:
        """初始化系统组件"""
        logger.info("🔧 初始化ReAct系统组件...")

        try:
            # 测试组件导入
            logger.info("📦 测试组件导入...")

            from src.agent.rag_engine import create_rag_engine, create_default_rag_config
            from src.agent.react_agent import MedicalReActAgent
            from src.agent.llm_manager import create_llm_manager
            from src.agent.retrieval_manager import create_retrieval_manager

            logger.info("✅ 所有组件导入成功")

            # 创建配置
            logger.info("⚙️  创建系统配置...")
            rag_config = create_default_rag_config()

            # 创建RAG引擎
            logger.info("🚀 创建RAG引擎...")
            rag_engine = create_rag_engine(rag_config)

            # 测试基本功能
            logger.info("🧪 测试基本功能...")
            from src.agent.rag_engine import RAGQuery

            test_query = RAGQuery(
                question="什么是ROSE技术？",
                query_id="test_001",
                user_id="test_user"
            )

            response = rag_engine.process_query_sync(test_query)

            if response.confidence > 0:
                logger.info(f"✅ 基本功能测试通过，置信度: {response.confidence}")
                logger.info(f"   答案预览: {response.answer[:100]}...")
                return True
            else:
                logger.error("❌ 基本功能测试失败")
                return False

        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            return False

    def start_api_service(self, port: int = 8000) -> bool:
        """启动API服务"""
        logger.info(f"🚀 启动API服务 (端口: {port})...")

        try:
            # 使用subprocess启动API服务
            import subprocess
            import signal

            # 启动API服务
            api_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "src.agent.api_service:app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--log-level", "info"
            ],
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
            )

            # 等待服务启动
            logger.info("⏳ 等待API服务启动...")
            time.sleep(5)

            # 检查服务是否启动成功
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=10)
                if response.status_code == 200:
                    logger.info("✅ API服务启动成功")
                    return True
                else:
                    logger.error(f"❌ API服务健康检查失败: {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ API服务连接失败: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ 启动API服务失败: {e}")
            return False

    def run_comprehensive_tests(self) -> bool:
        """运行综合测试"""
        logger.info("🧪 运行综合测试...")

        try:
            # 运行ReAct系统测试
            from src.agent.test_react_system import ReActSystemTester

            tester = ReActSystemTester()
            test_report = asyncio.run(tester.run_all_tests())

            # 保存测试报告
            report_file = self.logs_dir / f"react_system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(test_report, f, ensure_ascii=False, indent=2)

            overall_passed = test_report['test_execution']['overall_passed']
            logger.info(f"📋 测试报告已保存: {report_file}")

            if overall_passed:
                logger.info("✅ 综合测试通过")
                return True
            else:
                logger.error("❌ 综合测试失败")
                return False

        except Exception as e:
            logger.error(f"❌ 综合测试运行失败: {e}")
            return False

    def create_demo_script(self) -> bool:
        """创建演示脚本"""
        logger.info("📝 创建演示脚本...")

        demo_script = f'''#!/usr/bin/env python3
"""
ReAct智能代理系统演示脚本
演示系统的核心功能和API使用
"""

import requests
import json
import time

def demo_basic_query():
    """演示基础查询"""
    print("🚀 演示基础查询功能...")

    url = "http://localhost:8000/query/sync"
    payload = {{
        "question": "什么是肺部恶性肿瘤的ROSE细胞学特征？",
        "user_id": "demo_user",
        "metadata": {{"demo": True}}
    }}

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 查询成功!")
            print(f"问题: {{data['question']}}")
            print(f"答案: {{data['answer'][:200]}}...")
            print(f"置信度: {{data['confidence']}}")
            print(f"响应时间: {{data['response_time']:.2f}}s")
            print(f"使用模型: {{data['model_used']}}")
        else:
            print(f"❌ 查询失败: {{response.status_code}}")
    except Exception as e:
        print(f"❌ 查询异常: {{e}}")

def demo_search_types():
    """演示不同搜索类型"""
    print("\\n🔍 演示不同搜索类型...")

    search_configs = [
        {{"search_type": "keyword"}},
        {{"search_type": "semantic"}},
        {{"search_type": "hybrid", "keyword_weight": 0.5}}
    ]

    for config in search_configs:
        print(f"\\n测试搜索类型: {{config['search_type']}}")

        url = "http://localhost:8000/query/sync"
        payload = {{
            "question": "肺部恶性肿瘤的诊断方法",
            "search_config": config,
            "user_id": "demo_user"
        }}

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 成功，置信度: {{data['confidence']}}")
            else:
                print(f"❌ 失败: {{response.status_code}}")
        except Exception as e:
            print(f"❌ 异常: {{e}}")

def demo_query_suggestions():
    """演示查询建议"""
    print("\\n💡 演示查询建议...")

    test_queries = ["肺部", "ROSE", "细胞", "肿瘤"]

    for query in test_queries:
        print(f"\\n查询建议 for: '{{query}}'")

        url = f"http://localhost:8000/suggestions"
        params = {{"q": query, "max_suggestions": 3}}

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                suggestions = data.get('suggestions', [])
                print(f"建议: {{', '.join(suggestions)}}")
            else:
                print(f"❌ 失败: {{response.status_code}}")
        except Exception as e:
            print(f"❌ 异常: {{e}}")

def demo_batch_queries():
    """演示批量查询"""
    print("\\n📦 演示批量查询...")

    questions = [
        "肺部肿瘤的分类？",
        "ROSE技术的原理？",
        "细胞学检查的步骤？"
    ]

    payload = [
        {{"question": q, "user_id": "demo_user"}}
        for q in questions
    ]

    try:
        start_time = time.time()
        response = requests.post(f"http://localhost:8000/batch", json=payload)
        batch_time = time.time() - start_time

        if response.status_code == 200:
            results = response.json()
            print(f"✅ 批量查询成功!")
            print(f"查询数量: {{len(results)}}")
            print(f"总耗时: {{batch_time:.2f}}s")
            print(f"平均耗时: {{batch_time/len(results):.2f}}s")

            for i, result in enumerate(results):
                print(f"  {{i+1}}. {{result['question'][:30]}}... (置信度: {{result['confidence']}})")
        else:
            print(f"❌ 批量查询失败: {{response.status_code}}")
    except Exception as e:
        print(f"❌ 批量查询异常: {{e}}")

def demo_system_info():
    """演示系统信息"""
    print("\\nℹ️  系统信息")

    try:
        # 系统状态
        response = requests.get("http://localhost:8000/status")
        if response.status_code == 200:
            data = response.json()
            print(f"系统状态: {{data['status']}}")
            print(f"Elasticsearch文档: {{data['components']['retrieval']['elasticsearch']['document_count']}}")
            print(f"Milvus向量: {{data['components']['retrieval']['milvus']['vector_count']}}")
            print(f"LLM提供者: {{data['components']['llm']['active_provider']}}")

        # 统计信息
        response = requests.get("http://localhost:8000/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"\\n统计信息:")
            print(f"总查询数: {{data['total_queries']}}")
            print(f"成功查询数: {{data['successful_queries']}}")
            print(f"平均响应时间: {{data['average_response_time']:.2f}}s")

    except Exception as e:
        print(f"❌ 获取系统信息失败: {{e}}")

def main():
    """主演示函数"""
    print("🎉 ReAct智能代理系统演示")
    print("=" * 50)

    # 检查服务状态
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API服务未启动，请先启动服务")
            return
    except:
        print("❌ API服务连接失败，请先启动服务")
        return

    # 运行演示
    demo_basic_query()
    demo_search_types()
    demo_query_suggestions()
    demo_batch_queries()
    demo_system_info()

    print("\\n🎊 演示完成！")

if __name__ == "__main__":
    main()
'''

        demo_file = self.project_root / "demo_react_system.py"
        with open(demo_file, 'w', encoding='utf-8') as f:
            f.write(demo_script)

        # 设置执行权限
        os.chmod(demo_file, 0o755)

        logger.info(f"✅ 演示脚本已创建: {demo_file}")
        return True

    def create_usage_guide(self) -> bool:
        """创建使用指南"""
        logger.info("📖 创建使用指南...")

        guide_content = f"""# ReAct智能代理系统使用指南

## 🚀 快速开始

### 1. 启动系统
```bash
# 启动Docker服务（如果尚未启动）
cd /home/ubuntu/myproject/zhenlikeji2/docker
./start_services.sh

# 初始化ReAct系统
cd /home/ubuntu/myproject/zhenlikeji2/scripts
python3 init_react_system.py
```

### 2. 启动API服务
```bash
# 启动FastAPI服务
python3 -m uvicorn src.agent.api_service:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 测试系统
```bash
# 运行演示脚本
python3 /home/ubuntu/myproject/zhenlikeji2/demo_react_system.py

# 或使用API测试
python3 /home/ubuntu/myproject/zhenlikeji2/src/agent/test_react_system.py
```

## 📋 API端点

### 基础端点
- `GET /health` - 健康检查
- `GET /status` - 系统状态
- `GET /` - 服务信息

### 查询端点
- `POST /query` - 异步查询处理
- `POST /query/sync` - 同步查询处理
- `POST /batch` - 批量查询处理

### 功能端点
- `GET /suggestions` - 查询建议
- `GET /stats` - 统计信息
- `POST /sessions` - 创建会话
- `GET /sessions/{{session_id}}` - 获取会话

### LLM管理端点
- `GET /llm/providers` - 获取LLM提供者列表
- `POST /llm/providers/{{provider_name}}` - 切换LLM提供者

## 🔧 配置说明

### 默认配置
系统使用默认配置，支持以下LLM提供者：
- **mock**: 模拟LLM（默认，用于测试）
- **deepseek**: DeepSeek API（需要API密钥）
- **qwen**: 千问API（需要API密钥）

### 自定义配置
可以修改 `/home/ubuntu/myproject/zhenlikeji2/src/agent/llm_manager.py` 中的配置文件来添加新的LLM提供者。

## 📊 使用示例

### Python示例
```python
import requests

# 基础查询
response = requests.post("http://localhost:8000/query/sync", json={{
    "question": "什么是肺部恶性肿瘤的ROSE细胞学特征？",
    "user_id": "user123"
}})

result = response.json()
print(f"答案: {{result['answer']}}")
print(f"置信度: {{result['confidence']}}")
print(f"响应时间: {{result['response_time']}}s")
```

### cURL示例
```bash
# 基础查询
curl -X POST "http://localhost:8000/query/sync" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "question": "什么是ROSE技术？",
    "user_id": "user123"
  }}'
```

## 🎯 高级功能

### 混合检索
支持关键词搜索、语义搜索和混合搜索：
```json
{{
  "question": "肺部肿瘤诊断",
  "search_config": {{
    "search_type": "hybrid",
    "keyword_weight": 0.5
  }}
}}
```

### 批量查询
支持同时处理多个查询：
```json
[
  {{"question": "问题1", "user_id": "user1"}},
  {{"question": "问题2", "user_id": "user2"}},
  {{"question": "问题3", "user_id": "user3"}}
]
```

### 会话管理
支持多轮对话和上下文保持：
```bash
# 创建会话
SESSION_ID=$(curl -s -X POST "http://localhost:8000/sessions" | jq -r '.session_id')

# 在会话中查询
curl -X POST "http://localhost:8000/query/sync" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "question": "肺部肿瘤的特征？",
    "session_id": "'"$SESSION_ID"'"
  }}'
```

## 🔍 监控和调试

### 查看日志
```bash
# 查看系统状态报告
cat /home/ubuntu/myproject/zhenlikeji2/logs/system_status_report.txt

# 查看测试报告
ls -la /home/ubuntu/myproject/zhenlikeji2/logs/*test*.json
```

### 性能监控
- 平均响应时间：< 10秒
- 并发处理：支持5个并发查询
- 成功率：> 80%

### 故障排除
1. **服务无法连接**
   - 检查Docker服务：docker-compose ps
   - 检查端口占用：netstat -tulnp | grep 8000

2. **查询无结果**
   - 检查数据导入：python3 /home/ubuntu/myproject/zhenlikeji2/src/data_processing/simple_importer.py
   - 检查索引状态：curl -s "localhost:9200/medical_documents/_stats"

3. **LLM API错误**
   - 检查API密钥配置
   - 切换回模拟提供者进行测试

## 📞 技术支持

如遇到问题：
1. 查看系统日志：/home/ubuntu/myproject/zhenlikeji2/logs/
2. 运行测试脚本：python3 /home/ubuntu/myproject/zhenlikeji2/src/agent/test_react_system.py
3. 检查服务状态：python3 /home/ubuntu/myproject/zhenlikeji2/scripts/check_system_status.py

---

**系统版本**: 1.0.0
**最后更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**状态**: 🟢 运行中"""

        guide_file = self.project_root / "REACT_SYSTEM_USAGE_GUIDE.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)

        logger.info(f"✅ 使用指南已创建: {guide_file}")
        return True

    def run_initialization(self, start_api: bool = True, run_tests: bool = True) -> bool:
        """运行完整初始化流程"""
        logger.info("🚀 开始ReAct系统初始化...")
        logger.info("=" * 60)

        start_time = datetime.now()

        # 1. 检查先决条件
        logger.info("\n" + "="*40)
        logger.info("1. 检查系统先决条件")
        logger.info("="*40)
        if not self.check_prerequisites():
            logger.error("❌ 先决条件检查失败")
            return False

        # 2. 初始化组件
        logger.info("\n" + "="*40)
        logger.info("2. 初始化系统组件")
        logger.info("="*40)
        if not self.initialize_components():
            logger.error("❌ 组件初始化失败")
            return False

        # 3. 创建演示脚本和使用指南
        logger.info("\n" + "="*40)
        logger.info("3. 创建演示资源")
        logger.info("="*40)
        self.create_demo_script()
        self.create_usage_guide()

        # 4. 启动API服务（可选）
        if start_api:
            logger.info("\n" + "="*40)
            logger.info("4. 启动API服务")
            logger.info("="*40)
            if not self.start_api_service():
                logger.warning("⚠️ API服务启动失败，继续其他步骤")

        # 5. 运行综合测试（可选）
        if run_tests:
            logger.info("\n" + "="*40)
            logger.info("5. 运行综合测试")
            logger.info("="*40)
            if not self.run_comprehensive_tests():
                logger.warning("⚠️ 综合测试失败，系统可能存在问题")

        # 6. 生成完成报告
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n" + "="*60)
        logger.info("🎉 ReAct系统初始化完成！")
        logger.info("="*60)
        logger.info(f"总耗时: {duration:.2f}s")
        logger.info(f"完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        logger.info("\n📋 可用资源:")
        logger.info(f"   演示脚本: {self.project_root}/demo_react_system.py")
        logger.info(f"   使用指南: {self.project_root}/REACT_SYSTEM_USAGE_GUIDE.md")
        logger.info(f"   API文档: http://localhost:8000/docs")
        logger.info(f"   系统状态: http://localhost:8000/status")

        logger.info("\n🚀 快速开始:")
        logger.info("   1. 运行演示: python3 demo_react_system.py")
        logger.info("   2. 查看API文档: 打开 http://localhost:8000/docs")
        logger.info("   3. 测试查询: curl -X POST http://localhost:8000/query/sync \\")
        logger.info("      -H 'Content-Type: application/json' \\")
        logger.info("      -d '{\"question\": \"什么是ROSE技术？\"}'")

        return True

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ReAct智能代理系统初始化")
    parser.add_argument('--no-api', action='store_true', help='不启动API服务')
    parser.add_argument('--no-tests', action='store_true', help='不运行测试')
    parser.add_argument('--port', type=int, default=8000, help='API服务端口号')

    args = parser.parse_args()

    logger.info("🚀 启动ReAct智能代理系统初始化")

    initializer = ReActSystemInitializer()

    success = initializer.run_initialization(
        start_api=not args.no_api,
        run_tests=not args.no_tests
    )

    if success:
        logger.info("\n🎉 ReAct系统初始化成功！")
        sys.exit(0)
    else:
        logger.error("\n💥 ReAct系统初始化失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()