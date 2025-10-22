#!/bin/bash
# 启动脚本 - 确保所有模型配置正确加载

echo "🚀 启动医学RAG问答系统 - 带完整模型配置"

# 加载环境变量
if [ -f .env ]; then
    echo "📋 加载环境变量配置..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  未找到.env文件，使用默认配置"
fi

# 确保qwen3-80b模型配置可用
if [ -z "$QWEN3_80B_API_KEY" ]; then
    echo "🔧 设置QWEN3_80B_API_KEY为DASHSCOPE_API_KEY的值"
    export QWEN3_80B_API_KEY="$DASHSCOPE_API_KEY"
fi

# 验证模型配置
echo "✅ 模型配置验证:"
echo "  DeepSeek API密钥: ${#DEEPSEEK_API_KEY} 字符"
echo "  Qwen API密钥: ${#DASHSCOPE_API_KEY} 字符"
echo "  Qwen3-80B API密钥: ${#QWEN3_80B_API_KEY} 字符"

# 启动Docker服务
echo "🐳 启动Docker服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 启动后端API
echo "🔧 启动后端API服务..."
cd src/agent
python api_service.py

echo "✅ 系统启动完成！"
echo "📱 前端地址: http://localhost:12345"
echo "🔧 API文档: http://localhost:12345/docs"