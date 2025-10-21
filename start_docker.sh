#!/bin/bash

# Docker环境医学RAG系统启动脚本
# 自动处理端口配置和环境变量

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🏥 Docker环境医学RAG系统启动脚本${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# 检查Docker和Docker Compose
if ! command -v docker > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker未安装，请先安装Docker${NC}"
    exit 1
fi

if ! command -v docker-compose > /dev/null 2>1; then
    echo -e "${RED}❌ Docker Compose未安装，请先安装Docker Compose${NC}"
    exit 1
fi

# 检查.env文件是否存在
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ 未找到.env文件，请确保配置文件存在${NC}"
    exit 1
fi

# 备份当前配置
echo -e "${GREEN}💾 备份当前配置...${NC}"
cp docker-compose.yml docker-compose.yml.backup 2>/dev/null || true
cp nginx.conf nginx.conf.backup 2>/dev/null || true

# 设置端口配置
API_PORT=8001
FRONTEND_PORT=12345

echo -e "${GREEN}⚙️  配置端口映射...${NC}"
echo -e "${GREEN}   API服务端口: $API_PORT${NC}"
echo -e "${GREEN}   前端服务端口: $FRONTEND_PORT${NC}"

# 更新Docker Compose配置
echo -e "${GREEN}📝 更新Docker Compose配置...${NC}"
sed -i "s|\"800[0-9]:800[0-9]|\"$API_PORT:$API_PORT|g" docker-compose.yml
sed -i "s|localhost:800[0-9]|localhost:$API_PORT|g" docker-compose.yml
sed -i "s|app:800[0-9]|app:$API_PORT|g" docker-compose.yml
sed -i "s|test: \[\"CMD\", \"curl\", \"\-f\", \"http://localhost:800[0-9]/health\"\]|test: [\"CMD\", \"curl\", \"\-f\", \"http://localhost:$API_PORT/health\"]|g" docker-compose.yml

# 更新Nginx配置
echo -e "${GREEN}🌐 更新Nginx配置...${NC}"
sed -i "s|app:800[0-9]|app:$API_PORT|g" nginx.conf

# 更新测试脚本
echo -e "${GREEN}🧪 更新测试脚本...${NC}"
sed -i "s|localhost:800[0-9]|localhost:$API_PORT|g" test_server_interactive.py
sed -i "s|localhost:800[0-9]|localhost:$API_PORT|g" test_chunk_0019_simple.py
sed -i "s|localhost:800[0-9]|localhost:$API_PORT|g" test_internal_connection.py

# 设置环境变量
export API_PORT=$API_PORT
export FRONTEND_PORT=$FRONTEND_PORT

echo -e "${GREEN}🚀 启动Docker容器...${NC}"

# 启动Docker Compose服务
docker-compose up -d

# 等待服务启动
echo -e "${YELLOW}⏳ 等待服务启动...${NC}"
sleep 10

# 检查服务状态
echo -e "${GREEN}🔍 检查服务状态...${NC}"
if docker-compose ps | grep -q "Up (healthy)"; then
    echo -e "${GREEN}✅ Docker服务启动成功${NC}"
else
    echo -e "${YELLOW}⚠️  部分服务可能还在启动中，请稍后再检查${NC}"
fi

# 显示状态信息
echo -e "${GREEN}📊 Docker服务状态：${NC}"
docker-compose ps

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Docker环境医学RAG系统启动完成！${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 系统信息：${NC}"
echo -e "${GREEN}   • API服务地址: http://localhost:$API_PORT${NC}"
echo -e "${GREEN}   • 前端地址: http://localhost:$FRONTEND_PORT${NC}"
echo -e "${GREEN}   • 交互式测试: python test_server_interactive.py${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🔧 管理命令：${NC}"
echo -e "${GREEN}   • 查看日志: docker-compose logs -f app${NC}"
echo -e "${GREEN}   • 停止服务: docker-compose down${NC}"
echo -e "${GREEN}   • 系统状态: docker-compose ps${NC}"
echo -e "${GREEN}   • 重启服务: docker-compose restart${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ 系统运行中，享受使用！${NC}"

# 可选：运行快速测试
echo -e "${GREEN}🧪 运行快速测试...${NC}"
sleep 5
if curl -f -s "http://localhost:$API_PORT/health" > /dev/null; then
    echo -e "${GREEN}✅ API服务健康检查通过${NC}"
else
    echo -e "${YELLOW}⚠️  API服务可能还在启动中，请稍后再试${NC}"
fi

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}""${file_path}"/home/ubuntu/myproject/zhenlikeji2/start_docker.sh