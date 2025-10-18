#!/bin/bash

# RAG系统Docker服务启动脚本

echo "🚀 开始启动RAG系统Docker服务..."

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker服务未运行，请先启动Docker服务"
    exit 1
fi

# 检查docker-compose是否可用
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose未安装，请先安装docker-compose"
    exit 1
fi

# 创建必要的目录
echo "📁 创建数据目录..."
mkdir -p elasticsearch-data milvus-data etcd-data minio-data postgres-data

# 设置目录权限
chmod 777 elasticsearch-data milvus-data etcd-data minio-data postgres-data

# 停止已有的服务（如果存在）
echo "🛑 停止已有的服务..."
docker-compose down 2>/dev/null || true

# 拉取镜像
echo "📥 拉取Docker镜像..."
docker-compose pull

# 启动服务
echo "▶️ 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
echo "🔍 检查服务状态..."
services=("elasticsearch" "milvus" "etcd" "minio" "postgres")

for service in "${services[@]}"; do
    if docker-compose ps | grep -q "$service.*Up"; then
        echo "✅ $service 服务运行正常"
    else
        echo "❌ $service 服务未正常运行"
        echo "查看日志: docker-compose logs $service"
    fi
done

# 测试Elasticsearch
echo "🧪 测试Elasticsearch连接..."
if curl -s http://localhost:9200/_cluster/health > /dev/null; then
    echo "✅ Elasticsearch 连接成功"
    curl -s http://localhost:9200/_cluster/health | jq '.' 2>/dev/null || curl -s http://localhost:9200/_cluster/health
else
    echo "❌ Elasticsearch 连接失败"
fi

# 测试Milvus
echo "🧪 测试Milvus连接..."
python3 -c "
import sys
try:
    from pymilvus import connections
    connections.connect(alias='default', host='localhost', port='19530')
    print('✅ Milvus 连接成功')
except Exception as e:
    print(f'❌ Milvus 连接失败: {e}')
" 2>/dev/null || echo "⚠️ 需要安装pymilvus库进行完整测试"

# 显示服务访问信息
echo ""
echo "🎉 RAG系统Docker服务启动完成！"
echo ""
echo "📋 服务访问信息："
echo "  • Elasticsearch: http://localhost:9200"
echo "  • Milvus: localhost:19530"
echo "  • PostgreSQL: localhost:5432 (admin/password)"
echo "  • MinIO: http://localhost:9000 (minioadmin/minioadmin)"
echo "  • Kibana: http://localhost:5601"
echo ""
echo "🔧 常用命令："
echo "  • 查看状态: docker-compose ps"
echo "  • 查看日志: docker-compose logs [service-name]"
echo "  • 停止服务: docker-compose down"
echo "  • 重启服务: docker-compose restart [service-name]"
echo ""
echo "📖 下一步：运行数据导入脚本将PDF文本数据导入数据库"