#!/usr/bin/env bash
set -e
# 1. 建目录
mkdir -p volumes/{elasticsearch,milvus,postgres,minio,etcd}
# 2. 内核参数
sudo sysctl -w vm.max_map_count=262244
# 3. 拉起
docker compose up -d
# 4. 等待全部 healthy
until docker compose ps --services --filter 'health=healthy' | wc -l | grep -q "7"; do
  echo "等待服务就绪..."
  sleep 5
done
echo "✅ 全套 RAG 系统已就绪！"
echo "前端: http://localhost:12345  |  Kibana: http://localhost:5601"