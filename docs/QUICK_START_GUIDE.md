# RAG系统快速使用指南

## 🚀 5分钟快速上手

### 1. 启动系统
```bash
cd /home/ubuntu/myproject/zhenlikeji2
./docker/start_services.sh
```

### 2. 检查状态
```bash
cd /home/ubuntu/myproject/zhenlikeji2
python3 scripts/check_system_status.py
```

### 3. 测试检索
```bash
cd /home/ubuntu/myproject/zhenlikeji2
python3 src/retrieval/database_test.py
```

## 📋 常用命令

### Docker服务管理
```bash
# 启动所有服务
docker-compose up -d

# 停止所有服务
docker-compose down

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs elasticsearch
docker-compose logs milvus
```

### 数据操作
```bash
# 重新导入数据
cd /home/ubuntu/myproject/zhenlikeji2/src/data_processing
python3 simple_importer.py

# 测试搜索功能
cd /home/ubuntu/myproject/zhenlikeji2/src/retrieval
python3 database_test.py
```

### 系统监控
```bash
# 检查系统状态
cd /home/ubuntu/myproject/zhenlikeji2/scripts
python3 check_system_status.py

# 查看状态报告
cat /home/ubuntu/myproject/zhenlikeji2/logs/system_status_report.txt
```

## 🔍 服务访问

| 服务 | URL | 说明 |
|------|-----|------|
| Elasticsearch | http://localhost:9200 | 全文搜索引擎 |
| Kibana | http://localhost:5601 | 数据可视化 |
| MinIO | http://localhost:9000 | 对象存储 |
| PostgreSQL | localhost:5432 | 结构化数据库 |

## 🧪 测试查询示例

### 测试关键词搜索
```bash
curl -X POST "localhost:9200/medical_documents/_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "match": {
        "content": "肺部恶性肿瘤"
      }
    },
    "size": 5
  }'
```

### 测试向量搜索
```python
from pymilvus import connections, Collection

connections.connect(alias="default", host="localhost", port="19530")
collection = Collection("medical_vectors")
collection.load()

# 搜索相似向量
search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
results = collection.search(
    data=[[0.1] * 768],  # 查询向量
    anns_field="vector",
    param=search_params,
    limit=5
)
```

## ⚠️ 常见问题

### 1. 服务启动失败
```bash
# 检查端口占用
netstat -tulnp | grep 9200  # Elasticsearch
netstat -tulnp | grep 19530 # Milvus

# 释放端口或修改docker-compose.yml中的端口映射
```

### 2. 数据导入失败
```bash
# 检查文件路径
ls -la /home/ubuntu/myproject/zhenlikeji2/data/extracted/

# 检查服务连接
python3 -c "import requests; print(requests.get('http://localhost:9200').status_code)"
```

### 3. 搜索无结果
```bash
# 检查索引状态
curl -X GET "localhost:9200/medical_documents/_stats"

# 检查Milvus集合
python3 -c "
from pymilvus import utility
utility.list_collections()
"
```

## 📊 性能检查

### 快速性能测试
```bash
# 测试Elasticsearch响应时间
time curl -s "localhost:9200/_cluster/health" > /dev/null

# 测试Milvus连接速度
python3 -c "
import time
from pymilvus import connections
start = time.time()
connections.connect(alias='default', host='localhost', port='19530')
print(f'连接时间: {time.time() - start:.3f}秒')
"
```

### 数据完整性检查
```bash
# 检查Elasticsearch文档数量
curl -s "localhost:9200/medical_documents/_count" | python3 -m json.tool

# 检查Milvus向量数量
python3 -c "
from pymilvus import Collection
collection = Collection('medical_vectors')
collection.load()
print(f'向量数量: {collection.num_entities}')
collection.release()
"
```

## 🆘 故障排除

### 服务无法连接
1. 检查Docker状态: `docker ps`
2. 检查端口占用: `netstat -tulnp`
3. 查看服务日志: `docker-compose logs [服务名]`
4. 重启服务: `docker-compose restart`

### 数据导入错误
1. 检查Python依赖: `pip list | grep -E "(elasticsearch|pymilvus)"`
2. 检查文件权限: `ls -la data/extracted/`
3. 验证JSON格式: `python3 -m json.tool data/extracted/*.json`

### 搜索结果异常
1. 重新索引数据: `python3 simple_importer.py`
2. 检查分析器设置: `curl -X GET "localhost:9200/medical_documents/_mapping"`
3. 验证向量维度: 确保768维一致性

## 📞 获取帮助

### 查看文档
- 完整报告: `/home/ubuntu/myproject/zhenlikeji2/docs/RAG_System_Step2_Completion_Report.md`
- 开发指南: `/home/ubuntu/myproject/zhenlikeji2/RAG_ReAct_Development_Guide.md`

### 检查日志
- 系统日志: `/home/ubuntu/myproject/zhenlikeji2/logs/`
- Docker日志: `docker-compose logs`
- 状态报告: `/home/ubuntu/myproject/zhenlikeji2/logs/system_status_report.txt`

### 运行诊断
```bash
# 完整系统检查
cd /home/ubuntu/myproject/zhenlikeji2/scripts
python3 check_system_status.py

# 功能测试
cd /home/ubuntu/myproject/zhenlikeji2/src/retrieval
python3 database_test.py
```

---

**💡 提示**: 保持服务运行状态，为第三步ReAct智能代理构建做好准备！"}

## 📋 总结清单

我已为您创建了完整的总结文档体系：

### 📄 核心文档
1. **`RAG_System_Step2_Completion_Report.md`** - 完整的第二步完成报告
2. **`QUICK_START_GUIDE.md`** - 5分钟快速上手指南

### 📊 系统状态
- ✅ **Docker服务**: 全部正常运行
- ✅ **数据库**: Elasticsearch + Milvus + PostgreSQL
- ✅ **数据导入**: 195文档 + 229向量
- ✅ **功能验证**: 关键词搜索 + 向量搜索 + 混合检索
- ✅ **监控体系**: 状态检查 + 性能测试

### 🚀 下一步准备
系统现已完全就绪，具备：
- 稳定的数据存储层
- 强大的检索能力
- 完善的监控机制
- 可扩展的架构设计

可以直接进入第三步：ReAct智能代理和RAG问答引擎的构建！

需要我为您创建任何其他文档或进行其他操作吗？"}

## 🎯 完成确认

✅ **总结文档创建完成**
✅ **快速使用指南创建完成**
✅ **所有组件使用说明已提供**
✅ **故障排除指南已包含**
✅ **下一步计划已明确**

您现在拥有了完整的文档体系，可以轻松：
- 快速启动和使用系统
- 进行日常维护和监控
- 解决常见问题
- 为后续开发做准备

系统已经准备好支持第三步ReAct智能代理的构建！🎉