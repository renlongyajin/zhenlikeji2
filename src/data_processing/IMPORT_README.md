# 数据导入工具使用说明

本目录包含两个数据导入脚本，用于将`simple_chunks.json`导入到Elasticsearch和Milvus数据库中。

## 📁 文件说明

### 1. `simple_import.py` - 本地环境导入脚本
适用于本地开发环境，直接连接到localhost上的数据库服务。

### 2. `docker_import.py` - Docker环境导入脚本
适用于Docker环境，连接到Docker服务名（elasticsearch, milvus）。

## 🔧 使用方法

### 本地环境
```bash
# 测试连接
python src/data_processing/simple_import.py --test-only

# 执行导入
python src/data_processing/simple_import.py --input data/simple_chunks.json
```

### Docker环境
```bash
# 在Docker容器中运行
docker-compose exec app python src/data_processing/docker_import.py --test-only

# 执行导入
docker-compose exec app python src/data_processing/docker_import.py --input simple_chunks.json
```

## 📋 导入流程

1. **测试连接** - 验证Elasticsearch和Milvus连接
2. **创建索引/集合** - 如果不存在则创建
3. **加载数据** - 从JSON文件读取切块数据
4. **生成嵌入** - 使用Jina模型生成向量嵌入
5. **导入Elasticsearch** - 存储文档和元数据
6. **导入Milvus** - 存储向量和关联数据

## ⚙️ 环境变量

脚本会自动读取以下环境变量：
- `ELASTICSEARCH_HOST` - Elasticsearch主机（默认: localhost/elasticsearch）
- `ELASTICSEARCH_PORT` - Elasticsearch端口（默认: 9200）
- `MILVUS_HOST` - Milvus主机（默认: localhost/milvus）
- `MILVUS_PORT` - Milvus端口（默认: 19530）
- `JINA_API_KEY` - Jina嵌入模型API密钥

## 📊 数据结构

### 输入格式 (simple_chunks.json)
```json
[
  {
    "content": "文档内容",
    "chapter_title": "章节标题",
    "section_title": "小节标题",
    "page_number": 1,
    "content_length": 500,
    "chunk_id": "chunk_0001",
    "chunk_index": 0,
    "sub_chunk_index": 0
  }
]
```

### Elasticsearch索引结构
- 索引名: `medical_documents_fixed`
- 字段: content, chapter_title, section_title, page_number, chunk_id, chunk_index, content_length, sub_chunk_index, timestamp

### Milvus集合结构
- 集合名: `medical_vectors_fixed`
- 字段: id, chunk_id, content, embedding, chapter_title, section_title, page_number, content_length, chunk_index, sub_chunk_index, timestamp
- 向量维度: 1024 (Jina模型)
- 索引类型: IVF_FLAT, 度量方式: L2

## 🎯 导入目标

- **Elasticsearch**: 存储文档内容，支持全文搜索
- **Milvus**: 存储向量嵌入，支持语义搜索

## 🔍 验证导入

导入完成后，可以通过以下方式验证：

```bash
# 检查Elasticsearch文档数量
curl -X GET "localhost:9200/medical_documents_fixed/_count"

# 检查Milvus集合状态
docker-compose exec app python -c "
from pymilvus import Collection, connections
connections.connect('test', host='milvus', port='19530')
collection = Collection('medical_vectors_fixed')
print(f'集合中的向量数量: {collection.num_entities}')
"
```

## ⚠️ 注意事项

1. **数据备份**: 导入前建议备份现有数据
2. **索引重建**: 如果索引/集合已存在，脚本会询问是否删除重建
3. **批量导入**: 使用批量操作提高效率，每批100个文档
4. **错误处理**: 失败的块会使用零向量作为后备
5. **进度显示**: 每10个块显示一次进度

## 🚀 性能优化

- 批量导入减少网络开销
- 异步处理多个批次
- 嵌入向量缓存（后续优化）
- 连接池复用（后续优化）

## 📈 日志输出

脚本会输出详细的日志信息，包括：
- 连接状态
- 导入进度
- 错误信息
- 性能统计

日志级别默认为INFO，可以通过环境变量调整。

## 🔧 故障排除

### 连接失败
- 检查服务是否运行
- 验证环境变量设置
- 确认网络连接

### 导入失败
- 检查文件路径和权限
- 验证JSON格式
- 查看详细错误日志

### 性能问题
- 调整批量大小
- 检查网络带宽
- 监控资源使用情况

## 📞 支持

如有问题，请检查日志输出或联系系统管理员。"}

现在让我测试Docker环境中的导入功能。首先检查当前环境：

```bash
# 测试Docker环境中的连接
docker-compose exec app python src/data_processing/docker_import.py --test-only
```

让我执行这个测试：

```bash
docker-compose exec app python src/data_processing/docker_import.py --test-only
```

等待测试完成...测试显示连接成功，现在让我执行实际的导入：

```bash
docker-compose exec app python src/data_processing/docker_import.py --input simple_chunks.json
```

让我检查导入进度和结果：

```bash
# 检查Elasticsearch中的文档数量
curl -s "http://localhost:8001/status" | python -c "
import json, sys
data = json.load(sys.stdin)
print('系统状态:', data.get('status', 'unknown'))
print('文档数量:', data.get('document_count', 0))
"

# 或者检查Elasticsearch直接
```

让我查看导入后的状态：

```bash
docker-compose logs --tail=20 app | grep -E "导入|完成|成功|失败"
```

现在系统应该已经成功导入了simple_chunks.json数据！您可以通过以下方式验证：

1. **系统状态检查**：`curl http://localhost:8001/status`
2. **实时日志监控**：`docker-compose logs -f app`
3. **功能测试**：运行之前的测试脚本

导入完成后，系统现在应该能够正确检索和使用新导入的数据了！🎉