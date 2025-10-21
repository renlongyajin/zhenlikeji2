# Simple Import 使用指南

## 概述

`simple_import.py` 是一个将 `simple_chunks.json` 数据导入到 Elasticsearch 和 Milvus 数据库的脚本。该脚本由 `simple_chunker.py` 生成的切块数据文件作为输入，完成文本内容和向量嵌入的批量导入。

## 前置条件

1. **数据文件**: 确保 `data/simple_chunks.json` 已存在（由 `simple_chunker.py` 生成）
2. **数据库服务**: Elasticsearch 和 Milvus 服务正在运行
3. **Python依赖**: 确保已安装所需依赖包

## 使用方法

### 1. 基本导入（默认配置）

```bash
cd /home/ubuntu/myproject/zhenlikeji2
python src/data_processing/simple_import.py
```

### 2. 指定输入文件

```bash
# 使用相对路径（相对于项目根目录）
python src/data_processing/simple_import.py --input data/simple_chunks.json

# 使用绝对路径
python src/data_processing/simple_import.py --input /absolute/path/to/simple_chunks.json
```

### 3. 连接测试

在实际导入前，建议先测试数据库连接：

```bash
python src/data_processing/simple_import.py --test-only
```

## 运行环境

### Docker环境（推荐）

脚本默认配置适用于Docker环境：
- Elasticsearch: `http://elasticsearch:9200`
- Milvus: `milvus:19530`

```bash
# 在Docker容器中运行
docker-compose exec backend python src/data_processing/simple_import.py
```

### 本地环境

如果在本地运行，需要设置环境变量：

```bash
export ELASTICSEARCH_HOST=localhost
export ELASTICSEARCH_PORT=9200
export MILVUS_HOST=localhost
export MILVUS_PORT=19530

python src/data_processing/simple_import.py
```

## 数据流向

```
simple_chunks.json
    ↓
Elasticsearch (medical_documents_fixed索引)
    - 文本内容
    - 章节信息
    - 页码信息
    - 切块元数据

    ↓
Milvus (medical_vectors_fixed集合)
    - 向量嵌入 (768维)
    - 关联元数据
```

## 导入流程

1. **连接测试**: 验证Elasticsearch和Milvus连接
2. **数据加载**: 读取并解析JSON文件
3. **向量生成**: 使用Jina嵌入模型生成向量
4. **Elasticsearch导入**: 批量导入文本数据
5. **Milvus导入**: 批量导入向量数据
6. **结果验证**: 统计导入的文档和向量数量

## 输出信息

脚本会显示详细的执行日志：

```
✅ Elasticsearch连接成功
✅ Milvus连接成功，现有集合: X个
✅ 成功加载 XXX 个切块
✅ 嵌入向量生成完成: XXX 个
✅ ES批量导入成功: 1-100/XXX
✅ Milvus批量插入成功: 1-100/XXX
✅ Elasticsearch导入完成: XXX 个文档
✅ Milvus导入完成: XXX 个向量
✅ Elasticsearch索引 medical_documents_fixed 文档数量: XXX
✅ Milvus集合 medical_vectors_fixed 向量数量: XXX
🎉 导入流程完成！
```

## 常见问题

### 1. 连接失败

**问题**: Elasticsearch或Milvus连接失败
**解决**:
- 检查服务是否运行: `docker-compose ps`
- 验证网络连接: `curl http://localhost:9200`
- 检查环境变量配置

### 2. 文件不存在

**问题**: `FileNotFoundError: 文件不存在: data/simple_chunks.json`
**解决**:
- 先运行切块脚本: `python src/data_processing/simple_chunker.py`
- 确认文件路径正确
- 检查文件权限

### 3. 嵌入模型加载失败

**问题**: `无法导入嵌入模型`
**解决**:
- 检查Jina模型依赖: `pip install -r requirements.txt`
- 验证模型文件完整性
- 检查网络连接（首次下载模型）

### 4. 批量导入错误

**问题**: ES或Milvus批量导入失败
**解决**:
- 检查数据库状态和资源使用情况
- 减小批处理大小（修改代码中的 `batch_size`）
- 查看详细错误日志

## 性能优化建议

1. **批处理大小**: 默认100个文档/批次，可根据硬件调整
2. **并发控制**: 脚本为顺序处理，避免同时运行多个实例
3. **内存监控**: 大批量导入时监控内存使用情况
4. **网络优化**: 确保与数据库服务的网络连接稳定

## 验证导入结果

导入完成后，可以通过以下方式验证：

### Elasticsearch
```bash
curl -X GET "localhost:9200/medical_documents_fixed/_count"
```

### Milvus
```python
from pymilvus import Collection, connections
connections.connect("default", host="localhost", port="19530")
collection = Collection("medical_vectors_fixed")
print(f"向量数量: {collection.num_entities}")
```

## 故障排除

如遇到问题，请检查：

1. **日志文件**: 查看完整的错误日志输出
2. **服务状态**: 确认所有服务正常运行
3. **资源配置**: 检查内存和CPU使用情况
4. **网络连接**: 验证容器间网络通信
5. **数据完整性**: 确认JSON文件格式正确

## 相关文件

- `simple_chunker.py`: 数据切块脚本
- `simple_chunks.json`: 输入数据文件
- `embedding_models.py`: 嵌入模型配置
- `docker-compose.yml`: 服务配置