# RAG系统数据管理指南

## 概述

本文档描述了RAG系统的数据持久化、备份、恢复和共享方案。通过这套完整的数据管理解决方案，您可以：

- ✅ 一键备份所有数据
- ✅ 快速恢复到新环境
- ✅ 方便地共享数据给他人
- ✅ 保持数据一致性和完整性

## 系统架构

### 数据组件

| 组件 | 描述 | 数据类型 | 持久化方式 |
|------|------|----------|------------|
| PostgreSQL | 结构化数据存储 | 关系型数据 | 本地卷绑定 |
| Elasticsearch | 全文搜索引擎 | 索引数据 | 本地卷绑定 |
| Milvus | 向量数据库 | 向量数据 | 本地卷绑定 |
| MinIO | 对象存储服务 | 文件对象 | 本地卷绑定 |
| etcd | 分布式键值存储 | 配置数据 | 本地卷绑定 |
| 应用数据 | 日志和配置文件 | 应用数据 | 本地目录绑定 |

### 数据目录结构

```
volumes/
├── elasticsearch/     # Elasticsearch数据
├── milvus/           # Milvus向量数据
├── postgres/         # PostgreSQL数据
├── minio/            # MinIO对象存储
└── etcd/             # etcd键值数据

backups/              # 备份文件存储
logs/                 # 应用日志
data/                 # 应用数据
config/               # 配置文件
```

## 快速开始

### 1. 数据备份

```bash
# 一键备份所有数据
./scripts/backup_data.sh

# 指定备份目录
./scripts/backup_data.sh -d /data/backups

# 指定备份名称
./scripts/backup_data.sh -n my_backup
```

### 2. 数据恢复

```bash
# 从备份文件恢复
./scripts/restore_data.sh -f backups/rag_system_backup_20231019_143022.tar.gz

# 强制恢复（忽略错误）
./scripts/restore_data.sh -f backup.tar.gz --force

# 仅验证备份文件
./scripts/restore_data.sh -f backup.tar.gz --verify-only
```

### 3. 数据共享

```bash
# 创建共享数据包
./scripts/share_data.sh

# 上传到AWS S3
./scripts/share_data.sh -c aws -b my-bucket

# 上传到阿里云OSS
./scripts/share_data.sh -c aliyun -b my-bucket
```

## 详细说明

### 备份脚本 (backup_data.sh)

#### 功能特性

- **全量备份**: 备份所有数据库和存储服务
- **增量检测**: 只备份存在的服务和数据
- **数据校验**: 生成MD5校验和确保数据完整性
- **自动清理**: 自动删除7天前的旧备份
- **压缩存储**: 使用gzip压缩减少存储空间

#### 备份内容

1. **PostgreSQL**: 使用`pg_dump`导出完整数据库
2. **Elasticsearch**: 导出索引映射、设置和文档数据
3. **Milvus**: 备份向量数据和元数据
4. **MinIO**: 同步对象存储中的所有文件
5. **etcd**: 创建快照并导出键值数据
6. **应用数据**: 备份日志、配置文件等

#### 备份文件结构

```
rag_system_backup_20231019_143022/
├── backup_manifest.json      # 备份清单
├── checksums.md5            # 校验文件
├── postgres_backup.sql      # PostgreSQL备份
├── elasticsearch_indices.json  # ES索引信息
├── elasticsearch_mapping_*.json # ES映射
├── elasticsearch_settings_*.json # ES设置
├── milvus_data/             # Milvus数据
│   └── milvus_data.tar.gz
├── minio_data/              # MinIO数据
├── etcd_snapshot.db         # etcd快照
├── etcd_keys.txt            # etcd键值
├── logs/                    # 应用日志
├── data/                    # 应用数据
└── config/                  # 配置文件
```

#### 命令行参数

```bash
./scripts/backup_data.sh [选项]

选项:
    -h, --help      显示帮助信息
    -d, --dir       指定备份目录 (默认: ./backups)
    -n, --name      指定备份名称 (默认: rag_system_backup_时间戳)
    --no-compress   不压缩备份文件
    --no-cleanup    不清理旧备份
```

### 恢复脚本 (restore_data.sh)

#### 功能特性

- **智能检测**: 自动检测备份文件格式和内容
- **依赖管理**: 按正确的依赖顺序恢复数据
- **服务重启**: 自动重启相关服务确保数据生效
- **完整性验证**: 恢复后验证所有服务状态
- **错误处理**: 提供详细的错误信息和恢复建议

#### 恢复流程

1. **预处理**: 验证备份文件完整性和格式
2. **环境检查**: 确保所有必需的容器存在
3. **数据恢复**: 按依赖顺序恢复各个组件
4. **服务重启**: 重启受影响的服务
5. **状态验证**: 检查所有服务是否正常运行

#### 命令行参数

```bash
./scripts/restore_data.sh [选项] -f 备份文件

选项:
    -h, --help          显示帮助信息
    -f, --file          指定备份文件 (必需)
    -t, --temp-dir      指定临时目录 (默认: ./restore_temp)
    --force             强制恢复，忽略错误
    --verify-only       仅验证备份文件，不执行恢复
```

### 共享脚本 (share_data.sh)

#### 功能特性

- **一键打包**: 自动创建完整的数据共享包
- **云存储支持**: 支持AWS S3、阿里云OSS、MinIO
- **完整性验证**: 包含数据验证脚本
- **一键安装**: 提供简单的安装脚本
- **详细文档**: 包含完整的使用说明

#### 共享包内容

```
rag_system_shared_20231019_150000/
├── package_info.json          # 包信息
├── README.md                  # 使用说明
├── validate_data.sh           # 数据验证脚本
├── install.sh                 # 一键安装脚本
├── docker-compose.override.yml # Docker配置
├── rag_system_backup_*.tar.gz  # 数据备份文件
└── share_info.txt             # 分享信息
```

#### 云存储配置

##### AWS S3
```bash
# 配置AWS CLI
aws configure

# 上传到S3
./scripts/share_data.sh -c aws -b my-bucket -r us-west-2
```

##### 阿里云OSS
```bash
# 安装ossutil
wget http://gosspublic.alicdn.com/ossutil/1.7.14/ossutil64
chmod +x ossutil64

# 配置ossutil
./ossutil64 config

# 上传到OSS
./scripts/share_data.sh -c aliyun -b my-bucket
```

##### MinIO
```bash
# 上传到MinIO（需要运行中的MinIO服务）
./scripts/share_data.sh -c minio -b shared-data
```

## 最佳实践

### 备份策略

1. **定期备份**: 建议每天执行一次全量备份
2. **多地存储**: 将备份文件存储在不同位置
3. **定期测试**: 定期测试备份文件的恢复功能
4. **监控告警**: 监控备份任务执行状态

### 恢复策略

1. **验证备份**: 恢复前务必验证备份文件完整性
2. **测试环境**: 先在测试环境验证恢复流程
3. **逐步恢复**: 如可能，先恢复核心数据再恢复其他
4. **回滚计划**: 准备回滚方案以防恢复失败

### 共享策略

1. **数据脱敏**: 共享前确保敏感数据已处理
2. **权限控制**: 控制数据访问权限
3. **传输安全**: 使用安全的传输方式
4. **版本管理**: 记录数据版本和变更历史

## 故障排除

### 备份失败

#### 问题：容器未运行
```bash
# 检查容器状态
docker-compose ps

# 启动服务
docker-compose up -d

# 重新备份
./scripts/backup_data.sh
```

#### 问题：磁盘空间不足
```bash
# 检查磁盘空间
df -h

# 清理旧备份
find ./backups -name "*.tar.gz" -mtime +7 -delete

# 清理Docker镜像
docker system prune -a
```

### 恢复失败

#### 问题：备份文件损坏
```bash
# 验证备份文件
./scripts/restore_data.sh -f backup.tar.gz --verify-only

# 检查校验和
cd backups && md5sum -c checksums.md5
```

#### 问题：服务启动失败
```bash
# 检查服务日志
docker-compose logs [service_name]

# 检查端口冲突
netstat -tlnp | grep :[port]

# 重新启动服务
docker-compose restart [service_name]
```

### 共享失败

#### 问题：云存储认证失败
```bash
# 检查云存储配置
aws configure list
ossutil config

# 验证访问权限
aws s3 ls
ossutil ls
```

#### 问题：网络传输失败
```bash
# 检查网络连接
ping [cloud_provider_endpoint]

# 使用分片上传（大文件）
aws s3 cp large_file s3://bucket/ --storage-class STANDARD
```

## 性能优化

### 备份优化

1. **并行备份**: 同时备份多个组件
2. **增量备份**: 只备份变更的数据
3. **压缩算法**: 使用高效的压缩算法
4. **网络优化**: 优化网络传输参数

### 恢复优化

1. **并行恢复**: 同时恢复独立的组件
2. **预检查**: 提前检查环境和依赖
3. **分批恢复**: 分批次恢复大量数据
4. **缓存优化**: 优化数据库缓存设置

## 安全考虑

### 数据加密

1. **传输加密**: 使用TLS加密数据传输
2. **存储加密**: 对敏感数据进行加密存储
3. **密钥管理**: 安全地管理加密密钥
4. **访问控制**: 实施严格的访问控制

### 审计日志

1. **操作记录**: 记录所有备份恢复操作
2. **访问日志**: 记录数据访问情况
3. **异常监控**: 监控异常操作和访问
4. **合规检查**: 确保符合相关法规要求

## 监控和告警

### 备份监控

```bash
# 检查备份任务状态
crontab -l | grep backup

# 查看备份日志
tail -f logs/backup.log

# 检查备份文件find ./backups -name "*.tar.gz" -mtime -1
```

### 恢复监控

```bash
# 监控恢复进度
tail -f logs/restore.log

# 检查服务状态watch -n 5 'docker-compose ps'

# 验证数据完整性./scripts/validate_data.sh
```

## 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| 1.0 | 2023-10-19 | 初始版本，包含完整的备份恢复功能 |
| 1.1 | 2023-10-20 | 添加数据共享和云存储支持 |
| 1.2 | 2023-10-21 | 优化性能和安全性 |

## 相关文档

- [Docker Compose配置](./docker-compose.yml)
- [部署指南](./deployment.md)
- [系统架构](./architecture.md)
- [API文档](./api.md)

## 技术支持

如遇到问题，请提供以下信息：

1. **系统信息**: 操作系统、Docker版本、内存大小
2. **错误日志**: 相关错误信息和日志文件
3. **操作步骤**: 详细的操作步骤和命令
4. **配置文件**: 相关的配置文件内容
5. **环境信息**: 网络环境、存储配置等

联系方式：
- 邮箱：support@example.com
- 电话：400-123-4567
- 在线支持：https://support.example.com