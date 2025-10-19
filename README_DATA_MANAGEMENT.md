# RAG系统数据管理快速指南

## 🚀 快速开始

### 1. 备份当前数据
```bash
# 一键备份所有数据
./scripts/backup_data.sh

# 备份完成后，文件保存在 ./backups/ 目录
```

### 2. 恢复数据到新环境
```bash
# 从备份文件恢复
./scripts/restore_data.sh -f backups/rag_system_backup_*.tar.gz

# 启动服务
docker-compose up -d
```

### 3. 分享数据给他人
```bash
# 创建共享数据包
./scripts/share_data.sh

# 上传到云存储（可选）
./scripts/share_data.sh -c aws -b my-bucket
```

## 📋 常用命令

### 数据备份
```bash
# 默认备份
./scripts/backup_data.sh

# 指定备份目录
./scripts/backup_data.sh -d /data/backups

# 指定备份名称
./scripts/backup_data.sh -n my_backup
```

### 数据恢复
```bash
# 从备份文件恢复
./scripts/restore_data.sh -f backup.tar.gz

# 强制恢复（忽略错误）
./scripts/restore_data.sh -f backup.tar.gz --force

# 仅验证备份文件
./scripts/restore_data.sh -f backup.tar.gz --verify-only
```

### 数据共享
```bash
# 创建本地共享包
./scripts/share_data.sh

# 上传到AWS S3
./scripts/share_data.sh -c aws -b my-bucket

# 上传到阿里云OSS
./scripts/share_data.sh -c aliyun -b my-bucket
```

## 📁 文件结构

```
.
├── scripts/                    # 数据管理脚本
│   ├── backup_data.sh         # 数据备份脚本
│   ├── restore_data.sh        # 数据恢复脚本
│   └── share_data.sh          # 数据共享脚本
├── docs/                      # 文档
│   └── data_management.md     # 详细使用指南
├── volumes/                   # 数据卷（Docker配置优化后）
│   ├── elasticsearch/         # Elasticsearch数据
│   ├── milvus/               # Milvus向量数据
│   ├── postgres/             # PostgreSQL数据
│   ├── minio/                # MinIO对象存储
│   └── etcd/                 # etcd键值数据
├── backups/                   # 备份文件存储
├── logs/                      # 应用日志
├── data/                      # 应用数据
└── config/                    # 配置文件
```

## 🔧 系统要求

- **Docker**: >= 20.10
- **Docker Compose**: >= 1.29
- **内存**: >= 8GB
- **磁盘空间**: >= 50GB
- **端口**: 确保以下端口未被占用
  - 80, 443 (Nginx)
  - 5432 (PostgreSQL)
  - 9200 (Elasticsearch)
  - 19530 (Milvus)
  - 9000, 9001 (MinIO)
  - 5601 (Kibana)
  - 2379 (etcd)

## 🎯 使用场景

### 场景1：日常备份
```bash
# 设置定时任务（每天凌晨2点备份）
0 2 * * * /path/to/rag-system/scripts/backup_data.sh
```

### 场景2：环境迁移
```bash
# 在原环境备份
./scripts/backup_data.sh

# 复制备份文件到新环境
scp backups/*.tar.gz user@new-server:/path/to/rag-system/backups/

# 在新环境恢复
./scripts/restore_data.sh -f backups/*.tar.gz
docker-compose up -d
```

### 场景3：团队协作
```bash
# 创建共享数据包
./scripts/share_data.sh

# 分享给团队成员
# 他们收到后只需运行：
tar -xzf rag_system_shared_*.tar.gz
cd rag-system
./install.sh
```

## ⚠️ 注意事项

1. **备份前检查**：确保所有服务正常运行
2. **恢复前备份**：恢复前建议先备份当前数据
3. **磁盘空间**：确保有足够的磁盘空间进行备份
4. **网络连接**：共享数据时需要稳定的网络连接
5. **数据安全**：共享数据时注意敏感信息保护

## 🔍 故障排除

### 备份失败
```bash
# 检查服务状态
docker-compose ps

# 检查磁盘空间
df -h

# 查看详细日志
./scripts/backup_data.sh 2>&1 | tee backup.log
```

### 恢复失败
```bash
# 验证备份文件
./scripts/restore_data.sh -f backup.tar.gz --verify-only

# 检查端口冲突
netstat -tlnp

# 查看服务日志
docker-compose logs [service_name]
```

### 共享失败
```bash
# 检查云存储配置
aws configure list  # AWS
ossutil config      # 阿里云

# 检查网络连接
ping [cloud-provider-endpoint]
```

## 📚 详细文档

如需了解更多详细信息，请查看：[数据管理完整指南](./docs/data_management.md)

## 💡 最佳实践

1. **定期备份**：建议每天自动备份
2. **多地存储**：将备份文件存储在不同位置
3. **定期测试**：定期测试备份文件的恢复功能
4. **监控告警**：监控备份任务执行状态
5. **版本管理**：记录数据版本和变更历史

## 🆘 技术支持

如遇到问题：

1. 查看详细文档：[docs/data_management.md](./docs/data_management.md)
2. 检查日志文件：`logs/` 目录下的相关日志
3. 验证系统要求：确保满足所有系统要求
4. 提供以下信息寻求帮助：
   - 错误信息和日志
   - 系统配置信息
   - 操作步骤和命令

---

**现在您可以开始使用这些数据管理功能了！** 🎉

1. 先运行 `./scripts/backup_data.sh` 创建当前数据的备份
2. 如需分享，运行 `./scripts/share_data.sh` 创建共享包
3. 在新环境中，使用 `./scripts/restore_data.sh` 恢复数据