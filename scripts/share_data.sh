#!/bin/bash

# Docker数据共享脚本
# 用于打包和分享数据库备份数据

set -e

# 配置变量
BACKUP_DIR="./backups"
SHARE_DIR="./shared_data"
CLOUD_PROVIDER=""
BUCKET_NAME=""
ACCESS_KEY=""
SECRET_KEY=""
REGION="us-east-1"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# 检查依赖
check_dependencies() {
    local deps=("tar" "gzip" "openssl")
    local missing_deps=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            missing_deps+=("$dep")
        fi
    done

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "缺少依赖: ${missing_deps[*]}"
        log_info "请安装缺失的依赖包"
        exit 1
    fi
}

# 创建共享目录
create_share_dir() {
    mkdir -p "$SHARE_DIR"
    log_info "创建共享目录: $SHARE_DIR"
}

# 获取最新的备份文件
get_latest_backup() {
    local latest_backup=$(find "$BACKUP_DIR" -name "rag_system_backup_*.tar.gz" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$latest_backup" ]; then
        log_error "未找到备份文件，请先运行备份脚本"
        exit 1
    fi

    echo "$latest_backup"
}

# 创建数据包描述文件
create_package_info() {
    local backup_file="$1"
    local package_name="$2"
    local timestamp=$(date +"%Y%m%d_%H%M%S")

    cat > "$SHARE_DIR/package_info.json" << EOF
{
    "package_info": {
        "name": "${package_name}",
        "version": "1.0",
        "created_at": "$(date -Iseconds)",
        "timestamp": "${timestamp}",
        "system": "RAG System",
        "description": "RAG系统完整数据备份包",
        "backup_file": "$(basename "$backup_file")",
        "size": "$(du -h "$backup_file" | cut -f1)",
        "checksum": "$(sha256sum "$backup_file" | cut -d' ' -f1)"
    },
    "system_requirements": {
        "docker_version": ">=20.10",
        "docker_compose_version": ">=1.29",
        "memory": ">=8GB",
        "disk_space": ">=50GB"
    },
    "components": {
        "postgresql": {
            "version": "15",
            "description": "结构化数据存储",
            "default_port": 5432
        },
        "elasticsearch": {
            "version": "8.8.0",
            "description": "全文搜索引擎",
            "default_port": 9200
        },
        "milvus": {
            "version": "2.3.0",
            "description": "向量数据库",
            "default_port": 19530
        },
        "minio": {
            "version": "latest",
            "description": "对象存储服务",
            "default_port": 9000
        },
        "etcd": {
            "version": "3.5.5",
            "description": "分布式键值存储",
            "default_port": 2379
        }
    },
    "restore_instructions": {
        "steps": [
            "1. 安装Docker和Docker Compose",
            "2. 下载并解压数据包",
            "3. 运行恢复脚本: ./scripts/restore_data.sh -f backup_file.tar.gz",
            "4. 启动服务: docker-compose up -d",
            "5. 验证服务状态"
        ],
        "scripts": {
            "backup": "./scripts/backup_data.sh",
            "restore": "./scripts/restore_data.sh",
            "health_check": "curl http://localhost/health"
        }
    }
}
EOF

    log_info "数据包描述文件创建完成"
}

# 创建使用说明
create_readme() {
    local package_name="$1"

    cat > "$SHARE_DIR/README.md" << EOF
# RAG系统数据包

## 概述

这个数据包包含了RAG系统的完整数据备份，包括：

- **PostgreSQL**: 结构化数据存储
- **Elasticsearch**: 全文搜索引擎
- **Milvus**: 向量数据库
- **MinIO**: 对象存储服务
- **etcd**: 分布式键值存储
- **应用数据**: 日志、配置文件等

## 系统要求

- Docker >= 20.10
- Docker Compose >= 1.29
- 内存 >= 8GB
- 磁盘空间 >= 50GB

## 快速开始

### 1. 解压数据包

\`\`\`bash
tar -xzf ${package_name}.tar.gz
\`\`\`

### 2. 恢复数据

\`\`\`bash
# 进入解压后的目录
cd rag-system

# 运行数据恢复脚本
./scripts/restore_data.sh -f backups/*.tar.gz
\`\`\`

### 3. 启动服务

\`\`\`bash
# 启动所有服务
docker-compose up -d

# 检查服务状态
docker-compose ps
\`\`\`

### 4. 验证系统

\`\`\`bash
# 检查应用状态
curl http://localhost/health

# 检查Elasticsearch
curl http://localhost:9200/_cluster/health

# 检查PostgreSQL
docker exec rag-postgres pg_isready -U admin -d rag_system
\`\`\`

## 数据内容

| 组件 | 描述 | 默认端口 |
|------|------|----------|
| PostgreSQL | 结构化数据存储 | 5432 |
| Elasticsearch | 全文搜索引擎 | 9200 |
| Milvus | 向量数据库 | 19530 |
| MinIO | 对象存储服务 | 9000 |
| etcd | 分布式键值存储 | 2379 |

## 管理脚本

### 备份数据
\`\`\`bash
./scripts/backup_data.sh
\`\`\`

### 恢复数据
\`\`\`bash
./scripts/restore_data.sh -f backup_file.tar.gz
\`\`\`

### 查看日志
\`\`\`bash
docker-compose logs -f [service_name]
\`\`\`

## 注意事项

1. **数据安全**: 数据包包含完整的数据库备份，请妥善保管
2. **端口冲突**: 确保默认端口未被占用
3. **资源要求**: 确保系统有足够的内存和磁盘空间
4. **网络配置**: 检查docker-compose.yml中的网络配置

## 故障排除

### 服务启动失败
- 检查端口是否被占用
- 检查Docker服务状态
- 查看服务日志

### 数据恢复失败
- 验证备份文件完整性
- 检查容器状态
- 查看恢复脚本日志

### 性能问题
- 调整JVM内存设置
- 优化数据库配置
- 监控系统资源使用

## 技术支持

如需技术支持，请提供以下信息：
- 系统版本和配置
- 错误日志
- 操作步骤

## 许可证

本数据包遵循相应的开源许可证，请在使用前确认许可证条款。

---

**创建时间**: $(date -Iseconds)
**数据包**: ${package_name}
**校验值**: $(sha256sum "$SHARE_DIR/${package_name}.tar.gz" 2>/dev/null | cut -d' ' -f1 || echo "N/A")
EOF

    log_info "使用说明文档创建完成"
}

# 创建Docker Compose配置文件
create_docker_compose_config() {
    cat > "$SHARE_DIR/docker-compose.override.yml" << EOF
# Docker Compose覆盖配置
# 用于数据恢复后的服务配置

version: '3.8'

services:
  # 主应用服务
  app:
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./config:/app/config
      - ./backups:/app/backups

  # Elasticsearch服务
  elasticsearch:
    volumes:
      - ./volumes/elasticsearch:/usr/share/elasticsearch/data
      - ./backups:/usr/share/elasticsearch/backup

  # Milvus服务
  milvus:
    volumes:
      - ./volumes/milvus:/var/lib/milvus

  # PostgreSQL服务
  postgres:
    volumes:
      - ./volumes/postgres:/var/lib/postgresql/data
      - ./backups:/backups

  # MinIO服务
  minio:
    volumes:
      - ./volumes/minio:/data
      - ./backups:/backups

  # etcd服务
  etcd:
    volumes:
      - ./volumes/etcd:/etcd-data

volumes:
  elasticsearch-data:
    external: true
  milvus-data:
    external: true
  postgres-data:
    external: true
  minio-data:
    external: true
  etcd-data:
    external: true
EOF

    log_info "Docker Compose覆盖配置创建完成"
}

# 创建数据验证脚本
create_validation_script() {
    cat > "$SHARE_DIR/validate_data.sh" << 'EOF'
#!/bin/bash

# 数据验证脚本
# 用于验证数据包完整性和可用性

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 检查文件完整性
check_files() {
    log_info "检查数据文件完整性..."

    local required_files=(
        "package_info.json"
        "README.md"
        "docker-compose.override.yml"
        "validate_data.sh"
    )

    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            log_info "✓ $file 存在"
        else
            log_error "✗ $file 缺失"
            return 1
        fi
    done
}

# 验证备份文件
verify_backup() {
    log_info "验证备份文件..."

    local backup_file=$(jq -r '.package_info.backup_file' package_info.json)
    local expected_checksum=$(jq -r '.package_info.checksum' package_info.json)

    if [ -f "$backup_file" ]; then
        log_info "找到备份文件: $backup_file"

        # 计算实际校验值
        local actual_checksum=$(sha256sum "$backup_file" | cut -d' ' -f1)

        if [ "$actual_checksum" = "$expected_checksum" ]; then
            log_info "✓ 备份文件校验通过"
        else
            log_error "✗ 备份文件校验失败"
            log_error "期望: $expected_checksum"
            log_error "实际: $actual_checksum"
            return 1
        fi
    else
        log_error "✗ 备份文件不存在: $backup_file"
        return 1
    fi
}

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."

    # 检查Docker
    if command -v docker >/dev/null 2>&1; then
        local docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        log_info "✓ Docker版本: $docker_version"
    else
        log_error "✗ Docker未安装"
        return 1
    fi

    # 检查Docker Compose
    if command -v docker-compose >/dev/null 2>&1; then
        local compose_version=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        log_info "✓ Docker Compose版本: $compose_version"
    else
        log_error "✗ Docker Compose未安装"
        return 1
    fi

    # 检查磁盘空间
    local available_space=$(df -h . | tail -1 | awk '{print $4}')
    log_info "✓ 可用磁盘空间: $available_space"

    # 检查内存
    if command -v free >/dev/null 2>&1; then
        local total_memory=$(free -h | grep Mem | awk '{print $2}')
        log_info "✓ 总内存: $total_memory"
    fi
}

# 主函数
main() {
    log_info "开始数据包验证..."

    if check_files && verify_backup && check_system_requirements; then
        log_info "✓ 数据包验证通过，可以安全使用"
        exit 0
    else
        log_error "✗ 数据包验证失败，请检查问题"
        exit 1
    fi
}

# 显示帮助
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "数据验证脚本"
    echo "用法: $0 [--help]"
    echo "验证数据包完整性和系统要求"
    exit 0
fi

main "$@"
EOF

    chmod +x "$SHARE_DIR/validate_data.sh"
    log_info "数据验证脚本创建完成"
}

# 创建一键安装脚本
create_install_script() {
    cat > "$SHARE_DIR/install.sh" << 'EOF'
#!/bin/bash

# 一键安装脚本
# 自动解压和安装RAG系统数据

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 检查并解压数据包
check_and_extract() {
    log_info "检查数据包..."

    # 查找数据包文件
    local data_package=$(find . -name "*.tar.gz" -type f | grep -E "(rag_system|backup)" | head -1)

    if [ -z "$data_package" ]; then
        log_error "未找到数据包文件"
        exit 1
    fi

    log_info "找到数据包: $data_package"

    # 验证数据包
    if [ -f "validate_data.sh" ]; then
        ./validate_data.sh
        if [ $? -ne 0 ]; then
            log_error "数据包验证失败"
            exit 1
        fi
    fi

    # 解压数据包
    log_info "解压数据包..."
    tar -xzf "$data_package" -C /tmp/

    # 查找解压后的备份文件
    local backup_file=$(find /tmp -name "*.tar.gz" -path "*/backups/*" | head -1)

    if [ -z "$backup_file" ]; then
        log_error "未找到备份文件"
        exit 1
    fi

    echo "$backup_file"
}

# 创建目录结构
create_directory_structure() {
    log_info "创建目录结构..."

    mkdir -p volumes/{elasticsearch,milvus,postgres,minio,etcd}
    mkdir -p {logs,data,config,backups,scripts}

    log_info "目录结构创建完成"
}

# 恢复数据
restore_data() {
    local backup_file="$1"

    log_info "开始恢复数据..."

    # 检查恢复脚本是否存在
    if [ ! -f "scripts/restore_data.sh" ]; then
        log_error "恢复脚本不存在，请确保在正确的目录中"
        exit 1
    fi

    # 执行数据恢复
    ./scripts/restore_data.sh -f "$backup_file"

    if [ $? -eq 0 ]; then
        log_info "数据恢复完成"
    else
        log_error "数据恢复失败"
        exit 1
    fi
}

# 启动服务
start_services() {
    log_info "启动服务..."

    # 检查docker-compose文件
    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml文件不存在"
        exit 1
    fi

    # 启动服务
    docker-compose up -d

    # 等待服务启动
    log_info "等待服务启动..."
    sleep 30

    # 检查服务状态
    docker-compose ps
}

# 验证安装
verify_installation() {
    log_info "验证安装..."

    # 检查主要服务状态
    local services=("rag-postgres" "rag-elasticsearch" "rag-milvus" "rag-minio" "rag-etcd")
    local failed_services=()

    for service in "${services[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^${service}$"; then
            log_info "✓ $service 运行正常"
        else
            log_error "✗ $service 未运行"
            failed_services+=("$service")
        fi
    done

    if [ ${#failed_services[@]} -eq 0 ]; then
        log_info "✓ 所有服务运行正常"
    else
        log_warn "部分服务未运行: ${failed_services[*]}"
        log_warn "请检查日志以获取详细信息"
    fi

    # 检查应用状态
    if curl -s -f "http://localhost/health" >/dev/null 2>&1; then
        log_info "✓ 应用服务正常"
    else
        log_warn "⚠ 应用服务可能未完全就绪"
    fi
}

# 显示使用信息
show_usage() {
    log_info "安装完成！"
    echo
    echo "系统访问地址:"
    echo "  - 主应用: http://localhost"
    echo "  - Elasticsearch: http://localhost:9200"
    echo "  - MinIO控制台: http://localhost:9001"
    echo "  - Kibana: http://localhost:5601"
    echo
    echo "管理命令:"
    echo "  - 查看状态: docker-compose ps"
    echo "  - 查看日志: docker-compose logs -f [service_name]"
    echo "  - 停止服务: docker-compose down"
    echo "  - 重启服务: docker-compose restart"
    echo
    echo "数据管理:"
    echo "  - 备份数据: ./scripts/backup_data.sh"
    echo "  - 恢复数据: ./scripts/restore_data.sh -f backup_file.tar.gz"
    echo
}

# 主函数
main() {
    log_info "开始RAG系统一键安装..."

    # 检查Docker环境
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker未运行，请先启动Docker"
        exit 1
    fi

    # 检查并解压数据包
    local backup_file=$(check_and_extract)

    # 创建目录结构
    create_directory_structure

    # 恢复数据
    restore_data "$backup_file"

    # 启动服务
    start_services

    # 验证安装
    verify_installation

    # 显示使用信息
    show_usage

    log_info "RAG系统安装完成！"
}

# 显示帮助
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "一键安装脚本"
    echo "用法: $0 [--help]"
    echo "自动解压和安装RAG系统数据"
    echo
    echo "前提条件:"
    echo "  - Docker和Docker Compose已安装"
    echo "  - 足够的磁盘空间（>50GB）"
    echo "  - 端口未被占用（80, 443, 5432, 9200, 19530, 9000, 9001, 5601, 2379）"
    exit 0
fi

main "$@"
EOF

    chmod +x "$SHARE_DIR/install.sh"
    log_info "一键安装脚本创建完成"
}

# 创建压缩包
create_share_package() {
    local latest_backup=$(get_latest_backup)
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local package_name="rag_system_shared_${timestamp}"

    log_info "创建共享数据包: $package_name"

    # 复制备份文件到共享目录
    cp "$latest_backup" "$SHARE_DIR/"

    # 创建相关文件
    create_package_info "$latest_backup" "$package_name"
    create_readme "$package_name"
    create_docker_compose_config
    create_validation_script
    create_install_script

    # 创建最终的压缩包
    cd "$SHARE_DIR"
    tar -czf "../${package_name}.tar.gz" .
    cd - >/dev/null

    local package_size=$(du -h "${package_name}.tar.gz" | cut -f1)

    log_info "共享数据包创建完成"
    log_info "文件大小: $package_size"
    log_info "文件路径: ./${package_name}.tar.gz"

    # 清理临时文件
    rm -rf "$SHARE_DIR"

    echo "$package_name.tar.gz"
}

# 上传到云存储
upload_to_cloud() {
    local package_file="$1"

    if [ -z "$CLOUD_PROVIDER" ]; then
        log_info "未指定云存储提供商，跳过上传"
        return 0
    fi

    log_info "上传到云存储: $CLOUD_PROVIDER"

    case "$CLOUD_PROVIDER" in
        "aws"|"s3")
            upload_to_aws_s3 "$package_file"
            ;;
        "aliyun"|"oss")
            upload_to_aliyun_oss "$package_file"
            ;;
        "minio")
            upload_to_minio "$package_file"
            ;;
        *)
            log_error "不支持的云存储提供商: $CLOUD_PROVIDER"
            return 1
            ;;
    esac
}

# 上传到AWS S3
upload_to_aws_s3() {
    local package_file="$1"

    if ! command -v aws >/dev/null 2>&1; then
        log_error "AWS CLI未安装"
        return 1
    fi

    if [ -z "$BUCKET_NAME" ]; then
        log_error "未指定S3存储桶名称"
        return 1
    fi

    log_info "上传到AWS S3: s3://$BUCKET_NAME/"

    aws s3 cp "$package_file" "s3://$BUCKET_NAME/" --region "$REGION"

    if [ $? -eq 0 ]; then
        log_info "上传成功: s3://$BUCKET_NAME/$package_file"
        log_info "下载命令: aws s3 cp s3://$BUCKET_NAME/$package_file ./"
    else
        log_error "上传失败"
        return 1
    fi
}

# 上传到阿里云OSS
upload_to_aliyun_oss() {
    local package_file="$1"

    if ! command -v ossutil >/dev/null 2>&1; then
        log_error "阿里云OSS工具未安装"
        return 1
    fi

    if [ -z "$BUCKET_NAME" ]; then
        log_error "未指定OSS存储桶名称"
        return 1
    fi

    log_info "上传到阿里云OSS: oss://$BUCKET_NAME/"

    ossutil cp "$package_file" "oss://$BUCKET_NAME/"

    if [ $? -eq 0 ]; then
        log_info "上传成功: oss://$BUCKET_NAME/$package_file"
        log_info "下载命令: ossutil cp oss://$BUCKET_NAME/$package_file ./"
    else
        log_error "上传失败"
        return 1
    fi
}

# 上传到MinIO
upload_to_minio() {
    local package_file="$1"

    if [ -z "$BUCKET_NAME" ]; then
        log_error "未指定MinIO存储桶名称"
        return 1
    fi

    log_info "上传到MinIO: $BUCKET_NAME"

    # 使用docker运行mc客户端
    docker run --rm \
        -v "$(pwd)":/data \
        minio/mc:latest \
        /bin/sh -c "
            mc alias set share http://minio:9000 minioadmin minioadmin &&
            mc cp /data/$package_file share/$BUCKET_NAME/
        " 2>/dev/null || true

    log_info "MinIO上传完成（如果服务可用）"
}

# 生成分享信息
generate_share_info() {
    local package_file="$1"
    local package_size=$(du -h "$package_file" | cut -f1)
    local checksum=$(sha256sum "$package_file" | cut -d' ' -f1)

    cat > "share_info.txt" << EOF
=====================================
RAG系统数据共享包信息
=====================================

数据包文件: $package_file
文件大小: $package_size
创建时间: $(date -Iseconds)
校验值(SHA256): $checksum

使用说明:
1. 解压数据包: tar -xzf $package_file
2. 验证数据包: ./validate_data.sh
3. 一键安装: ./install.sh
4. 手动安装: 参考README.md

数据内容包括:
- PostgreSQL数据库备份
- Elasticsearch索引数据
- Milvus向量数据
- MinIO对象存储
- etcd键值数据
- 应用配置和日志

注意事项:
- 确保系统满足要求(Docker >= 20.10, 内存 >= 8GB)
- 检查端口是否被占用
- 备份文件较大，下载需要一定时间
- 使用前请验证文件完整性

技术支持:
如有问题，请提供以下信息:
- 数据包文件名: $package_file
- 校验值: $checksum
- 错误信息和日志

=====================================
EOF

    log_info "分享信息已保存到: share_info.txt"
}

# 主函数
main() {
    log_info "开始创建数据共享包..."

    check_dependencies
    create_share_dir

    local package_file=$(create_share_package)

    # 上传到云存储（如果配置了）
    upload_to_cloud "$package_file"

    # 生成分享信息
    generate_share_info "$package_file"

    log_info "数据共享包创建完成！"
    log_info "数据包文件: $package_file"
    log_info "分享信息: share_info.txt"
    echo
    log_info "使用说明:"
    log_info "1. 将数据包文件分享给他人"
    log_info "2. 接收方解压后运行 ./install.sh 一键安装"
    log_info "3. 或按照 README.md 中的说明手动安装"
}

# 显示帮助信息
show_help() {
    cat << EOF
Docker数据共享脚本

用法: $0 [选项]

选项:
    -h, --help              显示帮助信息
    -d, --backup-dir        指定备份目录 (默认: ./backups)
    -c, --cloud-provider    云存储提供商 (aws/aliyun/minio)
    -b, --bucket            存储桶名称
    -r, --region            区域 (默认: us-east-1)
    --access-key            访问密钥
    --secret-key            秘密密钥

示例:
    $0                              # 创建本地共享包
    $0 -c aws -b my-bucket          # 上传到AWS S3
    $0 -c aliyun -b my-bucket       # 上传到阿里云OSS
    $0 -c minio -b shared-data      # 上传到MinIO

数据包内容包括:
    - 最新备份文件
    - 使用说明文档
    - 数据验证脚本
    - 一键安装脚本
    - Docker配置覆盖文件

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        -c|--cloud-provider)
            CLOUD_PROVIDER="$2"
            shift 2
            ;;
        -b|--bucket)
            BUCKET_NAME="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        --access-key)
            ACCESS_KEY="$2"
            shift 2
            ;;
        --secret-key)
            SECRET_KEY="$2"
            shift 2
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 执行主函数
main

# 清理敏感信息（如果提供了）
if [ -n "$ACCESS_KEY" ] || [ -n "$SECRET_KEY" ]; then
    unset ACCESS_KEY
    unset SECRET_KEY
    log_info "已清理敏感信息"
fi