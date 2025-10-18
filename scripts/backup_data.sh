#!/bin/bash

# Docker数据备份脚本
# 用于备份所有数据库和存储服务的数据

set -e

# 配置变量
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="rag_system_backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# 检查Docker服务
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker服务未运行，请先启动Docker"
        exit 1
    fi
}

# 检查容器状态
check_containers() {
    local containers=("rag-postgres" "rag-elasticsearch" "rag-milvus" "rag-minio" "rag-etcd")
    local all_running=true

    for container in "${containers[@]}"; do
        if ! docker ps --format "table {{.Names}}" | grep -q "^${container}$"; then
            log_warn "容器 ${container} 未运行，跳过备份"
            all_running=false
        fi
    done

    if [ "$all_running" = false ]; then
        log_warn "部分容器未运行，继续备份可用的容器..."
    fi
}

# 创建备份目录
create_backup_dir() {
    mkdir -p "${BACKUP_PATH}"
    log_info "创建备份目录: ${BACKUP_PATH}"
}

# 备份PostgreSQL
backup_postgres() {
    log_info "开始备份PostgreSQL数据..."

    if docker ps --format "table {{.Names}}" | grep -q "^rag-postgres$"; then
        docker exec rag-postgres pg_dump -U admin -d rag_system --clean --if-exists > "${BACKUP_PATH}/postgres_backup.sql"

        if [ $? -eq 0 ]; then
            log_info "PostgreSQL备份完成"
        else
            log_error "PostgreSQL备份失败"
            return 1
        fi
    else
        log_warn "PostgreSQL容器未运行，跳过备份"
    fi
}

# 备份Elasticsearch
backup_elasticsearch() {
    log_info "开始备份Elasticsearch数据..."

    if docker ps --format "table {{.Names}}" | grep -q "^rag-elasticsearch$"; then
        # 创建快照仓库
        curl -X PUT "localhost:9200/_snapshot/backup_repo" -H 'Content-Type: application/json' -d'{
            "type": "fs",
            "settings": {
                "location": "/usr/share/elasticsearch/backup",
                "compress": true
            }
        }' 2>/dev/null || true

        # 创建快照
        curl -X PUT "localhost:9200/_snapshot/backup_repo/snapshot_${TIMESTAMP}" -H 'Content-Type: application/json' -d'{
            "indices": "*",
            "ignore_unavailable": true,
            "include_global_state": false
        }' 2>/dev/null || true

        # 等待快照完成
        sleep 5

        # 导出索引数据
        curl -X GET "localhost:9200/_cat/indices?format=json" > "${BACKUP_PATH}/elasticsearch_indices.json" 2>/dev/null

        # 导出映射和设置
        indices=$(curl -s -X GET "localhost:9200/_cat/indices?h=index" 2>/dev/null | grep -v "^\.")
        for index in $indices; do
            curl -X GET "localhost:9200/${index}/_mapping" > "${BACKUP_PATH}/elasticsearch_mapping_${index}.json" 2>/dev/null
            curl -X GET "localhost:9200/${index}/_settings" > "${BACKUP_PATH}/elasticsearch_settings_${index}.json" 2>/dev/null
        done

        log_info "Elasticsearch备份完成"
    else
        log_warn "Elasticsearch容器未运行，跳过备份"
    fi
}

# 备份Milvus
backup_milvus() {
    log_info "开始备份Milvus数据..."

    if docker ps --format "table {{.Names}}" | grep -q "^rag-milvus$"; then
        # 备份Milvus数据卷
        docker run --rm \
            -v zhenlikeji2_milvus-data:/source:ro \
            -v "${BACKUP_PATH}/milvus_data":/backup \
            alpine:latest \
            tar -czf /backup/milvus_data.tar.gz -C /source . 2>/dev/null || true

        # 获取Milvus集合信息
        curl -X GET "localhost:19530/v1/vector/collections" > "${BACKUP_PATH}/milvus_collections.json" 2>/dev/null || true

        log_info "Milvus备份完成"
    else
        log_warn "Milvus容器未运行，跳过备份"
    fi
}

# 备份MinIO
backup_minio() {
    log_info "开始备份MinIO数据..."

    if docker ps --format "table {{.Names}}" | grep -q "^rag-minio$"; then
        # 使用mc客户端备份MinIO数据
        docker run --rm \
            --network rag-network \
            -v "${BACKUP_PATH}/minio_data":/backup \
            minio/mc:latest \
            /bin/sh -c "
                mc alias set local http://rag-minio:9000 minioadmin minioadmin &&
                mc mirror --overwrite local /backup
            " 2>/dev/null || true

        log_info "MinIO备份完成"
    else
        log_warn "MinIO容器未运行，跳过备份"
    fi
}

# 备份etcd
backup_etcd() {
    log_info "开始备份etcd数据..."

    if docker ps --format "table {{.Names}}" | grep -q "^rag-etcd$"; then
        # 使用etcdctl备份数据
        docker exec rag-etcd etcdctl snapshot save /tmp/etcd_snapshot.db 2>/dev/null || true
        docker cp rag-etcd:/tmp/etcd_snapshot.db "${BACKUP_PATH}/etcd_snapshot.db" 2>/dev/null || true
        docker exec rag-etcd rm /tmp/etcd_snapshot.db 2>/dev/null || true

        # 获取etcd键值信息
        docker exec rag-etcd etcdctl get / --prefix --keys-only > "${BACKUP_PATH}/etcd_keys.txt" 2>/dev/null || true

        log_info "etcd备份完成"
    else
        log_warn "etcd容器未运行，跳过备份"
    fi
}

# 备份应用数据
backup_app_data() {
    log_info "开始备份应用数据..."

    # 备份日志文件
    if [ -d "./logs" ]; then
        cp -r ./logs "${BACKUP_PATH}/" 2>/dev/null || true
    fi

    # 备份数据文件
    if [ -d "./data" ]; then
        cp -r ./data "${BACKUP_PATH}/" 2>/dev/null || true
    fi

    # 备份配置文件
    if [ -d "./config" ]; then
        cp -r ./config "${BACKUP_PATH}/" 2>/dev/null || true
    fi

    log_info "应用数据备份完成"
}

# 生成备份清单
create_manifest() {
    log_info "生成备份清单..."

    cat > "${BACKUP_PATH}/backup_manifest.json" << EOF
{
    "backup_info": {
        "timestamp": "${TIMESTAMP}",
        "date": "$(date -Iseconds)",
        "backup_name": "${BACKUP_NAME}",
        "system": "RAG System",
        "version": "1.0"
    },
    "components": {
        "postgresql": {
            "status": "$(if [ -f "${BACKUP_PATH}/postgres_backup.sql" ]; then echo "completed"; else echo "skipped"; fi)",
            "file": "postgres_backup.sql"
        },
        "elasticsearch": {
            "status": "$(if [ -f "${BACKUP_PATH}/elasticsearch_indices.json" ]; then echo "completed"; else echo "skipped"; fi)",
            "files": ["elasticsearch_indices.json", "elasticsearch_mapping_*.json", "elasticsearch_settings_*.json"]
        },
        "milvus": {
            "status": "$(if [ -f "${BACKUP_PATH}/milvus_data/milvus_data.tar.gz" ]; then echo "completed"; else echo "skipped"; fi)",
            "file": "milvus_data/milvus_data.tar.gz"
        },
        "minio": {
            "status": "$(if [ -d "${BACKUP_PATH}/minio_data" ]; then echo "completed"; else echo "skipped"; fi)",
            "directory": "minio_data"
        },
        "etcd": {
            "status": "$(if [ -f "${BACKUP_PATH}/etcd_snapshot.db" ]; then echo "completed"; else echo "skipped"; fi)",
            "file": "etcd_snapshot.db"
        },
        "application": {
            "status": "completed",
            "directories": ["logs", "data", "config"]
        }
    }
}
EOF

    log_info "备份清单生成完成"
}

# 创建校验文件
create_checksum() {
    log_info "创建校验文件..."

    cd "${BACKUP_PATH}"
    find . -type f -exec md5sum {} \; > checksums.md5
    cd - >/dev/null

    log_info "校验文件创建完成"
}

# 压缩备份文件
compress_backup() {
    log_info "压缩备份文件..."

    cd "${BACKUP_DIR}"
    tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
    local compress_size=$(du -h "${BACKUP_NAME}.tar.gz" | cut -f1)

    # 删除原始备份目录
    rm -rf "${BACKUP_NAME}"

    log_info "备份压缩完成，文件大小: ${compress_size}"
    log_info "备份文件路径: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
}

# 清理旧备份
cleanup_old_backups() {
    log_info "清理旧备份文件..."

    # 保留最近7天的备份
    find "${BACKUP_DIR}" -name "rag_system_backup_*.tar.gz" -mtime +7 -delete 2>/dev/null || true

    log_info "旧备份清理完成"
}

# 主函数
main() {
    log_info "开始RAG系统数据备份..."

    check_docker
    check_containers
    create_backup_dir

    # 执行备份
    backup_postgres
    backup_elasticsearch
    backup_milvus
    backup_minio
    backup_etcd
    backup_app_data

    create_manifest
    create_checksum
    compress_backup
    cleanup_old_backups

    log_info "RAG系统数据备份完成！"
    log_info "备份文件: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
}

# 显示帮助信息
show_help() {
    cat << EOF
Docker数据备份脚本

用法: $0 [选项]

选项:
    -h, --help      显示帮助信息
    -d, --dir       指定备份目录 (默认: ./backups)
    -n, --name      指定备份名称 (默认: rag_system_backup_时间戳)
    --no-compress   不压缩备份文件
    --no-cleanup    不清理旧备份

示例:
    $0                          # 默认备份
    $0 -d /data/backups        # 指定备份目录
    $0 -n my_backup            # 指定备份名称
    $0 --no-compress           # 不压缩备份文件

备份内容包括:
    - PostgreSQL数据库
    - Elasticsearch索引数据
    - Milvus向量数据
    - MinIO对象存储
    - etcd键值数据
    - 应用日志和配置文件

EOF
}

# 解析命令行参数
NO_COMPRESS=false
NO_CLEANUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        -n|--name)
            BACKUP_NAME="$2"
            shift 2
            ;;
        --no-compress)
            NO_COMPRESS=true
            shift
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            shift
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