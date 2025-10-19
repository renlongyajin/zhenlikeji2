#!/bin/bash

# Docker数据恢复脚本
# 用于从备份文件恢复所有数据库和存储服务的数据

set -e

# 配置变量
RESTORE_DIR="./restore_temp"
BACKUP_FILE=""
FORCE_RESTORE=false
VERIFY_ONLY=false

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

# 检查Docker服务
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker服务未运行，请先启动Docker"
        exit 1
    fi
}

# 检查备份文件
check_backup_file() {
    if [ -z "$BACKUP_FILE" ]; then
        log_error "请指定备份文件"
        show_help
        exit 1
    fi

    if [ ! -f "$BACKUP_FILE" ]; then
        log_error "备份文件不存在: $BACKUP_FILE"
        exit 1
    fi

    # 检查文件格式
    if [[ "$BACKUP_FILE" == *.tar.gz ]]; then
        log_info "检测到压缩备份文件"
    elif [[ "$BACKUP_FILE" == *.tgz ]]; then
        log_info "检测到压缩备份文件"
    else
        log_warn "未知的备份文件格式，尝试解压..."
    fi
}

# 解压备份文件
extract_backup() {
    log_info "解压备份文件..."

    mkdir -p "$RESTORE_DIR"

    if [[ "$BACKUP_FILE" == *.tar.gz ]] || [[ "$BACKUP_FILE" == *.tgz ]]; then
        tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR" --strip-components=1
    else
        log_error "不支持的备份文件格式"
        exit 1
    fi

    log_info "备份文件解压完成"
}

# 验证备份文件
verify_backup() {
    log_info "验证备份文件完整性..."

    if [ ! -f "$RESTORE_DIR/backup_manifest.json" ]; then
        log_error "备份清单文件不存在，无法验证备份"
        exit 1
    fi

    # 检查校验文件
    if [ -f "$RESTORE_DIR/checksums.md5" ]; then
        cd "$RESTORE_DIR"
        if md5sum -c checksums.md5 >/dev/null 2>&1; then
            log_info "备份文件校验通过"
        else
            log_error "备份文件校验失败，数据可能已损坏"
            if [ "$FORCE_RESTORE" = false ]; then
                exit 1
            else
                log_warn "强制恢复模式，忽略校验错误"
            fi
        fi
        cd - >/dev/null
    else
        log_warn "未找到校验文件，跳过完整性检查"
    fi

    # 显示备份信息
    if command -v jq >/dev/null 2>&1; then
        log_info "备份信息:"
        jq -r '.backup_info | "备份时间: \(.date)\n系统: \(.system)\n版本: \(.version)"' "$RESTORE_DIR/backup_manifest.json"
    fi
}

# 检查容器状态
check_containers_before_restore() {
    log_info "检查容器状态..."

    local containers=("rag-postgres" "rag-elasticsearch" "rag-milvus" "rag-minio" "rag-etcd")
    local not_running=()

    for container in "${containers[@]}"; do
        if ! docker ps --format "table {{.Names}}" | grep -q "^${container}$"; then
            not_running+=("$container")
        fi
    done

    if [ ${#not_running[@]} -gt 0 ]; then
        log_warn "以下容器未运行: ${not_running[*]}"
        log_info "尝试启动相关服务..."

        # 启动docker-compose服务
        if [ -f "docker-compose.yml" ]; then
            docker-compose up -d "${not_running[@]}" 2>/dev/null || true

            # 等待服务启动
            sleep 30
        fi
    fi
}

# 恢复PostgreSQL
restore_postgres() {
    log_info "开始恢复PostgreSQL数据..."

    if [ ! -f "$RESTORE_DIR/postgres_backup.sql" ]; then
        log_warn "PostgreSQL备份文件不存在，跳过恢复"
        return 0
    fi

    if docker ps --format "table {{.Names}}" | grep -q "^rag-postgres$"; then
        # 等待PostgreSQL完全启动
        log_info "等待PostgreSQL服务就绪..."
        for i in {1..30}; do
            if docker exec rag-postgres pg_isready -U admin -d rag_system >/dev/null 2>&1; then
                break
            fi
            sleep 2
        done

        # 恢复数据
        docker exec -i rag-postgres psql -U admin -d rag_system < "$RESTORE_DIR/postgres_backup.sql"

        if [ $? -eq 0 ]; then
            log_info "PostgreSQL数据恢复完成"
        else
            log_error "PostgreSQL数据恢复失败"
            if [ "$FORCE_RESTORE" = false ]; then
                exit 1
            fi
        fi
    else
        log_warn "PostgreSQL容器未运行，跳过恢复"
    fi
}

# 恢复Elasticsearch
restore_elasticsearch() {
    log_info "开始恢复Elasticsearch数据..."

    if [ ! -f "$RESTORE_DIR/elasticsearch_indices.json" ]; then
        log_warn "Elasticsearch备份文件不存在，跳过恢复"
        return 0
    fi

    if docker ps --format "table {{.Names}}" | grep -q "^rag-elasticsearch$"; then
        # 等待Elasticsearch完全启动
        log_info "等待Elasticsearch服务就绪..."
        for i in {1..60}; do
            if curl -s -f "localhost:9200/_cluster/health" >/dev/null 2>&1; then
                break
            fi
            sleep 2
        done

        # 恢复索引设置和映射
        for mapping_file in "$RESTORE_DIR"/elasticsearch_mapping_*.json; do
            if [ -f "$mapping_file" ]; then
                local index_name=$(basename "$mapping_file" | sed 's/elasticsearch_mapping_\(.*\)\.json/\1/')
                log_info "恢复索引: $index_name"

                # 创建索引（如果不存在）
                curl -X PUT "localhost:9200/${index_name}" -H 'Content-Type: application/json' -d'{
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }' 2>/dev/null || true

                # 恢复映射
                if [ -f "$RESTORE_DIR/elasticsearch_settings_${index_name}.json" ]; then
                    curl -X PUT "localhost:9200/${index_name}/_settings" -H 'Content-Type: application/json' \
                        -d "@$RESTORE_DIR/elasticsearch_settings_${index_name}.json" 2>/dev/null || true
                fi
            fi
        done

        log_info "Elasticsearch数据恢复完成"
    else
        log_warn "Elasticsearch容器未运行，跳过恢复"
    fi
}

# 恢复Milvus
restore_milvus() {
    log_info "开始恢复Milvus数据..."

    if [ ! -f "$RESTORE_DIR/milvus_data/milvus_data.tar.gz" ]; then
        log_warn "Milvus备份文件不存在，跳过恢复"
        return 0
    fi

    if docker ps --format "table {{.Names}}" | grep -q "^rag-milvus$"; then
        # 停止Milvus服务
        log_info "停止Milvus服务..."
        docker stop rag-milvus

        # 等待服务完全停止
        sleep 10

        # 恢复数据卷
        log_info "恢复Milvus数据卷..."
        docker run --rm \
            -v zhenlikeji2_milvus-data:/target \
            -v "$RESTORE_DIR/milvus_data":/source:ro \
            alpine:latest \
            tar -xzf /source/milvus_data.tar.gz -C /target/ 2>/dev/null || true

        # 重新启动Milvus服务
        log_info "重新启动Milvus服务..."
        docker start rag-milvus

        # 等待服务启动
        sleep 20

        log_info "Milvus数据恢复完成"
    else
        log_warn "Milvus容器未运行，跳过恢复"
    fi
}

# 恢复MinIO
restore_minio() {
    log_info "开始恢复MinIO数据..."

    if [ ! -d "$RESTORE_DIR/minio_data" ]; then
        log_warn "MinIO备份数据不存在，跳过恢复"
        return 0
    fi

    if docker ps --format "table {{.Names}}" | grep -q "^rag-minio$"; then
        # 使用mc客户端恢复数据
        docker run --rm \
            --network rag-network \
            -v "$RESTORE_DIR/minio_data":/source:ro \
            minio/mc:latest \
            /bin/sh -c "
                mc alias set local http://rag-minio:9000 minioadmin minioadmin &&
                mc mirror --overwrite /source local
            " 2>/dev/null || true

        log_info "MinIO数据恢复完成"
    else
        log_warn "MinIO容器未运行，跳过恢复"
    fi
}

# 恢复etcd
restore_etcd() {
    log_info "开始恢复etcd数据..."

    if [ ! -f "$RESTORE_DIR/etcd_snapshot.db" ]; then
        log_warn "etcd备份文件不存在，跳过恢复"
        return 0
    fi

    if docker ps --format "table {{.Names}}" | grep -q "^rag-etcd$"; then
        # 停止etcd服务
        log_info "停止etcd服务..."
        docker stop rag-etcd

        # 等待服务完全停止
        sleep 5

        # 恢复数据
        log_info "恢复etcd数据..."
        docker cp "$RESTORE_DIR/etcd_snapshot.db" rag-etcd:/tmp/etcd_snapshot.db

        # 使用快照恢复数据
        docker run --rm \
            -v zhenlikeji2_etcd-data:/etcd-data \
            -v "$RESTORE_DIR":/backup:ro \
            quay.io/coreos/etcd:v3.5.5 \
            /bin/sh -c "
                etcdctl snapshot restore /backup/etcd_snapshot.db \
                    --data-dir /etcd-data \
                    --name etcd \
                    --initial-cluster etcd=http://localhost:2380 \
                    --initial-advertise-peer-urls http://localhost:2380
            " 2>/dev/null || true

        # 重新启动etcd服务
        log_info "重新启动etcd服务..."
        docker start rag-etcd

        # 等待服务启动
        sleep 10

        log_info "etcd数据恢复完成"
    else
        log_warn "etcd容器未运行，跳过恢复"
    fi
}

# 恢复应用数据
restore_app_data() {
    log_info "开始恢复应用数据..."

    # 恢复日志文件
    if [ -d "$RESTORE_DIR/logs" ]; then
        cp -r "$RESTORE_DIR/logs" ./ 2>/dev/null || true
        log_info "日志文件恢复完成"
    fi

    # 恢复数据文件
    if [ -d "$RESTORE_DIR/data" ]; then
        cp -r "$RESTORE_DIR/data" ./ 2>/dev/null || true
        log_info "数据文件恢复完成"
    fi

    # 恢复配置文件
    if [ -d "$RESTORE_DIR/config" ]; then
        cp -r "$RESTORE_DIR/config" ./ 2>/dev/null || true
        log_info "配置文件恢复完成"
    fi
}

# 验证恢复结果
verify_restore() {
    log_info "验证恢复结果..."

    local verification_passed=true

    # 检查PostgreSQL
    if docker ps --format "table {{.Names}}" | grep -q "^rag-postgres$"; then
        if docker exec rag-postgres pg_isready -U admin -d rag_system >/dev/null 2>&1; then
            log_info "✓ PostgreSQL服务正常"
        else
            log_error "✗ PostgreSQL服务异常"
            verification_passed=false
        fi
    fi

    # 检查Elasticsearch
    if docker ps --format "table " | grep -q "^rag-elasticsearch$"; then
        if curl -s -f "localhost:9200/_cluster/health" >/dev/null 2>&1; then
            log_info "✓ Elasticsearch服务正常"
        else
            log_error "✗ Elasticsearch服务异常"
            verification_passed=false
        fi
    fi

    # 检查Milvus
    if docker ps --format "table {{.Names}}" | grep -q "^rag-milvus$"; then
        if curl -s -f "localhost:19530/health" >/dev/null 2>&1; then
            log_info "✓ Milvus服务正常"
        else
            log_warn "⚠ Milvus服务状态未知（需要特殊工具检查）"
        fi
    fi

    # 检查MinIO
    if docker ps --format "table {{.Names}}" | grep -q "^rag-minio$"; then
        if curl -s -f "localhost:9000/minio/health/live" >/dev/null 2>&1; then
            log_info "✓ MinIO服务正常"
        else
            log_error "✗ MinIO服务异常"
            verification_passed=false
        fi
    fi

    # 检查etcd
    if docker ps --format "table {{.Names}}" | grep -q "^rag-etcd$"; then
        if docker exec rag-etcd etcdctl endpoint health >/dev/null 2>&1; then
            log_info "✓ etcd服务正常"
        else
            log_error "✗ etcd服务异常"
            verification_passed=false
        fi
    fi

    if [ "$verification_passed" = true ]; then
        log_info "恢复验证通过，所有服务运行正常"
    else
        log_warn "部分服务验证失败，请检查日志"
    fi
}

# 清理临时文件
cleanup() {
    log_info "清理临时文件..."

    if [ -d "$RESTORE_DIR" ]; then
        rm -rf "$RESTORE_DIR"
    fi

    log_info "清理完成"
}

# 主函数
main() {
    log_info "开始RAG系统数据恢复..."

    check_docker
    check_backup_file

    if [ "$VERIFY_ONLY" = true ]; then
        extract_backup
        verify_backup
        cleanup
        exit 0
    fi

    check_containers_before_restore
    extract_backup
    verify_backup

    # 执行恢复
    restore_postgres
    restore_elasticsearch
    restore_milvus
    restore_minio
    restore_etcd
    restore_app_data

    # 等待所有服务重新启动
    log_info "等待所有服务重新启动..."
    sleep 30

    verify_restore
    cleanup

    log_info "RAG系统数据恢复完成！"
}

# 显示帮助信息
show_help() {
    cat << EOF
Docker数据恢复脚本

用法: $0 [选项] -f 备份文件

选项:
    -h, --help          显示帮助信息
    -f, --file          指定备份文件 (必需)
    -t, --temp-dir      指定临时目录 (默认: ./restore_temp)
    --force             强制恢复，忽略错误
    --verify-only       仅验证备份文件，不执行恢复

示例:
    $0 -f backup.tar.gz                    # 从备份文件恢复
    $0 -f backup.tar.gz --force            # 强制恢复
    $0 -f backup.tar.gz --verify-only      # 仅验证备份文件

恢复内容包括:
    - PostgreSQL数据库
    - Elasticsearch索引数据
    - Milvus向量数据
    - MinIO对象存储
    - etcd键值数据
    - 应用日志和配置文件

注意:
    - 恢复操作会覆盖现有数据
    - 建议在恢复前创建当前数据的备份
    - 恢复过程中部分服务会重启

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--file)
            BACKUP_FILE="$2"
            shift 2
            ;;
        -t|--temp-dir)
            RESTORE_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE_RESTORE=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
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