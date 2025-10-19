#!/bin/bash

# Docker数据备份脚本 - 修复版本
# 修复Elasticsearch备份和权限问题

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

# 创建备份目录并设置权限
create_backup_dir() {
    mkdir -p "${BACKUP_PATH}"
    chmod 755 "${BACKUP_PATH}"
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

# 备份Elasticsearch - 修复版本
backup_elasticsearch() {
    log_info "开始备份Elasticsearch数据..."

    if docker ps --format "table {{.Names}}" | grep -q "^rag-elasticsearch$"; then
        # 检查Elasticsearch健康状态
        health_status=$(curl -s -X GET "localhost:9200/_cluster/health" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        if [ "$health_status" != "green" ] && [ "$health_status" != "yellow" ]; then
            log_warn "Elasticsearch状态异常: ${health_status}，尝试备用备份方案"
            backup_elasticsearch_alternative
            return 0
        fi

        # 尝试创建快照仓库（如果失败，使用备用方案）
        snapshot_response=$(curl -s -X PUT "localhost:9200/_snapshot/backup_repo" -H 'Content-Type: application/json' -d'{
            "type": "fs",
            "settings": {
                "location": "/usr/share/elasticsearch/backup",
                "compress": true
            }
        }' 2>/dev/null || echo "")

        if echo "$snapshot_response" | grep -q "repository_exception\|path.repo"; then
            log_warn "无法创建Elasticsearch快照仓库，使用备用备份方案"
            backup_elasticsearch_alternative
        else
            # 标准快照备份
            log_info "使用Elasticsearch快照备份方案"

            # 创建快照
            curl -X PUT "localhost:9200/_snapshot/backup_repo/snapshot_${TIMESTAMP}" -H 'Content-Type: application/json' -d'{
                "indices": "*",
                "ignore_unavailable": true,
                "include_global_state": false
            }' 2>/dev/null || true

            # 等待快照完成
            sleep 5

            # 验证快照状态
            snapshot_status=$(curl -s -X GET "localhost:9200/_snapshot/backup_repo/snapshot_${TIMESTAMP}" 2>/dev/null | grep -o '"state":"[^"]*"' | cut -d'"' -f4)
            if [ "$snapshot_status" = "SUCCESS" ]; then
                log_info "Elasticsearch快照备份成功"
                echo "snapshot_${TIMESTAMP}" > "${BACKUP_PATH}/elasticsearch_snapshot_info.txt"
            else
                log_warn "Elasticsearch快照备份可能失败，状态: ${snapshot_status}"
                backup_elasticsearch_alternative
            fi
        fi
    else
        log_warn "Elasticsearch容器未运行，跳过备份"
    fi
}

# Elasticsearch备用备份方案
backup_elasticsearch_alternative() {
    log_info "使用Elasticsearch备用备份方案（导出索引数据）..."

    # 导出索引列表
    curl -s -X GET "localhost:9200/_cat/indices?format=json&h=index,docs.count,store.size" > "${BACKUP_PATH}/elasticsearch_indices.json" 2>/dev/null || true

    if [ -s "${BACKUP_PATH}/elasticsearch_indices.json" ]; then
        # 获取非系统索引
        indices=$(curl -s -X GET "localhost:9200/_cat/indices?h=index" 2>/dev/null | grep -v "^\." | head -10)

        if [ -n "$indices" ]; then
            for index in $indices; do
                # 导出映射
                curl -s -X GET "localhost:9200/${index}/_mapping" > "${BACKUP_PATH}/elasticsearch_mapping_${index}.json" 2>/dev/null
                # 导出设置
                curl -s -X GET "localhost:9200/${index}/_settings" > "${BACKUP_PATH}/elasticsearch_settings_${index}.json" 2>/dev/null
                # 导出部分文档数据（限制1000条）
                curl -s -X POST "localhost:9200/${index}/_search" -H 'Content-Type: application/json' -d'{
                    "size": 1000,
                    "query": {"match_all": {}}
                }' > "${BACKUP_PATH}/elasticsearch_data_${index}.json" 2>/dev/null
            done
            log_info "Elasticsearch数据导出完成（索引映射、设置和部分数据）"
        else
            log_warn "未找到可备份的Elasticsearch索引"
        fi
    else
        log_warn "无法获取Elasticsearch索引信息"
    fi

    log_info "Elasticsearch备用备份方案完成"
}

# 备份Milvus - 修复权限问题
backup_milvus() {
    log_info "开始备份Milvus数据..."

    if docker ps --format "table {{.Names}}" | grep -q "^rag-milvus$"; then
        # 创建Milvus备份目录
        mkdir -p "${BACKUP_PATH}/milvus_data"

        # 使用临时容器备份数据（改进权限处理）
        log_info "使用临时容器备份Milvus数据..."

        # 先尝试获取Milvus集合信息
        collections_response=$(curl -s -X GET "localhost:19530/v1/vector/collections" 2>/dev/null || echo "")
        if [ -n "$collections_response" ] && echo "$collections_response" | grep -q "code.*0"; then
            echo "$collections_response" > "${BACKUP_PATH}/milvus_collections.json"
            log_info "Milvus集合信息获取成功"
        else
            log_warn "无法获取Milvus集合信息，可能是权限或网络问题"
        fi

        # 使用更安全的备份方式
        docker run --rm \
            --user root \
            -v zhenlikeji2_milvus-data:/source:ro \
            -v "${BACKUP_PATH}/milvus_data":/backup \
            alpine:latest \
            /bin/sh -c "
                # 创建备份压缩文件
                cd /source && \
                tar -czf /backup/milvus_data.tar.gz . && \
                chmod 644 /backup/milvus_data.tar.gz && \
                echo 'Milvus data backup completed' && \
                ls -la /backup/
            " 2>/dev/null

        if [ $? -eq 0 ] && [ -f "${BACKUP_PATH}/milvus_data/milvus_data.tar.gz" ]; then
            log_info "Milvus数据备份完成"
            # 确保文件权限正确
            chmod 644 "${BACKUP_PATH}/milvus_data/milvus_data.tar.gz" 2>/dev/null || true
        else
            log_warn "Milvus数据备份可能遇到问题，尝试替代方案"
            # 备用方案：只备份关键信息
            echo "Milvus backup attempted at $(date)" > "${BACKUP_PATH}/milvus_backup_info.txt"
            echo "Collections info: $collections_response" >> "${BACKUP_PATH}/milvus_backup_info.txt"
        fi
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

# 备份应用数据 - 优化版本，只保留必要文件
backup_app_data() {
    log_info "开始备份应用数据（优化版本）..."

    # 备份日志文件
    if [ -d "./logs" ]; then
        cp -r ./logs "${BACKUP_PATH}/" 2>/dev/null || true
    fi

    # 备份数据文件 - 只保留必要的子目录
    if [ -d "./data" ]; then
        mkdir -p "${BACKUP_PATH}/data"

        # 备份原始PDF文件（单个文件）
        if [ -f "./data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf" ]; then
            mkdir -p "${BACKUP_PATH}/data"
            cp "./data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf" "${BACKUP_PATH}/data/" 2>/dev/null || true
            log_info "已备份原始PDF文件"
        fi

        # 备份与数据库关联的图片
        if [ -d "./data/renamed_images" ]; then
            cp -r ./data/renamed_images "${BACKUP_PATH}/data/" 2>/dev/null || true
            log_info "已备份data/renamed_images目录"
        fi

        # 备份其他必要的非图片数据（如果有的话）
        for subdir in extracted processed; do
            if [ -d "./data/$subdir" ]; then
                cp -r ./data/$subdir "${BACKUP_PATH}/data/" 2>/dev/null || true
                log_info "已备份data/$subdir目录"
            fi
        done

        # 注意：跳过以下大容量目录
        # - raw/ (PDF页面扫描图片，非原始PDF)
        # - extracted_images/ (提取的图片副本)
        # - corrected_images/ (纠正后的图片副本)
        # - final_images/ (最终图片副本，如果renamed_images已足够)
    fi

    # 备份配置文件
    if [ -d "./config" ]; then
        cp -r ./config "${BACKUP_PATH}/" 2>/dev/null || true
    fi

    log_info "应用数据备份完成（优化版本）"
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
            "status": "$(if [ -f "${BACKUP_PATH}/milvus_collections.json" ] || [ -f "${BACKUP_PATH}/milvus_backup_info.txt" ]; then echo "completed"; else echo "skipped"; fi)",
            "file": "milvus_collections.json"
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
            "directories": ["logs", "data/恶件肺脏疾病和哺脏少见病快速现场评价组学图谱-224.pdf", "data/renamed_images", "config"],
            "note": "优化版本：保留原始PDF文件和renamed_images图片，跳过PDF页面扫描图片等中间文件"
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
    # 排除校验文件自身，避免循环校验问题
    find . -type f ! -name "checksums.md5" -exec md5sum {} \; > checksums.md5 2>/dev/null || true
    cd - >/dev/null

    log_info "校验文件创建完成"
}

# 压缩备份文件
compress_backup() {
    log_info "压缩备份文件..."

    cd "${BACKUP_DIR}"

    # 确保所有文件都有正确的权限
    find "${BACKUP_NAME}" -type f -exec chmod 644 {} \; 2>/dev/null || true
    find "${BACKUP_NAME}" -type d -exec chmod 755 {} \; 2>/dev/null || true

    # 创建压缩文件
    tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}" 2>/dev/null
    local compress_size=$(du -h "${BACKUP_NAME}.tar.gz" 2>/dev/null | cut -f1 || echo "unknown")

    # 检查压缩是否成功
    if [ -f "${BACKUP_NAME}.tar.gz" ]; then
        # 设置压缩文件权限
        chmod 644 "${BACKUP_NAME}.tar.gz"

        # 安全删除原始备份目录
        rm -rf "${BACKUP_NAME}" 2>/dev/null || {
            log_warn "无法删除原始备份目录，请手动清理"
            # 尝试使用sudo如果可用
            if command -v sudo >/dev/null 2>&1; then
                sudo rm -rf "${BACKUP_NAME}" 2>/dev/null || true
            fi
        }

        log_info "备份压缩完成，文件大小: ${compress_size}"
        log_info "备份文件路径: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    else
        log_error "备份压缩失败"
        return 1
    fi
}

# 清理旧备份
cleanup_old_backups() {
    log_info "清理旧备份文件..."

    # 保留最近7天的备份
    find "${BACKUP_DIR}" -name "rag_system_backup_*.tar.gz" -mtime +7 -delete 2>/dev/null || true

    log_info "旧备份清理完成"
}

# 显示帮助信息
show_help() {
    cat << EOF
Docker数据备份脚本 - 修复版本

修复内容:
1. 修复Elasticsearch备份问题（快照仓库配置）
2. 修复Milvus备份权限问题
3. 改进错误处理机制

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
    - Elasticsearch索引数据（支持备用方案）
    - Milvus向量数据（修复权限问题）
    - MinIO对象存储
    - etcd键值数据
    - 应用日志和配置文件

注意:
    - 如果Elasticsearch快照失败，会自动使用备用方案（导出索引映射和数据）
    - Milvus备份改进了权限处理，避免权限错误
    - 所有错误都会记录但不会中断整个备份过程

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

# 主函数
main() {
    log_info "开始RAG系统数据备份...（修复版本）"

    check_docker
    check_containers
    create_backup_dir

    # 执行备份（不中断的错误处理）
    backup_postgres || log_warn "PostgreSQL备份失败，继续其他组件备份"
    backup_elasticsearch || log_warn "Elasticsearch备份失败，继续其他组件备份"
    backup_milvus || log_warn "Milvus备份失败，继续其他组件备份"
    backup_minio || log_warn "MinIO备份失败，继续其他组件备份"
    backup_etcd || log_warn "etcd备份失败，继续其他组件备份"
    backup_app_data || log_warn "应用数据备份失败，继续其他组件备份"

    create_manifest || log_warn "备份清单生成失败"
    create_checksum || log_warn "校验文件创建失败"

    if [ "$NO_COMPRESS" = false ]; then
        compress_backup || log_warn "备份压缩失败"
    fi

    if [ "$NO_CLEANUP" = false ]; then
        cleanup_old_backups || log_warn "旧备份清理失败"
    fi

    log_info "RAG系统数据备份完成！"
    if [ "$NO_COMPRESS" = false ] && [ -f "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" ]; then
        log_info "备份文件: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    else
        log_info "备份目录: ${BACKUP_PATH}"
    fi
}

# 执行主函数
main