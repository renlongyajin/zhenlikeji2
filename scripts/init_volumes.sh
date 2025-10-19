#!/bin/bash

# Docker数据卷初始化脚本
# 用于创建和初始化本地数据卷目录

set -e

# 颜色输出
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

# 创建数据卷目录
create_volume_dirs() {
    log_info "创建数据卷目录..."

    local volumes=(
        "volumes/elasticsearch"
        "volumes/milvus"
        "volumes/postgres"
        "volumes/minio"
        "volumes/etcd"
    )

    for volume in "${volumes[@]}"; do
        mkdir -p "$volume"
        # 设置适当的权限
        chmod 755 "$volume"
        log_info "✓ 创建目录: $volume"
    done
}

# 创建应用目录
create_app_dirs() {
    log_info "创建应用目录..."

    local dirs=(
        "backups"
        "logs"
        "data"
        "config"
    )

    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        chmod 755 "$dir"
        log_info "✓ 创建目录: $dir"
    done
}

# 设置目录权限
set_permissions() {
    log_info "设置目录权限..."

    # 确保Docker可以访问这些目录
    if command -v chcon >/dev/null 2>&1; then
        # SELinux系统
        for dir in volumes/*; do
            if [ -d "$dir" ]; then
                chcon -Rt svirt_sandbox_file_t "$dir" 2>/dev/null || true
                log_info "✓ 设置SELinux上下文: $dir"
            fi
        done
    fi

    # 设置数据卷目录的所有者和权限
    if [ "$(id -u)" -eq 0 ]; then
        # 如果是root用户，设置为docker用户
        for dir in volumes/*; do
            if [ -d "$dir" ]; then
                chown -R 1000:1000 "$dir" 2>/dev/null || true
                chmod -R 755 "$dir"
                log_info "✓ 设置所有者: $dir"
            fi
        done
    fi
}

# 验证目录结构
verify_structure() {
    log_info "验证目录结构..."

    local all_good=true

    # 检查数据卷目录
    local volumes=(
        "volumes/elasticsearch"
        "volumes/milvus"
        "volumes/postgres"
        "volumes/minio"
        "volumes/etcd"
    )

    for volume in "${volumes[@]}"; do
        if [ -d "$volume" ]; then
            log_info "✓ $volume 存在"
        else
            log_error "✗ $volume 不存在"
            all_good=false
        fi
    done

    # 检查应用目录
    local dirs=(
        "backups"
        "logs"
        "data"
        "config"
    )

    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            log_info "✓ $dir 存在"
        else
            log_error "✗ $dir 不存在"
            all_good=false
        fi
    done

    if [ "$all_good" = true ]; then
        log_info "✓ 所有目录验证通过"
        return 0
    else
        log_error "✗ 部分目录验证失败"
        return 1
    fi
}

# 显示帮助
show_help() {
    cat << EOF
Docker数据卷初始化脚本

用法: $0 [选项]

选项:
    -h, --help      显示帮助信息
    -v, --verify    只验证目录结构，不创建

示例:
    $0              # 创建所有目录
    $0 --verify     # 只验证目录结构

这个脚本会创建以下目录结构:
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

EOF
}

# 主函数
main() {
    log_info "开始初始化Docker数据卷目录..."

    if [ "$VERIFY_ONLY" = true ]; then
        verify_structure
        exit $?
    fi

    create_volume_dirs
    create_app_dirs
    set_permissions

    if verify_structure; then
        log_info "Docker数据卷初始化完成！"
        echo
        log_info "现在您可以："
        echo "  1. 运行Docker Compose: docker-compose up -d"
        echo "  2. 创建数据备份: ./scripts/backup_data.sh"
        echo "  3. 恢复数据: ./scripts/restore_data.sh -f backup.tar.gz"
        echo
        log_info "数据卷目录已准备就绪，支持Docker绑定挂载。"
    else
        log_error "初始化失败，请检查权限和磁盘空间"
        exit 1
    fi
}

# 解析命令行参数
VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verify)
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

# 清理敏感信息
unset VERIFY_ONLY
log_info "初始化脚本执行完成""}

chmod +x scripts/init_volumes.sh