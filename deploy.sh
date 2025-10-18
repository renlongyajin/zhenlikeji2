#!/bin/bash

# RAG+ReAct智能问答系统一键部署脚本
# 版本: v1.0
# 功能: 自动化部署完整的医学RAG问答系统

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."

    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi

    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi

    # 检查内存
    MEMORY_GB=$(free -g | awk 'NR==2{print $2}')
    if [ "$MEMORY_GB" -lt 4 ]; then
        log_warning "系统内存小于4GB，可能影响性能"
    fi

    # 检查磁盘空间
    DISK_AVAIL=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$DISK_AVAIL" -lt 10 ]; then
        log_warning "磁盘空间小于10GB，可能影响系统运行"
    fi

    log_success "系统要求检查通过"
}

# 检查端口占用
check_port_availability() {
    log_info "检查端口可用性..."

    local ports=(80 443 8000 9200 19530 5432 5601 9000 2379)
    local occupied_ports=()

    for port in "${ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            occupied_ports+=("$port")
        fi
    done

    if [ ${#occupied_ports[@]} -gt 0 ]; then
        log_warning "以下端口已被占用: ${occupied_ports[*]}"
        log_info "系统将尝试使用这些端口，但可能需要手动处理冲突"
    else
        log_success "所有端口可用"
    fi
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."

    local dirs=("logs" "data" "ssl" "config" "backups")

    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log_info "创建目录: $dir"
    done

    log_success "目录创建完成"
}

# 生成SSL证书（开发环境）
generate_ssl_certificates() {
    log_info "生成SSL证书..."

    if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
        log_info "正在生成自签名SSL证书..."

        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem \
            -days 365 -nodes -subj "/C=CN/ST=Beijing/L=Beijing/O=RAG-ReAct/CN=localhost" \
            2>/dev/null || {
            log_warning "SSL证书生成失败，将使用HTTP模式"
            return 0
        }

        chmod 600 ssl/key.pem
        chmod 644 ssl/cert.pem

        log_success "SSL证书生成完成"
    else
        log_info "SSL证书已存在，跳过生成"
    fi
}

# 环境变量配置
setup_environment() {
    log_info "配置环境变量..."

    if [ ! -f ".env" ]; then
        cat > .env << EOF
# RAG+ReAct系统环境变量配置
# 数据库配置
ELASTICSEARCH_URL=http://elasticsearch:9200
MILVUS_URL=milvus:19530
POSTGRES_URL=postgresql://admin:password@postgres:5432/rag_system

# API密钥配置（请替换为您的实际密钥）
JINA_API_KEY=${JINA_API_KEY:-}
DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-}
QWEN_API_KEY=${QWEN_API_KEY:-}

# 应用配置
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# 性能配置
MAX_WORKERS=4
BATCH_SIZE=10
CACHE_TTL=3600

# 安全配置
SECRET_KEY=$(openssl rand -hex 32)
API_RATE_LIMIT=100

# 监控配置
ENABLE_METRICS=true
METRICS_PORT=9090
EOF
        log_success "环境变量配置文件已创建 (.env)"
        log_warning "请根据需要修改 .env 文件中的配置"
    else
        log_info "环境变量配置文件已存在"
    fi
}

# Docker镜像预拉取
pull_docker_images() {
    log_info "预拉取Docker镜像..."

    local images=(
        "python:3.11-slim"
        "nginx:alpine"
        "docker.elastic.co/elasticsearch/elasticsearch:8.8.0"
        "milvusdb/milvus:v2.3.0"
        "postgres:15"
        "minio/minio:latest"
        "quay.io/coreos/etcd:v3.5.5"
        "docker.elastic.co/kibana/kibana:8.8.0"
    )

    for image in "${images[@]}"; do
        log_info "拉取镜像: $image"
        if docker pull "$image"; then
            log_success "镜像拉取成功: $image"
        else
            log_warning "镜像拉取失败: $image"
        fi
    done
}

# 系统初始化
initialize_system() {
    log_info "初始化系统..."

    # 检查并启动Docker服务
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker服务未运行，请先启动Docker服务"
        exit 1
    fi

    # 清理旧的容器（如果存在）
    log_info "清理旧的容器..."
    docker-compose down --remove-orphans 2>/dev/null || true

    # 清理未使用的镜像和卷
    if [ "$CLEANUP_OLD_IMAGES" = "true" ]; then
        log_info "清理未使用的Docker资源..."
        docker system prune -f --volumes 2>/dev/null || true
    fi

    log_success "系统初始化完成"
}

# 构建应用镜像
build_application() {
    log_info "构建应用镜像..."

    # 确保frontend目录存在
    mkdir -p frontend

    # 构建应用镜像
    if docker-compose build --no-cache app; then
        log_success "应用镜像构建成功"
    else
        log_error "应用镜像构建失败"
        exit 1
    fi
}

# 启动服务
start_services() {
    log_info "启动服务..."

    # 启动所有服务
    if docker-compose up -d; then
        log_success "服务启动命令执行成功"
    else
        log_error "服务启动失败"
        exit 1
    fi

    # 等待服务就绪
    log_info "等待服务就绪..."
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost/health >/dev/null 2>&1; then
            log_success "应用服务已就绪"
            break
        fi

        log_info "等待服务就绪... (尝试 $attempt/$max_attempts)"
        sleep 10
        attempt=$((attempt + 1))
    done

    if [ $attempt -gt $max_attempts ]; then
        log_error "服务启动超时"
        show_service_logs
        exit 1
    fi
}

# 数据导入
import_sample_data() {
    log_info "导入示例数据..."

    # 检查是否有数据文件
    if [ -d "data/extracted" ] && [ "$(ls -A data/extracted)" ]; then
        log_info "发现数据文件，开始导入..."

        # 等待数据库服务就绪
        sleep 30

        # 执行数据导入
        if docker-compose exec -T app python3 /app/src/data_processing/simple_importer.py; then
            log_success "数据导入成功"
        else
            log_warning "数据导入失败，请手动导入数据"
        fi
    else
        log_info "未找到数据文件，跳过数据导入"
    fi
}

# 系统验证
verify_system() {
    log_info "验证系统状态..."

    # 健康检查
    local services=("app" "elasticsearch" "milvus" "postgres" "nginx")
    local all_healthy=true

    for service in "${services[@]}"; do
        log_info "检查服务: $service"
        if docker-compose exec -T "$service" timeout 10 bash -c "exit 0" 2>/dev/null; then
            log_success "$service 服务运行正常"
        else
            log_error "$service 服务异常"
            all_healthy=false
        fi
    done

    # API功能测试
    log_info "测试API功能..."
    if curl -f -X POST http://localhost/api/query/sync \
        -H "Content-Type: application/json" \
        -d '{"question": "系统测试", "user_id": "deploy_test"}' \
        -m 30 >/dev/null 2>&1; then
        log_success "API功能测试通过"
    else
        log_warning "API功能测试失败"
        all_healthy=false
    fi

    if [ "$all_healthy" = true ]; then
        log_success "系统验证通过"
    else
        log_warning "系统验证未完全通过，请检查服务状态"
    fi
}

# 显示服务状态
show_service_status() {
    log_info "服务状态概览:"
    echo
    docker-compose ps
    echo

    log_info "服务端口状态:"
    echo "应用服务: http://localhost"
    echo "Elasticsearch: http://localhost:9200"
    echo "Milvus: localhost:19530"
    echo "PostgreSQL: localhost:5432"
    echo "MinIO: http://localhost:9000 (admin/admin)"
    echo "Kibana: http://localhost:5601"
    echo
}

# 显示服务日志
show_service_logs() {
    log_info "显示服务日志 (最近50行):"
    echo
    docker-compose logs --tail=50
}

# 性能测试
run_performance_test() {
    log_info "运行性能测试..."

    # 简单的响应时间测试
    local test_query="系统性能测试"
    local start_time=$(date +%s.%N)

    if curl -f -X POST http://localhost/api/query/sync \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"$test_query\", \"user_id\": \"perf_test\"}" \
        -m 60 >/dev/null 2>&1; then

        local end_time=$(date +%s.%N)
        local response_time=$(echo "$end_time - $start_time" | bc -l)

        log_success "性能测试完成"
        log_info "响应时间: ${response_time}s"

        if (( $(echo "$response_time < 5.0" | bc -l) )); then
            log_success "响应时间符合要求 (< 5秒)"
        else
            log_warning "响应时间超过预期 (> 5秒)"
        fi
    else
        log_error "性能测试失败"
    fi
}

# 显示部署报告
show_deployment_report() {
    log_success "🎉 部署完成！"
    echo
    echo "=========================================="
    echo "    RAG+ReAct智能问答系统部署报告"
    echo "=========================================="
    echo
    echo "✅ 部署状态: 成功"
    echo "📅 部署时间: $(date)"
    echo "🐳 Docker版本: $(docker --version)"
    echo "🧩 Docker Compose版本: $(docker-compose --version)"
    echo
    echo "🌐 访问地址:"
    echo "   主应用: http://localhost"
    echo "   Elasticsearch: http://localhost:9200"
    echo "   Kibana: http://localhost:5601"
    echo "   MinIO: http://localhost:9000 (admin/admin)"
    echo
    echo "📊 服务状态:"
    docker-compose ps
    echo
    echo "📋 下一步操作:"
    echo "   1. 访问 http://localhost 测试系统功能"
    echo "   2. 查看日志: docker-compose logs -f"
    echo "   3. 停止服务: docker-compose down"
    echo "   4. 系统状态: docker-compose ps"
    echo
    echo "🔧 故障排除:"
    echo "   查看日志: docker-compose logs [service_name]"
    echo "   重启服务: docker-compose restart [service_name]"
    echo "   系统检查: curl http://localhost/api/status"
    echo
    echo "📖 文档和帮助:"
    echo "   API文档: http://localhost/docs"
    echo "   使用指南: docs/RAG_System_Usage_Guide.md"
    echo "   开发文档: docs/RAG_System_Step3_Development_Guide.md"
    echo
    echo "=========================================="
    echo "    欢迎使用RAG+ReAct智能问答系统！"
    echo "=========================================="
}

# 清理函数
cleanup() {
    log_info "正在清理..."
    docker-compose down --remove-orphans 2>/dev/null || true
    log_success "清理完成"
}

# 主函数
main() {
    # 显示欢迎信息
    echo "=========================================="
    echo "  RAG+ReAct智能问答系统一键部署脚本"
    echo "=========================================="
    echo

    # 解析命令行参数
    local CLEANUP_OLD_IMAGES="false"
    local SKIP_TESTS="false"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --cleanup)
                CLEANUP_OLD_IMAGES="true"
                shift
                ;;
            --skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            --help)
                echo "使用说明:"
                echo "  $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --cleanup      清理旧的Docker镜像和卷"
                echo "  --skip-tests   跳过系统测试"
                echo "  --help         显示帮助信息"
                echo ""
                echo "环境变量:"
                echo "  JINA_API_KEY      Jina API密钥"
                echo "  DEEPSEEK_API_KEY  DeepSeek API密钥"
                echo "  QWEN_API_KEY      千问API密钥"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done

    # 设置错误处理
    trap cleanup EXIT

    # 执行部署步骤
    check_system_requirements
    check_port_availability
    create_directories
    generate_ssl_certificates
    setup_environment
    pull_docker_images
    initialize_system
    build_application
    start_services
    import_sample_data

    # 系统验证和测试
    if [ "$SKIP_TESTS" != "true" ]; then
        verify_system
        run_performance_test
    else
        log_info "跳过系统测试"
    fi

    # 显示结果
    show_service_status
    show_deployment_report
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi