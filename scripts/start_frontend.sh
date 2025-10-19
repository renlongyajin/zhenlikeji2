#!/bin/bash

# 前端服务启动脚本
# 用于启动和管理RAG系统的前端服务

set -e

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

# 配置变量
FRONTEND_DIR="./frontend"
NGINX_CONF="./nginx.conf"
BACKEND_URL="http://localhost:8000"
FRONTEND_PORT="12345"
DOCKER_NETWORK="rag-network"

# 显示帮助信息
show_help() {
    cat << EOF
前端服务启动脚本

用法: $0 [选项]

选项:
    -h, --help          显示帮助信息
    -d, --dev           开发模式（本地HTTP服务器）
    -p, --prod          生产模式（Nginx容器）
    -s, --stop          停止前端服务
    -r, --restart       重启前端服务
    -c, --check         检查服务状态
    -b, --build         构建前端资源
    --ssl               启用SSL/HTTPS
    --port PORT         指定端口（默认: 12345）
    --backend URL       指定后端URL（默认: http://localhost:8000）

模式说明:
    开发模式: 使用Python HTTP服务器，适合开发调试
    生产模式: 使用Nginx容器，适合生产环境

示例:
    $0                    # 生产模式启动
    $0 -d                 # 开发模式启动
    $0 -s                 # 停止服务
    $0 -r                 # 重启服务
    $0 -p --ssl           # 生产模式带SSL
    $0 --port 8080        # 指定端口

EOF
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖项..."

    # 检查docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi

    # 检查docker-compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi

    # 检查前端文件
    if [ ! -d "$FRONTEND_DIR" ]; then
        log_error "前端目录 $FRONTEND_DIR 不存在"
        exit 1
    fi

    # 检查nginx配置文件
    if [ ! -f "$NGINX_CONF" ]; then
        log_warn "Nginx配置文件 $NGINX_CONF 不存在，将使用默认配置"
    fi

    log_info "依赖项检查完成"
}

# 检查服务状态
check_service_status() {
    log_info "检查服务状态..."

    # 检查后端服务
    if curl -s "http://localhost:8000/health" > /dev/null; then
        log_info "✅ 后端API服务运行正常"
    else
        log_warn "⚠️ 后端API服务可能未运行"
    fi

    # 检查前端服务（开发模式）
    if pgrep -f "python.*http.server" > /dev/null; then
        log_info "✅ 开发模式前端服务正在运行"
    fi

    # 检查nginx容器
    if docker ps | grep -q "rag-react-nginx"; then
        log_info "✅ Nginx容器正在运行"
    fi

    # 检查端口占用
    if netstat -tlnp 2>/dev/null | grep -q ":$FRONTEND_PORT "; then
        log_info "✅ 端口 $FRONTEND_PORT 已被占用"
    else
        log_info "✅ 端口 $FRONTEND_PORT 可用"
    fi
}

# 停止前端服务
stop_frontend() {
    log_info "停止前端服务..."

    # 停止开发模式
    if pgrep -f "python.*http.server" > /dev/null; then
        log_info "停止开发模式HTTP服务器..."
        pkill -f "python.*http.server" || true
        log_info "开发模式已停止"
    fi

    # 停止Nginx容器
    if docker ps | grep -q "rag-react-nginx"; then
        log_info "停止Nginx容器..."
        docker-compose stop nginx || true
        docker-compose rm -f nginx || true
        log_info "Nginx容器已停止"
    fi

    log_info "前端服务停止完成"
}

# 构建前端资源
build_frontend() {
    log_info "构建前端资源..."

    cd "$FRONTEND_DIR"

    # 检查是否有构建脚本
    if [ -f "package.json" ]; then
        log_info "检测到package.json，执行npm构建..."
        if command -v npm &> /dev/null; then
            npm install || log_warn "npm安装失败，继续执行"
            npm run build || log_warn "npm构建失败，使用原始文件"
        else
            log_warn "npm未安装，跳过构建步骤"
        fi
    fi

    # 检查是否有构建输出目录
    if [ -d "dist" ] || [ -d "build" ]; then
        log_info "✅ 检测到构建输出，使用构建后的文件"
    else
        log_info "✅ 使用原始前端文件"
    fi

    cd - > /dev/null
    log_info "前端资源构建完成"
}

# 开发模式启动
dev_mode() {
    log_info "启动开发模式..."

    # 停止已有服务
    stop_frontend

    # 构建前端资源
    build_frontend

    cd "$FRONTEND_DIR"

    log_info "启动Python HTTP服务器..."
    log_info "前端服务地址: http://localhost:$FRONTEND_PORT"
    log_info "后端API地址: $BACKEND_URL"

    # 启动HTTP服务器
    python3 -m http.server $FRONTEND_PORT --bind 0.0.0.0 &
    SERVER_PID=$!

    log_info "开发模式HTTP服务器已启动 (PID: $SERVER_PID)"
    log_info "按 Ctrl+C 停止服务"

    # 等待中断信号
    trap "log_info '收到中断信号，停止服务...'; pkill -f 'python.*http.server'; exit 0" INT TERM
    wait $SERVER_PID
}

# 生产模式启动
prod_mode() {
    log_info "启动生产模式..."

    # 停止已有服务
    stop_frontend

    # 构建前端资源
    build_frontend

    log_info "启动Nginx容器..."
    log_info "前端服务地址: http://localhost:$FRONTEND_PORT"
    log_info "后端API地址: $BACKEND_URL"

    # 检查docker-compose配置
    if ! grep -q "nginx:" docker-compose.yml; then
        log_error "docker-compose.yml中未找到nginx服务配置"
        exit 1
    fi

    # 启动nginx服务
    docker-compose up -d nginx

    # 等待服务启动
    log_info "等待Nginx服务启动..."
    for i in {1..30}; do
        if curl -s "http://localhost:$FRONTEND_PORT" > /dev/null; then
            log_info "✅ Nginx服务启动成功"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Nginx服务启动超时"
            docker-compose logs nginx
            exit 1
        fi
        sleep 2
    done

    log_info "生产模式Nginx服务已启动"
    log_info "使用 'docker-compose logs nginx' 查看日志"
}

# 重启服务
restart_service() {
    log_info "重启前端服务..."
    stop_frontend
    sleep 2

    # 根据当前模式重启
    if pgrep -f "python.*http.server" > /dev/null; then
        dev_mode
    else
        prod_mode
    fi
}

# 主函数
main() {
    local mode="prod"
    local ssl_enabled=false
    local custom_port=""
    local custom_backend=""

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dev)
                mode="dev"
                shift
                ;;
            -p|--prod)
                mode="prod"
                shift
                ;;
            -s|--stop)
                stop_frontend
                exit 0
                ;;
            -r|--restart)
                restart_service
                exit 0
                ;;
            -c|--check)
                check_dependencies
                check_service_status
                exit 0
                ;;
            -b|--build)
                check_dependencies
                build_frontend
                exit 0
                ;;
            --ssl)
                ssl_enabled=true
                shift
                ;;
            --port)
                custom_port="$2"
                FRONTEND_PORT="$2"
                shift 2
                ;;
            --backend)
                custom_backend="$2"
                BACKEND_URL="$2"
                shift 2
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 检查依赖
    check_dependencies

    # 根据模式启动
    case $mode in
        dev)
            dev_mode
            ;;
        prod)
            prod_mode
            ;;
        *)
            log_error "未知模式: $mode"
            exit 1
            ;;
    esac
}

# 如果直接运行脚本
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi