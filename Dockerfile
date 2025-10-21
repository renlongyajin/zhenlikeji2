# 使用官方Python运行时作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY docs/ ./docs/

# 创建必要的目录
RUN mkdir -p logs data extracted

# 设置文件权限
RUN chmod +x scripts/*.py

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

# 切换到非root用户
USER app

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 启动命令 - 使用环境变量端口
CMD ["sh", "-c", "python3 -m uvicorn src.agent.api_service:app --host 0.0.0.0 --port ${API_PORT:-8001} --log-level info"]