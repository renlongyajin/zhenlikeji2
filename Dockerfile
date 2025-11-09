FROM python:3.11-slim

# ========= 环境变量 =========
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ========= 换 APT 镜像源（兼容 debian.sources） =========
RUN set -eux; \
    MIRROR="https://mirrors.tuna.tsinghua.edu.cn"; \
    suite="$(. /etc/os-release; echo "${VERSION_CODENAME:-bookworm}")"; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i -e "s|http://deb.debian.org|${MIRROR}|g" \
               -e "s|https://deb.debian.org|${MIRROR}|g" \
               -e "s|http://security.debian.org|${MIRROR}|g" \
               -e "s|https://security.debian.org|${MIRROR}|g" /etc/apt/sources.list; \
    else \
        echo "Types: deb" > /etc/apt/sources.list.d/debian.sources; \
        echo "URIs: ${MIRROR}/debian" >> /etc/apt/sources.list.d/debian.sources; \
        echo "Suites: ${suite} ${suite}-updates" >> /etc/apt/sources.list.d/debian.sources; \
        echo "Components: main contrib non-free non-free-firmware" >> /etc/apt/sources.list.d/debian.sources; \
        echo "Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg" >> /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        gcc g++ curl netcat-traditional ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# ========= 可选：构建时代理 =========
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY}

# ========= Python 依赖安装 =========
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url "${PIP_INDEX_URL}" -r requirements.txt

# ========= 拷贝项目文件 =========
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY docs/ ./docs/

# ========= 创建目录、用户、权限 =========
RUN set -eux; \
    mkdir -p logs data extracted; \
    [ -d scripts ] && chmod +x scripts/*.py || true; \
    useradd --create-home --shell /bin/bash app; \
    chown -R app:app /app

USER app

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["sh", "-c", "python3 -m uvicorn src.agent.api_service:app --host 0.0.0.0 --port ${API_PORT:-8001} --log-level info"]
