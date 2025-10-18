#!/usr/bin/env python3
"""
Milvus连接修复脚本
等待Milvus服务完全启动并建立连接
"""

import time
import requests
import subprocess
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_milvus_health():
    """检查Milvus服务健康状态"""
    try:
        # 检查Milvus容器状态
        result = subprocess.run(
            ['docker', 'exec', 'rag-milvus', 'curl', '-f', 'http://localhost:9091/health'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            logger.info("✅ Milvus容器健康检查通过")
            return True
        else:
            logger.warning(f"⚠️  Milvus健康检查失败: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("⏰ Milvus健康检查超时")
        return False
    except Exception as e:
        logger.error(f"❌ Milvus健康检查异常: {e}")
        return False

def wait_for_milvus_ready(max_attempts=30, wait_time=10):
    """等待Milvus服务完全就绪"""
    logger.info("⏳ 等待Milvus服务完全就绪...")

    for attempt in range(1, max_attempts + 1):
        logger.info(f"🔄 第 {attempt}/{max_attempts} 次尝试连接Milvus...")

        # 检查容器状态
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=rag-milvus', '--filter', 'status=running', '--format', 'table {{.Names}}\t{{.Status}}'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if 'rag-milvus' in result.stdout and 'Up' in result.stdout:
                logger.info("✅ Milvus容器正在运行")
            else:
                logger.warning("⚠️  Milvus容器状态异常")
                return False

        except Exception as e:
            logger.error(f"❌ 检查容器状态失败: {e}")
            return False

        # 检查Milvus内部服务
        try:
            # 检查Milvus的proxy服务
            result = subprocess.run(
                ['docker', 'exec', 'rag-milvus', 'pgrep', 'milvus'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and 'milvus' in result.stdout:
                logger.info("✅ Milvus进程正在运行")
            else:
                logger.warning("⚠️  Milvus进程未找到")

        except Exception as e:
            logger.warning(f"⚠️  检查Milvus进程失败: {e}")

        # 尝试连接Milvus
        try:
            from pymilvus import connections, utility

            # 连接到Milvus
            connections.connect(alias="test", host="localhost", port="19530", timeout=5)

            # 尝试获取集合列表
            collections = utility.list_collections(using="test")
            logger.info(f"✅ Milvus连接成功！找到 {len(collections)} 个集合")

            # 关闭连接
            connections.disconnect("test")

            logger.info("🎉 Milvus服务已完全就绪！")
            return True

        except Exception as e:
            logger.warning(f"⏳ Milvus连接失败 (尝试 {attempt}): {e}")

            if attempt < max_attempts:
                logger.info(f"⏱️  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error("❌ 达到最大尝试次数，Milvus服务仍未就绪")
                return False

    logger.error("❌ Milvus服务启动超时")
    return False

def check_milvus_dependencies():
    """检查Milvus依赖服务状态"""
    logger.info("🔍 检查Milvus依赖服务...")

    dependencies = {
        'etcd': {'port': 2379, 'check_cmd': ['curl', '-f', 'http://localhost:2379/health']},
        'minio': {'port': 9000, 'check_cmd': ['curl', '-f', 'http://localhost:9000/minio/health/live']}
    }

    all_ready = True

    for service, config in dependencies.items():
        try:
            result = subprocess.run(
                ['docker', 'exec', f'rag-{service}'] + config['check_cmd'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                logger.info(f"✅ {service} 依赖服务正常")
            else:
                logger.warning(f"⚠️  {service} 依赖服务异常: {result.stderr}")
                all_ready = False

        except Exception as e:
            logger.error(f"❌ 检查{service}依赖失败: {e}")
            all_ready = False

    return all_ready

def restart_milvus_if_needed():
    """在必要时重启Milvus服务"""
    logger.info("🔄 检查是否需要重启Milvus服务...")

    try:
        # 检查Milvus日志是否有错误
        result = subprocess.run(
            ['docker', 'logs', 'rag-milvus', '--tail', '50'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if 'error' in result.stdout.lower() or 'failed' in result.stdout.lower():
            logger.warning("发现Milvus错误日志，尝试重启服务...")

            # 重启Milvus服务
            subprocess.run(['docker', 'restart', 'rag-milvus'], check=True)
            logger.info("✅ Milvus服务已重启")

            # 等待重启完成
            time.sleep(30)
            return True
        else:
            logger.info("✅ Milvus日志未发现严重错误")
            return False

    except Exception as e:
        logger.error(f"❌ 检查Milvus日志失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 Milvus连接修复工具")
    print("="*50)

    logger.info("开始修复Milvus连接问题...")

    # 步骤1: 检查依赖服务
    if not check_milvus_dependencies():
        logger.error("❌ 依赖服务检查失败，请先确保etcd和minio正常运行")
        return 1

    # 步骤2: 检查是否需要重启
    if restart_milvus_if_needed():
        logger.info("🔄 Milvus服务已重启，等待重新初始化...")

    # 步骤3: 等待Milvus就绪
    if wait_for_milvus_ready():
        logger.info("🎉 Milvus连接修复成功！")
        print("\\n✅ Milvus服务已完全就绪，可以继续使用RAG系统")
        print("📝 建议: 现在可以运行数据导入脚本")
        return 0
    else:
        logger.error("❌ Milvus连接修复失败")
        print("\\n⚠️  Milvus服务仍有问题，建议检查Docker日志")
        print("🔧 查看日志命令: docker-compose logs milvus")
        return 1

if __name__ == "__main__":
    exit(main())