#!/usr/bin/env python3
"""
RAG系统状态检查脚本
检查Docker服务、数据库连接和数据导入状态
"""

import subprocess
import requests
import json
from datetime import datetime
import logging
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemStatusChecker:
    """系统状态检查器"""

    def __init__(self):
        """初始化检查器"""
        self.services = {
            'elasticsearch': {'host': 'localhost', 'port': 9200, 'endpoint': '/_cluster/health'},
            'milvus': {'host': 'localhost', 'port': 19530, 'type': 'grpc'},
            'postgres': {'host': 'localhost', 'port': 5432, 'type': 'database'},
            'minio': {'host': 'localhost', 'port': 9000, 'endpoint': '/minio/health/live'},
            'kibana': {'host': 'localhost', 'port': 5601, 'endpoint': '/api/status'}
        }

    def check_docker_services(self) -> Dict[str, Any]:
        """检查Docker服务状态"""
        logger.info("🐳 检查Docker服务状态...")

        try:
            # 获取容器状态
            result = subprocess.run(
                ['docker-compose', '-f', '/home/ubuntu/myproject/zhenlikeji2/docker/docker-compose.yml', 'ps', '--format', 'json'],
                capture_output=True,
                text=True,
                check=True
            )

            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        container = json.loads(line)
                        containers.append({
                            'name': container.get('Name', ''),
                            'state': container.get('State', ''),
                            'status': container.get('Status', ''),
                            'ports': container.get('Publishers', [])
                        })
                    except json.JSONDecodeError:
                        continue

            logger.info(f"✅ 发现 {len(containers)} 个Docker容器")
            return {'status': 'running', 'containers': containers}

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Docker服务检查失败: {e}")
            return {'status': 'error', 'error': str(e)}
        except Exception as e:
            logger.error(f"❌ Docker服务检查失败: {e}")
            return {'status': 'error', 'error': str(e)}

    def check_elasticsearch(self) -> Dict[str, Any]:
        """检查Elasticsearch状态"""
        logger.info("🔍 检查Elasticsearch状态...")

        try:
            response = requests.get('http://localhost:9200/_cluster/health', timeout=10)
            if response.status_code == 200:
                health = response.json()
                logger.info(f"✅ Elasticsearch状态: {health['status']}")

                # 获取索引统计
                stats_response = requests.get('http://localhost:9200/medical_documents/_stats', timeout=10)
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    doc_count = stats['indices']['medical_documents']['primaries']['docs']['count']
                    logger.info(f"📊 文档数量: {doc_count}")

                    return {
                        'status': 'healthy',
                        'health': health['status'],
                        'document_count': doc_count,
                        'nodes': health['number_of_nodes'],
                        'data_nodes': health['number_of_data_nodes']
                    }
                else:
                    return {'status': 'warning', 'health': health['status'], 'message': '无法获取索引统计'}
            else:
                return {'status': 'error', 'message': f'HTTP {response.status_code}'}

        except Exception as e:
            logger.error(f"❌ Elasticsearch检查失败: {e}")
            return {'status': 'error', 'error': str(e)}

    def check_milvus(self) -> Dict[str, Any]:
        """检查Milvus状态"""
        logger.info("🎯 检查Milvus状态...")

        try:
            from pymilvus import connections, utility
            connections.connect(alias="default", host="localhost", port="19530")

            collections = utility.list_collections()
            logger.info(f"✅ Milvus连接成功，集合数量: {len(collections)}")

            collection_stats = {}
            for collection_name in collections:
                if collection_name == 'medical_vectors':
                    from pymilvus import Collection
                    collection = Collection(collection_name)
                    collection.load()
                    count = collection.num_entities
                    collection.release()
                    collection_stats[collection_name] = {'vector_count': count}
                    logger.info(f"📊 集合 {collection_name}: {count} 个向量")

            return {
                'status': 'healthy',
                'collections': collections,
                'collection_stats': collection_stats
            }

        except Exception as e:
            logger.error(f"❌ Milvus检查失败: {e}")
            return {'status': 'error', 'error': str(e)}

    def check_postgres(self) -> Dict[str, Any]:
        """检查PostgreSQL状态"""
        logger.info("🗄️  检查PostgreSQL状态...")

        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="rag_system",
                user="admin",
                password="password"
            )
            cur = conn.cursor()

            # 检查数据库版本
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            logger.info(f"✅ PostgreSQL连接成功: {version.split(',')[0]}")

            # 检查表数量
            cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
            table_count = cur.fetchone()[0]
            logger.info(f"📊 表数量: {table_count}")

            cur.close()
            conn.close()

            return {
                'status': 'healthy',
                'version': version,
                'table_count': table_count
            }

        except Exception as e:
            logger.error(f"❌ PostgreSQL检查失败: {e}")
            return {'status': 'error', 'error': str(e)}

    def check_service_ports(self) -> Dict[str, Dict[str, Any]]:
        """检查服务端口状态"""
        logger.info("🔌 检查服务端口状态...")

        port_status = {}

        for service_name, config in self.services.items():
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((config['host'], config['port']))
                sock.close()

                if result == 0:
                    port_status[service_name] = {'status': 'open', 'port': config['port']}
                    logger.info(f"✅ {service_name} 端口 {config['port']}: 开放")
                else:
                    port_status[service_name] = {'status': 'closed', 'port': config['port']}
                    logger.info(f"❌ {service_name} 端口 {config['port']}: 关闭")

            except Exception as e:
                port_status[service_name] = {'status': 'error', 'port': config['port'], 'error': str(e)}
                logger.info(f"❌ {service_name} 端口 {config['port']}: 错误 - {e}")

        return port_status

    def generate_system_report(self) -> str:
        """生成系统状态报告"""
        logger.info("📋 生成系统状态报告...")

        report = []
        report.append("=" * 80)
        report.append("RAG系统状态报告")
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Docker服务状态
        docker_status = self.check_docker_services()
        report.append("🐳 Docker服务状态:")
        if docker_status['status'] == 'running':
            for container in docker_status.get('containers', []):
                report.append(f"  • {container['name']}: {container['status']}")
        else:
            report.append(f"  ❌ {docker_status.get('error', '未知错误')}")
        report.append("")

        # 端口状态
        port_status = self.check_service_ports()
        report.append("🔌 服务端口状态:")
        for service, status in port_status.items():
            if status['status'] == 'open':
                report.append(f"  ✅ {service}: 端口 {status['port']} 开放")
            else:
                report.append(f"  ❌ {service}: 端口 {status['port']} {status['status']}")
        report.append("")

        # Elasticsearch状态
        es_status = self.check_elasticsearch()
        report.append("🔍 Elasticsearch状态:")
        if es_status['status'] == 'healthy':
            report.append(f"  ✅ 健康状态: {es_status['health']}")
            report.append(f"  📊 文档数量: {es_status['document_count']}")
            report.append(f"  🖥️  节点数: {es_status['nodes']}")
        else:
            report.append(f"  ❌ {es_status.get('error', '连接失败')}")
        report.append("")

        # Milvus状态
        milvus_status = self.check_milvus()
        report.append("🎯 Milvus状态:")
        if milvus_status['status'] == 'healthy':
            report.append(f"  ✅ 连接成功")
            report.append(f"  📋 集合数量: {len(milvus_status['collections'])}")
            for collection_name, stats in milvus_status.get('collection_stats', {}).items():
                report.append(f"  📊 {collection_name}: {stats['vector_count']} 个向量")
        else:
            report.append(f"  ❌ {milvus_status.get('error', '连接失败')}")
        report.append("")

        # PostgreSQL状态
        pg_status = self.check_postgres()
        report.append("🗄️  PostgreSQL状态:")
        if pg_status['status'] == 'healthy':
            report.append(f"  ✅ 连接成功")
            report.append(f"  📊 表数量: {pg_status['table_count']}")
        else:
            report.append(f"  ❌ {pg_status.get('error', '连接失败')}")
        report.append("")

        # 总体状态评估
        all_healthy = all([
            docker_status['status'] == 'running',
            es_status['status'] == 'healthy',
            milvus_status['status'] == 'healthy',
            pg_status['status'] == 'healthy'
        ])

        if all_healthy:
            report.append("🎉 系统状态: 所有服务运行正常")
        else:
            report.append("⚠️  系统状态: 部分服务存在问题，请检查日志")

        report.append("=" * 80)

        return "\n".join(report)

def main():
    """主函数"""
    checker = SystemStatusChecker()
    report = checker.generate_system_report()
    print(report)

    # 保存报告到文件
    with open('/home/ubuntu/myproject/zhenlikeji2/logs/system_status_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info("✅ 系统状态报告已生成并保存到 logs/system_status_report.txt")

if __name__ == "__main__":
    main()