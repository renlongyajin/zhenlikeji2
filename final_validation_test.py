#!/usr/bin/env python3
"""
RAG+ReAct智能问答系统 - 最终验证测试
用于验证系统部署成功并运行正常
"""

import requests
import json
import time
import sys
from datetime import datetime

def test_system_components():
    """测试系统各个组件"""
    print("🚀 开始RAG+ReAct系统最终验证测试...")
    print("="*60)

    # 1. 测试核心服务状态
    print("\n📊 1. 测试核心服务状态")
    print("-" * 40)

    services = {
        "Elasticsearch": "http://localhost:9200/_cluster/health",
        "Milvus": "http://localhost:19530/health",
        "PostgreSQL": "http://localhost:5432",  # 端口检查
        "MinIO": "http://localhost:9000/minio/health/live",
        "Kibana": "http://localhost:5601/api/status"
    }

    service_status = {}

    for service, url in services.items():
        try:
            if service == "PostgreSQL":
                # PostgreSQL端口检查
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', 5432))
                sock.close()
                if result == 0:
                    print(f"✅ {service}: 运行正常")
                    service_status[service] = True
                else:
                    print(f"❌ {service}: 连接失败")
                    service_status[service] = False
            else:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"✅ {service}: 运行正常")
                    service_status[service] = True
                else:
                    print(f"⚠️  {service}: 状态异常 ({response.status_code})")
                    service_status[service] = False
        except Exception as e:
            print(f"❌ {service}: 连接失败 - {str(e)}")
            service_status[service] = False

    # 2. 测试Milvus连接
    print("\n🎯 2. 测试Milvus向量数据库")
    print("-" * 40)

    try:
        from pymilvus import connections, utility
        connections.connect(alias="test", host="localhost", port="19530", timeout=5)
        collections = utility.list_collections(using="test")
        print(f"✅ Milvus连接成功")
        print(f"📋 现有集合: {collections}")
        connections.disconnect("test")
        milvus_ok = True
    except Exception as e:
        print(f"❌ Milvus连接失败: {str(e)}")
        milvus_ok = False

    # 3. 测试Elasticsearch
    print("\n🔍 3. 测试Elasticsearch检索")
    print("-" * 40)

    try:
        response = requests.get("http://localhost:9200/_cluster/health")
        health_data = response.json()
        status = health_data.get("status", "unknown")
        print(f"✅ Elasticsearch集群状态: {status}")

        # 检查索引
        indices_response = requests.get("http://localhost:9200/_cat/indices?format=json")
        indices = indices_response.json()
        if indices:
            print(f"📊 发现 {len(indices)} 个索引")
            for index in indices:
                print(f"   - {index['index']}: {index['docs.count']} 文档")
        else:
            print("ℹ️  暂无索引（这是正常的，数据可能尚未导入）")

        elasticsearch_ok = True
    except Exception as e:
        print(f"❌ Elasticsearch测试失败: {str(e)}")
        elasticsearch_ok = False

    # 4. 测试RAG API（如果可用）
    print("\n🌐 4. 测试RAG API服务")
    print("-" * 40)

    api_ok = False
    try:
        # 首先测试API状态
        status_response = requests.get("http://localhost/api/status", timeout=5)
        if status_response.status_code == 200:
            print("✅ RAG API状态正常")
            api_ok = True

            # 尝试一个简单的查询
            test_query = {
                "question": "什么是医学？",
                "user_id": "validation_test"
            }

            print("\n🧪 测试RAG查询功能...")
            start_time = time.time()
            query_response = requests.post(
                "http://localhost/api/query/sync",
                json=test_query,
                timeout=30
            )
            end_time = time.time()

            if query_response.status_code == 200:
                result = query_response.json()
                print(f"✅ RAG查询成功")
                print(f"⏱️  响应时间: {end_time - start_time:.2f}秒")
                print(f"📝 答案长度: {len(result.get('answer', ''))} 字符")
                print(f"📚 引用文档: {len(result.get('references', []))} 个")
            else:
                print(f"⚠️  RAG查询返回状态码: {query_response.status_code}")
        else:
            print(f"⚠️  API状态检查返回: {status_response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ RAG API服务未启动或无法连接")
    except Exception as e:
        print(f"❌ RAG API测试失败: {str(e)}")

    # 5. 总结
    print("\n📋 5. 验证测试总结")
    print("="*60)

    total_tests = len(service_status) + 2  # 服务 + Milvus + Elasticsearch
    passed_tests = sum(service_status.values()) + (1 if milvus_ok else 0) + (1 if elasticsearch_ok else 0)

    print(f"总测试项目: {total_tests}")
    print(f"通过项目: {passed_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")

    print("\n📊 详细结果:")
    for service, status in service_status.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {service}")

    print(f"{'✅' if milvus_ok else '❌'} Milvus向量数据库")
    print(f"{'✅' if elasticsearch_ok else '❌'} Elasticsearch检索")
    print(f"{'✅' if api_ok else '❌'} RAG API服务")

    # 6. 最终结论
    print("\n🏆 最终验证结论")
    print("="*60)

    if passed_tests == total_tests and api_ok:
        print("🎉 系统验证通过！")
        print("✅ RAG+ReAct智能问答系统已成功部署并正常运行")
        print("✅ 所有核心服务运行正常")
        print("✅ 向量数据库连接正常")
        print("✅ 检索服务运行正常")
        print("✅ API服务运行正常")
        return True
    else:
        print("⚠️  系统验证部分通过")
        print("❌ 部分服务存在问题，需要进一步检查")
        return False

def main():
    """主函数"""
    print("RAG+ReAct智能问答系统 - 最终验证测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    try:
        success = test_system_components()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()