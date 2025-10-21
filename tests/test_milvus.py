#!/usr/bin/env python3
"""
从Milvus检索数据 (向量搜索) - 修正版
"""
import numpy as np
from pymilvus import connections, Collection

# --- 配置 ---
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "medical_vectors_simple" # 确保这和您导入时用的集合名称一致

# 您的搜索查询
QUERY_TEXT = "黏液腺瘤的图像特征" # 试试语义化的问题

# ----------------

def generate_query_embedding(text: str) -> list[float]:
    """
    (模拟) 为查询文本生成向量
    
    !!! 警告 !!!
    这里必须使用和您导入(SimpleImporter)时完全相同的向量生成逻辑。
    """
    print(f"🔄 正在为查询 '{text}' 生成(模拟)向量...")
    # 基于文本内容生成可重复的模拟向量
    np.random.seed(hash(text) % 2**32)
    vector = np.random.randn(768).astype(np.float32)
    # 归一化向量
    vector = vector / np.linalg.norm(vector)
    return vector.tolist()

def search_milvus(query_vector: list[float]):
    """
    连接到Milvus并执行向量搜索
    """
    try:
        # 1. 连接Milvus
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("✅ Milvus连接成功")

        # 2. 获取集合并加载到内存
        collection = Collection(name=COLLECTION_NAME)
        collection.load()
        print(f"✅ 已加载集合 '{COLLECTION_NAME}'")

        # 3. 定义搜索参数
        search_params = {
            "metric_type": "COSINE", 
            "params": {"nprobe": 10},
        }

        # 4. 执行搜索
        response = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=5,
            # 确保这些字段在导入时已存在
            output_fields=["id", "chapter_title", "section_title"] 
        )

        # 5. 解析并打印结果
        hits = response[0]
        
        if not hits:
            print(f"🤷 未找到向量搜索结果。")
            return
            
        print(f"🎉 找到 {len(hits)} 条向量搜索结果:")
        
        # ******************
        # * 这里是修改点  *
        # ******************
        for i, hit in enumerate(hits):
            print(f"\n--- 结果 {i+1} (相似度: {hit.distance}) ---")
            
            # 1. 'id' 是 hit 对象的直接属性
            print(f"  ID (ES文档ID): {hit.id}")
            
            # 2. .get() 方法不接受 'N/A' 这样的默认值
            # 字段内容在 hit.entity 对象中
            entity = hit.entity
            print(f"  章节: {entity.get('chapter_title')}")
            print(f"  小节: {entity.get('section_title')}")
            
        # ******************
        # * 修改结束   *
        # ******************
            
        # 释放集合
        collection.release()

    except Exception as e:
        print(f"❌ Milvus搜索时发生错误: {e}")

if __name__ == "__main__":
    # 1. 为查询生成向量
    query_vec = generate_query_embedding(QUERY_TEXT)
    # 2. 执行搜索
    search_milvus(query_vec)