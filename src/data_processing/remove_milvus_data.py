import os
import logging
from pymilvus import utility, connections

# --- 配置 ---
# 从环境变量获取连接信息，如果未设置，则使用默认的'localhost'和'19530'
MILVUS_HOST = os.environ.get('MILVUS_HOST', 'localhost')
MILVUS_PORT = os.environ.get('MILVUS_PORT', '19530')
CONNECTION_ALIAS = "deleter_connection"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def delete_all_collections():
    """
    连接到Milvus并删除其中的所有集合。
    这是一个危险操作，请谨慎使用。
    """
    logging.info("--- 开始删除所有Milvus集合 ---")
    
    try:
        # 1. 连接到Milvus
        logging.info(f"正在连接到 Milvus ({MILVUS_HOST}:{MILVUS_PORT})...")
        connections.connect(alias=CONNECTION_ALIAS, host=MILVUS_HOST, port=str(MILVUS_PORT))
        logging.info("✅ Milvus 连接成功。")

        # 2. 获取所有集合的列表
        collections_list = utility.list_collections(using=CONNECTION_ALIAS)
        
        if not collections_list:
            logging.info("✅ 实例中没有任何集合，无需删除。")
            return

        logging.info(f"发现 {len(collections_list)} 个集合: {collections_list}")

        # 3. 遍历并删除每个集合
        for collection_name in collections_list:
            try:
                logging.warning(f"准备删除集合: '{collection_name}'...")
                utility.drop_collection(collection_name, using=CONNECTION_ALIAS)
                logging.info(f"✅ 成功删除集合: '{collection_name}'。")
            except Exception as e:
                logging.error(f"❌ 删除集合 '{collection_name}' 时发生错误: {e}")

        # 4. 再次验证
        remaining_collections = utility.list_collections(using=CONNECTION_ALIAS)
        if not remaining_collections:
            logging.info("🎉 所有集合已成功删除！")
        else:
            logging.warning(f"⚠️ 操作完成后仍有残留集合: {remaining_collections}")

    except Exception as e:
        logging.error(f"❌ 在操作过程中发生严重错误: {e}")
    finally:
        # 5. 断开连接
        if CONNECTION_ALIAS in connections.list_connections():
            connections.disconnect(CONNECTION_ALIAS)
            logging.info("--- 操作完成，已断开 Milvus 连接 ---")

if __name__ == "__main__":
    # 在执行危险操作前，增加一个确认环节
    print("🚨 警告：此脚本将删除Milvus实例中的所有集合！此操作不可恢复。")
    print(f"将要连接的目标实例是: {MILVUS_HOST}:{MILVUS_PORT}")
    
    confirm = input("请输入 'yes' 以确认执行删除操作: ")
    
    if confirm.lower() == 'yes':
        delete_all_collections()
    else:
        print("操作已取消。")
# ```

# ### 如何使用

# 1.  **保存脚本**：将上面的代码保存为一个 Python 文件，例如 `delete_all_milvus_collections.py`。
# 2.  **运行脚本**：在你的终端中，进入 Conda 环境 (`zhenlikeji`)，然后运行这个文件：
#     ```bash
#     python delete_all_milvus_collections.py
#     ```
# 3.  **确认操作**：脚本会提示你输入 `yes` 来确认删除。这是一个安全措施，防止你意外清空数据库。
#     ```
#     🚨 警告：此脚本将删除Milvus实例中的所有集合！此操作不可恢复。
#     将要连接的目标实例是: localhost:19530
#     请输入 'yes' 以确认执行删除操作: yes
    
