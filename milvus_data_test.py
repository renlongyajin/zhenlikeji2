import os
import sys
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 脚本需要放在项目的根目录（例如 zhenlikeji2/）下运行
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    # 【关键】从pymilvus导入utility模块
    from pymilvus import connections, Collection, utility
    from src.embedding.embedding_models import JinaEmbeddingModel
except ImportError as e:
    logger.error(f"无法导入所需模块，请确保已在项目根目录运行，并安装了依赖: {e}")
    sys.exit(1)

# --- 配置 ---
MILVUS_HOST = os.environ.get('MILVUS_HOST', 'localhost')
MILVUS_PORT = os.environ.get('MILVUS_PORT', '19530')
MILVUS_COLLECTION = "medical_vectors_fixed"
CONNECTION_ALIAS = "milvus_test"

# 搜索参数
QUERY_TEXT = "腺癌"
TOP_K = 3

def test_milvus_search():
    """
    连接到Milvus，将文本查询转换为向量，并执行向量搜索。
    """
    logger.info("--- 开始Milvus向量检索测试 ---")

    try:
        # 1. 初始化嵌入模型
        logger.info("正在初始化Jina嵌入模型...")
        embedding_model = JinaEmbeddingModel()
        logger.info("✅ 嵌入模型初始化成功。")

        # 2. 连接到Milvus
        logger.info(f"正在连接到Milvus ({MILVUS_HOST}:{MILVUS_PORT})...")
        if CONNECTION_ALIAS not in connections.list_connections():
            connections.connect(alias=CONNECTION_ALIAS, host=MILVUS_HOST, port=str(MILVUS_PORT))
        logger.info("✅ Milvus连接成功。")

        # 3. 获取集合并加载
        logger.info(f"正在获取Milvus集合: '{MILVUS_COLLECTION}'...")
        collection = Collection(name=MILVUS_COLLECTION, using=CONNECTION_ALIAS)
        
        logger.info("正在将集合加载到内存中...")
        collection.load()
        logger.info("✅ 集合加载指令已发送。")

        # --- 【最终修复】 ---
        # 使用官方推荐的utility.wait_for_loading_complete()来替代固定的time.sleep()
        # 这将确保我们等到集合100%加载完毕再继续，无论需要多长时间
        logger.info("正在等待集合完全加载...")
        utility.wait_for_loading_complete(
            collection_name=MILVUS_COLLECTION,
            using=CONNECTION_ALIAS,
            timeout=120  # 设置一个合理的超时时间，例如120秒
        )
        logger.info("✅ 集合已确认100%加载完成，准备搜索。")
        # --- 【修复结束】 ---

        # 4. 生成查询向量
        logger.info(f"正在为查询 '{QUERY_TEXT}' 生成向量...")
        query_vector = embedding_model.encode(QUERY_TEXT)[0]
        logger.info(f"✅ 查询向量生成成功 (维度: {len(query_vector)})。")

        # 5. 执行向量搜索
        logger.info(f"--- 正在执行向量搜索 (Top K = {TOP_K}) ---")
        search_params = { "metric_type": "L2", "params": {"nprobe": 10} }
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=TOP_K,
            output_fields=["id", "chapter_title", "section_title", "page_number"]
        )
        
        logger.info("--- 搜索结果 ---")
        if not results or not results[0]:
            logger.warning("⚠️ 搜索没有返回任何结果。")
            return
        print("res:",results)
        

        

    except Exception as e:
        logger.error(f"❌ 脚本执行过程中发生错误: {e}", exc_info=True)
    finally:
        if CONNECTION_ALIAS in connections.list_connections():
            connections.disconnect(CONNECTION_ALIAS)
            logger.info("\n--- 测试完成，已断开Milvus连接 ---")

if __name__ == "__main__":
    test_milvus_search()
# ```

# ### 下一步操作

# 1.  用上面这个最终修复版的代码，覆盖你之前的测试脚本 `milvus_data_test.py`。
# 2.  再次运行脚本：
#     ```bash
#     python milvus_data_test.py
    
