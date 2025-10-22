"""
RAGAS测试系统配置
"""

import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEST_DATA_DIR = BASE_DIR / "test_data"
REPORTS_DIR = BASE_DIR / "evaluation_reports"

# 测试参数配置
TEST_CONFIG = {
    "num_questions": 150,  # 生成问题数量
    "difficulty_distribution": {
        "basic": 0.6,      # 基础题60%
        "medium": 0.3,     # 中等题30%
        "hard": 0.1        # 难题10%
    },
    "disease_coverage": {
        "lung_cancer": 0.35,           # 肺部实体瘤35%
        "metastatic_tumor": 0.25,      # 转移性肿瘤25%
        "rare_disease": 0.25,          # 少见病25%
        "rose_technique": 0.15         # ROSE技术15%
    },
    "top_k_values": [3, 5, 10],        # 测试的top-k值
    "batch_size": 10,                  # 批处理大小
    "max_workers": 4,                  # 最大工作线程数
}

# 模型配置
MODEL_CONFIG = {
    "question_generation_model": "qwen3-max",  # 问题生成模型
    "embedding_model": "BAAI/bge-large-zh-v1.5",  # 嵌入模型
    "temperature": 0.7,                    # 生成温度
    "max_tokens": 1000,                    # 最大token数
}

# API配置（从环境变量读取）
API_CONFIG = {
    "siliconflow_api_key": os.getenv("SILICONFLOW_API_KEY"),
    "siliconflow_base_url": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    "dashscope_api_key": os.getenv("DASHSCOPE_API_KEY"),
    "dashscope_base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"),
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY"),
    "deepseek_base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
}

# 数据库配置
DB_CONFIG = {
    "elasticsearch_host": os.getenv("ELASTICSEARCH_HOST", "localhost"),
    "elasticsearch_port": int(os.getenv("ELASTICSEARCH_PORT", "9200")),
    "milvus_host": os.getenv("MILVUS_HOST", "localhost"),
    "milvus_port": int(os.getenv("MILVUS_PORT", "19530")),
    "index_name": "medical_documents_fixed",
    "collection_name": "medical_vectors_fixed",
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": BASE_DIR / "logs" / "ragas_test.log",
}