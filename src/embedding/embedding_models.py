#!/usr/bin/env python3
"""
嵌入模型配置和实现
支持多种嵌入模型：Jina、千问3等
"""

import os
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseEmbeddingModel(ABC):
    """嵌入模型基类"""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        编码文本为向量

        Args:
            texts: 输入文本或文本列表

        Returns:
            向量数组
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度"""
        pass

class JinaEmbeddingModel(BaseEmbeddingModel):
    """Jina嵌入模型实现"""

    #  model_name: str = "jina-embeddings-v2-base-zh"
    def __init__(self, model_name: str = "jina-embeddings-v2-base-zh", api_key: str = None):
        """
        初始化Jina嵌入模型

        Args:
            model_name: 模型名称
            api_key: API密钥
        """
        from dotenv import load_dotenv
        load_dotenv()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self._dimension = 768

        if not self.api_key:
            logger.warning("⚠️  未设置JINA_API_KEY，将使用模拟向量")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]

        if not self.api_key:
            # 使用模拟向量
            logger.info("🔄 使用模拟向量生成")
            return self._generate_mock_embeddings(texts)

        try:
            # 调用Jina API
            url = "https://api.jina.ai/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model_name,
                "input": texts
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            embeddings = []

            for item in result.get("data", []):
                embeddings.append(item.get("embedding", []))

            return np.array(embeddings)

        except Exception as e:
            logger.error(f"❌ Jina API调用失败: {e}，使用模拟向量")
            return self._generate_mock_embeddings(texts)

    def _generate_mock_embeddings(self, texts: List[str]) -> np.ndarray:
        """生成模拟嵌入向量（用于测试）"""
        np.random.seed(42)  # 确保可重复性
        return np.random.random((len(texts), self._dimension)).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension

class Qwen3EmbeddingModel(BaseEmbeddingModel):
    """千问3嵌入模型实现"""

    def __init__(self, model_name: str = "qwen3-0.6b-embedding", api_key: str = None):
        """
        初始化千问3嵌入模型

        Args:
            model_name: 模型名称
            api_key: API密钥
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("QWEN3_API_KEY")
        self._dimension = 768

        if not self.api_key:
            logger.warning("⚠️  未设置QWEN3_API_KEY，将使用模拟向量")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]

        if not self.api_key:
            # 使用模拟向量
            logger.info("🔄 使用模拟向量生成")
            return self._generate_mock_embeddings(texts)

        try:
            # 调用千问3 API
            url = "https://dashscope.aliyuncs.com/api/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model_name,
                "input": {
                    "texts": texts
                }
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            embeddings = result.get("output", {}).get("embeddings", [])

            return np.array(embeddings)

        except Exception as e:
            logger.error(f"❌ 千问3 API调用失败: {e}，使用模拟向量")
            return self._generate_mock_embeddings(texts)

    def _generate_mock_embeddings(self, texts: List[str]) -> np.ndarray:
        """生成模拟嵌入向量（用于测试）"""
        np.random.seed(42)  # 确保可重复性
        return np.random.random((len(texts), self._dimension)).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension

class EmbeddingModelFactory:
    """嵌入模型工厂类"""

    _models = {
        "jina": JinaEmbeddingModel,
        "jina-0.5b": JinaEmbeddingModel,
        "qwen3": Qwen3EmbeddingModel,
        "qwen3-0.6b": Qwen3EmbeddingModel
    }

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseEmbeddingModel:
        """
        创建嵌入模型实例

        Args:
            model_type: 模型类型
            **kwargs: 其他参数

        Returns:
            嵌入模型实例
        """
        if model_type not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"不支持的模型类型: {model_type}。可用模型: {available_models}")

        model_class = cls._models[model_type]
        return model_class(**kwargs)

    @classmethod
    def list_available_models(cls) -> List[str]:
        """返回可用模型列表"""
        return list(cls._models.keys())

class EmbeddingManager:
    """嵌入管理器"""

    def __init__(self, model_type: str = "jina", **model_kwargs):
        """
        初始化嵌入管理器

        Args:
            model_type: 模型类型
            **model_kwargs: 模型参数
        """
        self.model = EmbeddingModelFactory.create_model(model_type, **model_kwargs)
        logger.info(f"✅ 初始化嵌入模型: {model_type}")

    def encode_texts(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        批量编码文本

        Args:
            texts: 文本或文本列表
            batch_size: 批处理大小

        Returns:
            嵌入向量数组
        """
        if isinstance(texts, str):
            texts = [texts]

        logger.info(f"🔄 开始编码 {len(texts)} 个文本")

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.append(batch_embeddings)

        result = np.vstack(embeddings)
        logger.info(f"✅ 编码完成，获得 {result.shape} 的向量数组")
        return result

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.model.dimension

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算向量相似度（余弦相似度）

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            相似度分数
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 全局嵌入管理器实例
_embedding_manager = None

def get_embedding_manager(model_type: str = "jina", **model_kwargs) -> EmbeddingManager:
    """
    获取全局嵌入管理器实例

    Args:
        model_type: 模型类型
        **model_kwargs: 模型参数

    Returns:
        嵌入管理器实例
    """
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(model_type, **model_kwargs)
    return _embedding_manager

def main():
    """测试函数"""
    # 测试嵌入模型
    texts = [
        "肺部恶性肿瘤的细胞学特征包括细胞体积增大和核质比增加。",
        "ROSE（快速现场评价）是一种实时细胞学判读技术。",
        "腺癌是肺部最常见的恶性肿瘤类型之一。"
    ]

    # 使用Jina模型
    jina_manager = get_embedding_manager("jina")
    jina_embeddings = jina_manager.encode_texts(texts)
    print(f"Jina嵌入向量形状: {jina_embeddings.shape}")

    # 使用千问3模型
    qwen3_manager = get_embedding_manager("qwen3")
    qwen3_embeddings = qwen3_manager.encode_texts(texts)
    print(f"千问3嵌入向量形状: {qwen3_embeddings.shape}")

    # 计算相似度
    sim = jina_manager.similarity(jina_embeddings[0], jina_embeddings[1])
    print(f"文本1和文本2的相似度: {sim:.4f}")

if __name__ == "__main__":
    main()