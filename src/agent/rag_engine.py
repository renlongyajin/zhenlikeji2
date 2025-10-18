#!/usr/bin/env python3
"""
RAG问答引擎
集成检索、推理和生成的完整问答系统
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json

try:
    # 尝试相对导入
    from .react_agent import MedicalReActAgent, create_react_agent
    from .llm_manager import LLMManager, create_llm_manager
    from .retrieval_manager import MedicalRetrievalManager, create_retrieval_manager
except ImportError:
    # 回退到绝对导入
    try:
        from react_agent import MedicalReActAgent, create_react_agent
        from llm_manager import LLMManager, create_llm_manager
        from retrieval_manager import MedicalRetrievalManager, create_retrieval_manager
    except ImportError:
        # 最终回退
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from react_agent import MedicalReActAgent, create_react_agent
        from llm_manager import LLMManager, create_llm_manager
        from retrieval_manager import MedicalRetrievalManager, create_retrieval_manager

try:
    from embedding.embedding_models import get_embedding_manager
except ImportError:
    # Fallback for when running as script
    try:
        from ..embedding.embedding_models import get_embedding_manager
    except ImportError:
        # 最终回退 - 使用模拟嵌入管理器
        def get_embedding_manager(model_type="mock", **kwargs):
            """模拟嵌入管理器工厂函数"""
            import numpy as np
            class MockEmbeddingManager:
                def encode_texts(self, texts, **kwargs):
                    return np.random.random((len(texts), 768))
                def get_dimension(self):
                    return 768
            return MockEmbeddingManager()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RAGQuery:
    """RAG查询数据结构"""
    question: str
    query_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    search_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RAGResponse:
    """RAG响应数据结构"""
    query_id: str
    question: str
    answer: str
    confidence: float
    retrieved_documents: List[Dict[str, Any]]
    reasoning_steps: List[Dict[str, Any]]
    search_queries: List[str]
    response_time: float
    model_used: str
    timestamp: str
    metadata: Dict[str, Any]

class RAGEngine:
    """RAG问答引擎"""

    def __init__(self, config: Dict[str, Any]):
        """初始化RAG引擎"""
        self.config = config
        self.llm_manager = None
        self.retrieval_manager = None
        self.embedding_manager = None
        self.react_agent = None
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'average_response_time': 0.0,
            'model_usage': {}
        }

        self._initialize_components()

    def _initialize_components(self):
        """初始化各个组件"""
        try:
            # 初始化嵌入管理器
            embedding_config = self.config.get('embedding', {})
            embedding_type = embedding_config.get('type', 'mock')
            self.embedding_manager = get_embedding_manager(embedding_type)
            logger.info(f"✅ 嵌入管理器初始化成功: {embedding_type}")

            # 初始化LLM管理器
            llm_config = self.config.get('llm', {})
            self.llm_manager = create_llm_manager(llm_config)
            logger.info("✅ LLM管理器初始化成功")

            # 初始化检索管理器
            retrieval_config = self.config.get('retrieval', {})
            self.retrieval_manager = create_retrieval_manager(
                es_host=retrieval_config.get('es_host', 'localhost'),
                es_port=retrieval_config.get('es_port', 9200),
                milvus_host=retrieval_config.get('milvus_host', 'localhost'),
                milvus_port=retrieval_config.get('milvus_port', 19530),
                embedding_manager=self.embedding_manager
            )
            logger.info("✅ 检索管理器初始化成功")

            # 初始化ReAct代理
            self.react_agent = create_react_agent(
                llm_manager=self.llm_manager,
                retrieval_manager=self.retrieval_manager,
                embedding_manager=self.embedding_manager
            )
            logger.info("✅ ReAct代理初始化成功")

        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            raise

    async def process_query(self, query: RAGQuery) -> RAGResponse:
        """处理查询"""
        start_time = datetime.now()
        self.stats['total_queries'] += 1

        try:
            logger.info(f"🚀 开始处理查询: {query.question}")

            # 设置LLM提供者
            if query.search_config and 'model_provider' in query.search_config:
                self.llm_manager.set_active_provider(query.search_config['model_provider'])

            # 执行ReAct代理
            agent_result = await self.react_agent.process_question(query.question)

            if not agent_result['success']:
                raise Exception(f"ReAct代理处理失败: {agent_result.get('error', '未知错误')}")

            # 获取检索到的文档
            retrieved_docs = self._get_retrieved_documents(agent_result)

            # 构建响应
            response_time = (datetime.now() - start_time).total_seconds()

            rag_response = RAGResponse(
                query_id=query.query_id,
                question=query.question,
                answer=agent_result['answer'],
                confidence=agent_result['confidence'],
                retrieved_documents=retrieved_docs,
                reasoning_steps=agent_result.get('reasoning_steps', []),
                search_queries=agent_result.get('search_queries', [query.question]),
                response_time=response_time,
                model_used=self.llm_manager.active_provider,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'session_id': query.session_id,
                    'user_id': query.user_id,
                    'search_config': query.search_config,
                    'agent_metadata': agent_result.get('metadata', {})
                }
            )

            # 更新统计信息
            self._update_stats(True, response_time)

            logger.info(f"✅ 查询处理完成，响应时间: {response_time:.2f}s，置信度: {rag_response.confidence}")
            return rag_response

        except Exception as e:
            logger.error(f"❌ 查询处理失败: {e}")

            # 更新统计信息
            self._update_stats(False, (datetime.now() - start_time).total_seconds())

            # 返回错误响应
            return RAGResponse(
                query_id=query.query_id,
                question=query.question,
                answer=f"抱歉，处理您的查询时出现错误: {str(e)}",
                confidence=0.0,
                retrieved_documents=[],
                reasoning_steps=[],
                search_queries=[query.question],
                response_time=(datetime.now() - start_time).total_seconds(),
                model_used=self.llm_manager.active_provider,
                timestamp=datetime.now().isoformat(),
                metadata={'error': str(e)}
            )

    def process_query_sync(self, query: RAGQuery) -> RAGResponse:
        """同步处理查询"""
        start_time = datetime.now()
        self.stats['total_queries'] += 1

        try:
            logger.info(f"🚀 开始同步处理查询: {query.question}")

            # 设置LLM提供者
            if query.search_config and 'model_provider' in query.search_config:
                self.llm_manager.set_active_provider(query.search_config['model_provider'])

            # 执行ReAct代理
            agent_result = self.react_agent.process_question_sync(query.question)

            if not agent_result['success']:
                raise Exception(f"ReAct代理处理失败: {agent_result.get('error', '未知错误')}")

            # 获取检索到的文档
            retrieved_docs = self._get_retrieved_documents(agent_result)

            # 构建响应
            response_time = (datetime.now() - start_time).total_seconds()

            rag_response = RAGResponse(
                query_id=query.query_id,
                question=query.question,
                answer=agent_result['answer'],
                confidence=agent_result['confidence'],
                retrieved_documents=retrieved_docs,
                reasoning_steps=agent_result.get('reasoning_steps', []),
                search_queries=agent_result.get('search_queries', [query.question]),
                response_time=response_time,
                model_used=self.llm_manager.active_provider,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'session_id': query.session_id,
                    'user_id': query.user_id,
                    'search_config': query.search_config,
                    'agent_metadata': agent_result.get('metadata', {})
                }
            )

            # 更新统计信息
            self._update_stats(True, response_time)

            logger.info(f"✅ 同步查询处理完成，响应时间: {response_time:.2f}s")
            return rag_response

        except Exception as e:
            logger.error(f"❌ 同步查询处理失败: {e}")

            # 更新统计信息
            self._update_stats(False, (datetime.now() - start_time).total_seconds())

            # 返回错误响应
            return RAGResponse(
                query_id=query.query_id,
                question=query.question,
                answer=f"抱歉，处理您的查询时出现错误: {str(e)}",
                confidence=0.0,
                retrieved_documents=[],
                reasoning_steps=[],
                search_queries=[query.question],
                response_time=(datetime.now() - start_time).total_seconds(),
                model_used=self.llm_manager.active_provider,
                timestamp=datetime.now().isoformat(),
                metadata={'error': str(e)}
            )

    def _get_retrieved_documents(self, agent_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取检索到的文档"""
        # 这里可以扩展以从代理结果中提取实际检索到的文档
        # 目前返回空列表，实际实现中应该从代理的工具调用结果中提取
        return []

    def _update_stats(self, success: bool, response_time: float):
        """更新统计信息"""
        if success:
            self.stats['successful_queries'] += 1

        # 更新平均响应时间
        total_queries = self.stats['total_queries']
        current_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = (current_avg * (total_queries - 1) + response_time) / total_queries

        # 更新模型使用统计
        model = self.llm_manager.active_provider
        self.stats['model_usage'][model] = self.stats['model_usage'].get(model, 0) + 1

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            # 获取检索系统状态
            retrieval_stats = self.retrieval_manager.get_collection_stats()

            # 获取LLM提供者状态
            llm_status = self.llm_manager.get_provider_status()

            return {
                'status': 'healthy',
                'components': {
                    'retrieval': retrieval_stats,
                    'llm': llm_status,
                    'embedding': {'type': 'available'}
                },
                'stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ 获取系统状态失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """获取查询建议"""
        # 这里可以实现查询建议功能
        # 基于历史查询、医学术语库等
        suggestions = []

        # 简单的基于关键词的建议
        medical_keywords = [
            "肺部恶性肿瘤", "ROSE细胞学", "腺癌特征", "细胞核增大",
            "快速现场评价", "细胞学诊断", "病理分析", "癌症分期"
        ]

        for keyword in medical_keywords:
            if partial_query.lower() in keyword.lower():
                suggestions.append(keyword)
            if len(suggestions) >= max_suggestions:
                break

        return suggestions

    def create_query(self, question: str, **kwargs) -> RAGQuery:
        """创建查询对象"""
        import uuid

        return RAGQuery(
            question=question,
            query_id=str(uuid.uuid4()),
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id'),
            search_config=kwargs.get('search_config'),
            metadata=kwargs.get('metadata', {})
        )

# 创建默认配置
def create_default_rag_config() -> Dict[str, Any]:
    """创建默认RAG配置"""
    return {
        'llm': {
            'default_provider': 'mock',
            'deepseek': {
                'api_key': 'your-deepseek-api-key',
                'base_url': 'https://api.deepseek.com',
                'model': 'deepseek-reasoner'
            },
            'qwen': {
                'api_key': 'your-qwen-api-key',
                'base_url': 'https://dashscope.aliyuncs.com/api/v1',
                'model': 'qwen-max'
            }
        },
        'retrieval': {
            'es_host': 'localhost',
            'es_port': 9200,
            'milvus_host': 'localhost',
            'milvus_port': 19530
        },
        'embedding': {
            'type': 'jina'  # 使用可用的jina模型
        }
    }

# 创建RAG引擎实例
def create_rag_engine(config: Optional[Dict[str, Any]] = None) -> RAGEngine:
    """创建RAG引擎"""
    if config is None:
        config = create_default_rag_config()

    return RAGEngine(config)

# 快速测试函数
def test_rag_engine():
    """测试RAG引擎"""
    logger.info("🧪 开始测试RAG引擎...")

    try:
        # 创建RAG引擎
        engine = create_rag_engine()

        # 创建测试查询
        test_query = engine.create_query(
            "什么是肺部恶性肿瘤的ROSE细胞学特征？",
            user_id="test_user",
            session_id="test_session"
        )

        # 执行查询
        response = engine.process_query_sync(test_query)

        logger.info("✅ RAG引擎测试完成")
        logger.info(f"问题: {response.question}")
        logger.info(f"答案: {response.answer[:200]}...")
        logger.info(f"置信度: {response.confidence}")
        logger.info(f"响应时间: {response.response_time:.2f}s")

        return True

    except Exception as e:
        logger.error(f"❌ RAG引擎测试失败: {e}")
        return False

if __name__ == "__main__":
    test_rag_engine()