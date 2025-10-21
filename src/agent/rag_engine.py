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
import math

try:
    # 尝试相对导入
    from .react_agent import MedicalReActAgent, create_react_agent
    from .enhanced_react_agent import EnhancedMedicalReActAgent
    from .llm_manager import LLMManager, create_llm_manager
    from .retrieval_manager import MedicalRetrievalManager, create_retrieval_manager
    from .langgraph_rag_engine import LangGraphRAGEngine, create_langgraph_rag_engine
except ImportError:
    # 回退到绝对导入
    try:
        from react_agent import MedicalReActAgent, create_react_agent
        from enhanced_react_agent import EnhancedMedicalReActAgent
        from llm_manager import LLMManager, create_llm_manager
        from retrieval_manager import MedicalRetrievalManager, create_retrieval_manager
        from langgraph_rag_engine import LangGraphRAGEngine, create_langgraph_rag_engine
    except ImportError:
        # 最终回退
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from react_agent import MedicalReActAgent, create_react_agent
        from enhanced_react_agent import EnhancedMedicalReActAgent
        from llm_manager import LLMManager, create_llm_manager
        from retrieval_manager import MedicalRetrievalManager, create_retrieval_manager
        from langgraph_rag_engine import LangGraphRAGEngine, create_langgraph_rag_engine

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

def normalize_relevance_score(raw_score: float, max_score: Optional[float] = None) -> float:
    """
    标准化相关性分数到0-1范围

    Args:
        raw_score: 原始分数（可能超过1.0）
        max_score: 可选的最大分数用于标准化，如果不提供则自动计算

    Returns:
        标准化后的分数（0-1范围）
    """
    if raw_score <= 1.0:
        return raw_score

    # 如果提供了最大分数，使用它进行标准化
    if max_score and max_score > 0:
        return min(raw_score / max_score, 1.0)

    # 自动标准化：使用对数缩放处理极端值
    # 这样可以保持相对顺序，同时压缩极端高分数
    if raw_score <= 10.0:
        return raw_score / 10.0
    elif raw_score <= 100.0:
        return 0.9 + (raw_score - 10.0) / 900.0  # 0.9-1.0范围
    else:
        # 对于超过100的分数，使用对数缩放
        return min(0.95 + math.log10(raw_score) / 100.0, 1.0)

class RAGEngine:
    """RAG问答引擎 - 支持LangGraph和传统架构"""

    def __init__(self, config: Dict[str, Any], use_langgraph: bool = None):
        """初始化RAG引擎"""
        self.config = config
        # 从配置中读取架构设置，如果未提供参数则使用配置中的设置
        if use_langgraph is None:
            arch_config = config.get('architecture', {})
            self.use_langgraph = arch_config.get('use_langgraph', True)
        else:
            self.use_langgraph = use_langgraph
        self.llm_manager = None
        self.retrieval_manager = None
        self.embedding_manager = None
        self.react_agent = None
        self.langgraph_engine = None
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'average_response_time': 0.0,
            'model_usage': {},
            'langgraph_usage': 0,
            'legacy_usage': 0
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
                es_host=retrieval_config.get('es_host', 'elasticsearch'),
                es_port=retrieval_config.get('es_port', 9200),
                milvus_host=retrieval_config.get('milvus_host', 'milvus'),
                milvus_port=retrieval_config.get('milvus_port', 19530),
                embedding_manager=self.embedding_manager
            )
            logger.info("✅ 检索管理器初始化成功")

            # 初始化LangGraph引擎（如果启用）
            if self.use_langgraph:
                try:
                    self.langgraph_engine = create_langgraph_rag_engine(self.config, use_legacy_fallback=True)
                    logger.info("✅ LangGraph RAG引擎初始化成功")

                    # 如果使用LangGraph，就不需要传统的ReAct代理了
                    self.react_agent = None
                    return
                except Exception as e:
                    logger.warning(f"⚠️ LangGraph引擎初始化失败，回退到传统ReAct代理: {e}")
                    self.use_langgraph = False

            # 初始化ReAct代理（使用增强版）
            enhanced_react_config = self.config.get('enhanced_react', {'enabled': True})
            if enhanced_react_config.get('enabled', True):
                # 使用增强版ReAct代理
                self.react_agent = EnhancedMedicalReActAgent(
                    llm_manager=self.llm_manager,
                    retrieval_manager=self.retrieval_manager,
                    embedding_manager=self.embedding_manager
                )
                logger.info("✅ 增强版ReAct代理初始化成功")
            else:
                # 回退到原版ReAct代理
                self.react_agent = create_react_agent(
                    llm_manager=self.llm_manager,
                    retrieval_manager=self.retrieval_manager,
                    embedding_manager=self.embedding_manager
                )
                logger.info("✅ 标准ReAct代理初始化成功")

        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            raise

    async def process_query(self, query: RAGQuery) -> RAGResponse:
        """处理查询"""
        start_time = datetime.now()
        self.stats['total_queries'] += 1

        try:
            logger.info(f"🚀 开始处理查询: {query.question}")

            # 优先使用LangGraph引擎（如果启用）
            if self.use_langgraph and self.langgraph_engine:
                try:
                    logger.info("🚀 使用LangGraph引擎处理查询")
                    self.stats['langgraph_usage'] += 1

                    # 调用LangGraph引擎处理查询
                    langgraph_response = await self.langgraph_engine.process_query(query)

                    logger.info(f"✅ LangGraph引擎处理完成，响应时间: {langgraph_response.response_time:.2f}s")
                    return langgraph_response

                except Exception as e:
                    logger.error(f"❌ LangGraph引擎处理失败: {e}")
                    # 回退到传统ReAct代理
                    logger.info("🔄 回退到传统ReAct代理")

            # 传统ReAct代理处理流程
            # 设置LLM提供者
            if query.search_config and 'model_provider' in query.search_config:
                self.llm_manager.set_active_provider(query.search_config['model_provider'])

            # 执行ReAct代理（处理同步/异步API差异）
            if hasattr(self.react_agent, 'process_query'):
                # 增强版ReAct代理使用同步API
                logger.info(f"🚀 调用增强ReAct代理的process_query方法")
                agent_result = self.react_agent.process_query(
                    question=query.question,
                    user_id=query.user_id,
                    search_config=query.search_config
                )
                logger.info(f"📋 增强ReAct代理返回结果字段: {list(agent_result.keys()) if isinstance(agent_result, dict) else '非字典格式'}")
                logger.info(f"📋 增强ReAct代理返回的retrieved_docs数量: {len(agent_result.get('retrieved_docs', [])) if isinstance(agent_result, dict) else 'N/A'}")
                # 增强版ReAct代理不返回success字段，只返回结果或抛出异常
                # 如果执行到这里，说明处理成功
            else:
                # 原版ReAct代理使用异步API
                agent_result = await self.react_agent.process_question(query.question)
                # 原版ReAct代理返回success字段
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
                    'agent_metadata': agent_result.get('metadata', {}),
                    'architecture': 'legacy_react'
                }
            )

            # 更新统计信息
            self._update_stats(True, response_time)
            self.stats['legacy_usage'] += 1

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
                model_used=self.llm_manager.active_provider if hasattr(self, 'llm_manager') else 'unknown',
                timestamp=datetime.now().isoformat(),
                metadata={'error': str(e)}
            )

    def process_query_sync(self, query: RAGQuery) -> RAGResponse:
        """同步处理查询"""
        start_time = datetime.now()
        self.stats['total_queries'] += 1

        try:
            logger.info(f"🚀 开始同步处理查询: {query.question}")

            # 优先使用LangGraph引擎（如果启用）
            if self.use_langgraph and self.langgraph_engine:
                try:
                    logger.info("🚀 使用LangGraph引擎同步处理查询")
                    self.stats['langgraph_usage'] += 1

                    # 调用LangGraph引擎处理查询（同步版本）
                    langgraph_response = self.langgraph_engine.process_query_sync(query)

                    logger.info(f"✅ LangGraph引擎同步处理完成，响应时间: {langgraph_response.response_time:.2f}s")
                    return langgraph_response

                except Exception as e:
                    logger.error(f"❌ LangGraph引擎同步处理失败: {e}")
                    # 回退到传统ReAct代理
                    logger.info("🔄 回退到传统ReAct代理")

            # 传统ReAct代理处理流程
            # 设置LLM提供者
            if query.search_config and 'model_provider' in query.search_config:
                self.llm_manager.set_active_provider(query.search_config['model_provider'])

            # 执行ReAct代理
            agent_result = self.react_agent.process_question_sync(query.question)

            # 检查代理类型：原版ReAct代理返回success字段，增强版不返回
            if 'success' in agent_result and not agent_result['success']:
                raise Exception(f"ReAct代理处理失败: {agent_result.get('error', '未知错误')}")
            # 增强版ReAct代理不返回success字段，如果执行到这里说明成功

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
                    'agent_metadata': agent_result.get('metadata', {}),
                    'architecture': 'legacy_react'
                }
            )

            # 更新统计信息
            self._update_stats(True, response_time)
            self.stats['legacy_usage'] += 1

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
                model_used=self.llm_manager.active_provider if hasattr(self, 'llm_manager') else 'unknown',
                timestamp=datetime.now().isoformat(),
                metadata={'error': str(e)}
            )

    def _get_retrieved_documents(self, agent_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取检索到的文档"""
        try:
            # 从代理结果中提取检索到的文档
            # 注意：增强ReAct代理使用 'retrieved_documents'，原版使用 'retrieved_docs'
            retrieved_docs = agent_result.get("retrieved_documents", agent_result.get("retrieved_docs", []))

            logger.info(f"📋 从代理结果中提取文档，找到 {len(retrieved_docs)} 个原始文档")
            if retrieved_docs:
                logger.info(f"📋 第一个文档示例: {retrieved_docs[0] if isinstance(retrieved_docs[0], dict) else '非字典格式'}")
            else:
                logger.warning(f"📋 代理结果中没有找到retrieved_docs/retrieved_documents字段，可用字段: {list(agent_result.keys())}")

            # 如果代理结果中没有文档，尝试从其他字段获取
            if not retrieved_docs and "context" in agent_result:
                context_docs = agent_result.get("context", [])
                retrieved_docs = context_docs
                logger.info(f"📋 从context字段提取文档，找到 {len(retrieved_docs)} 个文档")

            # 格式化文档信息
            formatted_docs = []
            raw_scores = []

            # 首先收集所有原始分数
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    raw_score = doc.get("score", 0.0)
                    raw_scores.append(raw_score)

            # 计算最大分数用于标准化
            max_score = max(raw_scores) if raw_scores else 1.0

            # 格式化文档并标准化分数
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    raw_score = doc.get("score", 0.0)
                    # 标准化分数到0-1范围
                    normalized_score = normalize_relevance_score(raw_score, max_score)

                    formatted_doc = {
                        "content": doc.get("content", ""),
                        "chapter_title": doc.get("chapter_title", ""),
                        "section_title": doc.get("section_title", ""),
                        "page_number": doc.get("page_number", 0),
                        "score": normalized_score,  # 使用标准化后的分数
                        "source": doc.get("source", "unknown"),
                        "doc_id": doc.get("doc_id", ""),
                        "search_type": doc.get("search_type", "unknown")
                    }
                    formatted_docs.append(formatted_doc)

            logger.info(f"📄 提取到 {len(formatted_docs)} 个检索文档，分数已标准化")
            return formatted_docs

        except Exception as e:
            logger.error(f"❌ 提取检索文档失败: {e}")
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
            status_info = {
                'status': 'healthy',
                'architecture': 'langgraph' if self.use_langgraph else 'legacy_react',
                'components': {},
                'stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }

            # 获取检索系统状态
            if self.retrieval_manager:
                retrieval_stats = self.retrieval_manager.get_collection_stats()
                status_info['components']['retrieval'] = retrieval_stats

            # 获取LLM提供者状态
            if self.llm_manager:
                llm_status = self.llm_manager.get_provider_status()
                status_info['components']['llm'] = llm_status

            # 获取LangGraph引擎状态（如果启用）
            if self.use_langgraph and self.langgraph_engine:
                langgraph_status = self.langgraph_engine.get_system_status()
                status_info['components']['langgraph_engine'] = langgraph_status.get('components', {})

            # 获取嵌入管理器状态
            if self.embedding_manager:
                status_info['components']['embedding'] = {
                    'type': 'available',
                    'dimension': getattr(self.embedding_manager, 'get_dimension', lambda: 768)()
                }

            # 获取ReAct代理状态（如果存在）
            if self.react_agent:
                status_info['components']['react_agent'] = {
                    'type': 'enhanced' if hasattr(self.react_agent, 'process_query') else 'legacy',
                    'status': 'active'
                }

            return status_info

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

    def switch_architecture(self, use_langgraph: bool):
        """动态切换架构"""
        if self.use_langgraph == use_langgraph:
            logger.info(f"架构已经是 {'LangGraph' if use_langgraph else '传统ReAct'}，无需切换")
            return

        try:
            if use_langgraph:
                # 切换到LangGraph架构
                if not self.langgraph_engine:
                    # 需要重新初始化LangGraph引擎
                    self.langgraph_engine = create_langgraph_rag_engine(self.config, use_legacy_fallback=True)
                self.use_langgraph = True
                logger.info("✅ 切换到LangGraph架构")
            else:
                # 切换到传统架构
                if not self.react_agent:
                    # 需要重新初始化ReAct代理
                    enhanced_react_config = self.config.get('enhanced_react', {'enabled': True})
                    if enhanced_react_config.get('enabled', True):
                        self.react_agent = EnhancedMedicalReActAgent(
                            llm_manager=self.llm_manager,
                            retrieval_manager=self.retrieval_manager,
                            embedding_manager=self.embedding_manager
                        )
                        logger.info("✅ 增强版ReAct代理初始化成功")
                    else:
                        self.react_agent = create_react_agent(
                            llm_manager=self.llm_manager,
                            retrieval_manager=self.retrieval_manager,
                            embedding_manager=self.embedding_manager
                        )
                        logger.info("✅ 标准ReAct代理初始化成功")
                self.use_langgraph = False
                logger.info("✅ 切换到传统ReAct架构")

        except Exception as e:
            logger.error(f"❌ 架构切换失败: {e}")
            raise

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
    """创建默认RAG配置，从环境变量读取API密钥"""
    import os

    # 从环境变量读取API密钥
    deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY', '')
    qwen_api_key = os.environ.get('QWEN_API_KEY', '') or os.environ.get('DASHSCOPE_API_KEY', '')

    # 只有当API密钥存在时才启用相应的提供者
    llm_config = {
        'default_provider': 'mock',
        'deepseek': {
            'api_key': deepseek_api_key or 'your-deepseek-api-key',
            'base_url': 'https://api.deepseek.com',
            'model': 'deepseek-reasoner'
        },
        'qwen': {
            'api_key': qwen_api_key or 'your-qwen-api-key',
            'base_url': 'https://dashscope.aliyuncs.com/api/v1',
            'model': 'qwen-max'
        }
    }

    # 如果环境变量中有有效的API密钥，则设置为默认提供者
    if deepseek_api_key and len(deepseek_api_key) > 10:
        llm_config['default_provider'] = 'deepseek'
    elif qwen_api_key and len(qwen_api_key) > 10:
        llm_config['default_provider'] = 'qwen'

    return {
        'llm': llm_config,
        'retrieval': {
            'es_host': os.environ.get('ELASTICSEARCH_HOST', 'elasticsearch'),  # 使用环境变量，Docker环境默认为服务名
            'es_port': int(os.environ.get('ELASTICSEARCH_PORT', '9200')),
            'milvus_host': os.environ.get('MILVUS_HOST', 'milvus'),  # 使用环境变量，Docker环境默认为服务名
            'milvus_port': int(os.environ.get('MILVUS_PORT', '19530'))
        },
        'embedding': {
            'type': 'jina'  # 使用可用的jina模型
        },
        'architecture': {
            'use_langgraph': True,  # 默认启用LangGraph架构
            'max_iterations': 3,    # LangGraph最大迭代次数
            'fallback_to_legacy': True  # 失败时回退到传统架构
        },
        'langgraph_agent': {
            'max_iterations': 3,
            'enable_multi_round_search': True,
            'enhanced_title_weighting': True
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