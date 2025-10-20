#!/usr/bin/env python3
"""
LangGraph-based RAG引擎
集成新的LangGraph ReAct Agent，同时保持现有API兼容性
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json

# 尝试导入新的LangGraph Agent
try:
    from .langgraph_react_agent import LangGraphReActAgent, create_langgraph_react_agent
    from .enhanced_retrieval_manager import EnhancedMedicalRetrievalManager, create_enhanced_retrieval_manager
    from .llm_manager import LLMManager, create_llm_manager
except ImportError:
    try:
        from langgraph_react_agent import LangGraphReActAgent, create_langgraph_react_agent
        from enhanced_retrieval_manager import EnhancedMedicalRetrievalManager, create_enhanced_retrieval_manager
        from llm_manager import LLMManager, create_llm_manager
    except ImportError:
        # 最终回退
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from langgraph_react_agent import LangGraphReActAgent, create_langgraph_react_agent
        from enhanced_retrieval_manager import EnhancedMedicalRetrievalManager, create_enhanced_retrieval_manager
        from llm_manager import LLMManager, create_llm_manager

# 尝试导入旧的组件作为回退
try:
    from .rag_engine import RAGEngine, create_rag_engine, RAGQuery, RAGResponse, create_default_rag_config
except ImportError:
    try:
        from rag_engine import RAGEngine, create_rag_engine, RAGQuery, RAGResponse, create_default_rag_config
    except ImportError:
        # 如果旧的RAG引擎不可用，创建兼容的数据结构
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LangGraphRAGEngine:
    """基于LangGraph的RAG引擎"""

    def __init__(self, config: Dict[str, Any], use_legacy_fallback: bool = True):
        """初始化LangGraph RAG引擎"""
        self.config = config
        self.use_legacy_fallback = use_legacy_fallback
        self.llm_manager = None
        self.enhanced_retrieval_manager = None
        self.embedding_manager = None
        self.langgraph_agent = None
        self.legacy_engine = None
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'average_response_time': 0.0,
            'model_usage': {},
            'fallback_count': 0
        }

        self._initialize_components()

    def _initialize_components(self):
        """初始化各个组件"""
        try:
            # 初始化嵌入管理器
            embedding_config = self.config.get('embedding', {})
            embedding_type = embedding_config.get('type', 'jina')

            # 尝试导入嵌入管理器
            try:
                from embedding.embedding_models import get_embedding_manager
                self.embedding_manager = get_embedding_manager(embedding_type)
                logger.info(f"✅ 嵌入管理器初始化成功: {embedding_type}")
            except ImportError:
                logger.warning("⚠️ 嵌入管理器导入失败，使用模拟模式")
                self.embedding_manager = None

            # 初始化LLM管理器
            llm_config = self.config.get('llm', {})
            self.llm_manager = create_llm_manager(llm_config)
            logger.info("✅ LLM管理器初始化成功")

            # 初始化增强版检索管理器
            retrieval_config = self.config.get('retrieval', {})
            self.enhanced_retrieval_manager = create_enhanced_retrieval_manager(
                es_host=retrieval_config.get('es_host', 'elasticsearch'),
                es_port=retrieval_config.get('es_port', 9200),
                milvus_host=retrieval_config.get('milvus_host', 'milvus'),
                milvus_port=retrieval_config.get('milvus_port', 19530),
                embedding_manager=self.embedding_manager
            )
            logger.info("✅ 增强版检索管理器初始化成功")

            # 初始化LangGraph Agent
            agent_config = self.config.get('langgraph_agent', {})
            max_iterations = agent_config.get('max_iterations', 3)

            self.langgraph_agent = create_langgraph_react_agent(
                llm_manager=self.llm_manager,
                retrieval_manager=self.enhanced_retrieval_manager,
                embedding_manager=self.embedding_manager,
                max_iterations=max_iterations
            )
            logger.info("✅ LangGraph Agent初始化成功")

            # 初始化旧版引擎作为回退（如果启用）
            if self.use_legacy_fallback:
                try:
                    self.legacy_engine = create_rag_engine(self.config)
                    logger.info("✅ 旧版RAG引擎初始化成功（作为回退）")
                except Exception as e:
                    logger.warning(f"⚠️ 旧版RAG引擎初始化失败: {e}")
                    self.legacy_engine = None

        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            if self.use_legacy_fallback and hasattr(self, 'legacy_engine'):
                logger.info("尝试使用旧版引擎作为回退")
                # 降级到旧版引擎
                self.langgraph_agent = None
            else:
                raise

    async def process_query(self, query: RAGQuery) -> RAGResponse:
        """处理查询（异步接口）"""
        start_time = datetime.now()
        self.stats['total_queries'] += 1

        try:
            logger.info(f"🚀 开始处理LangGraph查询: {query.question}")

            # 优先使用LangGraph Agent（如果可用）
            if self.langgraph_agent:
                try:
                    # 设置LLM提供者
                    if query.search_config and 'model_provider' in query.search_config:
                        self.llm_manager.set_active_provider(query.search_config['model_provider'])

                    # 使用LangGraph Agent处理查询
                    agent_result = self.langgraph_agent.process_query(
                        question=query.question,
                        user_id=query.user_id,
                        search_config=query.search_config
                    )

                    # 转换为标准RAG响应格式
                    response_time = (datetime.now() - start_time).total_seconds()

                    rag_response = RAGResponse(
                        query_id=agent_result.get("query_id", query.query_id),
                        question=agent_result.get("question", query.question),
                        answer=agent_result.get("answer", {}).get('content', '') if isinstance(agent_result.get("answer"), dict) else str(agent_result.get("answer", "")),
                        confidence=agent_result.get("confidence", 0.0),
                        retrieved_documents=agent_result.get("retrieved_documents", []),
                        reasoning_steps=agent_result.get("reasoning_steps", []),
                        search_queries=agent_result.get("search_queries", [query.question]),
                        response_time=response_time,
                        model_used=agent_result.get("model_used", "langgraph_react"),
                        timestamp=agent_result.get("timestamp", datetime.now().isoformat()),
                        metadata={
                            'session_id': query.session_id,
                            'user_id': query.user_id,
                            'search_config': query.search_config,
                            'agent_metadata': agent_result.get("metadata", {}),
                            'architecture': 'langgraph'
                        }
                    )

                    self.stats['successful_queries'] += 1
                    logger.info(f"✅ LangGraph查询处理完成，响应时间: {response_time:.2f}s")
                    return rag_response

                except Exception as e:
                    logger.error(f"❌ LangGraph Agent处理失败: {e}")
                    self.stats['fallback_count'] += 1

                    # 回退到旧版引擎
                    if self.legacy_engine:
                        logger.info("回退到旧版RAG引擎")
                        return await self.legacy_engine.process_query(query)
                    else:
                        raise

            else:
                # LangGraph Agent不可用，直接使用旧版引擎
                if self.legacy_engine:
                    logger.info("使用旧版RAG引擎处理查询")
                    return await self.legacy_engine.process_query(query)
                else:
                    raise Exception("没有可用的RAG引擎")

        except Exception as e:
            logger.error(f"❌ 查询处理失败: {e}")
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
                model_used="error",
                timestamp=datetime.now().isoformat(),
                metadata={'error': str(e), 'architecture': 'error'}
            )

    def process_query_sync(self, query: RAGQuery) -> RAGResponse:
        """同步处理查询"""
        return asyncio.run(self.process_query(query))

    def _update_stats(self, success: bool, response_time: float):
        """更新统计信息"""
        if success:
            self.stats['successful_queries'] += 1

        # 更新平均响应时间
        total_queries = self.stats['total_queries']
        current_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = (current_avg * (total_queries - 1) + response_time) / total_queries

        # 更新模型使用统计
        model = "langgraph_react" if success else "error"
        self.stats['model_usage'][model] = self.stats['model_usage'].get(model, 0) + 1

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            status_info = {
                'status': 'healthy',
                'architecture': 'langgraph',
                'components': {},
                'stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }

            # 获取增强版检索系统状态
            if self.enhanced_retrieval_manager:
                retrieval_stats = self.enhanced_retrieval_manager.get_collection_stats()
                status_info['components']['enhanced_retrieval'] = retrieval_stats

            # 获取LLM提供者状态
            if self.llm_manager:
                llm_status = self.llm_manager.get_provider_status()
                status_info['components']['llm'] = llm_status

            # 获取LangGraph Agent状态
            if self.langgraph_agent:
                status_info['components']['langgraph_agent'] = {
                    'status': 'active',
                    'max_iterations': self.langgraph_agent.max_iterations,
                    'supports_tools': hasattr(self.llm_manager, 'bind_tools') if self.llm_manager else False
                }

            # 获取旧版引擎状态（如果可用）
            if self.legacy_engine and hasattr(self.legacy_engine, 'get_system_status'):
                legacy_status = self.legacy_engine.get_system_status()
                status_info['components']['legacy_engine'] = legacy_status.get('components', {})

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
        # 优先使用增强版检索系统的建议
        if self.enhanced_retrieval_manager:
            # 这里可以实现基于增强版检索的建议逻辑
            suggestions = []

            # 基于医学术语库的建议
            medical_terms = [
                "肺部恶性肿瘤", "ROSE细胞学", "腺癌特征", "细胞核增大",
                "快速现场评价", "细胞学诊断", "病理分析", "癌症分期",
                "黏液腺癌", "鳞状细胞癌", "小细胞癌", "大细胞癌"
            ]

            for term in medical_terms:
                if partial_query.lower() in term.lower():
                    suggestions.append(term)
                if len(suggestions) >= max_suggestions:
                    break

            return suggestions

        # 回退到基础建议
        return []

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

    def switch_architecture(self, use_langgraph: bool):
        """切换架构（用于测试和对比）"""
        if use_langgraph and self.langgraph_agent:
            logger.info("切换到LangGraph架构")
            # LangGraph Agent已经是默认的
        elif not use_langgraph and self.legacy_engine:
            logger.info("切换到传统架构")
            # 在process_query中会优先使用旧版引擎
            self.langgraph_agent = None
        else:
            logger.warning(f"无法切换到{'LangGraph' if use_langgraph else '传统'}架构：组件不可用")

# 创建LangGraph RAG引擎实例的工厂函数
def create_langgraph_rag_engine(config: Optional[Dict[str, Any]] = None,
                              use_legacy_fallback: bool = True) -> LangGraphRAGEngine:
    """创建LangGraph RAG引擎"""
    if config is None:
        config = create_default_rag_config()

    return LangGraphRAGEngine(config, use_legacy_fallback)

# 快速测试函数
def test_langgraph_rag_engine():
    """测试LangGraph RAG引擎"""
    logger.info("🧪 开始测试LangGraph RAG引擎...")

    try:
        # 创建引擎
        engine = create_langgraph_rag_engine()

        # 创建测试查询
        test_query = engine.create_query(
            "黏液腺癌的图像特征是什么？",
            user_id="test_user",
            session_id="test_session"
        )

        # 执行查询
        response = engine.process_query_sync(test_query)

        logger.info("✅ LangGraph RAG引擎测试完成")
        logger.info(f"问题: {response.question}")
        logger.info(f"答案: {response.answer[:200]}...")
        logger.info(f"置信度: {response.confidence}")
        logger.info(f"响应时间: {response.response_time:.2f}s")
        logger.info(f"架构: {response.metadata.get('architecture', 'unknown')}")

        return True

    except Exception as e:
        logger.error(f"❌ LangGraph RAG引擎测试失败: {e}")
        return False

if __name__ == "__main__":
    test_langgraph_rag_engine()