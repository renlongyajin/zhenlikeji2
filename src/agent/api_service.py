#!/usr/bin/env python3
"""
FastAPI服务层
提供RESTful API接口的RAG问答服务
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict

try:
    # 尝试相对导入（当作为模块运行时）
    from .rag_engine import RAGEngine, RAGQuery, create_rag_engine, create_default_rag_config
    from .llm_manager import LLMManager
except ImportError:
    # 回退到绝对导入（当直接运行时）
    try:
        from rag_engine import RAGEngine, RAGQuery, create_rag_engine, create_default_rag_config
        from llm_manager import LLMManager
    except ImportError:
        # 最终回退（当从项目根目录运行时）
        import sys
        import os
        # 添加src目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from agent.rag_engine import RAGEngine, RAGQuery, create_rag_engine, create_default_rag_config
        from agent.llm_manager import LLMManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量
rag_engine: Optional[RAGEngine] = None

# Pydantic模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., description="用户问题", min_length=2, max_length=1000)
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    search_config: Optional[Dict[str, Any]] = Field(None, description="搜索配置")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据")

class QueryResponse(BaseModel):
    """查询响应模型"""
    query_id: str
    question: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    retrieved_documents: List[Dict[str, Any]]
    reasoning_steps: List[Dict[str, Any]]
    response_time: float
    model_used: str
    timestamp: str

class SystemStatus(BaseModel):
    """系统状态模型"""
    status: str
    components: Dict[str, Any]
    stats: Dict[str, Any]
    timestamp: str

class QuerySuggestions(BaseModel):
    """查询建议模型"""
    suggestions: List[str]
    query: str

class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str
    detail: Optional[str] = None
    timestamp: str

# 会话管理
class SessionManager:
    """会话管理器"""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: Optional[str] = None) -> str:
        """创建会话"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'query_history': [],
            'context': {}
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话"""
        return self.sessions.get(session_id)

    def add_query_to_session(self, session_id: str, query_id: str, question: str):
        """添加查询到会话历史"""
        if session_id in self.sessions:
            self.sessions[session_id]['query_history'].append({
                'query_id': query_id,
                'question': question,
                'timestamp': datetime.now().isoformat()
            })

    def update_session_context(self, session_id: str, context: Dict[str, Any]):
        """更新会话上下文"""
        if session_id in self.sessions:
            self.sessions[session_id]['context'].update(context)

# 创建会话管理器实例
session_manager = SessionManager()

# API密钥验证
def verify_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """验证API密钥"""
    # 这里可以实现实际的API密钥验证逻辑
    # 目前允许所有请求
    return True

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global rag_engine

    # 启动时初始化RAG引擎
    logger.info("🚀 启动RAG问答服务...")
    try:
        config = create_default_rag_config()
        rag_engine = create_rag_engine(config)
        logger.info("✅ RAG引擎初始化成功")
    except Exception as e:
        logger.error(f"❌ RAG引擎初始化失败: {e}")
        raise

    yield

    # 关闭时清理资源
    logger.info("🛑 关闭RAG问答服务...")

# 创建FastAPI应用
app = FastAPI(
    title="医学RAG问答系统API",
    description="基于ReAct智能代理的医学文献问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康检查端点
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "medical-rag-api",
        "timestamp": datetime.now().isoformat()
    }

# 系统状态端点
@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """获取系统状态"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG引擎未初始化")

        status = rag_engine.get_system_status()
        return SystemStatus(**status)
    except Exception as e:
        logger.error(f"❌ 获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 查询端点
@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    api_key_valid: bool = Depends(verify_api_key)
):
    """处理查询请求"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG引擎未初始化")

        # 生成查询ID
        query_id = str(uuid.uuid4())
        logger.info(f"🚀 处理查询请求: {query_id}")

        # 创建会话（如果需要）
        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session(request.user_id)

        # 创建RAG查询
        rag_query = RAGQuery(
            question=request.question,
            query_id=query_id,
            user_id=request.user_id,
            session_id=session_id,
            search_config=request.search_config,
            metadata=request.metadata
        )

        # 处理查询
        response = await rag_engine.process_query(rag_query)

        # 后台任务：更新会话历史
        background_tasks.add_task(
            session_manager.add_query_to_session,
            session_id,
            query_id,
            request.question
        )

        # 转换为Pydantic模型
        return QueryResponse(**asdict(response))

    except Exception as e:
        logger.error(f"❌ 查询处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 同步查询端点（用于不支持异步的客户端）
@app.post("/query/sync", response_model=QueryResponse)
def process_query_sync(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    api_key_valid: bool = Depends(verify_api_key)
):
    """同步处理查询请求"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG引擎未初始化")

        # 生成查询ID
        query_id = str(uuid.uuid4())
        logger.info(f"🚀 处理同步查询请求: {query_id}")

        # 创建会话（如果需要）
        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session(request.user_id)

        # 创建RAG查询
        rag_query = RAGQuery(
            question=request.question,
            query_id=query_id,
            user_id=request.user_id,
            session_id=session_id,
            search_config=request.search_config,
            metadata=request.metadata
        )

        # 处理查询
        response = rag_engine.process_query_sync(rag_query)

        # 后台任务：更新会话历史
        background_tasks.add_task(
            session_manager.add_query_to_session,
            session_id,
            query_id,
            request.question
        )

        # 转换为Pydantic模型
        return QueryResponse(**asdict(response))

    except Exception as e:
        logger.error(f"❌ 同步查询处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 查询建议端点
@app.get("/suggestions", response_model=QuerySuggestions)
async def get_query_suggestions(
    q: str = "",
    max_suggestions: int = 5,
    api_key_valid: bool = Depends(verify_api_key)
):
    """获取查询建议"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG引擎未初始化")

        if not q:
            return QuerySuggestions(suggestions=[], query=q)

        suggestions = rag_engine.get_query_suggestions(q, max_suggestions)
        return QuerySuggestions(suggestions=suggestions, query=q)

    except Exception as e:
        logger.error(f"❌ 获取查询建议失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 会话管理端点
@app.post("/sessions", response_model=Dict[str, str])
async def create_session(
    user_id: Optional[str] = None,
    api_key_valid: bool = Depends(verify_api_key)
):
    """创建会话"""
    try:
        session_id = session_manager.create_session(user_id)
        return {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ 创建会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session(
    session_id: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """获取会话信息"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话未找到")

        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 获取会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# LLM提供者管理端点
@app.get("/llm/providers", response_model=Dict[str, Any])
async def get_llm_providers():
    """获取LLM提供者列表"""
    try:
        if not rag_engine or not rag_engine.llm_manager:
            raise HTTPException(status_code=503, detail="LLM管理器未初始化")

        return rag_engine.llm_manager.get_provider_status()
    except Exception as e:
        logger.error(f"❌ 获取LLM提供者列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/providers/{provider_name}")
async def set_llm_provider(
    provider_name: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """设置LLM提供者"""
    try:
        if not rag_engine or not rag_engine.llm_manager:
            raise HTTPException(status_code=503, detail="LLM管理器未初始化")

        rag_engine.llm_manager.set_active_provider(provider_name)
        return {
            "message": f"LLM提供者已切换到: {provider_name}",
            "active_provider": provider_name,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ 设置LLM提供者失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 统计信息端点
@app.get("/stats", response_model=Dict[str, Any])
async def get_stats(
    api_key_valid: bool = Depends(verify_api_key)
):
    """获取系统统计信息"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG引擎未初始化")

        return {
            "total_queries": rag_engine.stats['total_queries'],
            "successful_queries": rag_engine.stats['successful_queries'],
            "average_response_time": rag_engine.stats['average_response_time'],
            "model_usage": rag_engine.stats['model_usage'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ 获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 批量查询端点
@app.post("/batch", response_model=List[QueryResponse])
async def process_batch_queries(
    requests: List[QueryRequest],
    api_key_valid: bool = Depends(verify_api_key)
):
    """批量处理查询"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG引擎未初始化")

        logger.info(f"🚀 批量处理 {len(requests)} 个查询")

        # 创建异步任务
        tasks = []
        for request in requests:
            rag_query = RAGQuery(
                question=request.question,
                query_id=str(uuid.uuid4()),
                user_id=request.user_id,
                session_id=request.session_id,
                search_config=request.search_config,
                metadata=request.metadata
            )
            tasks.append(rag_engine.process_query(rag_query))

        # 并行执行所有查询
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"❌ 查询 {i} 处理失败: {response}")
                # 创建错误响应
                error_response = RAGResponse(
                    query_id=str(uuid.uuid4()),
                    question=requests[i].question,
                    answer=f"处理查询时出错: {str(response)}",
                    confidence=0.0,
                    retrieved_documents=[],
                    reasoning_steps=[],
                    search_queries=[requests[i].question],
                    response_time=0.0,
                    model_used="error",
                    timestamp=datetime.now().isoformat(),
                    metadata={'error': str(response)}
                )
                results.append(QueryResponse(**asdict(error_response)))
            else:
                results.append(QueryResponse(**asdict(response)))

        logger.info(f"✅ 批量查询处理完成，成功 {len([r for r in results if r.confidence > 0])} 个")
        return results

    except Exception as e:
        logger.error(f"❌ 批量查询处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局错误处理"""
    logger.error(f"❌ 未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# 根端点
@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "医学RAG问答系统API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "query": "/query",
            "query_sync": "/query/sync",
            "suggestions": "/suggestions",
            "sessions": "/sessions",
            "llm_providers": "/llm/providers",
            "stats": "/stats",
            "batch": "/batch"
        },
        "timestamp": datetime.now().isoformat()
    }

# 启动函数
def run_api_service(host: str = "0.0.0.0", port: int = None, reload: bool = False):
    """运行API服务"""
    import uvicorn
    import os

    # 从环境变量读取端口，默认8001避免冲突
    if port is None:
        port = int(os.environ.get('API_PORT', '8001'))

    logger.info(f"🚀 启动API服务: {host}:{port}")
    uvicorn.run(
        "api_service:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_api_service()