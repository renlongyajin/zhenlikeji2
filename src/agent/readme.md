 根据我的分析，现在使用的后端文件如下：

  🎯 当前主要后端架构

  核心引擎文件

  1. src/agent/api_service.py - FastAPI服务入口，提供RESTful API
  2. src/agent/rag_engine.py - 主要RAG引擎，支持双架构（LangGraph + 传统ReAct）
  3. src/agent/langgraph_rag_engine.py - LangGraph架构的RAG引擎

  检索管理器

  4. src/agent/retrieval_manager.py - 基础检索管理器（备用）
  5. src/agent/enhanced_retrieval_manager.py - 当前主要使用的增强版检索管理器

  Agent架构

  6. src/agent/langgraph_react_agent.py - LangGraph ReAct Agent（主要）
  7. src/agent/enhanced_react_agent.py - 增强版ReAct Agent（备用）
  8. src/agent/react_agent.py - 基础ReAct Agent（备用）

  LLM管理

  9. src/agent/llm_manager.py - LLM管理器，统一处理各种大模型

  🔧 当前运行逻辑

  根据 rag_engine.py:118-144 的初始化逻辑：

  1. 首选架构: LangGraph架构（use_langgraph=True）
  2. 检索器: 使用 enhanced_retrieval_manager（增强版检索管理器）
  3. Agent: 使用 LangGraphReActAgent
  4. 回退机制: 如果LangGraph失败，回退到增强版ReAct Agent

  📊 文件调用关系

  api_service.py (FastAPI入口)
      ↓
  rag_engine.py (主引擎，架构调度)
      ↓
  ├── langgraph_rag_engine.py (LangGraph架构)
  │   └── langgraph_react_agent.py
  │   └── enhanced_retrieval_manager.py (主要检索器)
  └── enhanced_react_agent.py (备用传统架构)
      └── retrieval_manager.py (备用基础检索器)

  llm_manager.py (被所有组件共用)

  总结: 当前主要使用的是 LangGraph架构，核心文件包括
  api_service.py、rag_engine.py、langgraph_rag_engine.py、langgraph_react_agent.py 和 enhanced_retrieval_manager.py。