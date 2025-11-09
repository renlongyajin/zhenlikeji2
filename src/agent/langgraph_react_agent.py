#!/usr/bin/env python3
"""
LangGraph-based ReAct Agent
基于LangGraph实现的现代化ReAct架构，支持LLM自主工具调用
遵循Thought → Action → Observation循环，但由LLM自主决策
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode  # 新版本不再提供ToolNode，自定义实现
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
import logging
import json
import asyncio
import re
from datetime import datetime
from dataclasses import dataclass, asdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # 尝试相对导入
    from .enhanced_retrieval_manager import EnhancedMedicalRetrievalManager, create_enhanced_retrieval_manager
    from .llm_manager import LLMManager, create_llm_manager
except ImportError:
    # 回退到绝对导入
    try:
        from enhanced_retrieval_manager import EnhancedMedicalRetrievalManager, create_enhanced_retrieval_manager
        from llm_manager import LLMManager, create_llm_manager
    except ImportError:
        # 最终回退
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from enhanced_retrieval_manager import EnhancedMedicalRetrievalManager, create_enhanced_retrieval_manager
        from llm_manager import LLMManager, create_llm_manager

class ReActAgentState(TypedDict):
    """ReAct Agent状态定义"""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    original_question: str
    query_type: str  # 查询类型：medical, general, comparison, etc.
    entities: List[str]  # 识别的实体列表
    keywords: List[str]  # LLM提取或回退生成的关键词
    need_search: bool  # 是否需要执行检索
    current_step: str  # 当前步骤
    tool_calls: List[Dict[str, Any]]  # 工具调用历史
    search_results: List[Dict[str, Any]]  # 搜索结果
    result_quality: float  # 结果质量评分
    iteration_count: int  # 迭代次数
    max_iterations: int  # 最大迭代次数
    final_answer: Optional[str]  # 最终答案
    confidence: float  # 置信度
    metadata: Dict[str, Any]  # 元数据

class ToolNode:
    """简单的工具节点实现，用于执行工具调用"""

    def __init__(self, tools: list):
        self.tools = {tool.name: tool for tool in tools}

    def __call__(self, state: ReActAgentState) -> Dict[str, Any]:
        """执行工具调用"""
        messages = state.get("messages", [])
        if messages is None:
            messages = []
        if messages is None:
            messages = []
        tool_calls = state.get("tool_calls", [])

        if not tool_calls:
            return {"messages": messages}

        # 执行工具调用
        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            if tool_name in self.tools:
                try:
                    # 执行工具
                    result = self.tools[tool_name].invoke(tool_args)

                    # 创建工具消息
                    tool_message = ToolMessage(
                        content=json.dumps(result) if isinstance(result, dict) else str(result),
                        tool_call_id=tool_call.get("id", f"call_{tool_name}_{id(tool_call)}"),
                        name=tool_name
                    )
                    tool_messages.append(tool_message)

                    logger.info(f"✅ 工具 {tool_name} 执行成功")

                except Exception as e:
                    logger.error(f"❌ 工具 {tool_name} 执行失败: {e}")
                    error_message = ToolMessage(
                        content=json.dumps({"success": False, "error": str(e)}),
                        tool_call_id=tool_call.get("id", f"call_{tool_name}_{id(tool_call)}"),
                        name=tool_name
                    )
                    tool_messages.append(error_message)
            else:
                logger.error(f"❌ 未知工具: {tool_name}")

        # 添加工具消息到消息列表
        messages.extend(tool_messages)

        return {"messages": messages}

@dataclass
class ToolCallResult:
    """工具调用结果"""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None

class LangGraphReActAgent:
    """基于LangGraph的ReAct Agent"""

    def __init__(self,
                 llm_manager=None,
                 retrieval_manager=None,
                 enhanced_retrieval_manager=None,
                 embedding_manager=None,
                 max_iterations: int = 3):
        """初始化LangGraph ReAct Agent"""
        self.llm_manager = llm_manager
        self.retrieval_manager = retrieval_manager
        self.enhanced_retrieval_manager = enhanced_retrieval_manager
        self.embedding_manager = embedding_manager
        self.max_iterations = max_iterations

        # 初始化工具
        self.tools = self._initialize_tools()
        self.tool_node = ToolNode(self.tools)

        # 构建图
        self._build_graph()

        logger.info("✅ LangGraph ReAct Agent 初始化完成")

    def _initialize_tools(self):
        """初始化工具"""

        @tool
        def rag_search(query: str,
                      search_type: str = "intelligent",
                      max_results: int = 5,
                      title_priority: bool = True) -> Dict[str, Any]:
            """搜索医学文档，支持标题优先级和多轮优化

            Args:
                query: 搜索查询
                search_type: 搜索类型 (keyword, semantic, hybrid, intelligent)
                max_results: 最大结果数量
                title_priority: 是否启用标题优先级

            Returns:
                搜索结果字典
            """
            try:
                logger.info(f"🔍 RAG搜索: '{query}' (类型: {search_type})")

                # 优先使用增强版检索管理器
                retrieval_mgr = self.enhanced_retrieval_manager or self.retrieval_manager

                if retrieval_mgr:
                    # 构建增强版搜索配置
                    search_config = {
                        'search_type': search_type,
                        'top_k': max_results,
                        'title_priority': title_priority,
                        'title_priority_config': {
                            'chapter_title_weight': 25.0,
                            'section_title_weight': 20.0,
                            'subsection_title_weight': 15.0,
                            'exact_match_boost': 3.0,
                            'medical_term_bonus': 5.0,
                            'descriptive_content_boost': 2.0,
                            'min_description_length': 150
                        }
                    }

                    # 执行增强版搜索
                    if hasattr(retrieval_mgr, 'enhanced_search'):
                        results = retrieval_mgr.enhanced_search(query, search_config)
                    else:
                        # 回退到普通搜索
                        results = retrieval_mgr.search(query, **search_config)

                    logger.info(f"✅ RAG搜索完成，找到 {len(results)} 个结果")

                    # 格式化结果（适配增强版搜索结果）
                    formatted_results = []
                    for result in results:
                        if isinstance(result, dict):
                            # 增强版搜索结果格式
                            formatted_results.append({
                                'content': result.get('content', ''),
                                'page_number': result.get('page_number', 0),
                                'chapter_title': result.get('chapter_title', ''),
                                'section_title': result.get('section_title', ''),
                                'score': result.get('score', 0.0),
                                'source': result.get('source', 'unknown'),
                                'doc_id': result.get('doc_id', ''),
                                'search_type': result.get('search_type', 'unknown'),
                                'title_match_score': result.get('title_match_score', 0.0),
                                'content_quality_score': result.get('content_quality_score', 0.0),
                                'is_descriptive': result.get('is_descriptive', False),
                                'has_medical_terms': result.get('has_medical_terms', False)
                            })
                        else:
                            # 普通搜索结果格式（回退）
                            formatted_results.append({
                                'content': getattr(result, 'content', ''),
                                'page_number': getattr(result, 'page_number', 0),
                                'chapter_title': getattr(result, 'chapter_title', ''),
                                'section_title': getattr(result, 'section_title', ''),
                                'score': getattr(result, 'score', 0.0),
                                'source': getattr(result, 'source', 'unknown'),
                                'doc_id': getattr(result, 'doc_id', ''),
                                'search_type': getattr(result, 'search_type', 'unknown')
                            })

                    return {
                        "success": True,
                        "query": query,
                        "search_type": search_type,
                        "results": formatted_results,
                        "count": len(formatted_results),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "error": "检索管理器未初始化"
                    }

            except Exception as e:
                logger.error(f"❌ RAG搜索失败: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @tool
        def evaluate_search_quality(search_results: List[Dict[str, Any]],
                                  original_query: str) -> Dict[str, Any]:
            """评估搜索结果质量

            Args:
                search_results: 搜索结果列表
                original_query: 原始查询

            Returns:
                质量评估结果
            """
            try:
                logger.info("🔍 评估搜索结果质量")

                if not search_results:
                    return {
                        "success": True,
                        "quality_score": 0.0,
                        "needs_optimization": True,
                        "feedback": "搜索结果为空"
                    }

                # 计算质量分数
                quality_score = 0.0
                feedback_items = []

                # 1. 结果数量评估
                if len(search_results) >= 3:
                    quality_score += 0.2
                else:
                    feedback_items.append("结果数量较少")

                # 2. 内容相关性评估
                relevant_results = 0
                for result in search_results:
                    content = result.get('content', '').lower()
                    if any(term in content for term in original_query.lower().split()):
                        relevant_results += 1

                relevance_ratio = relevant_results / len(search_results)
                quality_score += relevance_ratio * 0.3

                if relevance_ratio < 0.5:
                    feedback_items.append("相关性较低")

                # 3. 描述完整性评估
                descriptive_results = 0
                for result in search_results:
                    content = result.get('content', '')
                    # 检查是否包含详细描述（长度、标点、医学术语等）
                    if (len(content) > 200 and
                        any(punct in content for punct in ['。', '；', '：']) and
                        any(term in content for term in ['细胞', '组织', '病理'])):
                        descriptive_results += 1

                description_ratio = descriptive_results / len(search_results)
                quality_score += description_ratio * 0.3

                if description_ratio < 0.3:
                    feedback_items.append("详细描述较少")

                # 4. 标题匹配度评估
                title_matches = 0
                for result in search_results:
                    chapter_title = result.get('chapter_title', '').lower()
                    section_title = result.get('section_title', '').lower()
                    query_terms = original_query.lower().split()

                    if any(term in chapter_title or term in section_title for term in query_terms):
                        title_matches += 1

                title_match_ratio = title_matches / len(search_results)
                quality_score += title_match_ratio * 0.2

                if title_match_ratio < 0.3:
                    feedback_items.append("标题匹配度较低")

                # 综合评估
                needs_optimization = quality_score < 0.6 or len(feedback_items) > 0

                logger.info(f"质量评估完成: 分数={quality_score:.2f}, 需要优化={needs_optimization}")

                return {
                    "success": True,
                    "quality_score": quality_score,
                    "needs_optimization": needs_optimization,
                    "feedback": "; ".join(feedback_items) if feedback_items else "结果质量良好",
                    "details": {
                        "relevance_ratio": relevance_ratio,
                        "description_ratio": description_ratio,
                        "title_match_ratio": title_match_ratio,
                        "result_count": len(search_results)
                    }
                }

            except Exception as e:
                logger.error(f"❌ 质量评估失败: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "quality_score": 0.0,
                    "needs_optimization": True
                }

        return [rag_search, evaluate_search_quality]

    def _build_graph(self):
        """构建LangGraph工作流"""
        workflow = StateGraph(ReActAgentState)

        # 定义节点
        workflow.add_node("intent_analysis", self._intent_analysis_node)
        workflow.add_node("tool_selection", self._tool_selection_node)
        workflow.add_node("tool_execution", self.tool_node)
        workflow.add_node("result_observation", self._result_observation_node)
        workflow.add_node("quality_evaluation", self._quality_evaluation_node)
        workflow.add_node("response_generation", self._response_generation_node)

        # 定义条件边
        workflow.add_conditional_edges(
            "intent_analysis",
            self._should_use_tools,
            {
                "use_tools": "tool_selection",
                "direct_response": "response_generation"
            }
        )

        workflow.add_edge("tool_selection", "tool_execution")
        workflow.add_edge("tool_execution", "result_observation")
        workflow.add_edge("result_observation", "quality_evaluation")

        workflow.add_conditional_edges(
            "quality_evaluation",
            self._should_continue_search,
            {
                "continue": "tool_selection",  # 继续搜索优化
                "sufficient": "response_generation"
            }
        )

        workflow.add_edge("response_generation", END)

        # 设置入口点
        workflow.set_entry_point("intent_analysis")

        self.graph = workflow.compile()
        logger.info("✅ LangGraph工作流构建完成")

    def _intent_analysis_node(self, state: ReActAgentState) -> Dict[str, Any]:
        """意图分析节点 - Thought阶段"""
        logger.info("🧠 执行意图分析节点...")

        question = state["question"]
        messages = state.get("messages", [])

        try:
            classification = self._classify_query(question, state.get("metadata"))
            summary_text = classification.get("summary") or classification.get("reason") or ""
            query_type = classification.get("query_type") or self._determine_query_type(question, summary_text)
            keywords = classification.get("keywords") or self._fallback_keywords(question)
            need_search = classification.get("need_search")

            if need_search is None:
                # 默认：医学类查询或包含关键词则需要检索
                need_search = query_type != "general" or len(keywords) > 0

            analysis_summary = classification.get("summary") or classification.get("reason")
            if analysis_summary:
                messages.append(AIMessage(content=f"查询分析：{analysis_summary}"))
            else:
                preview_keywords = ", ".join(keywords[:5]) if keywords else "无"
                messages.append(AIMessage(
                    content=f"查询分析：类型={query_type}，关键词={preview_keywords}，需要检索={'是' if need_search else '否'}"
                ))

            logger.info(
                "✅ 意图分析完成 - 类型: %s, 关键词: %s, 需要检索: %s",
                query_type,
                keywords,
                need_search
            )

            return {
                "messages": messages,
                "query_type": query_type,
                "entities": keywords,  # 兼容旧字段
                "keywords": keywords,
                "need_search": need_search,
                "current_step": "intent_analysis",
                "iteration_count": 0
            }

        except Exception as e:
            logger.error(f"❌ 意图分析失败: {e}")
            fallback_keywords = self._fallback_keywords(question)
            fallback_query_type = self._determine_query_type(question, "")
            need_search = fallback_query_type != "general" or len(fallback_keywords) > 0

            messages.append(AIMessage(
                content="查询分析遇到异常，已使用回退规则提取关键词。"
            ))

            return {
                "messages": messages,
                "query_type": fallback_query_type,
                "entities": fallback_keywords,
                "keywords": fallback_keywords,
                "need_search": need_search,
                "current_step": "intent_analysis",
                "iteration_count": 0
            }

    def _generate_optimized_search_query(self,
                                         original_query: str,
                                         optimization_strategy: str,
                                         previous_results: List[Dict[str, Any]],
                                         iteration_count: int,
                                         keywords: Optional[List[str]] = None) -> str:
        """基于优化策略生成改进的搜索查询"""

        if optimization_strategy == "none" or iteration_count == 0:
            return original_query

        keywords = keywords or self._fallback_keywords(original_query)

        if optimization_strategy == "broaden_search":
            # 扩大搜索范围 - 使用更通用的术语
            return f"{original_query} 特征 表现 诊断"

        elif optimization_strategy == "refine_keywords":
            # 优化关键词 - 提取核心医学术语
            if keywords:
                return " ".join(keywords[:3])
            return original_query

        elif optimization_strategy == "target_descriptive":
            # 针对描述性内容 - 添加描述性关键词
            return f"{original_query} 详细描述 细胞形态 组织结构"

        elif optimization_strategy == "target_titles":
            # 针对标题匹配 - 使用章节结构关键词
            if keywords:
                return " ".join([keywords[0], "章节", "标题"]) if len(keywords) > 0 else original_query
            return original_query

        elif optimization_strategy == "adjust_weights":
            # 调整权重策略 - 使用更具体的医学术语
            if keywords:
                return f"{' '.join(keywords[:2])} 病理特征 显微镜下表现"
            return f"{original_query} 病理特征 显微镜下表现"

        else:
            return original_query

    def _tool_selection_node(self, state: ReActAgentState) -> Dict[str, Any]:
        """工具选择节点 - Action决策阶段"""
        logger.info("🛠️ 执行工具选择节点...")

        question = state["question"]
        query_type = state.get("query_type", "unknown")
        search_results = state.get("search_results", [])
        iteration_count = state.get("iteration_count", 0)
        messages = state.get("messages", [])
        optimization_strategy = state.get("optimization_strategy", "none")
        keywords = state.get("keywords", [])

        if messages is None:
            messages = []

        base_query = " ".join(keywords) if keywords else question
        base_query = base_query.strip() or question

        # 智能搜索词生成
        if iteration_count > 0 and optimization_strategy != "none":
            optimized_query = self._generate_optimized_search_query(
                base_query,
                optimization_strategy,
                search_results,
                iteration_count,
                keywords=keywords
            )
            logger.info(f"使用优化搜索词: '{optimized_query}' (策略: {optimization_strategy})")
        else:
            optimized_query = base_query
            logger.info(f"使用原始搜索词: '{optimized_query}'")

        # 构建工具选择提示
        tool_selection_prompt = f"""你是一个医学AI助手，需要选择合适的工具来帮助回答用户查询。

用户查询：{question}
优化搜索词：{optimized_query}
查询类型：{query_type}
当前迭代次数：{iteration_count}
已有搜索结果：{len(search_results)} 个
优化策略：{optimization_strategy}

    候选关键词：{', '.join(keywords[:5]) if keywords else '无'}
    可用工具：
1. rag_search: 搜索医学文档，支持标题优先级和多轮优化
2. evaluate_search_quality: 评估搜索结果质量

请分析：
1. 是否需要使用rag_search工具来搜索医学文档？
2. 如果已有搜索结果，是否需要评估其质量？
3. 应该使用什么搜索策略？（考虑当前优化策略）
4. 如何设置搜索参数以获得最佳结果？

请返回工具调用决策，格式如下：
TOOL_DECISION: [工具名称]
REASONING: [详细推理过程]
PARAMETERS: [工具参数]"""

        try:
            if self.llm_manager and hasattr(self.llm_manager, 'bind_tools'):
                # 使用支持工具调用的LLM
                tool_selection_messages = [
                    SystemMessage(content="你是一个专业的医学AI助手，擅长选择合适的工具。"),
                    HumanMessage(content=tool_selection_prompt)
                ]

                # 绑定工具并调用
                llm_with_tools = self.llm_manager.bind_tools(self.tools)
                response = llm_with_tools.invoke(tool_selection_messages)

                # 提取工具调用决策
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    tool_call = response.tool_calls[0]
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']

                    logger.info(f"✅ 工具选择完成 - 工具: {tool_name}, 参数: {tool_args}")

                    # 添加决策到消息历史
                    messages.append(AIMessage(
                        content=f"选择工具: {tool_name}",
                        tool_calls=[tool_call]
                    ))

                    return {
                        "messages": messages,
                        "tool_calls": [tool_call],
                        "current_step": "tool_selection"
                    }
                else:
                    # 没有工具调用，使用默认搜索（使用优化后的搜索词）
                    default_tool_call = {
                        "name": "rag_search",
                        "args": {
                            "query": optimized_query,
                            "search_type": "intelligent",
                            "max_results": 5,
                            "title_priority": True
                        }
                    }

                    messages.append(AIMessage(
                        content="使用默认rag_search工具",
                        tool_calls=[default_tool_call]
                    ))

                    return {
                        "messages": messages,
                        "tool_calls": [default_tool_call],
                        "current_step": "tool_selection"
                    }
            else:
                # 不支持工具调用的降级处理
                default_tool_call = {
                    "name": "rag_search",
                    "args": {
                        "query": question,
                        "search_type": "intelligent",
                        "max_results": 5,
                        "title_priority": True
                    }
                }

                return {
                    "tool_calls": [default_tool_call],
                    "current_step": "tool_selection"
                }

        except Exception as e:
            logger.error(f"❌ 工具选择失败: {e}")
            # 降级到默认搜索（使用优化后的搜索词）
            default_tool_call = {
                "name": "rag_search",
                "args": {
                    "query": optimized_query,
                    "search_type": "intelligent",
                    "max_results": 5,
                    "title_priority": True
                }
            }

            return {
                "tool_calls": [default_tool_call],
                "current_step": "tool_selection"
            }

    def _result_observation_node(self, state: ReActAgentState) -> Dict[str, Any]:
        """结果观察节点 - Observation阶段"""
        logger.info("👁️ 执行结果观察节点...")

        messages = state.get("messages", [])
        tool_calls = state.get("tool_calls", [])
        search_results = state.get("search_results", [])

        try:
            # 处理工具调用结果
            if tool_calls:
                last_tool_call = tool_calls[-1]
                tool_name = last_tool_call["name"]
                tool_args = last_tool_call["args"]

                # 从消息中提取工具执行结果
                tool_results = []
                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        tool_results.append(json.loads(msg.content) if isinstance(msg.content, str) else msg.content)

                if tool_results:
                    last_result = tool_results[-1]

                    if last_result.get("success"):
                        logger.info(f"✅ 工具 {tool_name} 执行成功")

                        # 提取搜索结果
                        if tool_name == "rag_search":
                            search_results = last_result.get("results", [])

                            # 生成观察报告
                            observation = f"""工具执行观察报告：
- 工具名称: {tool_name}
- 执行状态: 成功
- 搜索结果数量: {len(search_results)}
- 搜索查询: {tool_args.get('query', 'N/A')}
- 搜索类型: {tool_args.get('search_type', 'N/A')}
"""
                            if search_results:
                                observation += f"- 最佳结果分数: {search_results[0].get('score', 0):.3f}\n"
                                observation += f"- 结果来源分布: {len(set(r.get('source', 'unknown') for r in search_results))} 种来源"
                        else:
                            observation = f"工具 {tool_name} 执行成功，获得有效结果"

                        messages.append(AIMessage(content=observation))

                        return {
                            "messages": messages,
                            "search_results": search_results,
                            "current_step": "result_observation"
                        }
                    else:
                        # 工具执行失败
                        error_msg = last_result.get("error", "未知错误")
                        logger.error(f"❌ 工具 {tool_name} 执行失败: {error_msg}")

                        observation = f"工具 {tool_name} 执行失败: {error_msg}"
                        messages.append(AIMessage(content=observation))

                        return {
                            "messages": messages,
                            "current_step": "result_observation"
                        }

            # 没有工具结果的情况
            return {
                "current_step": "result_observation"
            }

        except Exception as e:
            logger.error(f"❌ 结果观察失败: {e}")
            return {
                "current_step": "result_observation"
            }

    def _quality_evaluation_node(self, state: ReActAgentState) -> Dict[str, Any]:
        """质量评估节点 - 多轮搜索优化核心"""
        logger.info("📊 执行质量评估节点...")

        search_results = state.get("search_results", [])
        question = state["question"]
        iteration_count = state.get("iteration_count", 0)
        messages = state.get("messages", [])
        previous_tool_calls = state.get("tool_calls", [])

        try:
            # 如果没有搜索结果，直接认为需要优化
            if not search_results:
                logger.info("没有搜索结果，需要优化")
                return {
                    "result_quality": 0.0,
                    "needs_optimization": True,
                    "optimization_strategy": "broaden_search",
                    "current_step": "quality_evaluation"
                }

            # 多维度质量评估
            quality_score = 0.0
            feedback_items = []
            optimization_strategy = "none"

            # 1. 结果数量评估 (15%)
            result_count = len(search_results)
            if result_count >= 5:
                quality_score += 0.15
            elif result_count >= 3:
                quality_score += 0.1
                feedback_items.append("结果数量偏少")
            else:
                quality_score += 0.05
                feedback_items.append("结果数量不足")
                optimization_strategy = "broaden_search"

            # 2. 内容相关性评估 (25%)
            relevant_results = 0
            query_terms = set(effective_query_text.lower().split())

            for result in search_results:
                content = result.get('content', '').lower()
                title = (result.get('chapter_title', '') + ' ' + result.get('section_title', '')).lower()

                # 检查查询词在内容和标题中的出现
                content_matches = sum(1 for term in query_terms if term in content)
                title_matches = sum(1 for term in query_terms if term in title)

                # 如果内容和标题都有匹配，认为是相关结果
                if content_matches >= 1 or title_matches >= 1:
                    relevant_results += 1

            relevance_ratio = relevant_results / result_count if result_count > 0 else 0
            quality_score += relevance_ratio * 0.25

            if relevance_ratio < 0.6:
                feedback_items.append("相关性较低")
                if optimization_strategy == "none":
                    optimization_strategy = "refine_keywords"

            # 3. 描述完整性评估 (35% - 最重要)
            descriptive_results = 0
            total_description_length = 0

            for result in search_results:
                content = result.get('content', '')

                # 检查是否为描述性内容
                is_descriptive = (
                    len(content) > 150 and  # 长度足够
                    any(punct in content for punct in ['。', '；', '：']) and  # 有完整句子
                    any(term in content for term in ['细胞', '组织', '病理', '诊断', '呈', '可见'])  # 有医学描述词
                )

                # 检查增强版搜索结果的特殊字段
                if result.get('is_descriptive', False) or result.get('content_quality_score', 0) > 0.5:
                    is_descriptive = True

                if is_descriptive:
                    descriptive_results += 1
                    total_description_length += len(content)

            description_ratio = descriptive_results / result_count if result_count > 0 else 0
            quality_score += description_ratio * 0.35

            if description_ratio < 0.4:
                feedback_items.append("详细描述较少")
                if optimization_strategy == "none":
                    optimization_strategy = "target_descriptive"

            # 4. 标题匹配度评估 (15%)
            title_matches = 0
            for result in search_results:
                chapter_title = result.get('chapter_title', '').lower()
                section_title = result.get('section_title', '').lower()

                # 检查标题匹配
                if any(term in chapter_title or term in section_title for term in query_terms):
                    title_matches += 1

            title_match_ratio = title_matches / result_count if result_count > 0 else 0
            quality_score += title_match_ratio * 0.15

            if title_match_ratio < 0.3:
                feedback_items.append("标题匹配度较低")
                if optimization_strategy == "none":
                    optimization_strategy = "target_titles"

            # 5. 分数分布评估 (10%)
            if search_results:
                scores = [r.get('score', 0) for r in search_results]
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)

                if avg_score > 0.5:
                    quality_score += 0.1
                elif avg_score > 0.3:
                    quality_score += 0.05
                else:
                    feedback_items.append("整体分数偏低")
                    if optimization_strategy == "none":
                        optimization_strategy = "adjust_weights"

            # 综合决策
            needs_optimization = quality_score < 0.6 or iteration_count == 0  # 第一轮总是尝试优化

            # 如果已经达到最大迭代次数，强制停止
            if iteration_count >= self.max_iterations - 1:
                needs_optimization = False
                optimization_strategy = "max_iterations_reached"
                feedback_items.append("达到最大迭代次数")

            logger.info(f"质量评估完成: 分数={quality_score:.2f}, 需要优化={needs_optimization}, 策略={optimization_strategy}")

            # 生成详细的评估报告
            evaluation_report = f"""搜索结果质量评估报告：
- 综合质量分数: {quality_score:.2f}/1.0
- 结果数量: {result_count} 个
- 相关性比例: {relevance_ratio:.2f}
- 描述性内容比例: {description_ratio:.2f}
- 标题匹配比例: {title_match_ratio:.2f}
- 平均分数: {avg_score:.3f}
- 迭代次数: {iteration_count}
- 优化策略: {optimization_strategy}
"""

            if feedback_items:
                evaluation_report += f"- 需要改进: {', '.join(feedback_items)}"

            # 添加评估结果到消息历史
            messages.append(AIMessage(content=evaluation_report))

            return {
                "messages": messages,
                "result_quality": quality_score,
                "needs_optimization": needs_optimization,
                "optimization_strategy": optimization_strategy,
                "current_step": "quality_evaluation"
            }

        except Exception as e:
            logger.error(f"❌ 质量评估失败: {e}")
            return {
                "result_quality": 0.0,
                "needs_optimization": True,
                "optimization_strategy": "evaluation_failed",
                "current_step": "quality_evaluation"
            }

    def _response_generation_node(self, state: ReActAgentState) -> Dict[str, Any]:
        """响应生成节点"""
        logger.info("📝 执行响应生成节点...")

        question = state["question"]
        search_results = state.get("search_results", [])
        messages = state.get("messages", [])
        result_quality = state.get("result_quality", 0.0)
        iteration_count = state.get("iteration_count", 0)

        try:
            if search_results and self.llm_manager:
                # 基于搜索结果生成回答
                context_text = self._build_context_from_results(search_results)

                response_prompt = f"""基于以下医学文献内容，回答用户的问题：

用户问题：{question}

参考内容：
{context_text}

回答要求：
1. 必须基于提供的医学文献内容
2. 使用专业但易于理解的医学术语
3. 提供准确、可靠的医学信息
4. 如果不确定，要明确说明
5. 建议咨询专业医疗人员获取个性化建议

请生成完整的回答："""

                response_messages = [
                    SystemMessage(content="你是一位专业的医学AI助手，基于医学文献提供准确的医学信息。"),
                    HumanMessage(content=response_prompt)
                ]

                # 使用同步方法生成响应
                response = self.llm_manager.generate_response_sync(response_messages)
                final_answer = response.content if hasattr(response, 'content') else str(response)

                logger.info(f"✅ 响应生成完成，长度: {len(final_answer)} 字符")

                # 添加生成过程到消息历史
                messages.append(AIMessage(content=f"生成最终回答 (质量分数: {result_quality:.2f})"))

                return {
                    "messages": messages,
                    "final_answer": final_answer,
                    "confidence": min(result_quality + 0.3, 0.95),  # 基于质量分数调整置信度
                    "current_step": "response_generation"
                }
            else:
                # 没有搜索结果或LLM不可用时的降级处理
                fallback_answer = "抱歉，未能找到相关的医学文献来回答您的问题。建议您咨询专业医疗人员获取准确信息。"

                return {
                    "final_answer": fallback_answer,
                    "confidence": 0.1,
                    "current_step": "response_generation"
                }

        except Exception as e:
            logger.error(f"❌ 响应生成失败: {e}")
            return {
                "final_answer": f"抱歉，生成回答时出现错误: {str(e)}",
                "confidence": 0.0,
                "current_step": "response_generation"
            }

    def _should_use_tools(self, state: ReActAgentState) -> str:
        """判断是否需要使用工具"""
        query_type = state.get("query_type", "unknown")
        entities = state.get("entities", [])
        need_search = state.get("need_search")
        iteration_count = state.get("iteration_count", 0)

        if need_search is True:
            return "use_tools"
        if need_search is False:
            return "direct_response"

        # 简单的决策逻辑
        if iteration_count > 0:
            # 已经在优化过程中，继续使用工具
            return "use_tools"

        if query_type in ["medical", "pathology", "diagnostic"] or len(entities) > 0:
            # 医学相关查询或包含实体，需要使用工具
            return "use_tools"

        # 其他情况，直接生成响应
        return "direct_response"

    def _should_continue_search(self, state: ReActAgentState) -> str:
        """判断是否需要继续搜索优化"""
        result_quality = state.get("result_quality", 0.0)
        iteration_count = state.get("iteration_count", 0)
        needs_optimization = state.get("needs_optimization", False)

        # 检查是否达到最大迭代次数
        if iteration_count >= self.max_iterations:
            logger.info(f"达到最大迭代次数 ({self.max_iterations})，停止搜索")
            return "sufficient"

        # 基于质量分数和优化需求决策
        if needs_optimization and result_quality < 0.6:
            optimization_strategy = state.get("optimization_strategy", "none")
            logger.info(f"结果质量不足 ({result_quality:.2f})，继续优化 (策略: {optimization_strategy})")
            return "continue"

        # 质量足够，停止优化
        logger.info(f"结果质量足够 ({result_quality:.2f})，停止优化")
        return "sufficient"

    def _classify_query(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """使用LLM对查询进行分类，返回结构化结果"""
        metadata = metadata or {}
        fallback_keywords = self._fallback_keywords(question)
        default = {
            "query_type": self._determine_query_type(question, ""),
            "need_search": None,
            "keywords": fallback_keywords,
            "reason": None,
            "summary": None
        }

        if not self.llm_manager:
            default["reason"] = "LLM不可用，使用回退关键词。"
            return default

        classification_prompt = f"""请扮演医学知识检索助手，对用户查询进行分类并输出JSON。

用户查询：{question}
附加元数据：{json.dumps(metadata, ensure_ascii=False)}

请以严格的JSON格式返回，字段包括：
- query_type: "medical"、"general"、"comparison"或其他自定义类别
- need_search: true/false，表示是否需要访问文档检索工具
- keywords: 3-6个用于检索的关键词数组（中文或英文，按重要性排序）
- summary: 一句话概括判断依据

示例输出：
{{
  "query_type": "medical",
  "need_search": true,
  "keywords": ["肺腺癌", "ROSE", "细胞学"],
  "summary": "问题询问肺腺癌的ROSE检查细节，需要查阅医学文献。"
}}

请只返回JSON。"""

        try:
            classification_messages = [
                SystemMessage(content="你是医学RAG系统的意图分析助手，回答必须是有效JSON。"),
                HumanMessage(content=classification_prompt)
            ]
            response = self.llm_manager.generate_response_sync(
                classification_messages,
                temperature=0.2,
                max_tokens=400
            )
            raw_content = response.content if hasattr(response, "content") else str(response)
            parsed = self._safe_json_parse(raw_content)

            if not parsed:
                raise ValueError(f"无法解析分类结果: {raw_content}")

            query_type = parsed.get("query_type") or default["query_type"]
            need_search = parsed.get("need_search")
            keywords = parsed.get("keywords") or fallback_keywords
            summary = parsed.get("summary") or parsed.get("reason")

            if isinstance(need_search, str):
                normalized = need_search.strip().lower()
                need_search = normalized in {"true", "yes", "需要", "是", "1"}

            if not isinstance(need_search, bool):
                need_search = None

            if isinstance(keywords, str):
                keywords = [kw.strip() for kw in re.split(r"[,\s]+", keywords) if kw.strip()]

            if not isinstance(keywords, list):
                keywords = fallback_keywords

            cleaned_keywords = []
            seen = set()
            for kw in keywords:
                if not isinstance(kw, str):
                    continue
                token = kw.strip()
                if not token:
                    continue
                lower_token = token.lower()
                if lower_token in seen:
                    continue
                seen.add(lower_token)
                cleaned_keywords.append(token)

            if not cleaned_keywords:
                cleaned_keywords = fallback_keywords

            return {
                "query_type": query_type,
                "need_search": need_search,
                "keywords": cleaned_keywords,
                "summary": summary,
                "reason": summary
            }

        except Exception as e:
            logger.warning(f"分类LLM调用失败，使用默认策略: {e}")
            default["reason"] = f"LLM调用失败，使用回退关键词: {fallback_keywords}"
            return default

    def _safe_json_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """安全解析JSON，兼容代码块或额外文本"""
        if not text:
            return None

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_match:
            try:
                return json.loads(fenced_match.group(1))
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        return None

    def _fallback_keywords(self, question: str, max_terms: int = 6) -> List[str]:
        """基于规则的关键词回退策略"""
        if not question:
            return []

        # 提取中文词块与英文/数字token
        candidates = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", question)
        cleaned: List[str] = []
        seen = set()

        for token in candidates:
            token = token.strip()
            if not token:
                continue

            lower_token = token.lower()
            if lower_token in {"什么", "有哪些", "怎么", "如何", "需要", "以及", "和", "的"}:
                continue

            if lower_token in seen:
                continue

            seen.add(lower_token)
            cleaned.append(token)

        if not cleaned:
            # 最少返回一些字符以避免空检索
            return [question[:8]]

        return cleaned[:max_terms]

    def _determine_query_type(self, question: str, analysis_content: str) -> str:
        """确定查询类型"""
        question_lower = question.lower()
        analysis_lower = analysis_content.lower() if analysis_content else ""
        combined_text = f"{question_lower} {analysis_lower}"

        # 医学相关关键词
        medical_keywords = ['癌', '细胞', '病理', '诊断', '治疗', '预后', 'ROSE']

        if any(keyword in combined_text for keyword in medical_keywords):
            return "medical"

        # 对比类查询
        comparison_keywords = ['区别', '差异', '不同', '比较', 'vs']
        if any(keyword in combined_text for keyword in comparison_keywords):
            return "comparison"

        return "general"

    def _evaluate_search_quality_internal(self, search_results: List[Dict[str, Any]], original_query: str) -> Dict[str, Any]:
        """内部质量评估实现"""
        # 简化的质量评估实现
        if not search_results:
            return {
                "success": True,
                "quality_score": 0.0,
                "needs_optimization": True,
                "feedback": "搜索结果为空"
            }

        # 基础质量评分
        quality_score = 0.0

        # 结果数量
        if len(search_results) >= 3:
            quality_score += 0.3

        # 内容长度（简单指标）
        avg_content_length = sum(len(r.get('content', '')) for r in search_results) / len(search_results)
        if avg_content_length > 150:
            quality_score += 0.3

        # 分数分布
        if search_results:
            avg_score = sum(r.get('score', 0) for r in search_results) / len(search_results)
            if avg_score > 0.5:
                quality_score += 0.4

        needs_optimization = quality_score < 0.6

        return {
            "success": True,
            "quality_score": quality_score,
            "needs_optimization": needs_optimization,
            "feedback": "需要优化" if needs_optimization else "质量良好"
        }

    def _build_context_from_results(self, search_results: List[Dict[str, Any]]) -> str:
        """从搜索结果构建上下文"""
        if not search_results:
            return "未找到相关医学文献。"

        context_parts = []

        for i, result in enumerate(search_results[:5], 1):  # 取前5个结果
            content = result.get('content', '')
            chapter_title = result.get('chapter_title', '')
            section_title = result.get('section_title', '')
            page_number = result.get('page_number', 0)
            score = result.get('score', 0)

            # 构建上下文片段
            context_part = f"""文档 {i} (页面: {page_number}, 分数: {score:.3f}):
章节: {chapter_title}
小节: {section_title}
内容: {content[:500]}{'...' if len(content) > 500 else ''}
"""
            context_parts.append(context_part)

        return "\n---\n".join(context_parts)

    def process_query(self, question: str, user_id: str = "default", search_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理查询（兼容现有API）"""
        logger.info(f"🚀 开始处理LangGraph查询: '{question}'")

        start_time = datetime.now()

        try:
            # 初始化状态
            initial_state = {
                "question": question,
                "original_question": question,
                "messages": [HumanMessage(content=question)],
                "search_results": [],
                "current_step": "start",
                "iteration_count": 0,
                "max_iterations": self.max_iterations,
                "metadata": {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "search_config": search_config or {}
                }
            }

            # 执行图
            result = self.graph.invoke(initial_state)

            # 计算响应时间
            response_time = (datetime.now() - start_time).total_seconds()

            # 构建响应
            response = {
                "query_id": f"langgraph_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(question) % 10000}",
                "question": question,
                "answer": result.get("final_answer", "无法生成答案"),
                "confidence": result.get("confidence", 0.0),
                "reasoning_steps": self._extract_reasoning_steps(result),
                "retrieved_documents": result.get("search_results", []),
                "response_time": response_time,
                "model_used": "langgraph_react",
                "metadata": {
                    "query_type": result.get("query_type", "unknown"),
                    "entities": result.get("entities", []),
                    "iteration_count": result.get("iteration_count", 0),
                    "result_quality": result.get("result_quality", 0.0)
                }
            }

            logger.info(f"✅ LangGraph查询处理完成，迭代次数: {result.get('iteration_count', 0)}")
            return response

        except Exception as e:
            logger.error(f"❌ LangGraph查询处理失败: {e}")
            return {
                "query_id": f"error_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "question": question,
                "answer": f"处理查询时出错: {str(e)}",
                "confidence": 0.0,
                "reasoning_steps": [],
                "retrieved_documents": [],
                "response_time": 0,
                "model_used": "langgraph_react",
                "metadata": {"error": str(e)}
            }

    def _extract_reasoning_steps(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从结果中提取推理步骤"""
        messages = result.get("messages", [])
        reasoning_steps = []

        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage):
                step_info = {
                    "step": f"step_{i}",
                    "thought": msg.content,
                    "timestamp": datetime.now().isoformat()
                }
                reasoning_steps.append(step_info)

        return reasoning_steps

    def process_question_sync(self, question: str) -> Dict[str, Any]:
        """同步处理问题（兼容现有API）"""
        return self.process_query(question, user_id="default", search_config={})

# 创建Agent实例的工厂函数
def create_langgraph_react_agent(llm_manager=None,
                               retrieval_manager=None,
                               embedding_manager=None,
                               max_iterations: int = 3) -> LangGraphReActAgent:
    """创建LangGraph ReAct Agent实例"""
    return LangGraphReActAgent(
        llm_manager=llm_manager,
        retrieval_manager=retrieval_manager,
        embedding_manager=embedding_manager,
        max_iterations=max_iterations
    )
