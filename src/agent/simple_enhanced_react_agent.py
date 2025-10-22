#!/usr/bin/env python3
"""
简化增强版ReAct Agent
基于LangGraph架构，集成轻量级章节增强功能
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
import logging
import json
import asyncio
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from .simple_chapter_enhancer import SimpleChapterEnhancer, get_chapter_enhancer
    from .enhanced_retrieval_manager import EnhancedMedicalRetrievalManager
    from .llm_manager import LLMManager
except ImportError:
    try:
        from simple_chapter_enhancer import SimpleChapterEnhancer, get_chapter_enhancer
        from enhanced_retrieval_manager import EnhancedMedicalRetrievalManager
        from llm_manager import LLMManager
    except ImportError:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from simple_chapter_enhancer import SimpleChapterEnhancer, get_chapter_enhancer
        from enhanced_retrieval_manager import EnhancedMedicalRetrievalManager
        from llm_manager import LLMManager

class SimpleEnhancedAgentState(TypedDict):
    """简化增强代理状态定义"""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    original_question: str
    query_type: str  # 查询类型
    entities: List[str]  # 识别的实体列表
    search_results: List[Dict[str, Any]]  # 搜索结果
    result_quality: float  # 结果质量评分
    iteration_count: int  # 迭代次数
    max_iterations: int  # 最大迭代次数
    final_answer: Optional[str]  # 最终答案
    confidence: float  # 置信度
    metadata: Dict[str, Any]  # 元数据

class ToolNode:
    """工具节点实现"""

    def __init__(self, tools: list):
        self.tools = {tool.name: tool for tool in tools}

    def __call__(self, state: SimpleEnhancedAgentState) -> Dict[str, Any]:
        """执行工具调用"""
        messages = state.get("messages", [])
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

class SimpleEnhancedReActAgent:
    """简化增强版ReAct Agent"""

    def __init__(self,
                 llm_manager=None,
                 retrieval_manager=None,
                 enhanced_retrieval_manager=None,
                 max_iterations: int = 2):
        """初始化简化增强代理"""
        self.llm_manager = llm_manager
        self.retrieval_manager = retrieval_manager
        self.enhanced_retrieval_manager = enhanced_retrieval_manager
        self.max_iterations = max_iterations

        # 初始化章节增强器
        self.chapter_enhancer = get_chapter_enhancer()

        # 初始化工具
        self.tools = self._initialize_tools()
        self.tool_node = ToolNode(self.tools)

        # 构建图
        self._build_graph()

        logger.info("✅ 简化增强ReAct Agent 初始化完成")

    def _initialize_tools(self):
        """初始化工具"""

        @tool
        def enhanced_rag_search(query: str,
                              search_type: str = "hybrid",
                              max_results: int = 5,
                              use_chapter_boost: bool = True) -> Dict[str, Any]:
            """增强版RAG搜索，集成章节增强功能"""
            try:
                logger.info(f"🔍 增强RAG搜索: '{query}' (类型: {search_type})")

                # 优先使用增强版检索管理器
                retrieval_mgr = self.enhanced_retrieval_manager or self.retrieval_manager

                if not retrieval_mgr:
                    return {
                        "success": False,
                        "error": "检索管理器未初始化"
                    }

                # 1. 生成增强查询
                enhanced_queries = self.chapter_enhancer.enhance_search_queries(query)
                logger.info(f"📚 增强查询: {enhanced_queries}")

                # 2. 执行多个查询并合并结果
                all_results = []
                for search_query in enhanced_queries:
                    search_config = {
                        'search_type': search_type,
                        'top_k': max_results,
                        'title_priority': True
                    }

                    # 执行搜索
                    if hasattr(retrieval_mgr, 'enhanced_search'):
                        results = retrieval_mgr.enhanced_search(search_query, search_config)
                    else:
                        results = retrieval_mgr.search(search_query, **search_config)

                    all_results.extend(results)

                # 3. 去重（基于内容）
                unique_results = self._deduplicate_results(all_results)

                # 4. 应用章节增强评分
                if use_chapter_boost:
                    boosted_results = self.chapter_enhancer.boost_result_scores(unique_results, query)
                else:
                    boosted_results = unique_results

                # 5. 格式化结果
                formatted_results = []
                for result in boosted_results[:max_results]:
                    if isinstance(result, dict):
                        formatted_results.append({
                            'content': result.get('content', ''),
                            'page_number': result.get('page_number', 0),
                            'chapter_title': result.get('chapter_title', ''),
                            'section_title': result.get('section_title', ''),
                            'score': result.get('score', 0.0),
                            'chapter_boost_score': result.get('chapter_boost_score', 0.0),
                            'source': result.get('source', 'unknown'),
                            'search_type': result.get('search_type', search_type)
                        })
                    else:
                        # 处理对象格式的结果
                        formatted_results.append({
                            'content': getattr(result, 'content', ''),
                            'page_number': getattr(result, 'page_number', 0),
                            'chapter_title': getattr(result, 'chapter_title', ''),
                            'section_title': getattr(result, 'section_title', ''),
                            'score': getattr(result, 'score', 0.0),
                            'chapter_boost_score': 0.0,
                            'source': getattr(result, 'source', 'unknown'),
                            'search_type': getattr(result, 'search_type', search_type)
                        })

                logger.info(f"✅ 增强搜索完成，找到 {len(formatted_results)} 个结果")

                return {
                    "success": True,
                    "query": query,
                    "enhanced_queries": enhanced_queries,
                    "results": formatted_results,
                    "count": len(formatted_results),
                    "chapter_boost_applied": use_chapter_boost,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"❌ 增强RAG搜索失败: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @tool
        def simple_quality_evaluate(search_results: List[Dict[str, Any]],
                                  original_query: str) -> Dict[str, Any]:
            """简化的质量评估"""
            try:
                logger.info("🔍 执行简化质量评估")

                if not search_results:
                    return {
                        "success": True,
                        "quality_score": 0.0,
                        "needs_optimization": True,
                        "feedback": "搜索结果为空"
                    }

                # 简化版质量评分
                quality_score = 0.0
                feedback_items = []

                # 1. 结果数量 (20%)
                result_count = len(search_results)
                if result_count >= 3:
                    quality_score += 0.2
                else:
                    feedback_items.append("结果数量较少")

                # 2. 内容相关性 (40%)
                relevant_results = 0
                query_terms = set(original_query.lower().split())

                for result in search_results:
                    content = result.get('content', '').lower()
                    title = (result.get('chapter_title', '') + ' ' + result.get('section_title', '')).lower()

                    # 简单的相关性检查
                    if any(term in content or term in title for term in query_terms):
                        relevant_results += 1

                relevance_ratio = relevant_results / result_count if result_count > 0 else 0
                quality_score += relevance_ratio * 0.4

                if relevance_ratio < 0.5:
                    feedback_items.append("相关性较低")

                # 3. 内容质量 (40%)
                quality_results = 0
                for result in search_results:
                    content = result.get('content', '')
                    # 简单的质量指标
                    if (len(content) > 100 and
                        any(punct in content for punct in ['。', '；', '：']) and
                        any(term in content for term in ['细胞', '病理', '诊断'])):
                        quality_results += 1

                quality_ratio = quality_results / result_count if result_count > 0 else 0
                quality_score += quality_ratio * 0.4

                if quality_ratio < 0.3:
                    feedback_items.append("内容质量较低")

                # 综合评估
                needs_optimization = quality_score < 0.6

                logger.info(f"质量评估完成: 分数={quality_score:.2f}, 需要优化={needs_optimization}")

                return {
                    "success": True,
                    "quality_score": quality_score,
                    "needs_optimization": needs_optimization,
                    "feedback": "; ".join(feedback_items) if feedback_items else "质量良好",
                    "details": {
                        "relevance_ratio": relevance_ratio,
                        "quality_ratio": quality_ratio,
                        "result_count": result_count
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

        return [enhanced_rag_search, simple_quality_evaluate]

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重搜索结果"""
        seen_content = set()
        unique_results = []

        for result in results:
            # 使用内容的前100字符作为去重键
            content_key = result.get('content', '')[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)

        logger.info(f"🔄 去重: {len(results)} -> {len(unique_results)}")
        return unique_results

    def _build_graph(self):
        """构建简化版LangGraph工作流"""
        workflow = StateGraph(SimpleEnhancedAgentState)

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
                "continue": "tool_selection",
                "sufficient": "response_generation"
            }
        )

        workflow.add_edge("response_generation", END)

        # 设置入口点
        workflow.set_entry_point("intent_analysis")

        self.graph = workflow.compile()
        logger.info("✅ 简化增强工作流构建完成")

    def _intent_analysis_node(self, state: SimpleEnhancedAgentState) -> Dict[str, Any]:
        """意图分析节点"""
        logger.info("🧠 执行意图分析节点...")

        question = state["question"]

        # 简化的实体提取
        entities = self._extract_entities(question)
        query_type = self._determine_query_type(question)

        logger.info(f"✅ 意图分析完成 - 类型: {query_type}, 实体: {entities}")

        return {
            "query_type": query_type,
            "entities": entities,
            "iteration_count": 0,
            "max_iterations": self.max_iterations
        }

    def _tool_selection_node(self, state: SimpleEnhancedAgentState) -> Dict[str, Any]:
        """工具选择节点"""
        logger.info("🛠️ 执行工具选择节点...")

        question = state["question"]
        iteration_count = state.get("iteration_count", 0)
        search_results = state.get("search_results", [])

        # 简单的工具选择逻辑
        if iteration_count == 0 or len(search_results) < 3:
            # 第一轮或结果不足，使用增强搜索
            tool_call = {
                "name": "enhanced_rag_search",
                "args": {
                    "query": question,
                    "search_type": "hybrid",
                    "max_results": 5,
                    "use_chapter_boost": True
                }
            }
        else:
            # 后续轮次，根据质量评估结果调整策略
            tool_call = {
                "name": "enhanced_rag_search",
                "args": {
                    "query": question,
                    "search_type": "semantic",
                    "max_results": 5,
                    "use_chapter_boost": True
                }
            }

        return {
            "tool_calls": [tool_call],
            "iteration_count": iteration_count
        }

    def _result_observation_node(self, state: SimpleEnhancedAgentState) -> Dict[str, Any]:
        """结果观察节点"""
        logger.info("👁️ 执行结果观察节点...")

        messages = state.get("messages", [])
        search_results = state.get("search_results", [])

        logger.info(f"📋 当前消息数量: {len(messages)}")
        logger.info(f"📋 当前搜索结果数量: {len(search_results)}")

        # 从工具消息中提取搜索结果
        for i, msg in enumerate(messages):
            logger.info(f"📨 检查消息 {i+1}: 类型={type(msg).__name__}")
            if isinstance(msg, ToolMessage) and hasattr(msg, 'content'):
                logger.info(f"📨 消息 {i+1} 是ToolMessage，工具名: {msg.name}")
                try:
                    result_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    logger.info(f"📨 消息 {i+1} 解析成功，包含键: {list(result_data.keys())}")
                    logger.info(f"📨 消息 {i+1} success值: {result_data.get('success', False)}")

                    # 修复：检查工具名应该在ToolMessage的name属性中，而不是content中
                    if msg.name == "enhanced_rag_search" and result_data.get("success"):
                        search_results = result_data.get("results", [])
                        logger.info(f"✅ 从工具消息 {i+1} 中提取到 {len(search_results)} 个搜索结果")
                        break
                    else:
                        logger.info(f"⚠️ 消息 {i+1} 不符合提取条件: 工具名={msg.name}, success={result_data.get('success', False)}")
                except Exception as e:
                    logger.error(f"解析消息 {i+1} 失败: {e}")
                    logger.error(f"消息 {i+1} 内容: {str(msg.content)[:200]}...")
                    continue
            else:
                logger.info(f"📨 消息 {i+1} 不是ToolMessage或没有content属性")

        logger.info(f"📊 最终提取到的搜索结果数量: {len(search_results)}")
        return {
            "search_results": search_results
        }

    def _quality_evaluation_node(self, state: SimpleEnhancedAgentState) -> Dict[str, Any]:
        """质量评估节点"""
        logger.info("📊 执行质量评估节点...")

        search_results = state.get("search_results", [])
        question = state["question"]
        iteration_count = state.get("iteration_count", 0)

        # 简化的质量评估
        if not search_results:
            return {
                "result_quality": 0.0,
                "needs_optimization": True,
                "iteration_count": iteration_count + 1
            }

        # 基础质量评分
        quality_score = 0.0

        # 结果数量
        if len(search_results) >= 3:
            quality_score += 0.3

        # 平均分数
        avg_score = sum(r.get('score', 0) for r in search_results) / len(search_results)
        quality_score += min(avg_score, 0.4)  # 最多0.4

        # 内容长度
        avg_length = sum(len(r.get('content', '')) for r in search_results) / len(search_results)
        if avg_length > 150:
            quality_score += 0.3

        needs_optimization = quality_score < 0.6 and iteration_count < self.max_iterations - 1

        logger.info(f"质量评估: 分数={quality_score:.2f}, 需要优化={needs_optimization}")

        return {
            "result_quality": quality_score,
            "needs_optimization": needs_optimization,
            "iteration_count": iteration_count + 1
        }

    def _response_generation_node(self, state: SimpleEnhancedAgentState) -> Dict[str, Any]:
        """响应生成节点"""
        logger.info("📝 执行响应生成节点...")

        question = state["question"]
        search_results = state.get("search_results", [])
        result_quality = state.get("result_quality", 0.0)

        if not search_results or not self.llm_manager:
            return {
                "final_answer": "抱歉，未能找到相关的医学文献来回答您的问题。",
                "confidence": 0.1
            }

        # 构建上下文
        context_text = self._build_context_from_results(search_results)

        # 生成回答
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

        try:
            response_messages = [
                {"role": "system", "content": "你是一位专业的医学AI助手，基于医学文献提供准确的医学信息。"},
                {"role": "user", "content": response_prompt}
            ]

            response = self.llm_manager.generate_response_sync(response_messages)
            final_answer = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"✅ 响应生成完成，长度: {len(final_answer)} 字符")

            return {
                "final_answer": final_answer,
                "confidence": min(result_quality + 0.2, 0.9)
            }

        except Exception as e:
            logger.error(f"❌ 响应生成失败: {e}")
            return {
                "final_answer": f"抱歉，生成回答时出错: {str(e)}",
                "confidence": 0.0
            }

    def _extract_entities(self, question: str) -> List[str]:
        """提取医学实体"""
        # 使用章节增强器中的实体列表
        entities = []
        question_lower = question.lower()

        # 检查关键医学实体
        for entity in self.chapter_enhancer.key_medical_entities:
            if entity in question_lower:
                entities.append(entity)

        return entities

    def _determine_query_type(self, question: str) -> str:
        """确定查询类型"""
        question_lower = question.lower()

        # 医学关键词
        medical_keywords = ['癌', '细胞', '病理', '诊断', 'ROSE', '图像', '特征']

        if any(keyword in question_lower for keyword in medical_keywords):
            return "medical"

        # 对比类查询
        comparison_keywords = ['区别', '差异', '不同', '比较', 'vs']
        if any(keyword in question_lower for keyword in comparison_keywords):
            return "comparison"

        return "general"

    def _should_use_tools(self, state: SimpleEnhancedAgentState) -> str:
        """判断是否需要使用工具"""
        query_type = state.get("query_type", "unknown")
        entities = state.get("entities", [])

        # 医学相关查询或包含实体，需要使用工具
        if query_type in ["medical", "comparison"] or len(entities) > 0:
            return "use_tools"

        return "direct_response"

    def _should_continue_search(self, state: SimpleEnhancedAgentState) -> str:
        """判断是否需要继续搜索"""
        result_quality = state.get("result_quality", 0.0)
        needs_optimization = state.get("needs_optimization", False)
        iteration_count = state.get("iteration_count", 0)

        # 检查是否达到最大迭代次数
        if iteration_count >= self.max_iterations:
            return "sufficient"

        # 基于质量需求决策
        if needs_optimization and result_quality < 0.6:
            return "continue"

        return "sufficient"

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
            boost_score = result.get('chapter_boost_score', 0)

            # 构建上下文片段
            context_part = f"""文档 {i} (页面: {page_number}, 分数: {score:.3f}):
章节: {chapter_title}
小节: {section_title}
章节增强分数: {boost_score:.3f}
内容: {content[:400]}{'...' if len(content) > 400 else ''}
"""
            context_parts.append(context_part)

        return "\n---\n".join(context_parts)

    def process_query(self, question: str, user_id: str = "default", search_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理查询（兼容现有API）"""
        logger.info(f"🚀 开始处理简化增强查询: '{question}'")

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
                "query_id": f"simple_enhanced_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(question) % 10000}",
                "question": question,
                "answer": result.get("final_answer", "无法生成答案"),
                "confidence": result.get("confidence", 0.0),
                "reasoning_steps": self._extract_reasoning_steps(result),
                "retrieved_documents": result.get("search_results", []),
                "response_time": response_time,
                "model_used": "simple_enhanced_react",
                "metadata": {
                    "query_type": result.get("query_type", "unknown"),
                    "entities": result.get("entities", []),
                    "iteration_count": result.get("iteration_count", 0),
                    "result_quality": result.get("result_quality", 0.0),
                    "search_results_count": len(result.get("search_results", []))
                }
            }

            logger.info(f"✅ 简化增强查询处理完成，迭代次数: {result.get('iteration_count', 0)}")
            return response

        except Exception as e:
            logger.error(f"❌ 简化增强查询处理失败: {e}")
            return {
                "query_id": f"error_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "question": question,
                "answer": f"处理查询时出错: {str(e)}",
                "confidence": 0.0,
                "reasoning_steps": [],
                "retrieved_documents": [],
                "response_time": 0,
                "model_used": "simple_enhanced_react",
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
def create_simple_enhanced_react_agent(llm_manager=None,
                                     retrieval_manager=None,
                                     enhanced_retrieval_manager=None,
                                     max_iterations: int = 2) -> SimpleEnhancedReActAgent:
    """创建简化增强ReAct Agent实例"""
    return SimpleEnhancedReActAgent(
        llm_manager=llm_manager,
        retrieval_manager=retrieval_manager,
        enhanced_retrieval_manager=enhanced_retrieval_manager,
        max_iterations=max_iterations
    )