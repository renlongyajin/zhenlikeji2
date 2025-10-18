#!/usr/bin/env python3
"""
ReAct智能代理框架
基于LangGraph的推理+行动智能代理系统
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
try:
    # 尝试新版本的导入方式
    from langgraph.prebuilt.tool_node import ToolNode
except ImportError:
    # 回退到旧版本的导入方式
    try:
        from langgraph.prebuilt import ToolNode
    except ImportError:
        # 如果都不存在，创建一个兼容的ToolNode
        from langgraph.prebuilt import ToolExecutor
        ToolNode = ToolExecutor

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
import json
import asyncio
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """代理状态定义"""
    messages: List[BaseMessage]
    question: str
    context: List[Dict[str, Any]]
    retrieved_docs: List[Dict[str, Any]]
    search_queries: List[str]
    current_step: str
    reasoning_steps: List[Dict[str, Any]]
    final_answer: Optional[str]
    confidence: float
    tool_calls: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class MedicalReActAgent:
    """医学ReAct智能代理"""

    def __init__(self,
                 llm_manager=None,
                 retrieval_manager=None,
                 embedding_manager=None):
        """初始化ReAct代理"""
        self.llm_manager = llm_manager
        self.retrieval_manager = retrieval_manager
        self.embedding_manager = embedding_manager
        self.graph = None
        self.tools = self._create_tools()
        self._build_graph()

    def _create_tools(self) -> List:
        """创建工具列表"""

        @tool
        def search_medical_documents(query: str, search_type: str = "hybrid") -> Dict[str, Any]:
            """搜索医学文档

            Args:
                query: 搜索查询词
                search_type: 搜索类型 (keyword, semantic, hybrid)

            Returns:
                搜索结果字典
            """
            try:
                logger.info(f"🔍 执行医学文档搜索: '{query}' (类型: {search_type})")

                if self.retrieval_manager:
                    results = self.retrieval_manager.search(
                        query=query,
                        search_type=search_type,
                        top_k=10
                    )

                    # 格式化结果
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            'content': result.get('content', ''),
                            'score': result.get('score', 0.0),
                            'source': result.get('source', ''),
                            'page_number': result.get('page_number', 0),
                            'chapter_title': result.get('chapter_title', ''),
                            'section_title': result.get('section_title', '')
                        })

                    logger.info(f"✅ 搜索完成，找到 {len(formatted_results)} 个结果")
                    return {
                        'success': True,
                        'query': query,
                        'search_type': search_type,
                        'results': formatted_results,
                        'count': len(formatted_results)
                    }
                else:
                    return {
                        'success': False,
                        'error': '检索管理器未初始化'
                    }

            except Exception as e:
                logger.error(f"❌ 搜索失败: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }

        @tool
        def analyze_medical_content(content: str, analysis_type: str = "general") -> Dict[str, Any]:
            """分析医学内容

            Args:
                content: 要分析的医学内容
                analysis_type: 分析类型 (general, pathology, diagnosis)

            Returns:
                分析结果字典
            """
            try:
                logger.info(f"🔬 执行医学内容分析 (类型: {analysis_type})")

                # 模拟分析逻辑
                analysis_result = {
                    'content_length': len(content),
                    'key_concepts': self._extract_key_concepts(content),
                    'medical_terms': self._extract_medical_terms(content),
                    'analysis_type': analysis_type,
                    'timestamp': datetime.now().isoformat()
                }

                logger.info(f"✅ 内容分析完成")
                return {
                    'success': True,
                    'analysis': analysis_result
                }

            except Exception as e:
                logger.error(f"❌ 内容分析失败: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }

        @tool
        def generate_embedding(text: str) -> Dict[str, Any]:
            """生成文本嵌入向量

            Args:
                text: 输入文本

            Returns:
                嵌入向量结果
            """
            try:
                if self.embedding_manager:
                    embedding = self.embedding_manager.encode_texts([text])[0]
                    return {
                        'success': True,
                        'embedding': embedding,
                        'dimension': len(embedding)
                    }
                else:
                    # 模拟嵌入向量
                    import numpy as np
                    np.random.seed(hash(text) % 1000)
                    embedding = np.random.random(768).tolist()
                    return {
                        'success': True,
                        'embedding': embedding,
                        'dimension': 768
                    }

            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }

        return [search_medical_documents, analyze_medical_content, generate_embedding]

    def _extract_key_concepts(self, content: str) -> List[str]:
        """提取关键概念"""
        # 简单的关键词提取逻辑
        medical_keywords = [
            "肺部", "肿瘤", "细胞", "病理", "诊断", "治疗",
            "恶性", "良性", "癌", "ROSE", "细胞学", "组织学"
        ]

        concepts = []
        content_lower = content.lower()
        for keyword in medical_keywords:
            if keyword in content_lower:
                concepts.append(keyword)

        return concepts[:5]  # 返回前5个

    def _extract_medical_terms(self, content: str) -> List[str]:
        """提取医学术语"""
        # 简单的医学术语提取
        terms = []
        if "腺癌" in content:
            terms.append("腺癌")
        if "鳞癌" in content:
            terms.append("鳞癌")
        if "ROSE" in content:
            terms.append("ROSE")
        if "细胞核" in content:
            terms.append("细胞核")
        if "细胞质" in content:
            terms.append("细胞质")

        return terms

    def _create_reasoning_prompt(self) -> ChatPromptTemplate:
        """创建推理提示模板"""
        system_message = """你是一位专业的医学AI助手，使用ReAct（推理+行动）框架来回答医学问题。

你的任务是：
1. 理解用户的医学问题
2. 制定搜索和分析策略
3. 使用工具搜索相关医学文档
4. 分析检索到的信息
5. 提供准确、专业的医学回答

可用工具：
- search_medical_documents: 搜索医学文档
- analyze_medical_content: 分析医学内容
- generate_embedding: 生成文本嵌入向量

请按照以下格式进行思考：

Thought: 我需要分析这个问题并制定策略
Action: 使用哪个工具
Action Input: 工具输入参数
Observation: 工具返回的结果

重复思考-行动循环，直到能够给出完整准确的答案。

最终答案应该：
- 基于检索到的医学文献
- 专业且易于理解
- 包含相关的医学概念解释
- 提供适当的上下文信息"""

        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{question}")
        ])

    def _build_graph(self):
        """构建LangGraph图"""
        # 创建工作流图
        workflow = StateGraph(AgentState)

        # 定义节点
        workflow.add_node("reasoner", self._reasoning_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("answer_generator", self._answer_generation_node)

        # 定义边（简化版流程）
        workflow.add_edge("reasoner", "answer_generator")
        workflow.add_edge("answer_generator", END)

        # 设置入口点
        workflow.set_entry_point("reasoner")

        # 编译图
        self.graph = workflow.compile()
        logger.info("✅ ReAct代理图构建完成")

    def _reasoning_node(self, state: AgentState) -> Dict[str, Any]:
        """推理节点"""
        logger.info("🧠 执行推理节点...")

        question = state["question"]

        # 直接执行搜索操作（简化版ReAct）
        logger.info(f"🔍 直接执行搜索: '{question}'")

        if self.retrieval_manager:
            search_results = self.retrieval_manager.search(
                query=question,
                search_type="hybrid",
                top_k=5
            )
            logger.info(f"✅ 搜索完成，找到 {len(search_results)} 个结果")
        else:
            search_results = []
            logger.warning("⚠️  检索管理器未初始化")

        # 更新状态
        reasoning_step = {
            "step": "reasoning_and_search",
            "thought": f"分析用户问题并执行搜索: {question}",
            "action": "search_medical_documents",
            "action_input": {"query": question, "search_type": "hybrid"},
            "observation": f"找到 {len(search_results)} 个相关文档",
            "timestamp": datetime.now().isoformat()
        }

        return {
            "question": question,
            "retrieved_docs": search_results,
            "reasoning_steps": state.get("reasoning_steps", []) + [reasoning_step],
            "current_step": "search_completed"
        }

    def _simulate_reasoning(self, question: str) -> str:
        """模拟推理过程"""
        return f"""Thought: 用户询问关于'{question}'的医学问题。我需要先搜索相关的医学文档来获取准确信息。

Action: search_medical_documents
Action Input: {{"query": "{question}", "search_type": "hybrid"}}

Observation: 等待搜索结果..."""

    def _answer_generation_node(self, state: AgentState) -> Dict[str, Any]:
        """答案生成节点"""
        logger.info("📝 执行答案生成节点...")

        question = state["question"]
        retrieved_docs = state.get("retrieved_docs", [])
        reasoning_steps = state.get("reasoning_steps", [])

        # 生成答案
        if self.llm_manager:
            # 构建医学回答的提示消息
            messages = [
                {
                    "role": "system",
                    "content": "你是一位专业的医学AI助手，基于提供的医学文献和推理过程来回答用户的问题。要求：1. 回答必须基于提供的医学文献内容 2. 使用专业但易于理解的医学术语 3. 提供准确、可靠的医学信息 4. 如果不确定，要明确说明 5. 建议咨询专业医疗人员获取个性化建议"
                },
                {
                    "role": "user",
                    "content": f"""
用户问题：{question}

相关医学文献：
{self._format_context(retrieved_docs)}

推理过程：
{self._format_reasoning_history(reasoning_steps)}

请基于以上信息提供专业的医学回答："""
                }
            ]

            llm_response = self.llm_manager.generate_response_sync(messages)
            answer = llm_response.content
        else:
            answer = self._generate_simulated_answer(question, retrieved_docs)

        # 更新状态
        final_state = {
            "final_answer": answer,
            "confidence": self._calculate_confidence(retrieved_docs),
            "current_step": "completed",
            "metadata": {
                "completion_time": datetime.now().isoformat(),
                "total_steps": len(reasoning_steps),
                "documents_used": len(retrieved_docs)
            }
        }

        logger.info("✅ 答案生成完成")
        return final_state

    def _generate_simulated_answer(self, question: str, docs: List[Dict]) -> str:
        """生成模拟答案"""
        return f"""基于医学文献检索，关于"{question}"的回答如下：

根据检索到的{len(docs)}个相关医学文档，我可以提供以下专业解答：

【主要发现】
- 相关医学概念和诊断要点
- 临床表现特征
- 治疗建议和注意事项

【详细信息】
由于这是模拟回答，具体的医学内容将基于实际的检索结果生成。

【建议】
请咨询专业医疗人员获取个性化的医疗建议。

此回答基于医学文献检索系统，置信度：中等"""

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """格式化上下文"""
        formatted = []
        for i, doc in enumerate(context[:5], 1):  # 只取前5个文档
            formatted.append(f"文档 {i}:")
            formatted.append(f"内容: {doc.get('content', '')[:200]}...")
            formatted.append(f"来源: {doc.get('chapter_title', '')} - {doc.get('section_title', '')}")
            formatted.append(f"页码: {doc.get('page_number', '未知')}")
            formatted.append("")
        return "\n".join(formatted)

    def _format_reasoning_history(self, reasoning_history: List[Dict[str, Any]]) -> str:
        """格式化推理历史"""
        formatted = []
        for i, step in enumerate(reasoning_history, 1):
            formatted.append(f"步骤 {i}: {step.get('thought', '')}")
            if step.get('action'):
                formatted.append(f"行动: {step['action']}")
            if step.get('observation'):
                formatted.append(f"观察: {step['observation'][:100]}...")
            formatted.append("")
        return "\n".join(formatted)

    def _calculate_confidence(self, retrieved_docs: List[Dict]) -> float:
        """计算置信度"""
        if not retrieved_docs:
            return 0.0

        # 基于检索结果数量和相关性计算置信度
        doc_count = len(retrieved_docs)
        avg_score = sum(doc.get('score', 0) for doc in retrieved_docs) / doc_count if doc_count > 0 else 0

        # 简单置信度计算
        confidence = min(0.9, (doc_count / 10) * avg_score)
        return round(confidence, 2)

    async def process_question(self, question: str) -> Dict[str, Any]:
        """处理问题"""
        logger.info(f"🚀 开始处理用户问题: {question}")

        # 初始化状态
        initial_state = {
            "messages": [],
            "question": question,
            "context": [],
            "retrieved_docs": [],
            "search_queries": [question],
            "current_step": "initial",
            "reasoning_steps": [],
            "final_answer": None,
            "confidence": 0.0,
            "tool_calls": [],
            "metadata": {}
        }

        try:
            # 执行图
            result = await self.graph.ainvoke(initial_state)

            logger.info("✅ 问题处理完成")
            return {
                "success": True,
                "question": question,
                "answer": result.get("final_answer", ""),
                "confidence": result.get("confidence", 0.0),
                "reasoning_steps": result.get("reasoning_steps", []),
                "metadata": result.get("metadata", {})
            }

        except Exception as e:
            logger.error(f"❌ 问题处理失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }

    def process_question_sync(self, question: str) -> Dict[str, Any]:
        """同步处理问题"""
        logger.info(f"🚀 开始处理用户问题: {question}")

        # 初始化状态
        initial_state = {
            "messages": [],
            "question": question,
            "context": [],
            "retrieved_docs": [],
            "search_queries": [question],
            "current_step": "initial",
            "reasoning_steps": [],
            "final_answer": None,
            "confidence": 0.0,
            "tool_calls": [],
            "metadata": {}
        }

        try:
            # 执行图
            result = self.graph.invoke(initial_state)

            logger.info("✅ 问题处理完成")
            return {
                "success": True,
                "question": question,
                "answer": result.get("final_answer", ""),
                "confidence": result.get("confidence", 0.0),
                "reasoning_steps": result.get("reasoning_steps", []),
                "metadata": result.get("metadata", {})
            }

        except Exception as e:
            logger.error(f"❌ 问题处理失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }

def create_react_agent(llm_manager=None, retrieval_manager=None, embedding_manager=None):
    """创建ReAct代理实例"""
    return MedicalReActAgent(
        llm_manager=llm_manager,
        retrieval_manager=retrieval_manager,
        embedding_manager=embedding_manager
    )