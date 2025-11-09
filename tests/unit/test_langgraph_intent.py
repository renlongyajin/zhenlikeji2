import json
from types import SimpleNamespace

from src.agent.langgraph_react_agent import LangGraphReActAgent


class StubLLMManager:
    def __init__(self, response_content=None, raise_error=False):
        self._response_content = response_content
        self._raise_error = raise_error

    def generate_response_sync(self, messages, **kwargs):
        if self._raise_error:
            raise RuntimeError("stub failure")
        content = self._response_content or json.dumps(
            {
                "query_type": "medical",
                "need_search": True,
                "keywords": ["默认", "关键词"],
                "summary": "回退响应",
            },
            ensure_ascii=False,
        )
        return SimpleNamespace(content=content)


class StubRetrievalManager:
    def enhanced_search(self, query, config):
        return []

    def search(self, query, **kwargs):
        return []


def _build_agent(llm_response=None, raise_error=False):
    llm_manager = StubLLMManager(response_content=llm_response, raise_error=raise_error)
    retrieval_manager = StubRetrievalManager()
    return LangGraphReActAgent(
        llm_manager=llm_manager,
        enhanced_retrieval_manager=retrieval_manager,
        max_iterations=2,
    )


def test_classify_query_llm_success():
    response = json.dumps(
        {
            "query_type": "medical",
            "need_search": True,
            "keywords": ["肺腺癌", "ROSE", "细胞学"],
            "summary": "医学文献相关问题需要检索。",
        },
        ensure_ascii=False,
    )
    agent = _build_agent(llm_response=response)

    result = agent._classify_query("肺腺癌的ROSE检查需要注意什么？")

    assert result["query_type"] == "medical"
    assert result["need_search"] is True
    assert result["keywords"][:2] == ["肺腺癌", "ROSE"]
    assert "检索" in (result["summary"] or "")


def test_classify_query_llm_failure_uses_fallback():
    agent = _build_agent(raise_error=True)
    question = "这种情况下需要注意哪些病理指标？"

    result = agent._classify_query(question)

    assert result["query_type"] in {"medical", "general"}
    assert isinstance(result["keywords"], list)
    assert result["keywords"], "fallback should provide at least one keyword"
    # need_search 在回退模式下可能为 None
    assert "回退关键词" in (result["reason"] or "")


def test_intent_analysis_populates_state():
    response = json.dumps(
        {
            "query_type": "comparison",
            "need_search": True,
            "keywords": ["ROSE", "细胞学", "病理诊断"],
            "summary": "用户要求比较ROSE细胞学与病理诊断，需要检索。",
        },
        ensure_ascii=False,
    )
    agent = _build_agent(llm_response=response)

    question = "请比较ROSE细胞学和常规病理诊断的优势？"
    initial_state = {
        "question": question,
        "original_question": question,
        "messages": [],
        "query_type": "",
        "entities": [],
        "keywords": [],
        "need_search": True,
        "current_step": "",
        "tool_calls": [],
        "search_results": [],
        "result_quality": 0.0,
        "iteration_count": 0,
        "max_iterations": 2,
        "final_answer": None,
        "confidence": 0.0,
        "metadata": {},
    }

    updated_state = agent._intent_analysis_node(initial_state)

    assert updated_state["query_type"] == "comparison"
    assert updated_state["need_search"] is True
    assert updated_state["keywords"][:2] == ["ROSE", "细胞学"]
    assert updated_state["entities"] == updated_state["keywords"]
    assert updated_state["current_step"] == "intent_analysis"
    assert updated_state["iteration_count"] == 0
    assert updated_state["messages"], "intent analysis should log a summary message"
