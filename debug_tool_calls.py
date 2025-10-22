#!/usr/bin/env python3
"""
调试工具调用结构
查看实际的工具调用结果格式
"""

import sys
import json
from datetime import datetime

# 添加项目路径
sys.path.append('/home/ubuntu/myproject/zhenlikeji2/src')

try:
    from agent.simple_enhanced_react_agent import SimpleEnhancedReActAgent, create_simple_enhanced_react_agent
    from agent.llm_manager import LLMManager
    from agent.enhanced_retrieval_manager import create_enhanced_retrieval_manager
    from embedding.embedding_models import get_embedding_manager
    from langchain_core.messages import ToolMessage
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

def debug_tool_message_structure():
    """调试工具消息结构"""
    print("🔍 调试工具消息结构")
    print("=" * 60)

    # 创建一个模拟的工具执行结果
    mock_tool_result = {
        "success": True,
        "query": "腺癌的图像特征是什么？",
        "enhanced_queries": ["腺癌的图像特征是什么？", "第.*节 腺癌", "腺癌 细胞形态 结构特征"],
        "results": [
            {
                "content": "腺癌的图像特征包括细胞核增大、核仁明显等。",
                "page_number": 45,
                "chapter_title": "第二章 肺部实体恶性肿瘤",
                "section_title": "第一节 腺癌",
                "score": 0.85,
                "chapter_boost_score": 0.3,
                "search_type": "hybrid"
            }
        ],
        "count": 1,
        "chapter_boost_applied": True,
        "timestamp": datetime.now().isoformat()
    }

    # 创建ToolMessage
    tool_message = ToolMessage(
        content=json.dumps(mock_tool_result),
        tool_call_id="call_enhanced_rag_search_123",
        name="enhanced_rag_search"
    )

    print("模拟ToolMessage结构:")
    print(f"  content: {tool_message.content[:100]}...")
    print(f"  tool_call_id: {tool_message.tool_call_id}")
    print(f"  name: {tool_message.name}")
    print(f"  type: {type(tool_message.content)}")

    # 解析内容
    try:
        parsed_content = json.loads(tool_message.content)
        print(f"\n解析后的内容:")
        print(f"  类型: {type(parsed_content)}")
        print(f"  键: {list(parsed_content.keys())}")
        print(f"  有name字段吗: {'name' in parsed_content}")
        print(f"  有success字段吗: {'success' in parsed_content}")
        if 'name' in parsed_content:
            print(f"  name值: {parsed_content['name']}")
    except Exception as e:
        print(f"解析失败: {e}")

def debug_actual_tool_execution():
    """调试实际的工具执行流程"""
    print("\n🔍 调试实际工具执行流程")
    print("=" * 60)

    try:
        # 初始化组件
        print("初始化组件...")
        embedding_manager = get_embedding_manager(model_type="jina")
        retrieval_manager = create_enhanced_retrieval_manager(
            es_host='localhost',
            es_port=9200,
            milvus_host='localhost',
            milvus_port=19530,
            embedding_manager=embedding_manager
        )
        llm_manager = LLMManager(config={
            'default_provider': 'deepseek',
            'deepseek': {'api_key': 'test_key', 'base_url': 'https://api.deepseek.com'}
        })

        agent = create_simple_enhanced_react_agent(
            llm_manager=llm_manager,
            enhanced_retrieval_manager=retrieval_manager,
            max_iterations=1
        )

        # 手动测试工具调用
        print("\n测试工具调用...")

        # 获取工具
        tools = agent.tools
        print(f"可用工具: {list(tools.keys())}")

        if 'enhanced_rag_search' in tools:
            tool = tools['enhanced_rag_search']
            print(f"工具类型: {type(tool)}")

            # 执行工具
            try:
                result = tool.invoke({
                    "query": "腺癌的图像特征是什么？",
                    "search_type": "hybrid",
                    "max_results": 2,
                    "use_chapter_boost": True
                })

                print(f"工具执行结果类型: {type(result)}")
                print(f"工具执行结果: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}...")

                # 检查结构
                if isinstance(result, dict):
                    print(f"结果键: {list(result.keys())}")
                    print(f"有name字段吗: {'name' in result}")
                    print(f"有success字段吗: {'success' in result}")

            except Exception as e:
                print(f"工具执行失败: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()

def debug_state_transitions():
    """调试状态转换逻辑"""
    print("\n🔍 调试状态转换逻辑")
    print("=" * 60)

    # 模拟状态转换
    print("模拟_langgraph工作流状态转换:")

    # 初始状态
    initial_state = {
        "question": "腺癌的图像特征是什么？",
        "original_question": "腺癌的图像特征是什么？",
        "messages": [],
        "search_results": [],
        "current_step": "start",
        "iteration_count": 0,
        "max_iterations": 2,
        "metadata": {}
    }

    print("1. 初始状态:")
    print(f"   - tool_calls: {initial_state.get('tool_calls', '不存在')}")
    print(f"   - search_results: {len(initial_state.get('search_results', []))}")

    # 工具选择后的状态
    tool_selection_state = {
        "tool_calls": [{
            "name": "enhanced_rag_search",
            "args": {"query": "腺癌的图像特征是什么？", "max_results": 5}
        }],
        "iteration_count": 0
    }

    print("\n2. 工具选择节点输出:")
    print(f"   - tool_calls: {tool_selection_state['tool_calls']}")

    # 工具执行后的状态
    tool_execution_state = {
        "messages": [
            {
                "type": "tool_message",
                "content": '{"success": true, "results": [...]}',
                "tool_call_id": "call_123",
                "name": "enhanced_rag_search"
            }
        ]
    }

    print("\n3. 工具执行节点输出:")
    print(f"   - messages数量: {len(tool_execution_state['messages'])}")
    if tool_execution_state['messages']:
        msg = tool_execution_state['messages'][0]
        print(f"   - 消息类型: {msg.get('type', '未知')}")
        print(f"   - 工具名: {msg.get('name', '未知')}")
        print(f"   - 内容长度: {len(str(msg.get('content', '')))}")

if __name__ == "__main__":
    print("🚀 工具调用调试脚本")
    print("=" * 60)

    debug_tool_message_structure()
    debug_actual_tool_execution()
    debug_state_transitions()

    print("\n" + "=" * 60)
    print("✅ 调试完成！")