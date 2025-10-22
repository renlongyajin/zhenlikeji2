#!/usr/bin/env python3
"""
大语言模型管理器
支持多种LLM API的集成和管理
"""

import openai
import requests
import json
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """LLM响应数据结构"""
    content: str
    model: str
    usage: Dict[str, int]
    response_time: float
    timestamp: str

class BaseLLMProvider(ABC):
    """LLM提供者基类"""

    @abstractmethod
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """生成响应"""
        pass

    @abstractmethod
    def generate_response_sync(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """同步生成响应"""
        pass

class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API提供者"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        """初始化DeepSeek提供者"""
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self.sync_client = None

        # 修复：使用最基础的客户端初始化方式
        try:
            # 使用基础参数创建客户端，避免任何可能导致问题的参数
            import openai

            # 创建同步客户端
            self.sync_client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )

            # 尝试创建异步客户端
            try:
                self.client = openai.AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
            except Exception as async_error:
                logger.warning(f"异步客户端创建失败，将使用同步客户端: {async_error}")
                self.client = None

        except Exception as e:
            logger.error(f"DeepSeek客户端初始化失败: {e}")
            # 如果所有客户端都创建失败，则回退到模拟模式
            logger.warning("⚠️  DeepSeek客户端创建失败，将使用模拟模式")
            self.client = None
            self.sync_client = None

    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """异步生成响应"""
        start_time = datetime.now()

        try:
            # 如果异步客户端不可用，使用同步客户端
            if self.client is None:
                logger.info("🔄 异步客户端不可用，使用同步客户端")
                return self.generate_response_sync(messages, **kwargs)

            response = await self.client.chat.completions.create(
                model=kwargs.get('model', 'deepseek-reasoner'),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000),
                stream=False
            )

            response_time = (datetime.now() - start_time).total_seconds()

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                response_time=response_time,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {e}")
            raise

    def generate_response_sync(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """同步生成响应"""
        start_time = datetime.now()

        # 如果同步客户端不可用，使用模拟响应
        if self.sync_client is None:
            logger.warning("⚠️  DeepSeek同步客户端不可用，使用模拟响应")
            return self._generate_mock_response(messages, start_time, **kwargs)

        try:
            response = self.sync_client.chat.completions.create(
                model=kwargs.get('model', 'deepseek-reasoner'),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000),
                stream=False
            )

            response_time = (datetime.now() - start_time).total_seconds()

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                response_time=response_time,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {e}，使用模拟响应")
            return self._generate_mock_response(messages, start_time, **kwargs)

    def _generate_mock_response(self, messages: List[Dict[str, str]], start_time, **kwargs) -> LLMResponse:
        """生成模拟响应"""
        import time

        # 模拟延迟
        time.sleep(0.5)

        # 获取最后一个用户消息
        last_user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break

        response_time = (datetime.now() - start_time).total_seconds()

        return LLMResponse(
            content=f"【DeepSeek模拟响应】基于您的问题：{last_user_message[:100]}...，这是一个专业的医学回答。",
            model="deepseek-reasoner-mock",
            usage={
                'prompt_tokens': len(str(messages)),
                'completion_tokens': 50,
                'total_tokens': len(str(messages)) + 50
            },
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )

class QwenProvider(BaseLLMProvider):
    """千问API提供者 - 支持阿里云DashScope和硅基流动API"""

    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/api/v1", model_name: str = None):
        """初始化千问提供者

        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称，支持qwen-max、qwen3-80b等，默认为qwen-max
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name or os.getenv("QWEN3_MODEL", "qwen-max")  # 支持环境变量配置
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        # 检测是否为硅基流动API
        self.is_siliconflow = 'siliconflow' in base_url or 'api.siliconflow.cn' in base_url

    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """异步生成响应"""
        start_time = datetime.now()

        try:
            if self.is_siliconflow:
                # 硅基流动API使用OpenAI兼容格式
                payload = {
                    'model': kwargs.get('model', self.model_name),
                    'messages': messages,
                    'temperature': kwargs.get('temperature', 0.7),
                    'max_tokens': kwargs.get('max_tokens', 2000)
                }

                response = requests.post(
                    f'{self.base_url}/chat/completions',
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
            else:
                # 阿里云DashScope API使用原始格式
                payload = {
                    'model': kwargs.get('model', self.model_name),
                    'input': {
                        'messages': messages
                    },
                    'parameters': {
                        'temperature': kwargs.get('temperature', 0.7),
                        'max_tokens': kwargs.get('max_tokens', 2000)
                    }
                }

                response = requests.post(
                    f'{self.base_url}/services/aigc/text-generation/generation',
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )

            response_time = (datetime.now() - start_time).total_seconds()

            if response.status_code == 200:
                result = response.json()

                if self.is_siliconflow:
                    # 硅基流动API响应格式（OpenAI兼容）
                    content = result['choices'][0]['message']['content']
                    usage = result.get('usage', {})
                    model_name = result.get('model', self.model_name)
                else:
                    # 阿里云DashScope API响应格式
                    if 'output' in result and 'text' in result['output']:
                        # 新的响应格式
                        content = result['output']['text']
                    elif 'output' in result and 'choices' in result['output'] and len(result['output']['choices']) > 0:
                        # 旧的响应格式
                        content = result['output']['choices'][0]['message']['content']
                    else:
                        # 其他格式，尝试直接访问
                        content = str(result.get('output', result))
                    usage = result.get('usage', {})
                    model_name = result.get('model', self.model_name)

                return LLMResponse(
                    content=content,
                    model=model_name,
                    usage=usage,
                    response_time=response_time,
                    timestamp=datetime.now().isoformat()
                )
            else:
                raise Exception(f"API调用失败: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"千问API调用失败: {e}")
            raise

    def generate_response_sync(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """同步生成响应"""
        # 避免在已有事件循环中使用asyncio.run()
        import concurrent.futures

        def run_async():
            """在单独线程中运行异步函数"""
            return asyncio.run(self.generate_response(messages, **kwargs))

        # 使用线程池在单独线程中运行异步代码
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async)
            return future.result()

class MockLLMProvider(BaseLLMProvider):
    """模拟LLM提供者（用于测试）"""

    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """异步生成模拟响应"""
        import asyncio
        await asyncio.sleep(0.1)  # 模拟延迟

        # 生成模拟响应
        last_message = messages[-1]['content'] if messages else "测试消息"
        mock_response = f"这是一个模拟的LLM响应，针对问题：{last_message[:50]}..."

        return LLMResponse(
            content=mock_response,
            model="mock-llm",
            usage={'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
            response_time=0.1,
            timestamp=datetime.now().isoformat()
        )

    def generate_response_sync(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """同步生成模拟响应"""
        # 生成模拟响应
        last_message = messages[-1]['content'] if messages else "测试消息"
        mock_response = f"这是一个模拟的LLM响应，针对问题：{last_message[:50]}..."

        return LLMResponse(
            content=mock_response,
            model="mock-llm",
            usage={'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
            response_time=0.01,
            timestamp=datetime.now().isoformat()
        )

class LLMManager:
    """LLM管理器"""

    def __init__(self, config: Dict[str, Any]):
        """初始化LLM管理器"""
        self.config = config
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.active_provider = config.get('default_provider', 'mock')
        self._initialize_providers()

    def _initialize_providers(self):
        """初始化各种LLM提供者"""
        # DeepSeek提供者
        if 'deepseek' in self.config:
            try:
                self.providers['deepseek'] = DeepSeekProvider(
                    api_key=self.config['deepseek']['api_key'],
                    base_url=self.config['deepseek'].get('base_url', 'https://api.deepseek.com')
                )
                logger.info("✅ DeepSeek提供者初始化成功")
            except Exception as e:
                logger.error(f"❌ DeepSeek提供者初始化失败: {e}")

        # 千问提供者 (qwen-max)
        if 'qwen' in self.config:
            try:
                self.providers['qwen'] = QwenProvider(
                    api_key=self.config['qwen']['api_key'],
                    base_url=self.config['qwen'].get('base_url', 'https://dashscope.aliyuncs.com/api/v1'),
                    model_name='qwen3-max'  # 明确指定qwen3-max模型
                )
                logger.info("✅ 千问提供者 (qwen-max) 初始化成功")
            except Exception as e:
                logger.error(f"❌ 千问提供者初始化失败: {e}")

        # 千问3-80b提供者
        if 'qwen-80b' in self.config:
            try:
                self.providers['qwen-80b'] = QwenProvider(
                    api_key=self.config['qwen-80b']['api_key'],
                    base_url=self.config['qwen-80b'].get('base_url', 'https://dashscope.aliyuncs.com/api/v1'),
                    model_name='Qwen/Qwen3-Next-80B-A3B-Thinking'  # 硅基流动的qwen3-80b模型名称
                )
                logger.info("✅ 千问3-80b提供者初始化成功")
            except Exception as e:
                logger.error(f"❌ 千问3-80b提供者初始化失败: {e}")

        # 模拟提供者（默认）
        self.providers['mock'] = MockLLMProvider()
        logger.info("✅ 模拟LLM提供者初始化成功")

    def set_active_provider(self, provider_name: str):
        """设置活动的LLM提供者"""
        if provider_name in self.providers:
            self.active_provider = provider_name
            logger.info(f"✅ 切换到LLM提供者: {provider_name}")
        else:
            logger.warning(f"⚠️ 提供者 {provider_name} 不可用，使用默认提供者")
            self.active_provider = 'mock'

    async def generate_response(self, messages, **kwargs) -> LLMResponse:
        """生成响应"""
        provider = self.providers.get(self.active_provider)
        if not provider:
            raise ValueError(f"LLM提供者 {self.active_provider} 未找到")

        # 转换消息格式
        converted_messages = self._convert_messages(messages)

        logger.info(f"🤖 使用 {self.active_provider} 提供者生成响应")
        return await provider.generate_response(converted_messages, **kwargs)

    def _convert_messages(self, messages):
        """转换消息格式 - 支持LangChain Message对象和字典格式"""
        converted_messages = []

        for msg in messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                # LangChain Message对象
                if msg.type == 'system':
                    role = 'system'
                elif msg.type == 'human':
                    role = 'user'
                elif msg.type == 'ai':
                    role = 'assistant'
                elif msg.type == 'tool':
                    role = 'tool'
                else:
                    role = 'user'  # 默认

                converted_messages.append({
                    'role': role,
                    'content': str(msg.content)
                })
            elif isinstance(msg, dict):
                # 字典格式
                converted_messages.append({
                    'role': msg.get('role', 'user'),
                    'content': str(msg.get('content', ''))
                })
            else:
                # 其他格式，尝试转换为字符串
                converted_messages.append({
                    'role': 'user',
                    'content': str(msg)
                })

        return converted_messages

    def generate_response_sync(self, messages, **kwargs) -> LLMResponse:
        """同步生成响应"""
        provider = self.providers.get(self.active_provider)
        if not provider:
            raise ValueError(f"LLM提供者 {self.active_provider} 未找到")

        # 转换消息格式
        converted_messages = self._convert_messages(messages)

        logger.info(f"🤖 使用 {self.active_provider} 提供者生成响应（同步）")
        return provider.generate_response_sync(converted_messages, **kwargs)

    def generate_medical_answer(self, question: str, context: List[Dict[str, Any]], reasoning_history: List[Dict[str, Any]]) -> str:
        """生成医学答案"""
        # 构建提示消息
        messages = [
            {
                "role": "system",
                "content": """你是一位专业的医学AI助手，基于提供的医学文献和推理过程来回答用户的问题。

要求：
1. 回答必须基于提供的医学文献内容
2. 使用专业但易于理解的医学术语
3. 提供准确、可靠的医学信息
4. 如果不确定，要明确说明
5. 建议咨询专业医疗人员获取个性化建议

请根据以下信息生成回答："""
            },
            {
                "role": "user",
                "content": f"""
用户问题：{question}

相关医学文献：
{self._format_context(context)}

推理过程：
{self._format_reasoning_history(reasoning_history)}

请基于以上信息提供专业的医学回答："""
            }
        ]

        try:
            response = self.generate_response_sync(messages, temperature=0.3)
            return response.content
        except Exception as e:
            logger.error(f"❌ 医学答案生成失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"

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

    def get_provider_status(self) -> Dict[str, Any]:
        """获取提供者状态"""
        status = {
            'active_provider': self.active_provider,
            'available_providers': list(self.providers.keys()),
            'provider_details': {}
        }

        for name, provider in self.providers.items():
            status['provider_details'][name] = {
                'type': provider.__class__.__name__,
                'status': 'active'
            }

        return status

# 创建默认配置
def create_default_llm_config() -> Dict[str, Any]:
    """创建默认LLM配置"""
    import os

    # 从环境变量读取API密钥
    deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY', '')
    qwen_api_key = os.environ.get('QWEN_API_KEY', '') or os.environ.get('DASHSCOPE_API_KEY', '')
    qwen3_80b_api_key = os.environ.get('QWEN3_80B_API_KEY', '') or qwen_api_key  # 复用qwen_api_key作为备选

    config = {
        'default_provider': 'mock',
        'deepseek': {
            'api_key': deepseek_api_key or 'your-deepseek-api-key',
            'base_url': 'https://api.deepseek.com',
            'model': 'deepseek-reasoner'
        },
        'qwen': {
            'api_key': qwen_api_key or 'your-qwen-api-key',
            'base_url': 'https://dashscope.aliyuncs.com/api/v1',
            'model': 'qwen3-max'  # 正确的qwen3-max模型名称
        }
    }

    # 添加千问3-80b模型配置（使用硅基流动API）
    siliconflow_api_key = os.environ.get('SILICONFLOW_API_KEY', '')
    if siliconflow_api_key and len(siliconflow_api_key) > 10:
        config['qwen-80b'] = {
            'api_key': siliconflow_api_key,
            'base_url': 'https://api.siliconflow.cn/v1',
            'model': 'Qwen/Qwen3-Next-80B-A3B-Thinking'  # 硅基流动的qwen3-80b模型
        }

    return config

def create_llm_manager(config: Optional[Dict[str, Any]] = None) -> LLMManager:
    """创建LLM管理器"""
    if config is None:
        config = create_default_llm_config()

    return LLMManager(config)