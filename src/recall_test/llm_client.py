"""
大语言模型客户端
用于调用各种LLM API生成问题和答案
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from config import API_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

class LLMClient:
    """大语言模型客户端"""

    def __init__(self):
        self.session = None
        self.api_config = API_CONFIG
        self.model_config = MODEL_CONFIG

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate_questions_batch(self, prompts: List[str], model: str = None) -> List[str]:
        """批量生成问题"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        model = model or self.model_config["question_generation_model"]
        tasks = []

        for prompt in prompts:
            task = self._generate_single_question(prompt, model)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"生成问题 {i} 失败: {result}")
                valid_results.append("")
            else:
                valid_results.append(result)

        return valid_results

    async def _generate_single_question(self, prompt: str, model: str) -> str:
        """生成单个问题"""
        try:
            if "qwen" in model.lower():
                return await self._call_qwen_api(prompt, model)
            elif "deepseek" in model.lower():
                return await self._call_deepseek_api(prompt, model)
            else:
                # 默认使用硅基流动API
                return await self._call_siliconflow_api(prompt, model)
        except Exception as e:
            logger.error(f"生成问题失败: {e}")
            raise e

    async def _call_qwen_api(self, prompt: str, model: str) -> str:
        """调用通义千问API"""
        url = f"{self.api_config['dashscope_base_url']}/services/aigc/text-generation/generation"

        headers = {
            "Authorization": f"Bearer {self.api_config['dashscope_api_key']}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "temperature": self.model_config["temperature"],
                "max_tokens": self.model_config["max_tokens"],
                "top_p": 0.9,
                "seed": 42
            }
        }

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("output", {}).get("text", "")
            else:
                error_text = await response.text()
                raise Exception(f"千问API调用失败: {response.status} - {error_text}")

    async def _call_deepseek_api(self, prompt: str, model: str) -> str:
        """调用DeepSeek API"""
        url = f"{self.api_config['deepseek_base_url']}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_config['deepseek_api_key']}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个医学专家，专注于肺部疾病的ROSE（快速现场评价）诊断。请根据提供的信息生成专业的医学测试问题。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.model_config["temperature"],
            "max_tokens": self.model_config["max_tokens"],
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                error_text = await response.text()
                raise Exception(f"DeepSeek API调用失败: {response.status} - {error_text}")

    async def _call_siliconflow_api(self, prompt: str, model: str) -> str:
        """调用硅基流动API"""
        url = f"{self.api_config['siliconflow_base_url']}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_config['siliconflow_api_key']}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个医学教育专家，专门创建高质量的医学测试问题。请确保问题专业、准确且具有教育意义。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.model_config["temperature"],
            "max_tokens": self.model_config["max_tokens"],
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                error_text = await response.text()
                raise Exception(f"硅基流动API调用失败: {response.status} - {error_text}")

    async def validate_question_quality(self, question: str, expected_answer: str) -> Dict[str, Any]:
        """验证问题质量"""
        validation_prompt = f"""
        请评估以下医学测试问题的质量，从1-5分评分：

        问题：{question}
        期望答案：{expected_answer}

        请评估以下方面：
        1. 医学准确性（1-5分）
        2. 问题清晰度（1-5分）
        3. 教育价值（1-5分）
        4. 难度适宜性（1-5分）

        请以JSON格式返回评估结果：
        {{
            "medical_accuracy": 分数,
            "clarity": 分数,
            "educational_value": 分数,
            "difficulty_appropriateness": 分数,
            "overall_score": 平均分,
            "suggestions": "改进建议"
        }}
        """

        try:
            result = await self._call_qwen_api(validation_prompt, "qwen3-max")
            # 尝试解析JSON结果
            if result.strip().startswith('{'):
                return json.loads(result)
            else:
                # 如果返回的不是JSON，创建默认评估
                return {
                    "medical_accuracy": 4,
                    "clarity": 4,
                    "educational_value": 4,
                    "difficulty_appropriateness": 4,
                    "overall_score": 4.0,
                    "suggestions": "需要进一步验证"
                }
        except Exception as e:
            logger.error(f"问题质量验证失败: {e}")
            return {
                "medical_accuracy": 3,
                "clarity": 3,
                "educational_value": 3,
                "difficulty_appropriateness": 3,
                "overall_score": 3.0,
                "suggestions": "验证失败，需要人工审核"
            }

    async def generate_question_variations(self, base_question: str, num_variations: int = 3) -> List[str]:
        """生成问题的多种表述方式"""
        variation_prompt = f"""
        请为以下医学测试问题创建{num_variations}种不同的表述方式，保持相同的医学内容和难度：

        原问题：{base_question}

        要求：
        1. 每种表述都要测试相同的知识点
        2. 使用不同的问法或角度
        3. 保持医学专业性和准确性
        4. 难度保持一致

        请按以下格式返回：
        1. 第一种表述：...
        2. 第二种表述：...
        3. 第三种表述：...
        """

        try:
            result = await self._call_qwen_api(variation_prompt, "qwen3-max")
            variations = []

            # 解析返回的变体
            lines = result.strip().split('\n')
            for line in lines:
                if line.strip() and ('.' in line or '：' in line):
                    # 提取表述内容
                    if '.' in line:
                        content = line.split('.', 1)[1].strip()
                    else:
                        content = line.split('：', 1)[1].strip()

                    if content and len(content) > 10:
                        variations.append(content)

            return variations[:num_variations]
        except Exception as e:
            logger.error(f"生成问题变体失败: {e}")
            return [base_question]  # 返回原问题作为备选

# 测试函数
async def test_llm_client():
    """测试LLM客户端"""
    async with LLMClient() as client:
        # 测试单个问题生成
        prompt = "请生成一个关于肺腺癌ROSE特征的医学测试问题，难度为基础水平。"
        result = await client._call_qwen_api(prompt, "qwen3-max")
        print(f"生成的问题: {result}")

        # 测试问题质量验证
        question = "肺腺癌的ROSE特征性表现是什么？"
        expected_answer = "肺腺癌在ROSE中表现为细胞较大，排列成乳头状或腺泡状结构，核质比增高。"
        validation = await client.validate_question_quality(question, expected_answer)
        print(f"质量验证结果: {validation}")

if __name__ == "__main__":
    asyncio.run(test_llm_client())