"""
基于LLM的医学测试数据生成器
使用大语言模型生成高质量的RAGAS测试问题
"""

import asyncio
import json
import logging
import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import random
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TEST_CONFIG, MODEL_CONFIG, API_CONFIG
from llm_client import LLMClient

logger = logging.getLogger(__name__)

@dataclass
class MedicalQuestion:
    """医学测试问题数据结构"""
    id: str
    question: str
    question_type: str  # 'concept', 'diagnosis', 'differential', 'case_analysis'
    difficulty: str  # 'basic', 'medium', 'hard'
    disease_category: str
    expected_answer: str
    related_concepts: List[str]
    source_chapter: str
    source_section: str
    expected_chunk_ids: List[str]  # 基于真实chunk内容的相关chunks
    keywords: List[str]
    created_at: str

class LLMQuestionGenerator:
    """基于LLM的医学问题生成器"""

    def __init__(self, model_name: str = "qwen3-max"):
        self.model_name = model_name
        self.llm_client = LLMClient()
        self.questions = []
        self.chunk_usage_stats = {}  # 记录chunk使用频率

        # 直接加载API密钥（绕过环境变量问题）
        self._load_api_keys()

    def _load_api_keys(self):
        """直接从.env文件加载API密钥"""
        try:
            env_path = Path("/home/ubuntu/myproject/zhenlikeji2/.env")
            if env_path.exists():
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            if key == 'DASHSCOPE_API_KEY':
                                self.dashscope_api_key = value.strip()
                                print(f"✅ 成功加载DashScope API密钥")
                                return

            # 如果文件不存在，使用硬编码的密钥（仅用于测试）
            self.dashscope_api_key = "sk-f15e302c67774c079d8888e0d09603d7"
            print(f"⚠️  使用硬编码API密钥")

        except Exception as e:
            print(f"❌ 加载API密钥失败: {e}")
            self.dashscope_api_key = "sk-f15e302c67774c079d8888e0d09603d7"  # 备用密钥

    async def load_chunks_data(self, chunks_file: str) -> List[Dict[str, Any]]:
        """加载chunks数据"""
        chunks_path = Path(chunks_file)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks文件不存在: {chunks_path}")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        logger.info(f"已加载 {len(chunks)} 个chunks")
        return chunks

    def create_comprehensive_prompt(self, chunk: Dict[str, Any],
                                  question_type: str,
                                  difficulty: str) -> str:
        """创建综合性的LLM prompt"""

        chapter = chunk.get('chapter_title', '')
        section = chunk.get('section_title', '')
        content = chunk.get('content', '')[:1000]  # 限制内容长度
        chunk_id = chunk.get('chunk_id', '')

        base_prompt = f"""你是一个专业的医学教育专家，专注于ROSE（快速现场评价）细胞组学领域。

请基于以下医学内容生成一个高质量的测试问题：

【章节信息】
章节：{chapter}
小节：{section}

【内容原文】
{content}

【生成要求】
问题类型：{question_type}
难度等级：{difficulty}
当前chunk ID：{chunk_id}

请生成一个符合以下标准的医学测试问题：

1. **问题质量要求**：
   - 问题必须基于提供的具体医学内容
   - 语言要专业、准确、清晰
   - 避免过于宽泛或模糊的问题

2. **问题类型规范**：
   - concept：概念解释类问题，要求解释医学概念的定义、特征或意义
   - diagnosis：诊断类问题，要求识别特定疾病的ROSE特征
   - differential：鉴别诊断类问题，要求比较不同疾病的区别
   - case_analysis：病例分析类问题，要求综合分析具体病例

3. **难度等级标准**：
   - basic：基础性问题，直接考察核心知识点
   - medium：需要一定理解和分析的中等问题
   - hard：需要综合分析和临床思维的复杂问题

4. **输出格式要求**：
请严格按照以下JSON格式输出，确保所有字段完整：
{json.dumps({
    "question": "具体问题文本",
    "question_type": question_type,
    "difficulty": difficulty,
    "expected_answer": "基于内容的准确答案",
    "related_concepts": ["相关概念1", "相关概念2"],
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "rationale": "为什么这个问题重要，考察什么知识点"
}, ensure_ascii=False, indent=2)}

请确保生成的答案必须可以从提供的内容中直接找到或合理推断出来。"""

        return base_prompt

    def create_multi_chunk_prompt(self, primary_chunk: Dict[str, Any],
                                 related_chunks: List[Dict[str, Any]],
                                 question_type: str,
                                 difficulty: str) -> str:
        """创建多chunk关联的prompt"""

        primary_content = primary_chunk.get('content', '')[:800]
        related_contents = []

        for i, chunk in enumerate(related_chunks[:3]):  # 最多3个相关chunks
            content = chunk.get('content', '')[:400]
            related_contents.append(f"\n【相关chunk {chunk.get('chunk_id', '')}】\n{content}")

        all_content = f"【主要内容】\n{primary_content}\n" + "\n".join(related_contents)

        multi_prompt = f"""你是一个专业的医学教育专家，专注于ROSE细胞组学领域。

请基于以下相关联的医学内容生成一个综合性的测试问题：

【内容集合】
{all_content}

【生成要求】
问题类型：{question_type}
难度等级：{difficulty}
主要chunk ID：{primary_chunk.get('chunk_id', '')}
相关chunks ID：{[c.get('chunk_id', '') for c in related_chunks]}

请生成一个能够综合考察多个相关医学知识点的测试问题，要求：

1. **综合性**：问题应该涉及多个chunks的内容，体现知识点的关联性
2. **临床相关性**：问题要有实际的临床意义和应用价值
3. **思维深度**：根据难度等级，考察不同层次的医学思维

4. **输出格式要求**：
请严格按照以下JSON格式输出：
{json.dumps({
    "question": "综合问题文本",
    "question_type": question_type,
    "difficulty": difficulty,
    "expected_answer": "基于所有相关内容的综合答案",
    "related_concepts": ["概念1", "概念2", "概念3"],
    "keywords": ["关键词1", "关键词2", "关键词3", "关键词4"],
    "rationale": "这个问题如何体现多个知识点的综合应用"
}, ensure_ascii=False, indent=2)}n
确保答案需要综合理解多个chunks的内容才能完整回答。"""

        return multi_prompt

    def find_related_chunks(self, primary_chunk: Dict[str, Any],
                           all_chunks: List[Dict[str, Any]],
                           max_related: int = 3) -> List[Dict[str, Any]]:
        """查找相关的chunks"""

        related_chunks = []
        primary_chapter = primary_chunk.get('chapter_title', '')
        primary_section = primary_chunk.get('section_title', '')
        primary_content = primary_chunk.get('content', '').lower()

        # 同章节的其他chunks优先
        same_chapter_chunks = [
            chunk for chunk in all_chunks
            if chunk.get('chapter_title', '') == primary_chapter and
               chunk.get('chunk_id', '') != primary_chunk.get('chunk_id', '')
        ]

        # 简单的相关性评分（基于章节和关键词）
        chunk_scores = []
        for chunk in same_chapter_chunks:
            score = 0

            # 同小节加分
            if chunk.get('section_title', '') == primary_section:
                score += 10

            # 内容关键词匹配（简单实现）
            chunk_content = chunk.get('content', '').lower()
            if len(primary_content) > 0 and len(chunk_content) > 0:
                # 简单的关键词重叠度计算
                primary_words = set(primary_content.split()[:50])  # 取前50个词
                chunk_words = set(chunk_content.split()[:50])
                overlap = len(primary_words & chunk_words) / max(len(primary_words), 1)
                score += overlap * 5

            chunk_scores.append((chunk, score))

        # 按分数排序，选择top N
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        related_chunks = [chunk for chunk, score in chunk_scores[:max_related]]

        return related_chunks

    async def generate_single_question(self, chunk: Dict[str, Any],
                                     question_type: str,
                                     difficulty: str) -> Dict[str, Any]:
        """为单个chunk生成问题 - 使用DashScope API"""

        # 根据问题类型决定是否使用多chunks
        if question_type in ['differential', 'case_analysis'] and difficulty in ['medium', 'hard']:
            # 查找相关chunks
            all_chunks = await self.load_chunks_data("data/simple_chunks.json")
            related_chunks = self.find_related_chunks(chunk, all_chunks)

            if related_chunks:
                prompt = self.create_multi_chunk_prompt(chunk, related_chunks, question_type, difficulty)
                expected_chunk_ids = [chunk.get('chunk_id', '')] + [c.get('chunk_id', '') for c in related_chunks]
            else:
                prompt = self.create_comprehensive_prompt(chunk, question_type, difficulty)
                expected_chunk_ids = [chunk.get('chunk_id', '')]
        else:
            prompt = self.create_comprehensive_prompt(chunk, question_type, difficulty)
            expected_chunk_ids = [chunk.get('chunk_id', '')]

        # 直接使用DashScope API调用qwen3-max - 使用正确的格式
        try:
            url = f"{API_CONFIG['dashscope_base_url']}/services/aigc/text-generation/generation"

            headers = {
                "Authorization": f"Bearer {self.dashscope_api_key}",
                "Content-Type": "application/json"
            }

            # 使用正确的DashScope API格式
            payload = {
                "model": "qwen3-max",
                "input": {
                    "prompt": prompt
                },
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1500,
                    "top_p": 0.9,
                    "seed": 42
                }
            }

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response_text = await response.text()
                    print(f"API响应状态: {response.status}")
                    print(f"API响应: {response_text[:200]}...")

                    if response.status == 200:
                        result = json.loads(response_text)

                        # DashScope API的正确响应格式
                        if "output" in result and "choices" in result["output"]:
                            result_text = result["output"]["choices"][0]["message"]["content"]
                        else:
                            result_text = result.get("output", {}).get("text", "")

                        if not result_text:
                            logger.error("LLM返回空结果")
                            return None

                        print(f"LLM响应成功，长度: {len(result_text)}")
                        # 移除调试输出，避免过多日志
                        # print(f"响应内容预览: {result_text[:200]}...")

                        # 解析JSON响应
                        try:
                            # 提取JSON部分（处理可能的markdown格式）
                            import re
                            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', result_text)
                            if json_match:
                                json_str = json_match.group(1)
                            else:
                                # 尝试直接解析整个响应为JSON
                                json_str = result_text.strip()

                            result = json.loads(json_str)

                            # 验证必需字段
                            required_fields = ['question', 'expected_answer', 'related_concepts', 'keywords']
                            for field in required_fields:
                                if field not in result:
                                    logger.warning(f"缺少必需字段: {field}")
                                    return None

                            # 构建完整的问题对象 - 只包含MedicalQuestion需要的字段
                            question_data = {
                                'id': '',  # 稍后分配
                                'question': result['question'],
                                'question_type': question_type,
                                'difficulty': difficulty,
                                'disease_category': self._categorize_chunk(chunk),
                                'expected_answer': result['expected_answer'],
                                'related_concepts': result['related_concepts'],
                                'source_chapter': chunk.get('chapter_title', ''),
                                'source_section': chunk.get('section_title', ''),
                                'expected_chunk_ids': expected_chunk_ids[:4],  # 最多4个
                                'keywords': result['keywords'],
                                'created_at': datetime.now().isoformat()
                                # 注意：不包含rationale字段，因为MedicalQuestion没有这个字段
                            }

                            return question_data

                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析失败: {e}")
                            logger.error(f"LLM响应: {result_text[:500]}...")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"DashScope API调用失败: {response.status} - {error_text}")
                        return None

        except Exception as e:
            logger.error(f"LLM生成问题失败: {e}")
            return None

    def _categorize_chunk(self, chunk: Dict[str, Any]) -> str:
        """根据chunk内容分类"""

        chapter = chunk.get('chapter_title', '').lower()
        section = chunk.get('section_title', '').lower()

        if '肺癌' in chapter or '肺部实体' in chapter:
            return 'lung_cancer'
        elif '转移性' in chapter or '其他' in chapter:
            return 'metastatic_tumor'
        elif '少见病' in chapter:
            return 'rare_disease'
        elif '技术' in chapter or '组学' in chapter:
            return 'rose_technique'
        else:
            return 'general'

    async def generate_balanced_questions(self, chunks_data: List[Dict[str, Any]],
                                        total_questions: int = 150) -> List[MedicalQuestion]:
        """生成均衡分布的测试问题"""

        print(f"开始生成 {total_questions} 个均衡分布的测试问题...")

        questions = []

        # 1. 按章节和难度分层
        layer_config = {
            'basic': {'count': int(total_questions * 0.6), 'types': ['concept', 'diagnosis']},
            'medium': {'count': int(total_questions * 0.3), 'types': ['diagnosis', 'differential']},
            'hard': {'count': int(total_questions * 0.1), 'types': ['differential', 'case_analysis']}
        }

        # 2. 按章节分布
        chapter_distribution = {
            'lung_cancer': 0.35,
            'metastatic_tumor': 0.25,
            'rare_disease': 0.25,
            'rose_technique': 0.15
        }

        # 3. 为每个chunk生成使用权重（避免过度集中）
        chunk_weights = {chunk['chunk_id']: 1.0 for chunk in chunks_data}

        question_id = 1

        for difficulty, config in layer_config.items():
            questions_per_difficulty = config['count']

            for question_type in config['types']:
                questions_per_type = questions_per_difficulty // len(config['types'])

                # 根据权重选择chunks
                for i in range(questions_per_type):
                    # 选择权重最高的chunks（但会逐渐降低已选chunks的权重）
                    candidate_chunks = sorted(chunks_data,
                                            key=lambda x: chunk_weights[x['chunk_id']],
                                            reverse=True)

                    primary_chunk = candidate_chunks[0]

                    # 生成问题
                    question_data = await self.generate_single_question(
                        primary_chunk, question_type, difficulty
                    )

                    if question_data:
                        question_data['id'] = f"med_q_{question_id:04d}"
                        questions.append(MedicalQuestion(**question_data))
                        question_id += 1

                        # 降低已使用chunks的权重，促进均衡分布
                        for chunk_id in question_data['expected_chunk_ids']:
                            if chunk_id in chunk_weights:
                                chunk_weights[chunk_id] *= 0.8  # 降低权重

                    # 每生成10个问题，重新平衡权重
                    if i % 10 == 0:
                        # 提升未使用chunks的权重
                        for chunk in chunks_data:
                            if chunk['chunk_id'] not in chunk_weights:
                                chunk_weights[chunk['chunk_id']] = 1.2

        print(f"成功生成 {len(questions)} 个测试问题")
        return questions

    def save_questions(self, questions: List[Any], output_path: str):
        """保存生成的问题"""
        # 处理两种类型：Dict列表或MedicalQuestion对象列表
        if questions and hasattr(questions[0], '__dict__'):
            # MedicalQuestion对象列表，转换为字典
            questions_data = [asdict(q) for q in questions]
        else:
            # 字典列表，直接保存
            questions_data = questions

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, ensure_ascii=False, indent=2)

        print(f"测试问题已保存到: {output_path}")

    def generate_statistics(self, questions: List[Any],
                          chunks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成详细的统计信息"""

        stats = {
            "total_questions": len(questions),
            "difficulty_distribution": {},
            "type_distribution": {},
            "category_distribution": {},
            "chunk_coverage": {},
            "avg_expected_chunks_per_question": 0,
            "generation_timestamp": datetime.now().isoformat(),
            "model_used": self.model_name
        }

        # 基础统计 - 支持Dict和MedicalQuestion对象
        for question in questions:
            if hasattr(question, '__dict__'):
                # MedicalQuestion对象
                difficulty = question.difficulty
                q_type = question.question_type
                category = question.disease_category
                chunk_count = len(question.expected_chunk_ids)
            else:
                # 字典对象
                difficulty = question["difficulty"]
                q_type = question["question_type"]
                category = question["disease_category"]
                chunk_count = len(question["expected_chunk_ids"])

            stats["difficulty_distribution"][difficulty] = stats["difficulty_distribution"].get(difficulty, 0) + 1
            stats["type_distribution"][q_type] = stats["type_distribution"].get(q_type, 0) + 1
            stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1
            stats["avg_expected_chunks_per_question"] += chunk_count

        # 计算平均值
        if questions:
            stats["avg_expected_chunks_per_question"] /= len(questions)

        # chunks覆盖统计
        all_expected_chunks = []
        for question in questions:
            if hasattr(question, '__dict__'):
                all_expected_chunks.extend(question.expected_chunk_ids)
            else:
                all_expected_chunks.extend(question["expected_chunk_ids"])

        available_chunks = set(chunk['chunk_id'] for chunk in chunks_data)
        used_chunks = set(all_expected_chunks)

        stats["chunk_coverage"] = {
            "total_available": len(available_chunks),
            "total_used": len(used_chunks),
            "coverage_percentage": len(used_chunks) / len(available_chunks) * 100,
            "unused_chunks": list(available_chunks - used_chunks),
            "chunk_usage_frequency": {}
        }

        # chunks使用频率
        chunk_freq = {}
        for chunk_id in all_expected_chunks:
            chunk_freq[chunk_id] = chunk_freq.get(chunk_id, 0) + 1

        stats["chunk_coverage"]["chunk_usage_frequency"] = chunk_freq

        return stats

async def main():
    """主函数 - 使用LLM生成测试数据"""

    print("🚀 启动LLM-based医学测试数据生成器")
    print("="*60)

    # 创建生成器
    generator = LLMQuestionGenerator(model_name="qwen3-max")

    # 加载chunks数据 - 修正路径
    chunks_file = "data/simple_chunks.json"
    print(f"📊 加载chunks数据: {chunks_file}")
    chunks_data = await generator.load_chunks_data(chunks_file)

    # 生成均衡分布的测试问题
    print("🤖 开始生成测试问题...")
    questions = await generator.generate_balanced_questions(chunks_data, total_questions=120)

    if questions:
        # 保存结果
        output_file = "test_data/generated_questions_chunk_llm.json"
        generator.save_questions(questions, output_file)

        # 生成统计
        stats = generator.generate_statistics(questions, chunks_data)
        print("\n📈 生成统计:")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

        # 保存统计
        stats_file = output_file.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 测试数据生成完成！")
        print(f"📁 输出文件: {output_file}")
        print(f"📊 统计文件: {stats_file}")
    else:
        print("❌ 测试数据生成失败")

if __name__ == "__main__":
    asyncio.run(main())