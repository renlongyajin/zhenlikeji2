"""
医学测试问题生成器
基于解析的医学数据生成高质量的RAGAS测试问题
"""

import json
import random
import asyncio
import logging
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from config import TEST_CONFIG, MODEL_CONFIG, API_CONFIG
from llm_client import LLMClient

logger = logging.getLogger(__name__)

@dataclass
class MedicalQuestion:
    """医学测试问题数据结构 - 支持真实chunk模式"""
    id: str
    question: str
    question_type: str  # 'concept', 'diagnosis', 'differential', 'case_analysis'
    difficulty: str  # 'basic', 'medium', 'hard'
    disease_category: str
    expected_answer: str
    related_concepts: List[str]
    source_chapter: str
    source_section: str
    expected_chunk_ids: List[str]  # 期望检索到的chunk ID（替代原有的doc_id）
    keywords: List[str]  # 关键词，用于检索
    created_at: str

class QuestionTemplate:
    """问题模板类"""

    def __init__(self):
        self.templates = {
            'concept': {
                'basic': [
                    "什么是{concept}？",
                    "{concept}的定义是什么？",
                    "请解释{concept}的含义。",
                    "{concept}主要包括哪些内容？",
                    "在临床实践中，{concept}有什么作用？"
                ],
                'medium': [
                    "{concept}与{related_concept}有什么区别？",
                    "如何在ROSE中识别{concept}？",
                    "{concept}的临床意义是什么？",
                    "哪些疾病会表现出{concept}的特征？",
                    "{concept}在诊断中的价值如何？"
                ],
                'hard': [
                    "结合具体病例，分析{concept}在鉴别诊断中的作用。",
                    "比较{concept}在不同疾病中的表现差异。",
                    "如何区分{concept}与其他相似特征？",
                    "在ROSE评估中，{concept}的误判原因有哪些？",
                    "阐述{concept}的病理生理机制及其临床意义。"
                ]
            },
            'diagnosis': {
                'basic': [
                    "{disease}的ROSE特征是什么？",
                    "如何在ROSE中识别{disease}？",
                    "{disease}的细胞学特点有哪些？",
                    "{disease}的典型表现是什么？",
                    "诊断{disease}需要哪些ROSE标准？"
                ],
                'medium': [
                    "{disease}与{differential_disease}在ROSE中如何鉴别？",
                    "{disease}的不同分化程度在ROSE中有何表现？",
                    "哪些细胞学特征支持{disease}的诊断？",
                    "在ROSE中，{disease}的背景特征有哪些？",
                    "{disease}的ROSE诊断陷阱有哪些？"
                ],
                'hard': [
                    "分析一个具体病例：患者表现出{symptoms}，ROSE发现{features}，最可能的诊断是什么？",
                    "比较{disease}在不同临床背景下的ROSE表现差异。",
                    "如何综合运用多种ROSE特征来确诊{disease}？",
                    "阐述{disease}ROSE特征的形成机制。",
                    "在疑难病例中，如何依靠ROSE特征排除{disease}的诊断？"
                ]
            },
            'differential': {
                'basic': [
                    "{disease1}和{disease2}在ROSE中如何区分？",
                    "哪些特征可以帮助鉴别{disease1}与{disease2}？",
                    "{disease1}与{disease2}的细胞学差异是什么？",
                    "在ROSE评估中，如何排除{disease2}而确诊{disease1}？",
                    "{disease1}和{disease2}的鉴别诊断要点有哪些？"
                ],
                'medium': [
                    "当ROSE同时表现出{feature1}和{feature2}时，应考虑哪些疾病？如何鉴别？",
                    "分析{disease1}、{disease2}和{disease3}的ROSE特征异同。",
                    "在{clinical_scenario}背景下，如何鉴别{disease1}和{disease2}？",
                    "哪些ROSE特征对{disease1}与{disease2}的鉴别诊断最有价值？",
                    "阐述{disease1}误诊为{disease2}的常见原因。"
                ],
                'hard': [
                    "复杂病例分析：患者{clinical_features}，ROSE显示{rose_features}，请制定详细的鉴别诊断策略。",
                    "比较{disease1}、{disease2}、{disease3}和{disease4}的ROSE特征谱系。",
                    "如何运用ROSE特征建立{disease1}与{disease2}的决策树？",
                    "在{special_population}中，{disease1}与{disease2}的鉴别诊断有哪些特殊考虑？",
                    "分析ROSE在{disease1}与{disease2}鉴别诊断中的局限性。"
                ]
            },
            'case_analysis': {
                'basic': [
                    "患者{age}岁，{gender}，{symptoms}。ROSE发现{rose_features}。最可能的诊断是什么？",
                    "分析以下ROSE表现：{rose_description}。考虑哪些疾病？",
                    "{imaging_findings}，ROSE显示{rose_features}。诊断思路如何？",
                    "患者有{risk_factors}，ROSE发现{rose_features}。如何诊断？",
                    "{laboratory_results}，ROSE表现{rose_features}。可能的疾病是什么？"
                ],
                'medium': [
                    "复杂病例：{clinical_presentation}，影像学显示{imaging}，ROSE发现{rose_features}。请制定诊疗方案。",
                    "患者{demographics}，有{medical_history}，现{symptoms}。ROSE：{rose_features}。鉴别诊断？",
                    "{procedure_type}标本ROSE显示{rose_features}，但{conflicting_evidence}。如何解释？",
                    "分析{number}个相似病例的ROSE特征差异及其诊断意义。",
                    "在{clinical_context}下，ROSE发现{rose_features}的诊疗策略是什么？"
                ],
                'hard': [
                    "疑难病例讨论：{complex_presentation}，多次检查{test_results}，ROSE表现{rose_features}。请制定诊断策略。",
                    "罕见病例分析：{rare_condition}患者{unusual_features}，ROSE显示{rose_features}。诊断和治疗的挑战？",
                    "误诊病例分析：最初ROSE诊断为{initial_diagnosis}，但{subsequent_findings}。分析误诊原因。",
                    "多中心研究：比较{number}个中心对相似ROSE特征{rose_features}的诊断一致性。",
                    "新技术应用：{new_technology}如何改变对{rose_features}的诊断准确性？"
                ]
            }
        }

class MedicalQuestionGenerator:
    """医学问题生成器"""

    def __init__(self, parsed_data_path: str):
        self.parsed_data_path = Path(parsed_data_path)
        self.parsed_data = None
        self.templates = QuestionTemplate()
        self.llm_client = LLMClient()
        self.questions = []

    async def load_parsed_data(self):
        """加载解析后的医学数据"""
        if not self.parsed_data_path.exists():
            raise FileNotFoundError(f"解析数据文件不存在: {self.parsed_data_path}")

        with open(self.parsed_data_path, 'r', encoding='utf-8') as f:
            self.parsed_data = json.load(f)

        print(f"已加载解析数据: {len(self.parsed_data.get('diseases', []))}种疾病, "
              f"{len(self.parsed_data.get('medical_concepts', []))}个概念")

    async def generate_questions(self, num_questions: int = 150) -> List[MedicalQuestion]:
        """生成医学测试问题"""
        print(f"开始生成{num_questions}个医学测试问题...")

        questions = []

        # 根据配置生成不同类别的问题
        category_counts = self._calculate_category_counts(num_questions)

        for category, count in category_counts.items():
            print(f"生成{category}类别问题: {count}个")
            category_questions = await self._generate_category_questions(category, count)
            questions.extend(category_questions)

        # 打乱问题顺序
        random.shuffle(questions)

        # 分配ID
        for i, question in enumerate(questions):
            question.id = f"med_q_{i+1:04d}"

        self.questions = questions
        print(f"问题生成完成，共生成{len(questions)}个问题")

        return questions

    def _calculate_category_counts(self, total_questions: int) -> Dict[str, int]:
        """计算各类别问题数量"""
        distribution = TEST_CONFIG["disease_coverage"]
        return {
            category: int(total_questions * ratio)
            for category, ratio in distribution.items()
        }

    async def _generate_category_questions(self, category: str, count: int) -> List[MedicalQuestion]:
        """生成特定类别的问题"""
        questions = []

        if category == "lung_cancer":
            questions = await self._generate_lung_cancer_questions(count)
        elif category == "metastatic_tumor":
            questions = await self._generate_metastatic_questions(count)
        elif category == "rare_disease":
            questions = await self._generate_rare_disease_questions(count)
        elif category == "rose_technique":
            questions = await self._generate_rose_technique_questions(count)

        return questions

    async def _generate_lung_cancer_questions(self, count: int) -> List[MedicalQuestion]:
        """生成肺癌相关问题"""
        questions = []
        diseases = [d for d in self.parsed_data["diseases"]
                   if d["category"] == "lung_cancer"]

        # 基础题 (60%)
        basic_count = int(count * 0.6)
        for i in range(basic_count):
            disease = random.choice(diseases)
            template = random.choice(self.templates.templates["diagnosis"]["basic"])

            # 生成问题文本（基础题）
            try:
                question_text = template.format(
                    disease=disease["name"],
                    features=random.choice(disease["rose_features"]) if disease["rose_features"] else "特征性表现"
                )
            except KeyError as e:
                # 如果模板缺少参数，使用简化格式
                logger.warning(f"模板缺少参数 {e}，使用简化格式")
                question_text = f"{disease['name']}的ROSE特征是什么？"

            question = MedicalQuestion(
                id="",  # 稍后分配
                question=question_text,
                question_type="diagnosis",
                difficulty="basic",
                disease_category="lung_cancer",
                expected_answer=f"{disease['name']}的ROSE特征包括：{', '.join(disease['rose_features'][:2])}",
                related_concepts=[disease["name"]] + disease["cell_types"],
                source_chapter=disease["chapter"],
                source_section=disease["section"],
                expected_chunk_ids=[],  # 稍后填充
                keywords=[disease["name"], "ROSE", "细胞学"],
                created_at=""
            )
            questions.append(question)

        # 中等题 (30%)
        medium_count = int(count * 0.3)
        for i in range(medium_count):
            disease1, disease2 = random.sample(diseases, 2)
            template = random.choice(self.templates.templates["differential"]["medium"])

            # 处理中等题模板，确保所有必需参数都存在
            try:
                if "{differential_disease}" in template:
                    question_text = template.format(
                        disease1=disease1["name"],
                        disease2=disease2["name"]
                    )
                elif "{clinical_scenario}" in template:
                    question_text = template.format(
                        disease1=disease1["name"],
                        disease2=disease2["name"],
                        clinical_scenario="肺部占位性病变"
                    )
                elif "{feature1}" in template and "{feature2}" in template:
                    question_text = template.format(
                        feature1="细胞异型性",
                        feature2="核分裂象",
                        disease1=disease1["name"],
                        disease2=disease2["name"],
                        disease3="肺腺癌"
                    )
                else:
                    # 默认格式
                    question_text = template.format(
                        disease1=disease1["name"],
                        disease2=disease2["name"]
                    )
            except KeyError as e:
                logger.warning(f"中等题模板缺少参数 {e}，使用简化格式")
                question_text = f"{disease1['name']}和{disease2['name']}在ROSE中如何鉴别？"

            question = MedicalQuestion(
                id="",
                question=question_text,
                question_type="differential",
                difficulty="medium",
                disease_category="lung_cancer",
                expected_answer=f"需要比较{disease1['name']}和{disease2['name']}的ROSE特征差异",
                related_concepts=[disease1["name"], disease2["name"]],
                source_chapter=disease1["chapter"],
                source_section=disease1["section"],
                expected_chunk_ids=[],
                keywords=[disease1["name"], disease2["name"], "鉴别诊断"],
                created_at=""
            )
            questions.append(question)

        # 难题 (10%)
        hard_count = count - basic_count - medium_count
        for i in range(hard_count):
            disease = random.choice(diseases)
            template = random.choice(self.templates.templates["case_analysis"]["hard"])

            # 处理难题模板，确保所有必需参数都存在
            try:
                if "{complex_presentation}" in template and "{rose_features}" in template:
                    question_text = template.format(
                        complex_presentation=f"肺部占位性病变患者",
                        rose_features=', '.join(disease["rose_features"][:2]) if disease["rose_features"] else "特征性细胞学表现"
                    )
                elif "{initial_diagnosis}" in template:
                    question_text = template.format(
                        initial_diagnosis=disease["name"],
                        subsequent_findings="进一步检查发现了新的特征"
                    )
                elif "{rare_condition}" in template:
                    question_text = template.format(
                        rare_condition=disease["name"],
                        unusual_features="不典型的细胞学表现"
                    )
                else:
                    # 默认格式
                    question_text = template.format(
                        complex_presentation=f"肺部占位性病变患者",
                        rose_features=', '.join(disease["rose_features"][:2]) if disease["rose_features"] else "特征性细胞学表现"
                    )
            except KeyError as e:
                logger.warning(f"难题模板缺少参数 {e}，使用简化格式")
                question_text = f"分析一个复杂病例：患者肺部占位性病变，ROSE显示{', '.join(disease['rose_features'][:2]) if disease['rose_features'] else '特征性表现'}，诊断策略是什么？"

            question = MedicalQuestion(
                id="",
                question=question_text,
                question_type="case_analysis",
                difficulty="hard",
                disease_category="lung_cancer",
                expected_answer=f"基于ROSE特征，考虑{disease['name']}的诊断，需要结合临床表现和其他检查",
                related_concepts=[disease["name"]],
                source_chapter=disease["chapter"],
                source_section=disease["section"],
                expected_chunk_ids=[],
                keywords=[disease["name"], "病例分析", "诊断策略"],
                created_at=""
            )
            questions.append(question)

        return questions

    async def _generate_metastatic_questions(self, count: int) -> List[MedicalQuestion]:
        """生成转移性肿瘤相关问题"""
        questions = []
        diseases = [d for d in self.parsed_data["diseases"]
                   if d["category"] == "metastatic_tumor"]

        # 类似肺癌的生成逻辑，但重点关注转移性特征
        for i in range(count):
            disease = random.choice(diseases)

            if i < int(count * 0.6):  # 基础题
                template = random.choice(self.templates.templates["diagnosis"]["basic"])
                difficulty = "basic"
                q_type = "diagnosis"
            elif i < int(count * 0.9):  # 中等题
                # 需要两个疾病进行对比
                if i + 1 < len(diseases):
                    disease1, disease2 = random.sample(diseases, 2)
                else:
                    disease1 = disease2 = disease

                template = random.choice(self.templates.templates["differential"]["medium"])
                difficulty = "medium"
                q_type = "differential"
                # 为中等题添加必要的模板参数
                if "{differential_disease}" in template:
                    template = template.format(
                        disease1=disease1["name"],
                        disease2=disease2["name"]
                    )
                elif "{clinical_scenario}" in template:
                    template = template.format(
                        disease1=disease1["name"],
                        disease2=disease2["name"],
                        clinical_scenario="肺部占位性病变"
                    )
                elif "{feature1}" in template and "{feature2}" in template:
                    template = template.format(
                        feature1="细胞异型性",
                        feature2="核分裂象",
                        disease1=disease1["name"],
                        disease2=disease2["name"],
                        disease3="肺腺癌"
                    )
            else:  # 难题
                template = random.choice(self.templates.templates["case_analysis"]["hard"])
                difficulty = "hard"
                q_type = "case_analysis"

            # 生成问题文本，处理不同的模板格式
            if q_type == "differential":
                # 对于中等题（鉴别诊断），已经在上面的条件分支中处理了模板
                question_text = template
            else:
                # 对于其他类型的问题
                try:
                    question_text = template.format(
                        disease=disease["name"],
                        features=random.choice(disease["rose_features"]) if disease["rose_features"] else "特征性表现"
                    )
                except KeyError as e:
                    # 如果模板缺少参数，使用简化格式
                    logger.warning(f"模板缺少参数 {e}，使用简化格式")
                    question_text = f"{disease['name']}的ROSE特征是什么？"

            question = MedicalQuestion(
                id="",
                question=question_text,
                question_type=q_type,
                difficulty=difficulty,
                disease_category="metastatic_tumor",
                expected_answer=f"{disease['name']}的ROSE特征分析",
                related_concepts=[disease["name"]],
                source_chapter=disease["chapter"],
                source_section=disease["section"],
                expected_chunk_ids=[],
                keywords=[disease["name"], "转移性肿瘤", "ROSE"],
                created_at=""
            )
            questions.append(question)

        return questions

    async def _generate_rare_disease_questions(self, count: int) -> List[MedicalQuestion]:
        """生成少见病相关问题"""
        questions = []
        diseases = [d for d in self.parsed_data["diseases"]
                   if d["category"] == "rare_disease"]

        for i in range(count):
            disease = random.choice(diseases)

            # 少见病更多关注诊断要点和鉴别诊断
            if i < int(count * 0.5):  # 基础题
                template = random.choice(self.templates.templates["diagnosis"]["basic"])
                q_type = "diagnosis"
                difficulty = "basic"
            elif i < int(count * 0.85):  # 中等题
                template = random.choice(self.templates.templates["differential"]["medium"])
                q_type = "differential"
                difficulty = "medium"
            else:  # 难题
                template = random.choice(self.templates.templates["case_analysis"]["hard"])
                q_type = "case_analysis"
                difficulty = "hard"

            # 生成问题文本，处理不同的模板格式
            if q_type == "differential":
                # 对于中等题（鉴别诊断），已经在上面的条件分支中处理了模板
                question_text = template
            else:
                # 对于其他类型的问题
                try:
                    question_text = template.format(
                        disease=disease["name"],
                        features=random.choice(disease["rose_features"]) if disease["rose_features"] else "特征性表现"
                    )
                except KeyError as e:
                    # 如果模板缺少参数，使用简化格式
                    logger.warning(f"模板缺少参数 {e}，使用简化格式")
                    question_text = f"{disease['name']}的ROSE特征是什么？"

            question = MedicalQuestion(
                id="",
                question=question_text,
                question_type=q_type,
                difficulty=difficulty,
                disease_category="rare_disease",
                expected_answer=f"{disease['name']}作为少见病，其ROSE特征需要特别注意",
                related_concepts=[disease["name"]],
                source_chapter=disease["chapter"],
                source_section=disease["section"],
                expected_chunk_ids=[],
                keywords=[disease["name"], "少见病", "ROSE诊断"],
                created_at=""
            )
            questions.append(question)

        return questions

    async def _generate_rose_technique_questions(self, count: int) -> List[MedicalQuestion]:
        """生成ROSE技术相关问题"""
        questions = []
        concepts = [c for c in self.parsed_data["medical_concepts"]
                   if c["category"] == "technique"]

        for i in range(count):
            concept = random.choice(concepts)

            if i < int(count * 0.7):  # 基础概念题
                template = random.choice(self.templates.templates["concept"]["basic"])
                q_type = "concept"
                difficulty = "basic"
            elif i < int(count * 0.95):  # 中等应用题
                template = random.choice(self.templates.templates["concept"]["medium"])
                q_type = "concept"
                difficulty = "medium"
            else:  # 难题
                template = random.choice(self.templates.templates["concept"]["hard"])
                q_type = "concept"
                difficulty = "hard"

            question_text = template.format(
                concept=concept["name"],
                related_concept=random.choice([c["name"] for c in concepts if c["name"] != concept["name"]])
            )

            question = MedicalQuestion(
                id="",
                question=question_text,
                question_type=q_type,
                difficulty=difficulty,
                disease_category="rose_technique",
                expected_answer=concept["description"],
                related_concepts=[concept["name"]],
                source_chapter=concept["chapter"],
                source_section=concept["section"],
                expected_chunk_ids=[],
                keywords=[concept["name"], "ROSE技术", "细胞组学"],
                created_at=""
            )
            questions.append(question)

        return questions

    def build_expected_chunk_mapping(self, chunk_data: List[Dict[str, Any]]):
        """构建问题-chunk映射关系 - 基于真实chunk数据"""
        print("构建问题-chunk映射关系...")

        # 为每个问题找到最相关的chunk
        for question in self.questions:
            relevant_chunks = []

            # 基于问题关键词和chunk内容进行匹配
            question_keywords = set(question.keywords + question.related_concepts)

            # 计算每个chunk的相关性分数
            chunk_scores = []
            for chunk in chunk_data:
                score = 0
                chunk_content = chunk.get('content', '').lower()
                chunk_title = f"{chunk.get('chapter_title', '')} {chunk.get('section_title', '')}".lower()

                # 关键词匹配评分
                for keyword in question_keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in chunk_content:
                        score += 2  # 内容匹配权重更高
                    if keyword_lower in chunk_title:
                        score += 3  # 标题匹配权重最高

                if score > 0:
                    chunk_scores.append((chunk['chunk_id'], score))

            # 按分数排序，选择最相关的chunk
            chunk_scores.sort(key=lambda x: x[1], reverse=True)

            # 选择top相关的chunk（最多4个）
            if chunk_scores:
                relevant_chunks = [chunk_id for chunk_id, score in chunk_scores[:4]]
            else:
                # 如果没有找到匹配的，随机选择一些chunk
                available_chunks = [chunk['chunk_id'] for chunk in chunk_data]
                if available_chunks:
                    relevant_chunks = random.sample(available_chunks, min(2, len(available_chunks)))

            question.expected_chunk_ids = relevant_chunks

            # 打印调试信息
            if len(relevant_chunks) < 2:
                print(f"⚠️ 问题 '{question.question[:50]}...' 只找到 {len(relevant_chunks)} 个相关chunk")

    def save_questions(self, output_path: str):
        """保存生成的问题"""
        questions_data = [asdict(q) for q in self.questions]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, ensure_ascii=False, indent=2)

        print(f"问题已保存到: {output_path}")

    def generate_statistics(self) -> Dict[str, Any]:
        """生成问题统计信息"""
        stats = {
            "total_questions": len(self.questions),
            "difficulty_distribution": {},
            "type_distribution": {},
            "category_distribution": {},
            "avg_keywords_per_question": 0,
            "avg_expected_chunks_per_question": 0
        }

        for question in self.questions:
            # 难度分布
            difficulty = question.difficulty
            stats["difficulty_distribution"][difficulty] = stats["difficulty_distribution"].get(difficulty, 0) + 1

            # 类型分布
            q_type = question.question_type
            stats["type_distribution"][q_type] = stats["type_distribution"].get(q_type, 0) + 1

            # 类别分布
            category = question.disease_category
            stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1

            # 平均关键词数
            stats["avg_keywords_per_question"] += len(question.keywords)

            # 平均期望chunk数
            stats["avg_expected_chunks_per_question"] += len(question.expected_chunk_ids)

        # 计算平均值
        if self.questions:
            stats["avg_keywords_per_question"] /= len(self.questions)
            stats["avg_expected_chunks_per_question"] /= len(self.questions)

        return stats

async def main():
    """主函数 - 使用真实chunk数据"""
    # 解析医学数据
    try:
        from data_parser import MedicalDataParser
    except ImportError:
        # 如果相对导入失败，尝试绝对导入
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data_parser import MedicalDataParser

    parser = MedicalDataParser("data/clean_data.md")
    parsed_data = parser.parse()

    # 保存解析数据
    parser.save_parsed_data("data/parsed_medical_data.json")

    # 生成问题
    generator = MedicalQuestionGenerator("data/parsed_medical_data.json")
    await generator.load_parsed_data()

    # 生成150个问题
    questions = await generator.generate_questions(150)

    # 加载真实chunk数据
    print("加载真实chunk数据...")
    chunk_file_path = "../../data/simple_chunks.json"  # 相对于当前文件的路径

    try:
        with open(chunk_file_path, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
        print(f"已加载 {len(chunk_data)} 个chunk")
    except FileNotFoundError:
        print(f"❌ 无法找到chunk数据文件: {chunk_file_path}")
        print("请确保 simple_chunks.json 文件存在")
        return

    # 构建chunk映射（使用真实的chunk ID）
    generator.build_expected_chunk_mapping(chunk_data)

    # 保存问题
    generator.save_questions("test_data/generated_questions_chunk.json")

    # 生成统计
    stats = generator.generate_statistics()
    print("\n问题生成统计:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())