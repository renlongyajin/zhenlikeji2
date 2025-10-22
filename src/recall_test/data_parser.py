"""
医学文档数据解析器
用于解析clean_data.md并提取医学概念和疾病特征
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MedicalConcept:
    """医学概念数据结构"""
    name: str
    category: str  # 'disease', 'technique', 'cell_feature', 'diagnostic_method'
    description: str
    chapter: str
    section: str
    related_terms: List[str]
    importance_score: float  # 重要性评分 0-1

@dataclass
class DiseaseInfo:
    """疾病信息结构"""
    name: str
    category: str
    rose_features: List[str]  # ROSE细胞学特征
    diagnostic_points: List[str]  # 诊断要点
    differential_diagnosis: List[str]  # 鉴别诊断
    chapter: str
    section: str
    cell_types: List[str]  # 相关细胞类型

class MedicalDataParser:
    """医学数据解析器"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content = ""
        self.chapters = {}
        self.medical_concepts = []
        self.diseases = []
        self.rose_features = set()
        self.cell_features = set()

    def parse(self) -> Dict:
        """解析医学文档"""
        print(f"开始解析医学文档: {self.file_path}")

        # 读取文件内容
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()

        # 提取章节结构
        self._extract_chapters()

        # 提取医学概念
        self._extract_medical_concepts()

        # 提取疾病信息
        self._extract_disease_info()

        # 提取ROSE特征
        self._extract_rose_features()

        # 提取细胞学特征
        self._extract_cell_features()

        print(f"解析完成，提取到:")
        print(f"- 章节数: {len(self.chapters)}")
        print(f"- 医学概念: {len(self.medical_concepts)}")
        print(f"- 疾病信息: {len(self.diseases)}")
        print(f"- ROSE特征: {len(self.rose_features)}")
        print(f"- 细胞特征: {len(self.cell_features)}")

        return {
            "chapters": self.chapters,
            "medical_concepts": [vars(concept) for concept in self.medical_concepts],
            "diseases": [vars(disease) for disease in self.diseases],
            "rose_features": list(self.rose_features),
            "cell_features": list(self.cell_features)
        }

    def _extract_chapters(self):
        """提取章节结构"""
        chapter_pattern = r'###\s+(第[一二三四]章)\s+([^\n]+)'
        section_pattern = r'\*\s+第([一二三四五六七八九十])节\s+([^\n]+)'

        chapters = re.findall(chapter_pattern, self.content)
        sections = re.findall(section_pattern, self.content)

        for chapter_num, chapter_title in chapters:
            self.chapters[chapter_num] = {
                "title": chapter_title.strip(),
                "sections": []
            }

        for section_num, section_title in sections:
            # 根据节号推断章节
            chapter_map = {"一": "第一章", "二": "第二章", "三": "第三章", "四": "第四章",
                          "五": "第四章", "六": "第四章", "七": "第四章", "八": "第四章",
                          "九": "第四章", "十": "第四章", "十一": "第四章"}

            chapter_num = chapter_map.get(section_num, "第四章")
            if chapter_num in self.chapters:
                self.chapters[chapter_num]["sections"].append({
                    "section_num": f"第{section_num}节",
                    "title": section_title.strip()
                })

    def _extract_medical_concepts(self):
        """提取医学概念"""
        # ROSE技术相关概念
        rose_concepts = [
            ("ROSE", "快速现场评价技术"),
            ("ROSE组学", "细胞组学分析方法"),
            ("定位细胞", "三维构象还原关键概念"),
            ("细胞组学", "细胞群空间关系分析"),
            ("肿瘤素质", "恶性细胞背景特征")
        ]

        for concept, description in rose_concepts:
            self.medical_concepts.append(MedicalConcept(
                name=concept,
                category="technique",
                description=description,
                chapter="第一章",
                section="前言",
                related_terms=[],
                importance_score=0.9
            ))

        # 细胞学特征概念
        cell_concepts = [
            ("核质比", "N/C比值，>1/2提示恶性"),
            ("核仁比例", "m/N比值，>0.25提示恶性"),
            ("深染", "染色质浓缩特征"),
            ("镶嵌样排列", "细胞排列方式"),
            ("乳头状结构", "腺癌特征性排列"),
            ("腺泡状结构", "腺癌排列方式"),
            ("桑葚状结构", "细胞聚集形态"),
            ("梭形细胞", "特定细胞形态"),
            ("透明细胞", "细胞质透明特征")
        ]

        for concept, description in cell_concepts:
            self.medical_concepts.append(MedicalConcept(
                name=concept,
                category="cell_feature",
                description=description,
                chapter="第一章",
                section="细胞学特点",
                related_terms=[],
                importance_score=0.8
            ))

    def _extract_disease_info(self):
        """提取疾病信息"""
        # 肺部实体恶性肿瘤
        lung_cancers = [
            ("腺癌", "分化较高时细胞较大，分化较低时细胞较小"),
            ("鳞癌", "分化较高时角化明显，分化较低时异型明显"),
            ("小细胞癌", "无质、无仁、鬼脸、镶嵌"),
            ("大细胞神经内分泌癌", "三大一少伴镶嵌"),
            ("典型类癌", "细胞大小较一致，异型性小"),
            ("不典型类癌", "细胞大小不一，异型性明显"),
            ("黏液表皮样癌", "含黏液细胞、表皮样细胞及中间型细胞"),
            ("腺样囊性癌", "体积小、核深染、黏液样基质"),
            ("黏液腺癌", "黏液湖中癌细胞成团"),
            ("腺泡细胞癌", "浆液细胞腺癌，低度恶性"),
            ("肉瘤样癌", "异型明显，可见巨细胞样、梭形细胞样")
        ]

        for disease, features in lung_cancers:
            self.diseases.append(DiseaseInfo(
                name=disease,
                category="lung_cancer",
                rose_features=[features],
                diagnostic_points=[f"ROSE特征：{features}"],
                differential_diagnosis=[],
                chapter="第二章",
                section=f"{disease}相关章节",
                cell_types=[disease]
            ))

        # 转移性肿瘤
        metastatic_tumors = [
            ("转移性平滑肌肉瘤", "梭形细胞，核细长"),
            ("转移性肾透明细胞癌", "细胞质透明或弱嗜酸性"),
            ("转移性宫颈绒毛腺管状腺癌", "低柱状细胞，绒毛状结构"),
            ("急性髓系白血病肺浸润", "髓系原始细胞特征"),
            ("弥漫性大B细胞淋巴瘤", "大B淋巴细胞特征"),
            ("转移性结肠腺癌", "柱形或杯状细胞"),
            ("上皮型间皮瘤", "立方状、多边形细胞")
        ]

        for disease, features in metastatic_tumors:
            self.diseases.append(DiseaseInfo(
                name=disease,
                category="metastatic_tumor",
                rose_features=[features],
                diagnostic_points=[f"ROSE特征：{features}"],
                differential_diagnosis=[],
                chapter="第三章",
                section=f"{disease}相关章节",
                cell_types=[disease]
            ))

        # 少见病
        rare_diseases = [
            ("肺泡蛋白沉积症", "粉染颗粒状蛋白样物质"),
            ("肺淀粉样物质沉积症", "致密无定形嗜氰物质"),
            ("肉芽肿性多血管炎", "坏死、中性粒细胞浸润"),
            ("过敏性肺炎", "淋巴细胞优势，非坏死性肉芽肿"),
            ("变态反应性支气管肺曲菌病", "嗜酸性粒细胞、曲霉菌丝"),
            ("急性纤维素性机化性肺炎", "纤维素样物质及机化性病变")
        ]

        for disease, features in rare_diseases:
            self.diseases.append(DiseaseInfo(
                name=disease,
                category="rare_disease",
                rose_features=[features],
                diagnostic_points=[f"ROSE特征：{features}"],
                differential_diagnosis=[],
                chapter="第四章",
                section=f"{disease}相关章节",
                cell_types=[disease]
            ))

    def _extract_rose_features(self):
        """提取ROSE特征"""
        rose_patterns = [
            r"ROSE特征.*?[:：]\s*([^\n]+)",
            r"特征.*?[:：]\s*([^\n]+)",
            r"表现为.*?[:：]\s*([^\n]+)",
            r"可见.*?[:：]\s*([^\n]+)",
            r"细胞学特点.*?[:：]\s*([^\n]+)"
        ]

        for pattern in rose_patterns:
            matches = re.findall(pattern, self.content, re.MULTILINE)
            for match in matches:
                features = match.strip().split('，')
                for feature in features:
                    feature = feature.strip().replace('"', '').replace("'", "")
                    if len(feature) > 2 and len(feature) < 100:
                        self.rose_features.add(feature)

    def _extract_cell_features(self):
        """提取细胞学特征"""
        cell_patterns = [
            "深染", "浓染", "淡染", "透明", "嗜酸", "嗜碱", "多形性", "异型性",
            "核大", "核小", "核仁", "核膜", "核质比", "镶嵌", "乳头状", "腺泡状",
            "桑葚状", "梭形", "圆形", "卵圆形", "立方状", "多边形", "巨细胞",
            "坏死", "凋亡", "有丝分裂", "核分裂", "细胞质", "细胞膜"
        ]

        for pattern in cell_patterns:
            if pattern in self.content:
                self.cell_features.add(pattern)

    def save_parsed_data(self, output_path: str):
        """保存解析后的数据"""
        data = self.parse()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"解析数据已保存到: {output_path}")

    def get_disease_categories(self) -> Dict[str, List[str]]:
        """获取疾病分类"""
        categories = defaultdict(list)
        for disease in self.diseases:
            categories[disease.category].append(disease.name)
        return dict(categories)

    def get_concepts_by_category(self, category: str) -> List[MedicalConcept]:
        """按类别获取概念"""
        return [concept for concept in self.medical_concepts if concept.category == category]

def main():
    """主函数"""
    parser = MedicalDataParser("data/clean_data.md")

    # 解析数据
    data = parser.parse()

    # 保存解析结果
    output_path = "data/parsed_medical_data.json"
    parser.save_parsed_data(output_path)

    # 打印统计信息
    categories = parser.get_disease_categories()
    print("\n疾病分类统计:")
    for category, diseases in categories.items():
        print(f"{category}: {len(diseases)}种疾病")
        for disease in diseases[:5]:  # 显示前5个
            print(f"  - {disease}")
        if len(diseases) > 5:
            print(f"  ... 还有{len(diseases)-5}种")

if __name__ == "__main__":
    main()