● 根据我对src/agent目录的深入分析，现在为你详细解释BM25和语义检索结果是如何进行rerank的：

  Rerank机制分析

  1. 双路检索架构

  BM25检索（Elasticsearch）：
  - 使用ES内置的BM25算法
  - 配置多字段权重：chapter_title^12.0, section_title^10.0, content^3.0
  - 医学术语扩展和同义词处理

  语义检索（Milvus）：
  - 将查询文本转换为768维向量
  - 使用余弦相似度在Milvus中搜索
  - 返回语义相似的文档

  2. 核心Rerank算法

  主要的rerank逻辑在 enhanced_retrieval_manager.py 的 _calculate_enhanced_score 方法中：

  def _calculate_enhanced_score(self, base_score, title_score, content_score, concept_score):
      # 根据ES基础分数质量动态调整权重
      if base_score >= 50:  # 高质量结果
          base_weight = 0.85
          title_weight = 0.10
          content_weight = 0.03
          concept_weight = 0.02
      elif base_score >= 20:  # 中等质量结果
          base_weight = 0.75
          title_weight = 0.15
          content_weight = 0.07
          concept_weight = 0.03
      else:  # 低质量结果
          base_weight = 0.60
          title_weight = 0.30
          content_weight = 0.05
          concept_weight = 0.05

      # 计算综合分数
      final_score = (base_score * base_weight +
                     title_score * title_weight +
                     content_score * content_weight +
                     concept_score * concept_weight)

      return final_score

  3. 多维度评分体系

  标题匹配评分：
  - 完全匹配：×2.0倍率
  - 包含匹配：高优先级
  - 词语级匹配：×0.3倍率
  - 医学术语特殊加权

  内容质量评分：
  - 描述性内容识别（"呈"、"可见"、"表现为"等）
  - 医学术语密度计算
  - 图表引用惩罚机制

  概念相关性评分：
  - 医学术语匹配度
  - 专业概念相关性
  - 语义相似度补充

  4. 智能融合策略

  动态权重分配：
  - 高质量BM25结果（≥50分）：保留85%的原始分数
  - 中等质量结果（20-50分）：平衡融合策略
  - 低质量结果（<20分）：更多依赖标题和语义信息

  医学内容优化：
  - ROSE技术内容识别
  - 诊断标准内容加权
  - 治疗指南内容优先
  - 病例研究内容特殊处理

  5. 结果归一化

  最终所有分数归一化到0-100范围，确保：
  - 不同类型查询结果可比
  - 置信度统一表示
  - 用户体验一致性

  这个rerank系统的核心创新是动态权重调整，它解决了传统线性融合的问题，能够根据BM25结果的质量智能调整融合策略
  ，既保留了高质量关键词匹配结果，又通过语义信息补充了低质量情况下的检索效果。