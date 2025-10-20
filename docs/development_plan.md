# LangGraph-based ReAct Agent 开发计划

## 项目概述
重构现有医学RAG系统，从硬编码的ReAct代理升级为基于LangGraph的现代化Agent架构，支持LLM自主工具调用，同时保持前端API兼容性。

## 架构目标
- **保留ReAct思想**：Thought → Action → Observation 循环
- **现代化实现**：使用LangGraph和LLM工具调用标准
- **智能工具调用**：Agent自主判断何时调用何种工具
- **多轮优化**：支持搜索结果不满意时的自动重试
- **API兼容**：保持现有前端接口不变

## 详细开发阶段

### 阶段1：架构设计与核心节点实现 (3-4天)

#### 1.1 图结构设计
```
LangGraph工作流：
用户输入 → 意图分析 → 工具选择 → 工具执行 → 结果观察 → 质量评估 → 响应生成
                     ↑           ↓           ↓
                     ←←←← 重新分析 (如果质量不达标)
```

#### 1.2 核心节点定义
- **intent_analysis_node**: 分析查询类型、提取实体、判断搜索需求
- **tool_selection_node**: 基于LLM推理选择合适工具（rag_search等）
- **tool_execution_node**: 执行工具调用，处理返回结果
- **result_observation_node**: 分析工具结果，提取关键信息
- **quality_evaluation_node**: 评估结果质量，决定是否继续优化
- **response_generation_node**: 生成最终响应

#### 1.3 条件边实现
- `need_search`: 判断是否需要搜索工具
- `quality_sufficient`: 评估结果质量是否达标
- `max_iterations_reached`: 防止无限循环

### 阶段2：增强rag_search工具 (2-3天)

#### 2.1 标题权重增强
```python
# 新的搜索配置
search_config = {
    "title_weights": {
        "chapter_title": 20.0,      # 章节标题最高权重
        "section_title": 15.0,      # 节标题高权重
        "subsection_title": 10.0,   # 小节标题中等权重
    },
    "content_analysis": {
        "descriptive_content_boost": 2.0,    # 描述性内容加分
        "figure_reference_penalty": 0.5,     # 图表引用降权
        "minimum_description_length": 100,   # 最小描述长度
    }
}
```

#### 2.2 内容质量识别
- 自动识别文档是否包含详细医学描述
- 区分描述性内容与纯索引内容
- 实现内容完整性和详细程度评分

#### 2.3 多策略搜索
- 首轮：智能混合搜索
- 优化轮：针对性关键词搜索
- 补充轮：语义相似度搜索

### 阶段3：多轮搜索优化机制 (2-3天)

#### 3.1 结果质量评估算法
```python
def evaluate_search_quality(results, original_query):
    quality_score = 0.0

    # 内容相关性评分
    relevance_score = calculate_relevance(results, original_query)

    # 描述完整性评分
    description_score = evaluate_description_completeness(results)

    # 标题匹配度评分
    title_match_score = evaluate_title_match(results, original_query)

    # 综合评分
    quality_score = (
        relevance_score * 0.4 +
        description_score * 0.4 +
        title_match_score * 0.2
    )

    return quality_score, detailed_feedback
```

#### 3.2 搜索策略优化
- **首轮搜索**：基础混合搜索
- **优化搜索**：基于首轮结果调整关键词
- **深度搜索**：针对特定医学术语的精确搜索
- **补充搜索**：语义扩展搜索

#### 3.3 失败处理机制
- 搜索结果为空时的替代策略
- 多次搜索失败时的降级处理
- 提供搜索失败原因和建议

### 阶段4：LLM集成与工具调用 (2-3天)

#### 4.1 模型集成
```python
# 支持的模型配置
SUPPORTED_MODELS = {
    "qwen3-80b": {
        "provider": "qwen",
        "model": "qwen3-80b",
        "supports_tools": True,
        "max_tokens": 8192
    },
    "qwen3-max": {
        "provider": "qwen",
        "model": "qwen3-max",
        "supports_tools": True,
        "max_tokens": 32768
    },
    "deepseek-reasoner": {
        "provider": "deepseek",
        "model": "deepseek-reasoner",
        "supports_tools": True,
        "max_tokens": 65536
    }
}
```

#### 4.2 工具定义
```python
# rag_search工具定义
tools = [{
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": "搜索医学文档，支持标题优先级和多轮优化",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询字符串"
                },
                "search_type": {
                    "type": "string",
                    "enum": ["keyword", "semantic", "hybrid", "intelligent"],
                    "description": "搜索类型"
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回结果数",
                    "default": 5
                },
                "title_priority": {
                    "type": "boolean",
                    "description": "是否启用标题优先级",
                    "default": True
                }
            },
            "required": ["query"]
        }
    }
}]
```

#### 4.3 工具调用逻辑
- Agent自主决定是否调用工具
- 支持单次和多次工具调用
- 提供工具调用结果的自然语言解释

### 阶段5：API兼容性保持 (1-2天)

#### 5.1 接口兼容
```python
# 保持现有API端点
@app.post("/query/sync")
async def process_query_sync(request: QueryRequest):
    # 内部转发到新的LangGraph Agent
    result = await langgraph_agent.process(request)
    # 转换为原有响应格式
    return convert_to_legacy_format(result)
```

#### 5.2 响应格式兼容
- 保持原有的响应字段结构
- 支持现有的查询参数和配置
- 提供新旧架构的功能开关

#### 5.3 渐进式迁移
```python
# 功能开关配置
FEATURE_FLAGS = {
    "use_langgraph_agent": True,      # 启用新Agent
    "enable_multi_round_search": True, # 启用多轮搜索
    "enhanced_title_weighting": True,  # 启用标题权重增强
    "fallback_to_legacy": True        # 失败时回退到旧系统
}
```

### 阶段6：测试与验证 (2-3天)

#### 6.1 功能测试
```python
# 测试用例设计
test_cases = [
    {
        "query": "黏液腺癌的图像特征是什么？",
        "expected": "应该返回第71页的详细描述",
        "criteria": ["包含黏液湖描述", "包含细胞形态", "来自正确页面"]
    },
    {
        "query": "ROSE技术的诊断标准",
        "expected": "返回相关诊断标准内容",
        "criteria": ["内容相关性", "信息完整性", "标题匹配度"]
    },
    {
        "query": "腺癌和鳞癌的区别",
        "expected": "返回对比分析内容",
        "criteria": ["对比内容", "区别描述", "准确性"]
    }
]
```

#### 6.2 性能测试
- 响应时间对比测试（新旧架构）
- 并发请求处理能力测试
- 内存使用效率测试
- 工具调用成功率测试

#### 6.3 兼容性测试
- API接口兼容性验证
- 前端功能完整性测试
- 数据格式一致性检查
- 错误处理机制测试

## 技术栈

### 核心依赖
- **LangGraph**: 工作流管理和节点编排
- **Pydantic**: 数据验证和序列化
- **FastAPI**: Web服务框架（保持现有）
- **Elasticsearch**: 搜索引擎（保持现有）
- **Milvus**: 向量数据库（保持现有）

### LLM集成
- **OpenAI API标准**: 工具调用接口
- **Qwen API**: 通义千问模型集成
- **DeepSeek API**: DeepSeek模型集成

### 开发工具
- **Docker**: 容器化部署
- **pytest**: 单元测试框架
- **locust**: 性能测试工具

## 预期成果

### 功能改进
- ✅ 更智能的医学查询处理
- ✅ 显著改善搜索结果质量
- ✅ 支持多轮自动优化
- ✅ 加强标题权重优先级
- ✅ 保持前端完全兼容

### 技术指标
- 搜索结果相关性提升30%+
- 详细描述内容返回率提升50%+
- 标题匹配准确度提升40%+
- 响应时间保持在2秒以内

### 可扩展性
- 支持未来添加更多工具（图片返回、高级分析等）
- 支持新LLM模型的快速集成
- 支持新的搜索策略和优化算法

## 风险评估与缓解

### 技术风险
1. **LangGraph学习曲线**: 需要时间熟悉新框架
   - 缓解：充分调研和原型验证

2. **LLM工具调用稳定性**: 不同模型工具调用能力差异
   - 缓解：多模型支持和降级机制

3. **性能影响**: 多轮搜索可能增加响应时间
   - 缓解：设置最大迭代次数，优化搜索算法

### 业务风险
1. **API兼容性**: 可能影响现有前端功能
   - 缓解：充分的兼容性测试和渐进式迁移

2. **搜索结果变化**: 用户可能需要适应新的结果格式
   - 缓解：保持响应格式一致，提供配置选项

## 后续规划

### 短期（1-2个月）
- 图片返回工具集成
- 高级搜索过滤器
- 用户查询历史分析

### 中期（3-6个月）
- 多模态搜索支持（文本+图片）
- 个性化搜索结果
- 高级医学知识图谱集成

### 长期（6个月以上）
- 支持更多医学领域
- 多语言支持
- 边缘计算部署

---

**开发团队**: 基于用户需求定制开发
**开始时间**: 2025年10月20日
**预计完成**: 2025年11月初
**维护周期**: 持续优化和功能扩展