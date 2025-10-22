// 全局变量
let queryHistory = JSON.parse(localStorage.getItem('queryHistory') || '[]');
let currentQueryId = null;
let isLoading = false;

// DOM元素
const elements = {
    queryInput: document.getElementById('query-input'),
    searchBtn: document.getElementById('search-btn'),
    modelSelect: document.getElementById('model-select'),
    embeddingModelSelect: document.getElementById('embedding-model-select'),  // 新增嵌入模型选择
    includeImages: document.getElementById('include-images'),
    showReasoning: document.getElementById('show-reasoning'),
    historySection: document.getElementById('history-section'),
    historyList: document.getElementById('history-list'),
    resultsSection: document.getElementById('results-section'),
    answerContent: document.getElementById('answer-content'),
    confidenceBadge: document.getElementById('confidence-badge'),
    responseTime: document.getElementById('response-time'),
    modelUsed: document.getElementById('model-used'),
    reasoningBox: document.getElementById('reasoning-box'),
    reasoningSteps: document.getElementById('reasoning-steps'),
    sourcesList: document.getElementById('sources-list'),
    imagesSection: document.getElementById('images-section'),
    imagesGrid: document.getElementById('images-grid'),
    loadingOverlay: document.getElementById('loading-overlay'),
    statusIndicator: document.getElementById('status-indicator'),
    statusText: document.querySelector('.status-text'),
    statusDot: document.querySelector('.status-dot')
};

// 初始化
function init() {
    console.log('🚀 初始化医学RAG智能问答系统');

    // 添加全局错误处理
    window.addEventListener('error', function(event) {
        console.error('🚨 全局JavaScript错误:', {
            message: event.message,
            filename: event.filename,
            lineno: event.lineno,
            colno: event.colno,
            error: event.error
        });
    });

    // 绑定事件监听器
    bindEventListeners();

    // 检查系统状态
    checkSystemStatus();

    // 加载历史记录
    loadQueryHistory();

    // 设置定时器
    setInterval(checkSystemStatus, 30000); // 每30秒检查一次系统状态

    console.log('✅ 系统初始化完成');
}

// 事件监听器绑定
function bindEventListeners() {
    // 查询按钮
    elements.searchBtn.addEventListener('click', handleSearch);

    // 回车键查询
    elements.queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSearch();
        }
    });

    // 快速建议标签
    document.querySelectorAll('.suggestion-tag').forEach(tag => {
        tag.addEventListener('click', () => {
            const question = tag.dataset.question;
            elements.queryInput.value = question;
            handleSearch();
        });
    });

    // 图片模态框
    document.getElementById('image-modal').addEventListener('click', (e) => {
        if (e.target.id === 'image-modal') {
            closeImageModal();
        }
    });

    document.querySelector('.modal-close').addEventListener('click', closeImageModal);

    // ESC键关闭模态框
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeImageModal();
        }
    });
}

// 系统状态检查
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        if (response.ok && data.status === 'healthy') {
            updateStatusIndicator('healthy', '系统运行正常');
            updateStats(data);
        } else {
            updateStatusIndicator('warning', '系统状态异常');
        }
    } catch (error) {
        console.error('系统状态检查失败:', error);
        updateStatusIndicator('error', '系统连接失败');
    }
}

// 更新状态指示器
function updateStatusIndicator(status, message) {
    elements.statusText.textContent = message;
    elements.statusIndicator.className = 'status-indicator';

    if (status === 'healthy') {
        elements.statusDot.style.background = '#28a745';
    } else if (status === 'warning') {
        elements.statusDot.style.background = '#ffc107';
        elements.statusIndicator.classList.add('warning');
    } else {
        elements.statusDot.style.background = '#dc3545';
        elements.statusIndicator.classList.add('error');
    }
}

// 更新统计数据
function updateStats(data) {
    if (data.components && data.stats) {
        const stats = data.stats;
        document.getElementById('query-count').textContent = stats.total_queries || 0;
        document.getElementById('avg-response-time').textContent =
            (stats.average_response_time || 0).toFixed(2);
    }
}

// 查询处理
async function handleSearch() {
    if (isLoading) return;

    const question = elements.queryInput.value.trim();
    if (!question) {
        showNotification('请输入查询内容', 'warning');
        return;
    }

    if (question.length < 2) {
        showNotification('查询内容太短', 'warning');
        return;
    }

    console.log('🔍 开始查询:', question);

    // 更新UI状态
    setLoading(true);
    hideResults();

    // 生成查询ID
    currentQueryId = generateQueryId();

    // 记录到历史
    addToHistory(question);

    try {
        // 构建请求数据
        const requestData = {
            question: question,
            user_id: 'web_user_' + Date.now(),
            search_config: {
                search_type: 'hybrid',
                keyword_weight: 0.5
            },
            metadata: {
                source: 'web_frontend',
                include_images: elements.includeImages.checked,
                show_reasoning: elements.showReasoning.checked
            }
        };

        // 添加模型选择
        const selectedModel = elements.modelSelect.value;
        if (selectedModel !== 'mock') {
            requestData.search_config.model_provider = selectedModel;
        }

        // 添加嵌入模型选择
        const selectedEmbeddingModel = elements.embeddingModelSelect.value;
        requestData.search_config.embedding_model = selectedEmbeddingModel;

        // 发送查询请求
        const startTime = Date.now();
        const response = await fetch('/api/query/sync', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        const endTime = Date.now();
        const responseTime = (endTime - startTime) / 1000;

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        console.log('✅ 查询成功:', data);

        // 显示结果
        displayResults(data, responseTime);

        // 更新统计
        updateQueryStats();

    } catch (error) {
        console.error('❌ 查询失败:', error);
        showError('查询失败: ' + error.message);
    } finally {
        setLoading(false);
    }
}

// 显示查询结果
function displayResults(data, responseTime) {
    // 显示结果区域
    elements.resultsSection.style.display = 'block';
    elements.resultsSection.classList.add('fade-in-up');

    // 显示答案
    displayAnswer(data, responseTime);

    // 显示推理过程（如果启用）
    console.log('🔍 检查推理过程显示条件:', {
        showReasoningChecked: elements.showReasoning.checked,
        reasoningStepsExist: !!data.reasoning_steps,
        reasoningStepsLength: data.reasoning_steps ? data.reasoning_steps.length : 0,
        reasoningStepsData: data.reasoning_steps
    });

    if (elements.showReasoning.checked && data.reasoning_steps) {
        console.log('✅ 条件满足，开始显示推理过程');
        displayReasoning(data.reasoning_steps);
    } else {
        console.log('❌ 条件不满足，不显示推理过程');
    }

    // 显示参考来源
    if (data.retrieved_documents && data.retrieved_documents.length > 0) {
        displaySources(data.retrieved_documents);
    }

    // 显示图片（如果启用且有图片）
    if (elements.includeImages.checked && data.metadata && data.metadata.images) {
        displayImages(data.metadata.images);
    }

    // 滚动到结果区域
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 格式化置信度显示
function formatConfidence(confidence) {
    const percentage = Math.round(confidence * 100);

    // 根据置信度确定级别和颜色
    let level, color;
    if (confidence >= 0.9) {
        level = '极高';
        color = '#28a745';
    } else if (confidence >= 0.8) {
        level = '高';
        color = '#20c997';
    } else if (confidence >= 0.6) {
        level = '中等';
        color = '#ffc107';
    } else if (confidence >= 0.4) {
        level = '一般';
        color = '#fd7e14';
    } else {
        level = '较低';
        color = '#dc3545';
    }

    return {
        percentage: percentage,
        level: level,
        color: color,
        display: `${percentage}% (${level})`
    };
}

// 显示答案
function displayAnswer(data, responseTime) {
    elements.answerContent.innerHTML = formatAnswer(data.answer);

    // 格式化置信度显示
    const confidenceInfo = formatConfidence(data.confidence);
    elements.confidenceBadge.textContent = confidenceInfo.display;
    elements.confidenceBadge.style.background = `linear-gradient(135deg, ${confidenceInfo.color}, ${confidenceInfo.color})`;

    elements.responseTime.textContent = `响应时间: ${responseTime.toFixed(2)}s`;
    elements.modelUsed.textContent = `模型: ${data.model_used || '未知'}`;
}

// 格式化答案
function formatAnswer(answer) {
    // 处理可能的LLMResponse对象字符串表示
    if (typeof answer === 'string' && answer.includes('LLMResponse(')) {
        console.log('🔍 检测到LLMResponse对象格式，开始解析...');

        try {
            // 方法1: 尝试解析为类似Python对象的格式
            let extractedContent = null;

            // 尝试匹配content='...' 或 content="..."
            let contentRegex = /content=(['"])([^\1]*?)\1/;
            let match = answer.match(contentRegex);

            if (match) {
                extractedContent = match[2];
            } else {
                // 方法2: 尝试找到content=后面的完整内容
                const contentStart = answer.indexOf('content=');
                if (contentStart !== -1) {
                    let remaining = answer.substring(contentStart + 8);

                    // 如果剩余部分以引号开始，找到匹配的结束引号
                    if (remaining.startsWith('"') || remaining.startsWith("'")) {
                        const quote = remaining[0];
                        let endIndex = -1;
                        let escapeCount = 0;

                        // 查找未转义的结束引号
                        for (let i = 1; i < remaining.length; i++) {
                            if (remaining[i] === '\\') {
                                escapeCount++;
                                i++; // 跳过下一个字符
                            } else if (remaining[i] === quote && escapeCount % 2 === 0) {
                                endIndex = i;
                                break;
                            } else {
                                escapeCount = 0;
                            }
                        }

                        if (endIndex !== -1) {
                            extractedContent = remaining.substring(1, endIndex);
                        }
                    } else {
                        // 如果没有引号，匹配到下一个逗号或右括号
                        const endMatch = remaining.match(/^([^,\)]+)/);
                        if (endMatch) {
                            extractedContent = endMatch[1].trim();
                        }
                    }
                }
            }

            // 方法3: 如果上述方法都失败，尝试粗暴提取
            if (!extractedContent) {
                // 尝试找到content=和model=之间的内容
                const modelStart = answer.indexOf(', model=');
                const contentStart = answer.indexOf('content=');

                if (contentStart !== -1 && modelStart !== -1 && modelStart > contentStart) {
                    let contentSection = answer.substring(contentStart + 8, modelStart);

                    // 去除开头的引号
                    if (contentSection.startsWith('"') || contentSection.startsWith("'")) {
                        const quote = contentSection[0];
                        if (contentSection.endsWith(quote)) {
                            extractedContent = contentSection.slice(1, -1);
                        }
                    }
                }
            }

            if (extractedContent) {
                answer = extractedContent;
                // 处理转义字符
                answer = answer.replace(/\\n/g, '\n').replace(/\\'/g, "'").replace(/\\"/g, '"').replace(/\\\\/g, '\\');
                console.log('✅ 成功提取content内容:', answer.substring(0, 100) + '...');
            } else {
                console.log('❌ 无法提取content，尝试备用解析方案');

                // 备用方案：尝试找到第一个model=之前的所有内容作为content
                const modelIndex = answer.indexOf(', model=');
                const contentIndex = answer.indexOf('content=');

                if (contentIndex !== -1 && modelIndex !== -1) {
                    let potentialContent = answer.substring(contentIndex + 8, modelIndex);

                    // 清理开头和结尾的引号
                    potentialContent = potentialContent.replace(/^['"]|['"]$/g, '');

                    if (potentialContent.length > 10) { // 确保内容足够长
                        answer = potentialContent;
                        answer = answer.replace(/\\n/g, '\n').replace(/\\'/g, "'").replace(/\\"/g, '"');
                        console.log('✅ 使用备用方案提取content:', answer.substring(0, 100) + '...');
                    }
                }
            }
        } catch (error) {
            console.log('❌ 解析LLMResponse对象时出错:', error);
        }
    }

    // 将答案分段并添加适当的HTML格式
    const paragraphs = answer.split('\n\n').filter(p => p.trim());
    let formattedHtml = '';

    paragraphs.forEach(paragraph => {
        const trimmed = paragraph.trim();
        if (trimmed.startsWith('【') && trimmed.endsWith('】')) {
            // 标题格式
            formattedHtml += `<h4>${trimmed}</h4>`;
        } else if (trimmed.startsWith('###')) {
            // Markdown风格的标题
            const title = trimmed.replace(/^###\s*/, '');
            formattedHtml += `<h4>${title}</h4>`;
        } else if (trimmed.startsWith('####')) {
            // Markdown风格的子标题
            const title = trimmed.replace(/^####\s*/, '');
            formattedHtml += `<h5>${title}</h5>`;
        } else if (trimmed.includes('：')) {
            // 包含冒号的可能是列表或重点
            const parts = trimmed.split('：');
            if (parts.length === 2) {
                formattedHtml += `<p><strong>${parts[0]}：</strong>${parts[1]}</p>`;
            } else {
                formattedHtml += `<p>${trimmed}</p>`;
            }
        } else if (trimmed.startsWith('- ')) {
            // 列表项
            const listItems = trimmed.split('\n').filter(item => item.trim().startsWith('- '));
            formattedHtml += '<ul class="answer-list">';
            listItems.forEach(item => {
                const content = item.replace(/^-\s*/, '');
                formattedHtml += `<li>${content}</li>`;
            });
            formattedHtml += '</ul>';
        } else if (trimmed.startsWith('1. ')) {
            // 有序列表
            const listItems = trimmed.split('\n').filter(item => item.trim().match(/^\d+\.\s/));
            formattedHtml += '<ol class="answer-list">';
            listItems.forEach(item => {
                const content = item.replace(/^\d+\.\s*/, '');
                formattedHtml += `<li>${content}</li>`;
            });
            formattedHtml += '</ol>';
        } else if (trimmed.includes('\n')) {
            // 多行文本，每行可能是一个要点
            const lines = trimmed.split('\n').filter(line => line.trim());
            if (lines.length > 1) {
                formattedHtml += '<div class="answer-multi-line">';
                lines.forEach(line => {
                    const cleanLine = line.trim();
                    if (cleanLine) {
                        formattedHtml += `<p>${cleanLine}</p>`;
                    }
                });
                formattedHtml += '</div>';
            } else {
                formattedHtml += `<p>${trimmed}</p>`;
            }
        } else {
            formattedHtml += `<p>${trimmed}</p>`;
        }
    });

    return formattedHtml;
}

// 显示推理过程
function displayReasoning(reasoningSteps) {
    console.log('🚀 开始显示推理过程，步骤数:', reasoningSteps.length);

    try {
        elements.reasoningBox.style.display = 'block';
        console.log('✅ 推理盒子已设置为显示状态');

        elements.reasoningSteps.innerHTML = '';
        console.log('✅ 推理步骤容器已清空');

        reasoningSteps.forEach((step, index) => {
            console.log(`📋 处理步骤 ${index + 1}:`, step);

            const stepElement = document.createElement('div');
            stepElement.className = 'reasoning-step slide-in-right';
            stepElement.style.animationDelay = `${index * 0.1}s`;

            const stepTitle = step.step || '步骤 ' + (index + 1);
            const stepTime = formatTime(step.timestamp);
            const stepContent = step.thought || step.content || '';

            console.log(`  步骤标题: ${stepTitle}`);
            console.log(`  步骤时间: ${stepTime}`);
            console.log(`  步骤内容: ${stepContent}`);

            stepElement.innerHTML = `
                <div class="step-header">
                    <span class="step-title">${stepTitle}</span>
                    <span class="step-time">${stepTime}</span>
                </div>
                <div class="step-content">
                    ${stepContent}
                </div>
            `;

            elements.reasoningSteps.appendChild(stepElement);
            console.log(`✅ 步骤 ${index + 1} 添加完成`);
        });

        console.log('✅ 推理过程显示完成');
    } catch (error) {
        console.error('❌ 显示推理过程时出错:', error);
        console.error('错误堆栈:', error.stack);
    }
}

// 格式化相关性分数显示
function formatRelevanceScore(score) {
    const percentage = (score * 100).toFixed(1);

    // 根据分数确定级别和颜色
    let level, color, icon;
    if (score >= 0.9) {
        level = '极高';
        color = '#28a745';
        icon = '🔥';
    } else if (score >= 0.8) {
        level = '高';
        color = '#20c997';
        icon = '⭐';
    } else if (score >= 0.6) {
        level = '中等';
        color = '#ffc107';
        icon = '👍';
    } else if (score >= 0.4) {
        level = '一般';
        color = '#fd7e14';
        icon = '📄';
    } else {
        level = '较低';
        color = '#6c757d';
        icon = '🔍';
    }

    return {
        percentage: percentage,
        level: level,
        color: color,
        icon: icon,
        display: `${icon} ${percentage}% (${level})`
    };
}

// 显示参考来源
function displaySources(sources) {
    elements.sourcesList.innerHTML = '';

    sources.forEach((source, index) => {
        const sourceElement = document.createElement('div');
        sourceElement.className = 'source-item';
        sourceElement.style.animationDelay = `${index * 0.1}s`;

        // 构建章节显示HTML
        let chapterHtml = '';
        if (source.chapter_title) {
            chapterHtml += `<div class="chapter-title"><i class="fas fa-book"></i> ${source.chapter_title}</div>`;
        }
        if (source.section_title) {
            chapterHtml += `<div class="section-title"><i class="fas fa-bookmark"></i> ${source.section_title}</div>`;
        }

        // 格式化相关性分数
        const scoreInfo = formatRelevanceScore(source.score);

        sourceElement.innerHTML = `
            <div class="source-header">
                <div class="source-info">
                    ${chapterHtml || '<span class="no-chapter">📄 医学文献内容</span>'}
                </div>
                <span class="source-score" style="background: ${scoreInfo.color} !important;">
                    ${scoreInfo.percentage}%
                </span>
            </div>
            <div class="source-content">${source.content || '暂无内容'}</div>
            <div class="source-meta">
                <span><i class="fas fa-file"></i> 第${source.page_number || '?'}页</span>
                <span><i class="fas fa-search"></i> ${source.search_type || '混合搜索'}</span>
                <span><i class="fas fa-star"></i> ${scoreInfo.display}</span>
            </div>
        `;

        elements.sourcesList.appendChild(sourceElement);
    });
}

// 显示图片
function displayImages(images) {
    if (!images || images.length === 0) return;

    elements.imagesSection.style.display = 'block';
    elements.imagesGrid.innerHTML = '';

    images.forEach((image, index) => {
        const imageElement = document.createElement('div');
        imageElement.className = 'image-item';
        imageElement.style.animationDelay = `${index * 0.1}s`;

        imageElement.innerHTML = `
            <img src="${image.url}" alt="${image.caption || '医学图片'}" onclick="openImageModal('${image.url}', '${image.caption || ''}')">
            <div class="image-caption">${image.caption || '医学图片'}</div>
        `;

        elements.imagesGrid.appendChild(imageElement);
    });
}

// 图片模态框
function openImageModal(imageUrl, caption) {
    const modal = document.getElementById('image-modal');
    const modalImage = document.getElementById('modal-image');
    const modalCaption = document.getElementById('modal-caption');

    modalImage.src = imageUrl;
    modalCaption.textContent = caption;
    modal.style.display = 'block';

    // 添加动画效果
    setTimeout(() => {
        modal.style.opacity = '1';
    }, 10);
}

function closeImageModal() {
    const modal = document.getElementById('image-modal');
    modal.style.opacity = '0';

    setTimeout(() => {
        modal.style.display = 'none';
    }, 300);
}

// 查询历史管理
function loadQueryHistory() {
    if (queryHistory.length > 0) {
        elements.historySection.style.display = 'block';
        renderQueryHistory();
    }
}

function renderQueryHistory() {
    elements.historyList.innerHTML = '';

    // 显示最近10条记录
    const recentHistory = queryHistory.slice(-10).reverse();

    recentHistory.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';

        historyItem.innerHTML = `
            <span class="history-question">${item.question}</span>
            <span class="history-time">${formatTime(item.timestamp)}</span>
        `;

        historyItem.addEventListener('click', () => {
            elements.queryInput.value = item.question;
            handleSearch();
        });

        elements.historyList.appendChild(historyItem);
    });
}

function addToHistory(question) {
    const historyItem = {
        question: question,
        timestamp: new Date().toISOString(),
        queryId: currentQueryId
    };

    // 避免重复
    queryHistory = queryHistory.filter(item => item.question !== question);
    queryHistory.push(historyItem);

    // 限制历史记录数量
    if (queryHistory.length > 50) {
        queryHistory = queryHistory.slice(-50);
    }

    // 保存到本地存储
    localStorage.setItem('queryHistory', JSON.stringify(queryHistory));

    // 更新界面
    elements.historySection.style.display = 'block';
    renderQueryHistory();
}

// UI状态管理
function setLoading(loading) {
    isLoading = loading;
    elements.searchBtn.disabled = loading;
    elements.loadingOverlay.style.display = loading ? 'flex' : 'none';

    if (loading) {
        elements.searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 查询中...';
    } else {
        elements.searchBtn.innerHTML = '<i class="fas fa-search"></i> 查询';
    }
}

function hideResults() {
    elements.resultsSection.style.display = 'none';
    elements.reasoningBox.style.display = 'none';
    elements.imagesSection.style.display = 'none';
}

// 工具函数
function generateQueryId() {
    return 'web_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return '刚刚';
    if (diffMins < 60) return `${diffMins}分钟前`;
    if (diffHours < 24) return `${diffHours}小时前`;
    if (diffDays < 7) return `${diffDays}天前`;

    return date.toLocaleDateString('zh-CN');
}

function showNotification(message, type = 'info') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;

    // 添加样式
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#28a745' : type === 'warning' ? '#ffc107' : '#17a2b8'};
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        z-index: 1000;
        animation: slideInRight 0.3s ease-out;
    `;

    document.body.appendChild(notification);

    // 自动移除
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

function showError(message) {
    showNotification(message, 'error');
}

function updateQueryStats() {
    // 更新查询统计（这里可以添加更复杂的统计逻辑）
    const currentCount = parseInt(document.getElementById('query-count').textContent);
    document.getElementById('query-count').textContent = currentCount + 1;
}

// 辅助功能
function showAPIStatus() {
    // 显示API状态信息
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            showNotification(`系统状态: ${data.status}`, 'info');
        })
        .catch(error => {
            showNotification('无法获取系统状态', 'error');
        });
}

function showSystemStats() {
    // 显示系统统计信息
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            const message = `总查询数: ${data.total_queries}\n成功率: ${((data.successful_queries / data.total_queries) * 100).toFixed(1)}%\n平均响应时间: ${data.average_response_time.toFixed(2)}s`;
            showNotification(message, 'info');
        })
        .catch(error => {
            showNotification('无法获取系统统计', 'error');
        });
}

function showHelp() {
    // 显示使用帮助
    const helpMessage = `使用帮助：
1. 在输入框中输入医学问题
2. 选择适当的AI模型
3. 点击查询按钮或按回车键
4. 查看AI回答和参考来源
5. 可以查看推理过程和图片（如果有）

快捷键：Enter - 查询，ESC - 关闭图片`;

    showNotification(helpMessage, 'info');
}

// 添加CSS动画样式
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }

    .notification {
        font-family: inherit;
        max-width: 300px;
    }

    .notification-content {
        display: flex;
        align-items: center;
        gap: 10px;
    }
`;
document.head.appendChild(style);

// 初始化系统
document.addEventListener('DOMContentLoaded', init);