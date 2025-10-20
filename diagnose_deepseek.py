#!/usr/bin/env python3
"""
诊断DeepSeek客户端问题
"""

import sys
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

print("=" * 80)
print("DeepSeek客户端诊断")
print("=" * 80)

# 检查openai版本
try:
    import openai
    print(f"✓ OpenAI库版本: {openai.__version__}")
except Exception as e:
    print(f"✗ 无法导入OpenAI库: {e}")
    sys.exit(1)

# 检查API密钥
api_key = os.getenv('DEEPSEEK_API_KEY')
base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

if api_key:
    print(f"✓ 找到API密钥: {api_key[:8]}...{api_key[-4:]}")
    print(f"✓ Base URL: {base_url}")
else:
    print("✗ 未找到DEEPSEEK_API_KEY")
    sys.exit(1)

print("\n" + "=" * 80)
print("尝试1: 最简单的客户端初始化")
print("=" * 80)

try:
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    print("✓ 成功创建同步客户端！")

    # 尝试一个简单的API调用
    print("\n测试API调用...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "你好，请回复'测试成功'"}
        ],
        max_tokens=20
    )
    print(f"✓ API调用成功！")
    print(f"  响应: {response.choices[0].message.content}")

except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("尝试2: 检查OpenAI客户端支持的参数")
print("=" * 80)

try:
    import inspect
    sig = inspect.signature(openai.OpenAI.__init__)
    print("OpenAI.__init__支持的参数:")
    for param_name, param in sig.parameters.items():
        if param_name != 'self':
            print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
except Exception as e:
    print(f"✗ 无法获取参数信息: {e}")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
