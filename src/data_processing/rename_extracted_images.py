#!/usr/bin/env python3
"""
重命名已提取的图像文件
根据具体要求：
1. 页面3、5、6的人物图像分别重命名为：冯靖.png、靳芳.png、植佳丽.png
2. 其余肿瘤图像识别底部文字，提取图号命名（如"图2-1 腺癌（分化较高）（1）"→"图2-1.png"）
"""

import os
import re
import cv2
import numpy as np
from paddleocr import PaddleOCR
from pathlib import Path


def extract_text_from_image_bottom(image_path: str) -> str:
    """从图像底部提取文字"""
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return ""

        # 获取图像尺寸
        height, width = img.shape[:2]

        # 提取底部25%区域（通常包含图注）
        bottom_height = int(height * 0.25)
        bottom_region = img[height-bottom_height:height, 0:width]

        # 使用OCR提取文字
        ocr = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False)
        result = ocr.ocr(bottom_region, cls=False)

        if not result or not result[0]:
            return ""

        # 提取所有识别到的文字
        texts = []
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0]
                confidence = line[1][1]
                if confidence > 0.5:  # 置信度过滤
                    texts.append(text)

        return ' '.join(texts)

    except Exception as e:
        print(f"OCR处理失败 {image_path}: {e}")
        return ""


def extract_figure_number(text: str) -> str:
    """从文字中提取图号"""
    if not text:
        return ""

    # 匹配图号模式，允许"图"字后有空格
    patterns = [
        r'图\s*(\d+-\d+[a-z]?)',      # 图 2-1, 图2-1, 图 2-1a
        r'图\s*(\d+[a-z]?)',         # 图 1, 图1, 图 3a
        r'图\s*([一二三四五六七八九十]+[a-z]?)',  # 图 一, 图一, 图 二a
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return f"图{matches[0]}"

    return ""


def rename_portrait_images(source_dir: str, target_dir: str):
    """重命名人物肖像图像"""
    portrait_mappings = {
        'page_3_detected_0.png': '人物_冯靖.png',
        'page_5_detected_0.png': '人物_靳芳.png',
        'page_6_detected_0.png': '人物_植佳丽.png'
    }

    for old_name, new_name in portrait_mappings.items():
        old_path = os.path.join(source_dir, old_name)
        new_path = os.path.join(target_dir, new_name)

        if os.path.exists(old_path):
            try:
                # 复制并重命名文件
                import shutil
                shutil.copy2(old_path, new_path)
                print(f"✓ 重命名: {old_name} -> {new_name}")
            except Exception as e:
                print(f"✗ 重命名失败 {old_name}: {e}")
        else:
            print(f"✗ 文件不存在: {old_path}")


def rename_tumor_images(source_dir: str, target_dir: str):
    """重命名肿瘤图像"""
    # 获取所有图像文件
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

    # 过滤出非人物肖像的图像（即肿瘤图像）
    portrait_files = {'page_3_detected_0.png', 'page_5_detected_0.png', 'page_6_detected_0.png'}
    tumor_files = [f for f in image_files if f not in portrait_files]

    print(f"\n开始处理肿瘤图像，共 {len(tumor_files)} 个文件...")

    for image_file in tumor_files:
        old_path = os.path.join(source_dir, image_file)

        try:
            # 从图像底部提取文字
            text = extract_text_from_image_bottom(old_path)

            if text:
                print(f"\n处理 {image_file}:")
                print(f"  提取的文字: {text}")

                # 提取图号
                figure_num = extract_figure_number(text)
                print(f"  提取的图号: {figure_num}")

                if figure_num:
                    new_name = f"{figure_num}.png"
                else:
                    # 如果无法提取图号，使用原文件名作为备份
                    base_name = image_file.replace('.png', '')
                    new_name = f"{base_name}.png"
                    print(f"  警告: 无法提取有效图号，使用原文件名")
            else:
                # OCR未识别到文字，使用原文件名
                base_name = image_file.replace('.png', '')
                new_name = f"{base_name}.png"
                print(f"  警告: OCR未识别到文字，使用原文件名")

            new_path = os.path.join(target_dir, new_name)

            # 复制文件
            import shutil
            shutil.copy2(old_path, new_path)
            print(f"  重命名: {image_file} -> {new_name}")

        except Exception as e:
            print(f"处理失败 {image_file}: {e}")


def main():
    """主函数"""
    # 设置路径
    source_dir = "/home/ubuntu/myproject/zhenlikeji2/data/extracted_images"
    target_dir = "/home/ubuntu/myproject/zhenlikeji2/data/renamed_images"

    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    print("开始重命名图像文件...")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print("-" * 50)

    # 1. 重命名人物肖像
    print("\n1. 重命名人物肖像图像...")
    rename_portrait_images(source_dir, target_dir)

    # 2. 重命名肿瘤图像
    print("\n2. 重命名肿瘤图像...")
    rename_tumor_images(source_dir, target_dir)

    print("\n" + "=" * 50)
    print("重命名完成！")
    print(f"结果保存在: {target_dir}")


if __name__ == "__main__":
    main()