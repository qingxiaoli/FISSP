#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
download_fissp.py

示例脚本：从公开可用的大规模细粒度数据集和搜索引擎（结合 ConceptNet）收集 FISSP 数据集所需的图片。

Usage:
    python scripts/download_fissp.py --data_dir ./data
"""

import os
import argparse
import requests
import shutil
import hashlib
import random
import time
from urllib.parse import quote

# ==========================
# 1. 公开可用数据集下载示例
# ==========================
def download_public_finegrained_datasets(data_dir: str):
    """
    从公开可用的细粒度数据集（iNaturalist、FGVC Workshop、ImageCLEF 等）中下载数据的示例函数。
    这里仅提供示例思路，具体如何下载请参考这些数据集的官方文档。
    """

    # 例如：iNaturalist 数据集（官方链接示例）
    # https://github.com/visipedia/inat_comp
    # 伪代码示例：
    print("[INFO] Downloading iNaturalist dataset (示例)...")
    # inat_url = "http://example.com/inat_dataset.zip"
    # local_zip_path = os.path.join(data_dir, "inat_dataset.zip")
    # download_file(inat_url, local_zip_path)
    # 解压等后续处理...

    # 例如：FGVC Workshop 官方下载
    # https://www.kaggle.com/c/fgvc8-bitgrayscale
    # ...此处同理，放置下载与解压代码...

    # 例如：ImageCLEF
    # 同理...

    print("[INFO] Public dataset download finished (示例)!\n")


def download_file(url: str, local_path: str):
    """
    下载文件的辅助函数。
    """
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


# ==========================
# 2. ConceptNet 查询与图像爬取
# ==========================

def get_related_concepts_from_conceptnet(target_concept: str):
    """
    从 ConceptNet 获取相关概念的示例。
    这里演示如何通过查询 ConceptNet API 获取和目标概念相关的概念。
    文档参考：https://github.com/commonsense/conceptnet5/wiki/API
    """
    # 这里仅写示例 URL，实际使用时需根据官方文档调整 endpoint 和参数。
    base_url = "http://api.conceptnet.io/query"
    params = {
        "start": f"/c/en/{target_concept}",  # 假设目标概念是英文
        "rel": "/r/RelatedTo",
        "limit": 10,
    }
    related_concepts = []

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        edges = data.get("edges", [])

        for edge in edges:
            end = edge.get("end", {})
            label = end.get("label", "")
            # 只要有具体词汇且与目标概念不重复，就添加
            if label and label.lower() != target_concept.lower():
                related_concepts.append(label)

    except Exception as e:
        print(f"[WARNING] Failed to query ConceptNet for {target_concept}: {e}")

    # 返回去重后的概念列表
    return list(set(related_concepts))


def generate_search_queries(target_concept: str, related_concepts: list):
    """
    针对目标概念及其相关概念，生成图像搜索的关键词列表。
    """
    queries = []
    # 目标概念本身
    queries.append(f"{target_concept} images")

    # 相关概念
    for rc in related_concepts:
        # 简单处理一下，生成更具体的查询
        queries.append(f"{rc} {target_concept} images")

    return queries


def crawl_images_for_queries(queries, data_dir, concept):
    """
    给定查询列表，爬取搜索引擎图片。
    这里仅展示爬虫示例逻辑，实际实现中需要考虑搜索引擎反爬、翻页爬取、API 限制、法律与版权等问题。
    """
    # 假设我们支持多家搜索引擎，做一个简单的循环。
    search_engines = [
        ("Google", "https://www.google.com/search?tbm=isch&q="),
        ("Bing", "https://www.bing.com/images/search?q="),
        ("Baidu", "https://image.baidu.com/search/index?tn=baiduimage&word="),
        # ("Yandex", "https://yandex.com/images/search?text="), # 其他搜索引擎...
    ]

    concept_dir = os.path.join(data_dir, "conceptnet", concept)
    os.makedirs(concept_dir, exist_ok=True)

    # 仅示意：爬取若干图片链接
    # 实际中需要解析 HTML、JSON 或者使用搜索引擎官方 API
    for engine_name, base_url in search_engines:
        for query in queries:
            encoded_query = quote(query)
            search_url = base_url + encoded_query
            print(f"[INFO] Crawling {engine_name} with query: {query}")
            
            # 伪代码：请求页面并解析图片链接
            try:
                resp = requests.get(search_url, timeout=10)
                # 解析 resp.text 中的图片链接 <img ... src="xxx" ...> 
                # 这里仅做模拟
                fake_image_links = simulate_extract_image_links(resp.text, max_images=5)

                # 下载每张图片
                for link in fake_image_links:
                    download_and_save_image(link, concept_dir)

                # 适当等待，避免触发反爬机制
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                print(f"[WARNING] Failed to crawl {engine_name}: {e}")
                continue


def simulate_extract_image_links(html_content, max_images=5):
    """
    这里做一个简单的模拟，假设我们从搜索结果页面中提取到的链接列表。
    实际中需要使用 HTML 解析库（如 BeautifulSoup、lxml 等）来提取。
    """
    # 模拟生成一些图片链接
    # 在真实情况下，需要根据搜索引擎返回的 HTML/JSON 格式进行解析
    dummy_links = []
    for i in range(max_images):
        dummy_links.append(f"https://example.com/image_{i}.jpg")
    return dummy_links


def download_and_save_image(img_url, save_dir):
    """
    从 img_url 下载图片并存储到指定目录。
    """
    try:
        response = requests.get(img_url, stream=True, timeout=10)
        response.raise_for_status()
        # 构造文件名
        file_name = os.path.basename(img_url)
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            file_name = file_name + ".jpg"
        local_path = os.path.join(save_dir, file_name)

        with open(local_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
    except Exception as e:
        print(f"[WARNING] Failed to download {img_url}: {e}")


# ==========================
# 3. 图片过滤和去重
# ==========================

def filter_and_deduplicate_images(root_dir):
    """
    简单地对图片进行去重（基于哈希），并可以进行一些规则过滤。
    例如过滤文件过小、带有水印等（此处仅做示例）。
    """
    print("[INFO] Starting image deduplication and filtering...")
    seen_hashes = set()

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, file)
                # 读取图片并计算哈希
                try:
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                        file_hash = hashlib.md5(file_bytes).hexdigest()

                    if file_hash in seen_hashes:
                        # 删除重复文件
                        os.remove(file_path)
                    else:
                        seen_hashes.add(file_hash)

                    # 其他过滤逻辑示例：若文件过小，删除
                    if os.path.getsize(file_path) < 1024:  # 小于1KB认定为无效图片
                        os.remove(file_path)

                except Exception as e:
                    print(f"[WARNING] Failed to process {file_path}: {e}")
                    continue


# ==========================
# 4. 主函数逻辑
# ==========================

def download_fissp(data_dir: str):
    """
    整体流程：
    1. 下载公开可用的细粒度数据集（示例：iNaturalist、FGVC、ImageCLEF）。
    2. 利用 ConceptNet 查询并生成搜索词，然后从搜索引擎爬取图片。
    3. 对所有下载图片进行过滤和去重。
    """
    # 创建根目录
    os.makedirs(data_dir, exist_ok=True)

    # 第一步：从公开数据源下载
    download_public_finegrained_datasets(data_dir)

    # 假设在我们的场景中，需要补充一些在公开数据集中还不够充分的概念
    target_concepts = ["car", "butterfly", "flower"]  # 仅作示例
    for tc in target_concepts:
        print(f"\n[INFO] Processing target concept: {tc}")
        related_concepts = get_related_concepts_from_conceptnet(tc)
        queries = generate_search_queries(tc, related_concepts)
        crawl_images_for_queries(queries, data_dir, tc)

    # 去重和简单过滤
    filter_and_deduplicate_images(data_dir)

    print("[INFO] FISSP data download & processing complete!")


def main():
    parser = argparse.ArgumentParser(description="FISSP Data Collection Script")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to save the downloaded FISSP dataset.")
    args = parser.parse_args()

    download_fissp(args.data_dir)


if __name__ == "__main__":
    main()