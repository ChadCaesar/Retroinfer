# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import shutil
import glob
import json
import time
import requests
import html2text
from bs4 import BeautifulSoup
from tqdm import tqdm

os.environ["PYTHONHASHSEED"] = "42"

temp_folder_repo = "essay_repo"
temp_folder_html = "essay_html"
TIMEOUT = (10, 60)      # (连接超时, 读取超时)
RETRIES = 3             # 最大重试次数
BACKOFF_FACTOR = 1      # 重试等待时间倍数
DELAY_BETWEEN = 0.5     # 请求间隔，避免触发反爬

headers = {
    "User-Agent": "Mozilla/5.0 (compatible; NeedleHaystack/1.0)"
}
# -------------

os.makedirs(temp_folder_repo, exist_ok=True)
os.makedirs(temp_folder_html, exist_ok=True)

h = html2text.HTML2Text()
h.ignore_images = True
h.ignore_tables = True
h.escape_all = True
h.reference_links = False
h.mark_code = False

def convert_github_url(url):
    if url.startswith("https://github.com/") and "/raw/" in url:
        parts = url.replace("https://github.com/", "").split("/")
        user, repo, _, branch = parts[:4]
        path = "/".join(parts[4:])
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url

def download_with_retry(url, mode="text"):
    for attempt in range(RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT)
            resp.raise_for_status()
            
            if mode == "html":
                return resp.text
            else:
                return resp.content.decode("utf-8", errors="replace")
        except requests.exceptions.SSLError as e:
            print(f"SSL 错误，跳过 {url}: {e}")
            raise
        except Exception as e:
            print(f"[尝试 {attempt+1}/{RETRIES}] 下载 {url} 失败: {e}")
            if attempt < RETRIES - 1:
                sleep_time = BACKOFF_FACTOR * (2 ** attempt)
                print(f"  等待 {sleep_time:.1f} 秒后重试...")
                time.sleep(sleep_time)
            else:
                print(f"  重试耗尽，最终失败。")
                raise

with open("PaulGrahamEssays_URLs.txt") as f:
    urls = [line.strip() for line in f if line.strip()]

failed_urls = []

for url in tqdm(urls, desc="下载进度"):
    url = convert_github_url(url)
    if ".html" in url:
        filename = url.split("/")[-1].replace(".html", ".txt")
        dest_path = os.path.join(temp_folder_html, filename)
    else:
        filename = url.split("/")[-1]
        dest_path = os.path.join(temp_folder_repo, filename)
    if os.path.exists(dest_path):
        continue
    try:
        if ".html" in url:
            filename = url.split("/")[-1].replace(".html", ".txt")
            html_content = download_with_retry(url, mode="html")
            soup = BeautifulSoup(html_content, "html.parser")
            specific_tag = soup.find("font")
            text = h.handle(str(specific_tag)) if specific_tag else ""
            dest_path = os.path.join(temp_folder_html, filename)
        else:
            filename = url.split("/")[-1]
            text = download_with_retry(url, mode="text")
            dest_path = os.path.join(temp_folder_repo, filename)

        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(text)

        time.sleep(DELAY_BETWEEN)

    except Exception as e:
        print(f"最终失败，跳过 {url}: {e}")
        failed_urls.append(url)

files_repo = glob.glob(os.path.join(temp_folder_repo, "*.txt"))
files_html = glob.glob(os.path.join(temp_folder_html, "*.txt"))
print(f"下载完成: 仓库文章 {len(files_repo)} 篇, HTML文章 {len(files_html)} 篇")
print(f"失败 URL 数量: {len(failed_urls)}")

if failed_urls:
    with open("failed_urls.txt", "w") as f:
        f.write("\n".join(failed_urls))
    print("失败 URL 已保存至 failed_urls.txt，可稍后手动处理。")

text = ""
for file in sorted(files_repo + files_html):    # sort by filename to ensure text is the same
    with open(file, "r", encoding="utf-8") as f:
        text += f.read()

with open("PaulGrahamEssays.json", "w", encoding="utf-8") as f:
    json.dump({"text": text}, f)


shutil.rmtree(temp_folder_repo)
shutil.rmtree(temp_folder_html)
