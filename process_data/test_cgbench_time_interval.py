

import json
import csv
from collections import defaultdict
import argparse
import os
import requests
import time
import re
from tqdm import tqdm
import base64
import oss2
from pathlib import Path
import copy
import torch

from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord
import numpy as np

import threading
import time
import json
import os
from queue import Queue
import random


def iou_1d(a, b):
    # a: [start, end]
    # b: [start, end]
    start_a, end_a = a
    start_b, end_b = b

    # 计算交集范围
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)

    # 如果没有重叠
    if inter_start >= inter_end:
        return 0.0

    # 计算交集和并集长度
    inter_length = inter_end - inter_start
    union_length = (end_a - start_a) + (end_b - start_b) - inter_length

    # 计算 IoU
    return inter_length / union_length if union_length != 0 else 0.0



MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number

auth = oss2.Auth()
bucket = oss2.Bucket()


# 批量获取URL
def get_oss_url(oss_base_path, local_file_path):
    local_file_name = os.path.basename(local_file_path)
    file_oss_path = os.path.join(oss_base_path, local_file_name)
    print(file_oss_path)
    bucket.put_object_from_file(file_oss_path, local_file_path) # upload 
    url = bucket.sign_url('GET', file_oss_path, 60*60*24*1000000)
    return url

    

def get_image_url(image_path):
    if 'file://' in image_path:
        image_path = image_path.replace('file://', '')
    with open(image_path, 'rb') as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"



url = URL
token = TOKEN



def encode_video(vreader, 
                 video_path,
            cache_dir,
            frame_ids,
        ):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_cache_path = os.path.join(cache_dir, video_name)

    if not os.path.exists(video_cache_path):
        os.makedirs(video_cache_path)

    image_urls = []
    patch_images = [Image.fromarray(f) for f in vreader.get_batch(frame_ids).asnumpy()]
    
    for i, img in enumerate(patch_images):
        frame_number = frame_ids[i]
        img_path = os.path.join(video_cache_path, f"{video_name}_frame{frame_number}.png")
        if not os.path.exists(img_path):
            img.save(img_path)

        url = get_oss_url("data/qize.yqz/work_dirs/shenghao_image/", img_path)
        image_urls.append(url)

    return image_urls




def gpt_api(messages, model_name):
    success = False
    max_try = 3
    tries = 0
    response_message = ""
    while (not success and tries <max_try):
        try:
            
            data = {
                    # "model": "gpt-4o",
                    # "model": "qwen2.5-72b-instruct",
                    "model": model_name,
                    "messages":messages,
                    # "n": 1
                }
            
            headers = {
                    "Content-Type": "application/json",
                        "Authorization": 'Bearer ' + token}
            response = requests.post(url, json=data, headers=headers)
            # print(response.json())
            response = response.json()
            response_message = response['choices'][0]['message']['content']
            

            success = True
        except Exception as e:
            print(f'{response},{e}')
            time.sleep(20)
            tries +=1

   
    return response_message


def build_cot(example):
    question = example["question"]
    
    video_path = '/mnt/data1/shenghao/datasets/CG-Bench/videos/' + example['video_uid'] + '.mp4'
    vreader = VideoReader(video_path, ctx=cpu(0))
    fps = vreader.get_avg_fps()

    start_time = example["clue_intervals"][0][0]
    end_time = example["clue_intervals"][0][1]
    start_frame = max(int(start_time * fps), 0)
    end_frame = min(int(end_time * fps), len(vreader) - 1)
    frame_ids = torch.linspace(start_frame, end_frame, min(16, int((end_time - start_time) * 2))).round().long().tolist()
    cache_dir = './cache/'
    images = encode_video(vreader, video_path, cache_dir, frame_ids)
    for i in range(3):
        try:

            prompt = "Question:\n" + question + "\nAnswer:\n" + example["answer"] + '\n\nDo the provided frames contain the visual clues for the answer of the question? Yes or No?'
            
            content = []
            for image in images:
                content.append({"type": "image_url","image_url": {"url":  image, "detail": "low"}})
            content.append({"type": "text", "text": prompt})

            message = [
                {
                    "role": "user",
                    "content": content
                    
                }
            ]
        
            print(prompt)
            # info = gpt_api(message, "qwen-vl-max-latest")
            # info = gpt_api(message, "gemini-2.5-pro")
            # info = gpt_api(message, "qwen3-235b-a22b-instruct-2507")
            info = gpt_api(message, "gpt-4o-2024-11-20")
            print(info)
            if 'yes' in info.lower() or 'no' in info.lower():
                break
                # return info

        except Exception as e:
            print(e)
            # message = None

    return info


import json
import threading
from queue import Queue
import os

test_case = [
    {
        "qid": 5607,
        "video_uid": "Y4y2e4rUxfo",
        "question": "In the video, what four points about Socrates are most impressive?",
        "answer": "Appearance, character, behavior, and his views and methods",
        "choices": [
            "Wisdom, trials, achievements, and his methods",
            "Speech, attire, interests, and his skepticism",
            "Teaching, attire, beliefs, and his controversies",
            "Lifestyle, ethics, dialogue, and his punishments",
            "Debate, attire, philosophy, and his punishments",
            "Reputation, dialogue, trial, and his skepticism",
            "Appearance, character, behavior, and his views and methods",
            "Philosophy, logic, rhetoric, and his controversies"
        ],
        "right_answer": "G",
        "clue_intervals": [
            [
                90,
                100
            ]
        ],
        "domain": "Art & Culture",
        "sub_category": "Event Perception",
        "duration": 1477,
        "relevant": False
    },
    {
        "qid": 6632,
        "video_uid": "Y4y2e4rUxfo",
        "question": "In the video, when the screen shows \"EACH AGE PRODUCES A SOCRATES\", what is the yellow pattern on the screen?",
        "answer": "Light bulb pattern",
        "choices": [
            "Light bulb pattern",
            "Lightning bolt pattern",
            "Star pattern",
            "Sunburst pattern",
            "Candle flame pattern",
            "Spark pattern"
        ],
        "right_answer": "A",
        "clue_intervals": [
            [
                130,
                135
            ]
        ],
        "domain": "Art & Culture",
        "sub_category": "2D Spatial Perception",
        "duration": 1477,
        "relevant": True
    },
    {
        "qid": 10511,
        "video_uid": "Y4y2e4rUxfo",
        "question": "In the video, when the subtitle \"Socrates said, a courageous man\" appeared, what was the second point written by the protagonist in red pen?",
        "answer": "BECOURAGEOUS",
        "choices": [
            "BEBOLD",
            "BEINSPIRED",
            "BEFEARLESS",
            "BEVALIANT",
            "BECOURAGEOUS",
            "BEBRAVE",
            "BEDETERMINED",
            "BECONFIDENT"
        ],
        "right_answer": "E",
        "clue_intervals": [
            [
                350,
                360
            ]
        ],
        "domain": "Art & Culture",
        "sub_category": "Event Cognition",
        "duration": 1477,
        "relevant": True
    },
    {
        "qid": 11002,
        "video_uid": "Y4y2e4rUxfo",
        "question": "In the video, when a man with white hair and beard appears in the lower left corner of the screen at the beginning, it is introduced that Socrates lived between which and which years?",
        "answer": "Between 469 BC and 399 BC",
        "choices": [
            "Between 475 BC and 405 BC",
            "Between 465 BC and 395 BC",
            "Between 470 BC and 400 BC",
            "Between 468 BC and 398 BC",
            "Between 469 BC and 399 BC",
            "Between 470 BC and 390 BC"
        ],
        "right_answer": "E",
        "clue_intervals": [
            [
                0,
                5
            ]
        ],
        "domain": "Art & Culture",
        "sub_category": "Event Perception",
        "duration": 1477,
        "relevant": False
    },
]

# 配置
INPUT_JSON = "time_interval_subset1.json"          # 输入的原始数据文件（包含一个列表）
OUTPUT_JSONL = "time_interval_output1.jsonl"      # 输出文件（JSON Lines 格式）
PROGRESS_FILE = "progress1.log"    # 已完成ID记录文件
NUM_THREADS = 1                   # 并发线程数


# 线程任务函数
def worker(task_queue, result_queue, done_ids):
    while not task_queue.empty():
        idx, data = task_queue.get()
        try:
            result = build_cot(data)
            data['relevant'] = result
            result_queue.put(data)
            done_ids.add(idx)
        except Exception as e:
            print(f"[ERROR] 处理数据 {idx} 时出错: {e}")
        finally:
            task_queue.task_done()

# 写入结果到文件（实时追加）
def result_writer(result_queue, done_ids):
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        while True:
            result = result_queue.get()
            if result is None:
                break
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            save_progress(done_ids)

# 保存已完成 ID 到文件
def save_progress(done_ids):
    with open(PROGRESS_FILE, "w") as f:
        f.write("\n".join(map(str, sorted(done_ids))))

# 读取已处理 ID
def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, "r") as f:
        return set(map(int, f.read().splitlines()))

# 主函数
def main():
    # 读取输入数据
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_data = []
    for da in data:
        all_data.append(da)

    # 加载已处理的任务 ID
    done_ids = load_progress()

    # 创建任务队列
    task_queue = Queue()
    for idx, data in enumerate(all_data):
        if idx not in done_ids:
            task_queue.put((idx, data))

    print(f"总数据量：{len(all_data)}，剩余待处理：{task_queue.qsize()}")

    # 启动写入线程
    result_queue = Queue()
    writer_thread = threading.Thread(target=result_writer, args=(result_queue, done_ids), daemon=True)
    writer_thread.start()

    # 启动工作线程
    threads = []
    for _ in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(task_queue, result_queue, done_ids))
        t.start()
        threads.append(t)

    # 等待所有任务完成
    task_queue.join()

    # 结束写入线程
    result_queue.put(None)
    writer_thread.join()

    print("所有任务已完成或已中断。")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print(f"总耗时: {time.time() - start_time:.2f}s")



