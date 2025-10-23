import os
import json
import re
from tqdm import tqdm
import torch
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from vision_process import process_vision_info
import argparse
from decord import VideoReader, cpu    # pip install decord
import requests
import time
BSZ = 16
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from vision_process import get_video_hw
import random



def add_frame_numbers_batch(frame_batch, frame_numbers):

    results = []
    for frame, frame_number in zip(frame_batch, frame_numbers):
        # Add frame number
        numbered_frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(numbered_frame)
        
        
        # Calculate text position
        width, height = numbered_frame.size
        # 动态字体大小
        font_size = int(min(width, height) / 8)
        font = ImageFont.truetype("/mnt/data1/shenghao/Arial.Unicode.ttf", size=font_size)
        text = str(frame_number)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0] + 10
        text_height = text_bbox[3] - text_bbox[1] + 10
        
        x = width - text_width
        y = height - text_height - text_height/3

        draw.text((x, y), text, font=font, fill='red')
        results.append(torch.tensor(np.array(numbered_frame)))
    results = torch.stack(results).permute(0, 3, 1, 2)
    return results



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


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)], [i for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks, start_ids = split_list(lst, n)
    return chunks[k], start_ids[k]


def get_sample_interval(intervals, total_length):
    for _ in range(10):
        interval = max(random.randint(0, int(total_length // 8)), 16)
        start = random.randint(0, total_length - interval - 1)
        end = start + interval
        flag = [iou_1d([start, end], interval) for interval in intervals]
        if sum(flag) == 0:
            return [start, end]
    return [start, end]

parser = argparse.ArgumentParser(description="Evaluation benchmark")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--file_name', type=str, required=True, help="Name of the file")
parser.add_argument('--total_gpu', type=int, required=True, help="Name of the file")
parser.add_argument('--gpu_id', type=int, required=True, help="Name of the file")

args = parser.parse_args()

MODEL_PATH = args.model_path
file_name = args.file_name


TYPE_TEMPLATE = {
    "multiple choice": "\nProvide only the single option letter (e.g., A, B, C, D, etc.) as the answer.",
    "numerical": "\nProvide the numerical value (e.g., 42 or 3.14) without units as the answer.",
    "OCR": " \nTranscribe text from the image/video clearly and provide your text answer.",
    "free-form": "\nProvide your text answer.",
    "regression": "\nProvide the numerical value (e.g., 42 or 3.14) without units as the answer.",
    "tal": "\nProvide the answer in list format, ie, [start_time, end_time].",
}

SYSTEM_PROMPT = """
Based on the video and the user question, determine whether the visual content is sufficient to answer the question. If you have enough information, reason using the visual content and provide your final answer within \\boxed{}. Otherwise, you may zoom in a specific interval of the video related to the question for more details based on your reasoning. Also provide the time span of the interval in seconds within \\boxed{[start_time, end_time]}."""


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation='flash_attention_2'
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)
pad_token_id = processor.tokenizer.pad_token_id
processor.pad_token_id = pad_token_id
processor.eos_token_id = processor.tokenizer.eos_token_id

# generation_config = GenerationConfig(
#     max_new_tokens=1536,
#     pad_token_id=processor.pad_token_id,
#     eos_token_id=processor.eos_token_id,
#     temperature=0.01,
#     top_k=1,
#     repetition_penalty=1.0,
#     )

generation_config = GenerationConfig(
    max_new_tokens=1536,
    do_sample=True,
    pad_token_id=processor.pad_token_id,
    eos_token_id=processor.eos_token_id,
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    repetition_penalty=1.0,
    )

OUTPUT_PATH = f"{file_name}_greedy_output_{args.gpu_id}.json"
PROMPT_PATH = '/mnt/data1/shenghao/datasets/videomarathon/data/moviechat_subset_mc.json'
# PROMPT_PATH = '/mnt/data1/shenghao/datasets/NExT-GQA/datasets/nextgqa/next_gqa.json'

if PROMPT_PATH.endswith('.json'):
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    raise ValueError("Input file must be .json or .jsonl")


data, start_id = get_chunk(data, args.total_gpu, args.gpu_id)
data = data[::-1]

final_output = []
mean_mra = []
for da in tqdm(data, total=len(data), desc="Processing batches"):
    try:
        question = da["question"]
        answer = da['answer']
        prompt = question + SYSTEM_PROMPT

        answers = []
        sample_frame = []
        vreader = VideoReader(da['video'], ctx=cpu(0))
        fps = vreader.get_avg_fps()
        duration = len(vreader) / vreader.get_avg_fps()

        num_fast_frames = 620 # limit total pixels to 16k
        num_slow_frames = 8
        fast_video_resolution = 32
        slow_video_resolution = 256
        if duration < 60:
            factor = 2
        else:
            factor = 1
    except:
        continue

    accs = []
    zoomins = []
    for _ in range(4):
        response = 0.0
        history = []
        try:
            start_frame = 0
            end_frame = len(vreader) - 1
            num_frames = min(int(duration * 2), num_fast_frames)
            fast_frame_ids = torch.linspace(start_frame, end_frame, num_frames).round().long().tolist()
            fast_video_sample_fps = round(num_frames / (end_frame - start_frame) * fps, 2)
            fast_video = vreader.get_batch(fast_frame_ids).asnumpy()
            fast_frame_seconds = [int(frame_id / fps) for frame_id in fast_frame_ids]
            fast_video = add_frame_numbers_batch(fast_video, fast_frame_seconds)
            # fast_video = torch.tensor(fast_video).permute(0, 3, 1, 2)
            H, W = fast_video.shape[-2:]
            resized_height, resized_width = get_video_hw(H, W, fast_video_resolution * 28 * 28 * factor)
            fast_video = transforms.functional.resize(
                fast_video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            zoomin_intervals = []
            
            for k in range(1):
                # frame_ids = torch.linspace(start_frame, end_frame, min(num_frames, int((end_frame - start_frame) / vreader.get_avg_fps() * 2))).round().long().tolist()
                messages = []
                content = []
                video_sample_fps_list = [fast_video_sample_fps]
                content.append({"type": "text", "text": "Full video [%d, %d]:" % (0, int(duration))})
                content.append({"type": "video", "video": fast_video})
                if len(zoomin_intervals) > 1:
                    num_slow_frames = 16
                for interval in zoomin_intervals:
                    start_frame = max(0, int(interval[0] * fps))
                    end_frame = min(len(vreader) - 1, int(interval[1] * fps))
                    num_frames = min(int((interval[1] - interval[0]) * 2), num_slow_frames)
                    slow_frame_ids = torch.linspace(start_frame, end_frame, num_frames).round().long().tolist()
                    sample_fps = round(num_frames / (end_frame - start_frame) * fps, 2)
                    video_sample_fps_list.append(sample_fps)
                    slow_video = vreader.get_batch(slow_frame_ids).asnumpy()
                    slow_frame_seconds = [int(frame_id / fps) for frame_id in slow_frame_ids]
                    slow_video = add_frame_numbers_batch(slow_video, slow_frame_seconds)
                    # slow_video = torch.tensor(slow_video).permute(0, 3, 1, 2)
                    H, W = slow_video.shape[-2:]
                    resized_height, resized_width = get_video_hw(H, W, slow_video_resolution * 28 * 28 * factor)
                    slow_video = transforms.functional.resize(
                        slow_video,
                        [resized_height, resized_width],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ).float()
                    content.append({"type": "text", "text": "Subset zoom-in video clip [%d, %d]:" % (int(interval[0]), int(interval[1]))})
                    content.append({"type": "video", "video": slow_video})

                prompt = question + SYSTEM_PROMPT
                content.append({"type": "text", "text": prompt})
                messages.append({"role": "system", "content": "You are a helpful assistant. The red numbers on each frame represent the timestamp in seconds and you can refer them during temporal grounding."})
                
                messages.append({"role": 'user', "content": content})
                video_kwargs = {'fps': video_sample_fps_list}
                
                text = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]# + "I need to zoom in on the video.\n\n"
                # print(text)
                images, videos = process_vision_info([messages])
                inputs = processor(text=text, images=images, videos=videos, padding=True, padding_side="left", return_tensors='pt', **video_kwargs)  # noqa: E501
                inputs = inputs.to('cuda')

                generated_ids = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    use_cache=True,        # 启用 cache
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                out = processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                info = out[0]
                print("==========================")
                print(info)
                print("==========================")
                if "\\boxed{" in info:
                    history.append({"round": k, "frame_ids": zoomin_intervals, "info": info})
                    resp = info.split('\\boxed{')[-1]
                    lt = len(resp)
                    counter, end = 1, None
                    for i in range(lt):
                        if resp[i] == '{':
                            counter += 1
                        elif resp[i] == '}':
                            counter -= 1
                        if counter == 0:
                            end = i
                            break
                        elif i == lt - 1:
                            end = lt
                            break
                    if end is not None:
                        response = resp[:end]
                    else:
                        response = resp
                    response = response.strip()
                    if not response.startswith('['):
                        break
                    # assert False
                    zoomin_interval = response[1:-1]
                    zoomin_interval = zoomin_interval.split(',')
                    zoomin_interval = [int(frame.strip()) for frame in zoomin_interval]
                    zoomin_interval = sorted(zoomin_interval)

                    
                    start_time = max(zoomin_interval[0] - 5, 0)
                    end_time = min(zoomin_interval[1] + 5, int(duration))
                    if end_time - start_time < 3:
                        break

                    if end_time - start_time < min(30, duration / 8):
                        interval = min(30, duration / 8)
                        mid_time = (start_time + end_time) / 2
                        start_time = int(max(mid_time - interval / 2, 0))
                        end_time = int(min(mid_time + interval / 2, duration))

                    ious = [iou_1d([start_time, end_time], interval) for interval in zoomin_intervals]
                    if sum([iou > 0.3 for iou in ious]):
                        break
                    zoomin_intervals.append([start_time, end_time])
                    zoomin_intervals = sorted(zoomin_intervals, key=lambda x: x[0])
                else:
                    response = "do not find any operation"
                    break
                


            try:
                if (response in ['more than 10 round', "do not find any operation", "do not find <select_frame>", "no new frames", '', "the interval is too small"]) or response.startswith('['):
                    # frame_ids = torch.linspace(0, len(vreader) - 1, num_frames).round().long().tolist()
                    
                    # if len(zoomin_intervals) == 0:
                    #     zoomin_intervals = [[0, int(duration)]]

                    content = []
                    video_sample_fps_list = [fast_video_sample_fps]
                    content.append({"type": "text", "text": "Full video [%d, %d]:" % (0, int(duration))})
                    content.append({"type": "video", "video": fast_video})
                    if len(zoomin_intervals) > 1:
                        num_slow_frames = 16
                    for interval in zoomin_intervals:
                        start_frame = max(0, int(interval[0] * fps))
                        end_frame = min(len(vreader) - 1, int(interval[1] * fps))
                        num_frames = min(int((interval[1] - interval[0]) * 2), num_slow_frames)
                        slow_frame_ids = torch.linspace(start_frame, end_frame, num_frames).round().long().tolist()
                        sample_fps = round(num_frames / (end_frame - start_frame) * fps, 2)
                        video_sample_fps_list.append(sample_fps)
                        slow_video = vreader.get_batch(slow_frame_ids).asnumpy()
                        slow_frame_seconds = [int(frame_id / fps) for frame_id in slow_frame_ids]
                        slow_video = add_frame_numbers_batch(slow_video, slow_frame_seconds)
                        # slow_video = torch.tensor(slow_video).permute(0, 3, 1, 2)
                        H, W = slow_video.shape[-2:]
                        resized_height, resized_width = get_video_hw(H, W, slow_video_resolution * 28 * 28 * factor)
                        slow_video = transforms.functional.resize(
                            slow_video,
                            [resized_height, resized_width],
                            interpolation=InterpolationMode.BICUBIC,
                            antialias=True,
                        ).float()
                        content.append({"type": "text", "text": "Subset zoom-in video clip [%d, %d]:" % (int(interval[0]), int(interval[1]))})
                        content.append({"type": "video", "video": slow_video})

                    prompt = question + SYSTEM_PROMPT
                    content.append({'type': 'text', 'text': prompt})
                    messages = []
                    messages.append({"role": "system", "content": "You are a helpful assistant. The red numbers on each frame represent the timestamp in seconds and you can refer them during temporal grounding."})
                
                    messages.append({"role": 'user', "content": content})
                    video_kwargs = {'fps': video_sample_fps_list}


                    text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
                    text[0] = text[0] + 'I get the answer.\n\n'# + "\\boxed{"
                    images, videos = process_vision_info([messages])
                    # print(len(images))
                    # print(images[0].size)
                    inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt', **video_kwargs)  # noqa: E501
                    inputs = inputs.to('cuda')

                    generated_ids = model.generate(
                        **inputs,
                        generation_config=generation_config,
                        use_cache=True,        # 启用 cache
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    out = processor.tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    info = out[0]
                    print("==========================")
                    print(info)
                    history.append({"round": 3, "frame_ids": zoomin_intervals, "info": info})
                    print("==========================")
                    # response = info[0] 
                    if "\\boxed" in info:
                        resp = info.split('\\boxed{')[-1]
                        lt = len(resp)
                        counter, end = 1, None
                        for i in range(lt):
                            if resp[i] == '{':
                                counter += 1
                            elif resp[i] == '}':
                                counter -= 1
                            if counter == 0:
                                end = i
                                break
                            elif i == lt - 1:
                                end = lt
                                break
                        if end is not None:
                            response = resp[:end]
                        else:
                            response = resp
                    else:
                        response = ''
            except Exception as e:
                print(e)
                response = ''
                # response = ''
        except Exception as e:
            print(e)
            continue
        if response in ['A', 'B', "C", "D", "E", 'F', 'G', 'H', "I"]:
            if response == answer:
                accs.append(True)
            else:
                accs.append(False)
            
            zoomins.append(zoomin_intervals)
    print(accs)


    fps = vreader.get_avg_fps()
    
    if sum(accs) > 0 and sum(accs) < len(accs):
        # da['model_output'] = model_output
        da['duration'] = len(vreader) / fps
        da['fps'] = fps
        da['frames'] = len(vreader)
        da['correctness'] = accs
        da['answer_zoomin_intervals'] = zoomins
        final_output.append(da)
        
        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")

# final_acc={'mean_acc': 0.0, 'mean_mra': 0.0}
# final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item()
# if mean_mra != []:
#     final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()

try:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
    print(f"Final accuracy saved to {OUTPUT_PATH}")
except Exception as e:
    print(f"Error writing final accuracy to output file: {e}")

print(f"Results saved to {OUTPUT_PATH}")

