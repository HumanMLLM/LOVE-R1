import os
import json
import re
from tqdm import tqdm
import torch
import math
from PIL import Image
import copy
import numpy as np
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from vllm import LLM, SamplingParams
# from qwen_vl_utils import process_vision_info
from vision_process import process_vision_info, get_video_hw
import argparse
from decord import VideoReader, cpu    # pip install decord
import requests
import time
BSZ = 4


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)], [i for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks, start_ids = split_list(lst, n)
    return chunks[k], start_ids[k]


parser = argparse.ArgumentParser(description="Evaluation benchmark")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--file_name', type=str, required=True, help="Name of the file")
parser.add_argument('--total_gpu', type=int, required=True, help="Name of the file")
parser.add_argument('--gpu_id', type=int, required=True, help="Name of the file")
parser.add_argument('--clip_size', type=int, required=True, help="seconds per clip")

args = parser.parse_args()

MODEL_PATH = args.model_path
file_name = args.file_name


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation='flash_attention_2'
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)
pad_token_id = processor.tokenizer.pad_token_id
processor.pad_token_id = pad_token_id
processor.eos_token_id = processor.tokenizer.eos_token_id

generation_config = GenerationConfig(
    max_new_tokens=2048,
    pad_token_id=processor.pad_token_id,
    eos_token_id=processor.eos_token_id,
    temperature=0.01,
    top_k=1,
    repetition_penalty=1.0,
    )

OUTPUT_PATH = f"{file_name}_greedy_output_{args.gpu_id}.json"
# PROMPT_PATH = '/mnt/data1/shenghao/multimodal-cot/ET-instruct-select-gvq.json'
PROMPT_PATH = '/mnt/data1/shenghao/multimodal-cot2/captions/cg_bench_videos.json'

if PROMPT_PATH.endswith('.json'):
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    raise ValueError("Input file must be .json or .jsonl")


data, start_id = get_chunk(data, args.total_gpu, args.gpu_id)


final_output = []
mean_mra = []
for da in tqdm(data, total=len(data), desc="Processing batches"):
    video_path = '/mnt/data1/shenghao/datasets/CG-Bench/videos/' + da + '.mp4'
    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
        fps = vreader.get_avg_fps()
        captions = {}
        total_seconds = (len(vreader) - 1) / fps
        num_clip = int(total_seconds // args.clip_size)

        for i in tqdm(range(num_clip), desc="Processing batches"):
            try:
                start_time = i * args.clip_size
                end_time = (i + 1) * args.clip_size
                start_frame = max(0, int(start_time * fps))
                end_frame = min(len(vreader) - 1, int(end_time * fps))
                if end_frame <= start_frame:
                    break
                frame_ids = torch.linspace(start_frame, end_frame, args.clip_size * 2).round().long().tolist()
                content = []
                video_clip = vreader.get_batch(frame_ids).asnumpy()
                video_clip = torch.tensor(video_clip).permute(0, 3, 1, 2)
                H, W = video_clip.shape[-2:]
                resized_height, resized_width = get_video_hw(H, W, 512 * 28 * 28 )
                video_clip = transforms.functional.resize(
                    video_clip,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).float()
                content.append({"type": "video", "video": video_clip})
                content.append({"type": "text", "text": "Elaborate on the visual and narrative elements of the video briefly."})
                msg = [{"role": 'user', "content": content}]

                prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)]
                image_inputs, video_inputs, video_kwargs = process_vision_info(msg, return_video_kwargs=True)
                inputs = processor(text=prompts, videos=video_inputs, padding=True, padding_side="left", return_tensors='pt')  # noqa: E501
                inputs = inputs.to('cuda')


                generated_ids = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    use_cache=True,        # 启用 cache
                    # # past_key_values=past_key_values,
                    # return_dict_in_generate=True
                )
                # past_key_values = generated_ids.past_key_values
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)#.sequences)
                ]
                out = processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                info = out[0]
                print(info)
                captions[start_time] = {'time_interval': [start_time, end_time], 'caption': info.strip()} 
            except Exception as e:
                print(e)
        
        new_da = {'video': video_path, 'captions': captions}
        final_output.append(new_da)
        
        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")
    else:
        print(video_path, "path not find")
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
