# Copyright 2024. All rights reserved.
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
# limitations under the License.
"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    train_video_llm.py \
    --dataset_name mfarre/simplevideoshorts \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir video-llm-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

from qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
monkey_patch_qwen2_5vl_flash_attn()

import os
import json
import random
import requests
import math
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer import safe_globals, ParallelMode, set_rng_state_for_device
import numpy as np
import transformers
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from trainer import process_vision_info, get_video_hw
from typing import Optional, Tuple
# from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset
import wandb
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

from typing import List, Dict, Any
from decord import VideoReader, cpu    # pip install decord
# from torchcodec.decoders import VideoDecoder
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import copy


def _new_load_rng_state(self, resume_from_checkpoint):
    # Load RNG states from `checkpoint`
    if checkpoint is None:
        return

    if self.args.world_size > 1:
        process_index = self.args.process_index
        rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
        if not os.path.isfile(rng_file):

            return
    else:
        rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            return

    with safe_globals():
        checkpoint_rng_state = torch.load(rng_file)
    random.setstate(checkpoint_rng_state["python"])
    np.random.set_state(checkpoint_rng_state["numpy"])
    torch.random.set_rng_state(checkpoint_rng_state["cpu"])

    is_distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
    if torch.cuda.is_available():
        set_rng_state_for_device("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)


transformers.Trainer._load_rng_state = _new_load_rng_state




from dataclasses import dataclass, field
@dataclass
class SFTModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

@dataclass
class SFTScriptArguments(ScriptArguments):
    data_folder: str = field(
        default="/mnt/data1/shenghao/",
        metadata={"help": "folder to training data"},
    )
    video_sft: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using video_sft"},
    )



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
        font = ImageFont.truetype("../../Arial.Unicode.ttf", size=font_size)
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




SYSTEM_PROMPT1 = """
<Video Details>:
- Duration: %.2f seconds
- Frame Rate: %.2f fps
- Total Frames: %d frames.

Question: 
"""
SYSTEM_PROMPT2 = """
Based on the video and the user question, determine whether the visual content is sufficient to answer the question. If you have enough information, reason using the visual content and provide your final answer within \\boxed{}. Otherwise, you may zoom in a specific interval of the video related to the question for more details based on your reasoning. Also provide the time span of the interval in seconds within \\boxed{[start_time, end_time]}."""


def get_random_zoomin_interval(duration):
    interval_size = max(random.randint(0, int(duration // 8)), 16)
    start_time = random.randint(0, int(duration - interval_size - 1))
    end_time = start_time + interval_size
    return [start_time, end_time]


class LazySupervisedDataset(Dataset):

    def __init__(self, data_path: str, script_args: SFTScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        with open(data_path, "r") as file:
            self.list_data_dict = json.load(file)
        # random.shuffle(self.list_data_dict)


    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i):
        # Format into conversation
        num_base_retries = 3
        try:
            return self._get_item(i)
        except Exception as e:
            print(e)
            print(i)


        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass
        

    def _get_item(self, i):
        source = copy.deepcopy(self.list_data_dict[i])
        # print(source)
        vreader = VideoReader(os.path.join(self.script_args.data_folder, source["video"]), ctx=cpu(0))
        duration = len(vreader) / vreader.get_avg_fps()
        fps = vreader.get_avg_fps()
        total_num_frames = len(vreader)
        # decoder = VideoDecoder(os.path.join(self.script_args.data_folder, source["video"]), device='cpu')
        # duration = decoder.metadata.duration_seconds
        # fps = decoder.metadata.average_fps
        # total_num_frames = decoder.metadata.num_frames


        if 'cot_data' in source:
            source['messages'][0]['content'][0]['text'] = source['messages'][0]['content'][0]['text'] + SYSTEM_PROMPT2
        
        num_fast_frames = 768 # limit total pixels to 16k
        num_slow_frames = 16
        fast_video_resolution = 32
        slow_video_resolution = 256
        if duration < 120:
            factor = 2
        else:
            factor = 1
        
        if 'cot_data' in source:
            zoomin_intervals = []
            for zoomin_interval in source["zoomin_intervals"]:
                start_time = int(round(zoomin_interval[0]))
                end_time = int(round(zoomin_interval[1]))
                if end_time - start_time < 5:
                    mid_time = int((start_time + end_time) / 2)
                    start_time = max(mid_time - 3, 0)
                    end_time = min(mid_time + 3, int(duration))
                zoomin_intervals.append([start_time, end_time])
            if len(zoomin_intervals) == 1:
                num_slow_frames = 32
            elif len(zoomin_intervals) == 2:
                num_slow_frames = 24
            else:
                num_slow_frames = 16
        elif 'tgt' in source:

            def merge_intervals(intervals):
                if not intervals:
                    return []
                
                # 按区间的起始位置排序
                intervals.sort(key=lambda x: x[0])
                
                merged = [intervals[0]]  # 初始化合并结果
                
                for current in intervals[1:]:
                    last = merged[-1]
                    # 如果当前区间与上一个区间重叠
                    if current[0] <= last[1]:
                        # 合并：更新结束位置为两者最大值
                        merged[-1][1] = max(last[1], current[1])
                    else:
                        # 不重叠，添加新区间
                        merged.append(current)
    
                return merged
            
            random_seed = random.random()
            if random_seed < 0.6:
                zoomin_intervals = [[int(tgt[0]), int(tgt[1])] for tgt in source['tgt']]
                zoomin_intervals = merge_intervals(zoomin_intervals)
                if len(zoomin_intervals) > 3:
                    zoomin_intervals = random.sample(zoomin_intervals, 3)
                zoomin_intervals = [[max(0, zoomin_interval[0] - random.randint(0, 5)), min(int(duration), zoomin_interval[1] + random.randint(0, 5))] for zoomin_interval in zoomin_intervals]
            elif random_seed < 0.8:
                zoomin_intervals = [[0, int(duration)]]
                num_slow_frames = 32
            else:
                zoomin_intervals = []
        else:
            if random.random() < 0.1:
                if duration < 30:
                    num_zoomin_intervals = random.choice([0, 1])
                elif duration < 180:
                    num_zoomin_intervals = random.choice([0, 1, 2])
                else:
                    num_zoomin_intervals = random.choice([0, 1, 2, 3])

                zoomin_intervals = [get_random_zoomin_interval(duration) for _ in range(num_zoomin_intervals)]
            else:
                zoomin_intervals = [[0, int(duration)]]
                num_slow_frames = 32
            
        zoomin_intervals = sorted(zoomin_intervals, key=lambda x: x[0])
        if len(zoomin_intervals) <= 1:
            num_slow_frames = 32

        video_sample_fps_list = []
        content = []
        start_frame = 0
        end_frame = len(vreader) - 1
        num_frames = min(int(duration * 2), num_fast_frames)
        fast_frame_ids = torch.linspace(start_frame, end_frame, num_frames).round().long().tolist()
        sample_fps = round(num_frames / (end_frame - start_frame) * fps, 2)
        video_sample_fps_list.append(sample_fps)
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
        content.append({"type": "text", "text": "Full video [%d, %d]:" % (0, int(duration))})
        content.append({"type": "video", "video": fast_video})

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
            resized_height, resized_width = get_video_hw(H, W, slow_video_resolution * 28 * 28)
            slow_video = transforms.functional.resize(
                slow_video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            content.append({"type": "text", "text": "Subset zoom-in video clip [%d, %d]:" % (int(interval[0]), int(interval[1]))})
            content.append({"type": "video", "video": slow_video})
        
        vreader.seek(0)
        del vreader
        video_kwargs = {'fps': video_sample_fps_list}
        messages = source['messages']
        messages[0]['content'] = content + messages[0]['content']
        messages = [{"role": "system", "content": "You are a helpful assistant. The red numbers on each frame represent the timestamp in seconds and you can refer them during temporal grounding."}] + messages
        image_inputs, video_inputs = process_vision_info(messages)
        # return {
        #     'image_inputs': image_inputs,
        #     'video_inputs': video_inputs,
        #     'messages': messages,
        # }
        texts = [processor.apply_chat_template(messages, tokenize=False)]
        # print(zoomin_intervals, video_sample_fps_list)
        # print(texts)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            **video_kwargs
        )
        # print(inputs["input_ids"].shape)
        if inputs["input_ids"].shape[1] > 20500:
            assert False, 'input too long'
        
        return {
            'image_inputs': image_inputs,
            'video_inputs': video_inputs,
            'messages': messages,
            'video_kwargs': video_kwargs,
        }



def find_assistant_spans(input_ids, start_tokens=[151644, 77091, 198], end_token=151645):
    """
    在 input_ids (Tensor) 中查找所有符合 assistant 模板的 span。
    
    Args:
        input_ids (Tensor): shape (seq_len,)
        start_tokens (list): assistant 内容的起始 token IDs
        end_token (int): assistant 内容的结束 token ID
    
    Returns:
        List of tuples: [(start_idx, end_idx), ...]
    """
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("input_ids 必须是 torch.Tensor")

    device = input_ids.device
    seq_len = input_ids.shape[0]
    spans = []

    # 转为 Tensor
    start_tensor = torch.tensor(start_tokens, device=device)
    start_len = len(start_tensor)

    i = 0
    while i <= seq_len - start_len:
        # 检查是否匹配 start_tokens
        if torch.equal(input_ids[i:i + start_len], start_tensor):
            # 找到开始位置，继续找结束符
            j = i + start_len
            while j < seq_len:
                if input_ids[j].item() == end_token:
                    spans.append((i + len(start_tokens), j + 1))  # 包含 j 的下一个位置（切片左闭右开）
                    break
                j += 1
            # 移动指针到当前 span 后
            i = j + 1
        else:
            i += 1

    return spans

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    image_inputs = []
    video_inputs = []

    for i, example in enumerate(examples):
        texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
        image_inputs = example["image_inputs"]
        video_inputs = example['video_inputs']
        video_kwargs = example['video_kwargs']

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        **video_kwargs
    )

    labels = inputs["input_ids"].clone()
    # labels[:] = -100
    # # only apply supervision on assistant part
    # for i in range(len(inputs["input_ids"])):
    #     spans = find_assistant_spans(inputs["input_ids"][i])
    #     for left, right in spans:
    #         labels[i, left: right] = inputs["input_ids"][i, left: right]
    labels[labels == processor.tokenizer.pad_token_id] = -100

    visual_tokens = [151652, 151653, 151656, 151655] # ['<|vision_start|><|vision_end|><|video_pad|><|image_pad|>']

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((SFTScriptArguments, SFTConfig, SFTModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # Model initialization
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    # Prepare dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    # prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    if model_config.freeze_vision_modules:
        print("Freezing vision modules...")
        for n, p in model.named_parameters():
            if any(keyword in n for keyword in ['visual']):
                p.requires_grad = False
    total_trainable_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total_trainable_params += p.numel()
    print("total_trainable_params: ", total_trainable_params)

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
    )

    # Train model
    if training_args.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
