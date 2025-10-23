# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import Dataset
import json
import random
import torch
import numpy as np
import copy
from PIL import Image
import torchvision.transforms as T
from decord import VideoReader, cpu    # pip install decord
from PIL import Image

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint


from trainer import GRPOTrainer
from trl import GRPOConfig,  ModelConfig, ScriptArguments, TrlParser, get_peft_config


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
monkey_patch_qwen2_5vl_flash_attn()

import requests
import time




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



@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    data_folder: str = field(
        default="/mnt/data1/shenghao/",
        metadata={"help": "folder to training data"},
    )



def accuracy_reward(format_rewards, pred_answers, solution, problem_type, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '').replace('. ', '.')
            return float(eval(num_str))
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure

    rewards = []
    for content, sol, format, question_type in zip(pred_answers, solution, format_rewards, problem_type):
        if format > 0 and content != '':
            try:
                
                if question_type == "multiple choice":
                    output_ans = content
                    gt_ans = extract_answer(sol)
                    reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
                elif question_type == "numerical":
                    output_ans = content
                    gt_ans = extract_answer(sol)
                    gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                    out_has_decimal = ("." in output_ans) or ("," in output_ans)
                    if gt_has_decimal != out_has_decimal:
                        reward = 0.0
                    else:
                        gt_number = normalize_number(gt_ans)
                        out_number = normalize_number(output_ans)
                        if gt_number is None or out_number is None:
                            reward = 0.0
                        else:
                            reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
                elif question_type == "OCR":
                    output_ans = content
                    gt_ans = extract_answer(sol)
                    error_rate = wer(gt_ans, output_ans)
                    reward = 1 - error_rate
                    reward = max(0.0, min(1.0, reward))
                elif question_type == "free-form":
                    output_ans = content
                    gt_ans = extract_answer(sol)
                    score = ai(gt_ans, output_ans)
                    if score is not None and "yes" in score.lower():
                        reward = 1.0
                    else:
                        reward = 0.0
                elif question_type == "regression":
                    output_ans = content
                    gt_ans = extract_answer(sol)
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                    rel_diff = min(1.0, max(0.0, rel_diff))
                    reward = 1 - rel_diff
                elif question_type == "tal" or question_type == 'tvg':
                    if content[0] == '[':
                        content = content[1:]
                    if content[-1] == ']':
                        content = content[:-1]
                    new_frames = content.split(',')
                    new_frames = [int(frame.strip()) for frame in new_frames]
                    new_frames = sorted(new_frames)

                    frame_id1 = new_frames[0]
                    frame_id2 = new_frames[1]
                    gt_ans = eval(extract_answer(sol).strip())
                    reward = iou_1d(gt_ans, [frame_id1, frame_id2])
                    print(content, sol, reward)
                else:
                    reward = 0.0
            except Exception as e:
                print(f"Error in reward_fn for question_type '{question_type}': {e}")
                reward = 0.0
        else:
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards



def format_reward(format_rewards, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    return format_rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}



SYSTEM_PROMPT = """
Based on the video and the user question, determine whether the visual content is sufficient to answer the question. If you have enough information, reason using the visual content and provide your final answer within \\boxed{}. Otherwise, you may zoom in a specific interval of the video related to the question for more details based on your reasoning. Also provide the time span of the interval in seconds within \\boxed{[start_time, end_time]}."""



Grounding_SYSTEM_PROMPT1 = """
<Video Details>:
- Duration: %.2f seconds
- Frame Rate: %.2f fps
- Total Frames: %d frames.

Query:
"""
Grounding_SYSTEM_PROMPT2 = """\nPlease reason and determine the precise time peroid [START_FRAME, END_FRAME] (in frame ids) related to the query and provide the answer within \\boxed{}."""


# SYSTEM_PROMPT2 = """
# Based on the selected video frames and the user question, analyze the visual content and determine whether the visual content is sufficient to answer the question. If you have enough information, reason step-by-step using the visual content and provide your final answer within \\boxed{}. Otherwise, you may request additional frames based on your reasoning through spatial co-occurrence, temporal proximity, attribute dependency, causal order, etc. Use the following format for such requests, for example <select_frame> 10, 20, 30, 40 </select_frame> (Replace the FRAME_ID with the frame ids you needed. The interval between frames should be larger than 5 and the number of frames should be less than 5.)"""

class LazySupervisedDataset(Dataset):

    TYPE_TEMPLATE = {
        "multiple choice": "\nProvide only the single option letter (e.g., A, B, C, D, etc.) as the answer.",
        "numerical": "\nProvide the numerical value (e.g., 42 or 3.14) without units as the answer.",
        "OCR": " \nTranscribe text from the image/video clearly and provide your text answer.",
        "free-form": "\nProvide your text answer.",
        "regression": "\nProvide the numerical value (e.g., 42 or 3.14) without units as the answer.",
        "tal": "\nProvide the answer in list format, ie, [start_time, end_time].",
    }


    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        with open(data_path, "r") as file:
            self.list_data_dict = json.load(file)
        
        # random.shuffle(self.list_data_dict)
        self.index = 0
        self.step = 0


    def __len__(self):
        return len(self.list_data_dict)



    def _make_conversation_image_and_video(self, example, example_id):

        if example["problem_type"] == 'multiple choice' and len(example["options"]) > 0:
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        # vreader = VideoReader(example["path"], ctx=cpu(0))
        # duration = len(vreader) / vreader.get_avg_fps()
        # fps = vreader.get_avg_fps()
        # if example["problem_type"] == 'tvg':
        #     prompt = Grounding_SYSTEM_PROMPT1 % (len(vreader) / fps, fps, len(vreader)) + question + Grounding_SYSTEM_PROMPT2
        # else:
        #     prompt = question + self.TYPE_TEMPLATE[example['problem_type']] + SYSTEM_PROMPT
        prompt = question + self.TYPE_TEMPLATE[example['problem_type']] + SYSTEM_PROMPT

        zoomin_intervals = example['zoomin_intervals']

        
        return prompt, zoomin_intervals

    def __getitem__(self, i):
        # Format into conversation
        num_base_retries = 3
        try:
            sample = self._get_item(i)
            self.index += 1
            return sample
            
        except Exception as e:
            print(i)

        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                self.index += 1
                return sample
            except Exception as e:
                # no need to sleep
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass



    def _get_item(self, i):
        source = self.list_data_dict[i]

        prompt, zoomin_intervals = self._make_conversation_image_and_video(source, i)
        problem_type = source["problem_type"]
        # if 'process' not in source:
        # image_inputs, _, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        solution = source["solution"]
        return {
            'path': source['path'],
            'prompt': prompt,
            'solution': solution,
            "problem_type": problem_type,
            "data_type": source["data_type"],
            'tgt': source['tgt'] if 'tgt' in source else None,
            'zoomin_intervals': zoomin_intervals,
        }




def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    
    trainer_cls = GRPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        freeze_vision_modules=model_args.freeze_vision_modules,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
