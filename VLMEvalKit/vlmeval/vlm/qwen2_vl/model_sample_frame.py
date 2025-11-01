from __future__ import annotations

import os
import sys
import warnings
import math
import logging
from decord import VideoReader, cpu    # pip install decord
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
import re 
import numpy as np
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from .vision_process import get_video_hw
import random

import json
import torch
from transformers import StoppingCriteria
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, listinstr
from ...dataset import DATASET_MODALITY

VLLM_MAX_IMAGE_INPUT_NUM = 64



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

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type


def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)


def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        import tempfile
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images

def setup_visible_devices_per_rank():
    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    assert world_size == 1, "Only support world_size == 1 for vLLM inference"
    num_gpus = total_gpus // world_size
    start_idx = rank * num_gpus
    assigned_devices = list(range(start_idx, start_idx + num_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in assigned_devices)
    logging.info(f"[Rank {rank}] Visible GPUs: {assigned_devices}")
    return num_gpus


def crop_image(img, coordinate):

    width, height = img.size
    
    # 将 0-1000 的坐标转换为像素坐标
    x1 = int(coordinate[0] / 1000 * width)
    y1 = int(coordinate[1] / 1000 * height)
    x2 = int(coordinate[2] / 1000 * width)
    y2 = int(coordinate[3] / 1000 * height)
    
    # 裁剪图像
    cropped_img = img.crop((x1, y1, x2, y2))

    return cropped_img

SYSTEM_PROMPT1 = """
<Video Details>:
- Duration: %.2f seconds
- Frame Rate: %.2f fps
- Total Frames: %d frames.

Question: 
"""
SYSTEM_PROMPT2 = """
Based on the video and the user question, determine whether the visual content is sufficient to answer the question. If you have enough information, reason using the visual content and provide your final answer within \\boxed{}. Otherwise, you may zoom in a specific interval of the video related to the question for more details based on your reasoning. Also provide the time span of the interval in seconds within \\boxed{[start_time, end_time]}."""




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
        font = ImageFont.truetype("../Arial.Unicode.ttf", size=font_size)
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




class Qwen2VLChatSampleFrame(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        use_audio_in_video: bool = False,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.max_new_tokens = max_new_tokens
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        # print(self.generate_kwargs)
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        print("max_number_of_frames: ", self.nframe)
        if self.fps is None and self.nframe is None:
            print("Warning: fps and nframe are both None, \
                  using default nframe/fps setting in qwen-vl-utils/qwen-omni-utils, \
                  the fps/nframe setting in video dataset is omitted")
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        print(model_path)
        MODEL_CLS = None

        if listinstr(['2.5', '2_5', 'qwen25'], model_path.lower()):
            # from .qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
            # monkey_patch_qwen2_5vl_flash_attn()
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0
        self.use_vllm = kwargs.get('use_vllm', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM

        if self.use_vllm:
            from vllm import LLM
            # gpu_count = setup_visible_devices_per_rank()
            # if gpu_count >= 8:
            #     tp_size = 8
            # elif gpu_count >= 4:
            #     tp_size = 4
            # elif gpu_count >= 2:
            #     tp_size = 2
            # else:
            #     tp_size = 1
            # logging.info(
            #     f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            # )
            # import os
            # if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
            #     logging.warning(
            #         'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn.'
            #         'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
            #     )
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=5,
                max_model_len=32768,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt, "video": self.limit_mm_per_prompt},
                tensor_parallel_size=1,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
            )

        else:
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation='flash_attention_2'
            )
            self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': ensure_video_url(s['value'])
                }
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            elif s['type'] == 'audio':
                item = {'type':'audio','audio':s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _prepare_content_vllm(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        return
        content = []
        video_inputs = [s for s in inputs if s['type'] == 'video']
        video_count = len(video_inputs)
        cur_image_count = 0
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if cur_image_count < self.limit_mm_per_prompt:
                    content.append(item)
                    cur_image_count += 1
                else:
                    logging.warning(
                        f"Number of images exceeds the limit of {self.limit_mm_per_prompt}. "
                        f"Only the first {self.limit_mm_per_prompt} images will be used."
                    )
            elif s['type'] == 'video':
                if video_count > 1:
                    logging.warning(
                        "Multiple videos detected. Using video frames for each video"
                    )
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()

                    frames_per_video = max(1, self.limit_mm_per_prompt // video_count)
                    content.append({"type": "text", "text": "<video frames start>"})
                    content.extend(process_video(s['value'], frames_per_video, min_pixels, max_pixels))
                    content.append({"type": "text", "text": "<video frames end>"})

                else:
                    item = {
                        'type': 'video',
                        'video': ensure_video_url(s['value'])
                    }
                    content.append(item)
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        return content

    def generate_inner_transformers(self, message, dataset=None):

        try:
            from .vision_process import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
            raise err
        response = 'more than 10 round'
        history = []
        k=0
        try:
            
            question = []
            flag = 0
            video_path = None
            for msg in message:
                if msg['type'] == 'video':
                    video_path = msg['value']
                    flag = 1
                if flag and msg['type'] == 'text':
                    question.append(msg['value'])
            question = ''.join(question)
            if dataset == "Video-MME":
                question = message[-1]['value'].split('Question: ')[-1].split('Answer:')[0] + '\nProvide only the single option letter (e.g., A, B, C, D, etc.) as the answer.'
            elif dataset == "MVBench":
                question = message[-3]['value'].split('Question: ')[-1] + '\nProvide only the single option letter (e.g., A, B, C, D, etc.) as the answer.'
            elif dataset == 'MLVU_MCQ':
                question = message[-2]['value'].split('Question: ')[-1] + '\nProvide only the single option letter (e.g., A, B, C, D, etc.) as the answer.'
            elif dataset == 'LVBench':
                question = message[-1]['value'].split('Question: ')[-1].split('Answer:')[0] + 'Respond with only the letter (A, B, C, or D) of the correct option.'


            vreader = VideoReader(video_path, ctx=cpu(0))
            fps = vreader.get_avg_fps()
            duration = len(vreader) / vreader.get_avg_fps()
            # question = SYSTEM_PROMPT1 % (len(vreader) / fps, fps, len(vreader)) + question
            print(question)
            if duration < 60:
                factor = 2
            else:
                factor = 1
            num_fast_frames = 768 # limit total pixels to 16k
            num_slow_frames = 32
            fast_video_resolution = 32
            slow_video_resolution = 256
            if dataset == 'MLVU_MCQ':
                num_fast_frames = 384
                fast_video_resolution = 64
            
            if dataset == "LongVideoBench":
                num_slow_frames = 16
            
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
            
            # assert False
            for k in range(2):
                # frame_ids = torch.linspace(start_frame, end_frame, min(num_frames, int((end_frame - start_frame) / vreader.get_avg_fps() * 2))).round().long().tolist()
                messages = []
                content = []
                video_sample_fps_list = [fast_video_sample_fps]
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
                    resized_height, resized_width = get_video_hw(H, W, slow_video_resolution * 28 * 28 * factor)
                    slow_video = transforms.functional.resize(
                        slow_video,
                        [resized_height, resized_width],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ).float()
                    content.append({"type": "text", "text": "Subset zoom-in video clip [%d, %d]:" % (int(interval[0]), int(interval[1]))})
                    content.append({"type": "video", "video": slow_video})

                prompt = question + SYSTEM_PROMPT2
                content.append({"type": "text", "text": prompt})
                messages.append({"role": "system", "content": "You are a helpful assistant. The red numbers on each frame represent the timestamp in seconds and you can refer them during temporal grounding."})
                messages.append({"role": 'user', "content": content})
                video_kwargs = {'fps': video_sample_fps_list}
                
                text = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]# + "I need to zoom in on the video.\n\n"
                # print(text)
                images, videos = process_vision_info([messages])
                inputs = self.processor(text=text, images=images, videos=videos, padding=True, padding_side="left", return_tensors='pt', **video_kwargs)  # noqa: E501
                inputs = inputs.to('cuda')

                generated_ids = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                    use_cache=True,        # 启用 cache
                    # past_key_values=past_key_values,
                    return_dict_in_generate=True
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids.sequences)
                ]
                out = self.processor.tokenizer.batch_decode(
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

                    
                    # start_time = max(zoomin_interval[0], 0)
                    # end_time = min(zoomin_interval[1], int(duration))
                    start_time = max(zoomin_interval[0] - 5, 0)
                    end_time = min(zoomin_interval[1] + 5, int(duration))
                    if end_time - start_time < 3:
                        break

                    if dataset != "LongVideoBench":
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
                    assert False, "do not find any operation"
                
        except Exception as e:
            print(e)
            history.append({"round": k, "frame_ids": zoomin_intervals, "info": info})
            # response = ''

        try:
            if (response in ['more than 10 round', "do not find any operation", "do not find <select_frame>", "no new frames", '', "the interval is too small"]) or response.startswith('['):
                # frame_ids = torch.linspace(0, len(vreader) - 1, num_frames).round().long().tolist()
                
                if len(zoomin_intervals) == 0:
                    zoomin_intervals = [[0, int(duration)]]

                content = []
                video_sample_fps_list = [fast_video_sample_fps]
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
                    resized_height, resized_width = get_video_hw(H, W, slow_video_resolution * 28 * 28 * factor)
                    slow_video = transforms.functional.resize(
                        slow_video,
                        [resized_height, resized_width],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ).float()
                    content.append({"type": "text", "text": "Subset zoom-in video clip [%d, %d]:" % (int(interval[0]), int(interval[1]))})
                    content.append({"type": "video", "video": slow_video})

                prompt = question + SYSTEM_PROMPT2
                content.append({'type': 'text', 'text': prompt})
                messages = []
                messages.append({"role": "system", "content": "You are a helpful assistant. The red numbers on each frame represent the timestamp in seconds and you can refer them during temporal grounding."})
                messages.append({"role": 'user', "content": content})
                video_kwargs = {'fps': video_sample_fps_list}


                text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
                text[0] = text[0] + 'I get the answer.\n\n'# + "\\boxed{"
                images, videos = process_vision_info([messages])
                # print(len(images))
                # print(images[0].size)
                inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt', **video_kwargs)  # noqa: E501
                inputs = inputs.to('cuda')

                if listinstr(['omni'], self.model_path.lower()):
                    self.generate_kwargs['use_audio_in_video'] = self.use_audio_in_video
                    self.generate_kwargs['return_audio'] = False
                generated_ids = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                    use_cache=True,  
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                out = self.processor.tokenizer.batch_decode(
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

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        with open("1_%s_%s_%d.jsonl" % (self.model_path.replace('/', "_"), dataset, rank), "a") as f:
            f.write(json.dumps({"video_path": video_path,
                                   "response": response,
                                   "history": history,}) + "\n")

        print(response)
        return response


    def generate_inner_vllm(self, message, dataset=None):
        # return
        from vllm import SamplingParams

        # try:
        #     from .vision_process import process_vision_info
        # except Exception as err:
        #     logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
        #     raise err
        

        # response = 'more than 10 round'
        # try:
            
        #     question = []
        #     flag = 0
        #     video_path = None
        #     for msg in message:
        #         if msg['type'] == 'video':
        #             video_path = msg['value']
        #             flag = 1
        #         if flag and msg['type'] == 'text':
        #             question.append(msg['value'])
        #     question = ''.join(question)
        #     if dataset == "Video-MME":
        #         question = message[-1]['value'].split('Question: ')[-1].split('Answer:')[0] + '\nProvide only the single option letter (e.g., A, B, C, D, etc.) as the answer.'
        #     elif dataset == "MVBench":
        #         question = message[-3]['value'].split('Question: ')[-1] + '\nProvide only the single option letter (e.g., A, B, C, D, etc.) as the answer.'
        #     elif dataset == 'MLVU_MCQ':
        #         question = message[-2]['value'].split('Question: ')[-1] + '\nProvide only the single option letter (e.g., A, B, C, D, etc.) as the answer.'
        #     elif dataset == 'LVBench':
        #         question = message[-1]['value'].split('Question: ')[-1].split('Answer:')[0] + 'Respond with only the letter (A, B, C, or D) of the correct option.'


        #     vreader = VideoReader(video_path, ctx=cpu(0))
        #     fps = vreader.get_avg_fps()
            
        #     frame_ids = []
        #     num_frames = min(64, int(len(vreader) / vreader.get_avg_fps() * 2))
        #     frame_ids = [round(i * len(vreader) / (num_frames - 1)) for i in range(num_frames)]
        #     frame_ids = frame_ids[1:-1]
        #     # assert False
        #     for k in range(6):
        #         messages = []
        #         content = []
        #         frame_ids = sorted(frame_ids)
        #         small_num_frames = 16
        #         small_num_frames = int(min(small_num_frames, (frame_ids[1] - frame_ids[0]) / vreader.get_avg_fps() * 2))
        #         if small_num_frames >= 2:
        #             small_frame_ids = []
        #             for frame_id1, frame_id2 in zip(frame_ids[:-1], frame_ids[1:]):
        #                 small_frame_ids += torch.linspace(frame_id1, frame_id2, small_num_frames + 2).round().long().tolist()[1:-1]
                    
        #             patch_images = vreader.get_batch(frame_ids + small_frame_ids).asnumpy()
        #             small_video = patch_images[len(frame_ids):]
        #             patch_images = [Image.fromarray(f) for f in patch_images[:len(frame_ids)]]
                    
        #             small_video = torch.tensor(small_video).permute(0, 3, 1, 2)
        #             W, H = patch_images[0].size
        #             resized_height, resized_width = get_video_hw(H, W)
        #             small_video = transforms.functional.resize(
        #                 small_video,
        #                 [resized_height, resized_width],
        #                 interpolation=InterpolationMode.BICUBIC,
        #                 antialias=True,
        #             ).float()
        #             small_video = small_video.split([small_num_frames for _ in range(len(frame_ids)-1)])
        #         else:
        #             patch_images_numpy = vreader.get_batch(frame_ids).asnumpy()
        #             patch_images = [Image.fromarray(f) for f in patch_images_numpy]


        #         content = []
        #         if small_num_frames < 2:
        #                 W, H = patch_images[0].size
        #                 resized_height, resized_width = get_video_hw(H, W)
        #                 small_video = transforms.functional.resize(
        #                     torch.tensor(patch_images_numpy[:1]).permute(0, 3, 1, 2),
        #                     [resized_height, resized_width],
        #                     interpolation=InterpolationMode.BICUBIC,
        #                     antialias=True,
        #                 ).float()
        #                 content.append({"type": "video", "video": small_video})
                
        #         for j, (image, frame_id) in enumerate(zip(patch_images, frame_ids)):
        #             content.append({"type": "text", "text": "Frame %d: " % (frame_id)})
        #             content.append({"type": "image", "image": image})
        #             if frame_id != frame_ids[-1] and small_num_frames >= 2:
        #                 content.append({"type": "video", "video": small_video[j]})
                
        #         prompt = SYSTEM_PROMPT1 % (len(vreader) / fps, fps, len(vreader)) + question + SYSTEM_PROMPT2
        #         content.append({"type": "text", "text": prompt})
        #         # print(prompt)
        #         past_key_values = None
        #         messages.append({"role": 'user', "content": content})
                
            

        #         text = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        #         # print(text)
        #         images, videos = process_vision_info([messages])
        #         videos_nd = [video.detach().cpu().numpy().transpose(0, 2, 3, 1) for video in videos]
        #         video_inputs = {
        #             "prompt": text[0],
        #             "multi_modal_data": {"image": images, "video": videos_nd[0]},
        #             "mm_processor_kwargs":{}
        #         }
                
        #         outputs = self.llm.generate(
        #             video_inputs,
        #             sampling_params=sampling_params,
        #         )
        #         for o in outputs:
        #             info = o.outputs[0].text
        #         # info = out[0]
        #         print("==========================")
        #         print(info)
        #         print("==========================")
        #         if "<select_frame>" in info and "</select_frame>" in info:
        #             # split frames
        #             pattern = r'<select_frame>\s*(.*?)\s*</select_frame>'
        #             match = re.search(pattern, info, re.DOTALL)
        #             if match:
        #                 new_frames = match.group(1).strip()
        #             else:
        #                 print("do not find <select_frame>")
        #                 response = "do not find <select_frame>"
        #                 assert False

        #             new_frames = new_frames.split(',')
        #             new_frames = [int(frame.strip()) for frame in new_frames]
        #             new_frames = sorted(new_frames)

        #             frame_id1 = new_frames[0]
        #             frame_id2 = new_frames[1]
        #             if frame_id1 < frame_ids[0]:
        #                 frame_id1 = 0
        #                 frame_id2 = frame_ids[0]
        #             elif frame_id1 >= frame_ids[-1]:
        #                 frame_id2 = len(vreader) - 1
        #                 frame_id1 = frame_ids[-1]
        #             else:
        #                 for i, exist_frame in enumerate(frame_ids):
        #                     if frame_id1 < exist_frame:
        #                         frame_id1 = frame_ids[i-1]
        #                         frame_id2 = frame_ids[i]
        #                         break
        #             if frame_id2 - frame_id1 < 20:
        #                 print("the interval is too small")
        #                 response = "the interval is too small"
        #                 assert False

        #             new_frames = np.linspace(frame_id1, frame_id2, 6, dtype=int).tolist()
        #             new_frames = new_frames[1:-1]

        #             frame_ids += new_frames

                
        #         elif "\\boxed{" in info:
        #             resp = info.split('\\boxed{')[-1]
        #             lt = len(resp)
        #             counter, end = 1, None
        #             for i in range(lt):
        #                 if resp[i] == '{':
        #                     counter += 1
        #                 elif resp[i] == '}':
        #                     counter -= 1
        #                 if counter == 0:
        #                     end = i
        #                     break
        #                 elif i == lt - 1:
        #                     end = lt
        #                     break
        #             if end is not None:
        #                 response = resp[:end]
        #             else:
        #                 response = resp
        #             break
        #         else:
        #             response = "do not find any operation"
        #             assert False, "do not find any operation"
        # except Exception as e:
        #     print(e)
        #     # response = ''

        # try:
        #     if response in ['more than 10 round', "do not find any operation", "do not find <select_frame>", "no new frames", '', "the interval is too small"]:
        #         frame_ids = torch.linspace(0, len(vreader) - 1, num_frames).round().long().tolist()
        #         small_num_frames = 16
        #         small_num_frames = int(min(small_num_frames, (frame_ids[1] - frame_ids[0]) / vreader.get_avg_fps() * 2))
        #         if small_num_frames >= 2:
        #             small_frame_ids = []
        #             for frame_id1, frame_id2 in zip(frame_ids[:-1], frame_ids[1:]):
        #                 small_frame_ids += torch.linspace(frame_id1, frame_id2, small_num_frames + 2).round().long().tolist()[1:-1]
                    
        #             patch_images = vreader.get_batch(frame_ids + small_frame_ids).asnumpy()
        #             small_video = patch_images[len(frame_ids):]
        #             patch_images = [Image.fromarray(f) for f in patch_images[:len(frame_ids)]]
                    
        #             small_video = torch.tensor(small_video).permute(0, 3, 1, 2)
        #             W, H = patch_images[0].size
        #             resized_height, resized_width = get_video_hw(H, W)
        #             small_video = transforms.functional.resize(
        #                 small_video,
        #                 [resized_height, resized_width],
        #                 interpolation=InterpolationMode.BICUBIC,
        #                 antialias=True,
        #             ).float()
        #             small_video = small_video.split([small_num_frames for _ in range(len(frame_ids)-1)])
        #         else:
        #             patch_images_numpy = vreader.get_batch(frame_ids).asnumpy()
        #             patch_images = [Image.fromarray(f) for f in patch_images_numpy]


        #         content = []
        #         if small_num_frames < 2:
        #                 W, H = patch_images[0].size
        #                 resized_height, resized_width = get_video_hw(H, W)
        #                 small_video = transforms.functional.resize(
        #                     torch.tensor(patch_images_numpy[:1]).permute(0, 3, 1, 2),
        #                     [resized_height, resized_width],
        #                     interpolation=InterpolationMode.BICUBIC,
        #                     antialias=True,
        #                 ).float()
        #                 content.append({"type": "video", "video": small_video})
                
        #         for j, (image, frame_id) in enumerate(zip(patch_images, frame_ids)):
        #             content.append({"type": "text", "text": "Frame %d: " % (frame_id)})
        #             content.append({"type": "image", "image": image})
        #             if frame_id != frame_ids[-1] and small_num_frames >= 2:
        #                 content.append({"type": "video", "video": small_video[j]})
        #         content.append({"type": "text", "text": question})
        #         messages = []
                
        #         messages.append({"role": 'user', "content": content})


        #         text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        #         # text[0] = text[0] + "\\boxed{"
        #         images, videos = process_vision_info([messages])
        #         videos_nd = [video.detach().cpu().numpy().transpose(0, 2, 3, 1) for video in videos]
        #         video_inputs = {
        #             "prompt": text[0],
        #             "multi_modal_data": {"image": images, "video": videos_nd[0]},
        #             "mm_processor_kwargs":{}
        #         }
                
        #         outputs = self.llm.generate(
        #             video_inputs,
        #             sampling_params=sampling_params,
        #         )
        #         for o in outputs:
        #             info = o.outputs[0].text
        #         response = info[0]
        # except:
        #     response = ''

        # print(response)

        return response

    def generate_inner(self, message, dataset=None):
        # print(message)
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)






