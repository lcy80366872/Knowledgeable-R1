# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
from data_process.hotpot import hotpotqa
from data_process.musique import musique
from data_process.conflict_qa import conflict,conflict_mix
#from data_process.math import gsm8k,reward_correct
from data_process.choice import choice


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
        if_augment: bool = False,
        dataset_name: str = None,
        is_val: bool = False,
        use_rag: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.if_aug =if_augment  #this controls whether to use parametric knowledge sampling
        self.dataset_name =dataset_name
        self.use_rag =use_rag
        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"
        if dataset_name in ['hotpotqa' , '2wiki']:
            load_data= hotpotqa
        elif dataset_name == 'musique':
            load_data= musique
        elif dataset_name == 'explainpe':
            load_data= choice
        elif dataset_name == 'confiqa':
            load_data= conflict
        elif dataset_name == 'confiqa_sc':
            load_data= conflict_mix
        else:
            load_data=load_dataset
        # if os.path.isdir(data_path):
        #     self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        # elif os.path.isfile(data_path):
        #     self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        # else:  # remote dataset
        #     self.dataset = load_dataset(data_path, split=data_split)
        if is_val:
            self.dataset = load_data(data_path)
        else:
            self.dataset = load_data(data_path)[:4800]  #we only use up to 4800 samples for training
        print("len_self.dataset",len(self.dataset))
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        
        passage =""
        if "passage" in row_dict:
            passage = row_dict["passage"]
        if self.use_rag:
            messages = [{"role": "user", "content": passage+row_dict[self.prompt_key]}]
        else:
            messages = [{"role": "user", "content": row_dict[self.prompt_key]}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        # print(messages)
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.if_aug:
            messages_aug = [{"role": "user", "content": row_dict[self.prompt_key]}]
            if self.system_prompt:
                messages_aug.insert(0, {"role": "system", "content": self.system_prompt})
            prompt_aug = self.tokenizer.apply_chat_template(messages_aug, add_generation_prompt=True, tokenize=False)
        if self.image_key in row_dict:
            prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            row_dict["multi_modal_data"] = {
                "image": [
                    process_image(image, self.max_pixels, self.min_pixels) for image in row_dict.pop(self.image_key)
                ]
            }
            model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)
            if self.if_aug:
                prompt_aug = prompt_aug.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
                row_dict["multi_modal_data_aug"] = {
                    "image": [
                        process_image(image, self.max_pixels, self.min_pixels) for image in row_dict.pop(self.image_key)
                    ]
                }
                model_inputs_aug = self.processor(row_dict["multi_modal_data_aug"]["image"], prompt_aug, return_tensors="pt")
                input_ids_aug = model_inputs_aug.pop("input_ids")[0]
                attention_mask_aug = model_inputs_aug.pop("attention_mask")[0]
                row_dict["multi_modal_inputs_aug"] = dict(model_inputs_aug)
                position_ids_aug = get_rope_index(
                    self.processor,
                    input_ids=input_ids_aug,
                    image_grid_thw=model_inputs_aug["image_grid_thw"],
                    attention_mask=attention_mask_aug,
                )  # (3, seq_length)
        else:
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
            if self.if_aug:
                model_inputs_aug = self.tokenizer([prompt_aug], add_special_tokens=False, return_tensors="pt")
                input_ids_aug = model_inputs_aug.pop("input_ids")[0]
                attention_mask_aug = model_inputs_aug.pop("attention_mask")[0]
                position_ids_aug = torch.clip(attention_mask_aug.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        if self.if_aug:
            input_ids_aug, attention_mask_aug, position_ids_aug = VF.postprocess_data(
                input_ids=input_ids_aug,
                attention_mask=attention_mask_aug,
                position_ids=position_ids_aug,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = row_dict.pop(self.answer_key)
        row_dict["processed_prompt"] = prompt
        if self.if_aug:
            row_dict["input_ids_aug"]= input_ids
            row_dict["attention_mask_aug"] = attention_mask
            row_dict["position_ids_aug"] = position_ids
            row_dict["raw_prompt_ids_aug"] = self.tokenizer.encode(prompt_aug, add_special_tokens=False)
            row_dict["ground_truth_aug"] = row_dict["ground_truth"]
            row_dict["processed_prompt_aug"] = prompt_aug
            
            # this 3 lines are added for computing J_pk  
            row_dict["input_ids_pk"]= input_ids_aug
            row_dict["attention_mask_pk"] = attention_mask_aug
            row_dict["position_ids_pk"] = position_ids_aug


        return row_dict
