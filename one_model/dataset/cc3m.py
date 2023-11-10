import os
import copy
import json
import pathlib
import random
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import Dataset
from one_model.common.registries import DATASET_REGISTRY
from .dataset import BaseDataset
from loguru import logger
from PIL import Image
from one_model.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from one_model.common import conversation as conversation_lib
import cv2
from transformers import CLIPImageProcessor


def preprocess_multimodal(source, mm_use_im_start_end):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source


@DATASET_REGISTRY.register("cc3m")
class CC3MDataset(BaseDataset):
    img_size = 1024
    ignore_label = 255

    def __init__(self, data_dir: str, max_sample_size, vision_tower) -> None:
        """data dir structure
           data_dir:
               chat.json
               images

        Args:
            data_dir (str): _description_
            max_sample_size (_type_): _description_
            visual_processor (_type_): _description_
            text_tokenizer (_type_): _description_
        """
        self.data_dir = data_dir
        self.visual_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.max_sample_size = max_sample_size
        # load json
        self.chat_data_path = os.path.join(data_dir, "chat.json")
        chat_data_dict = json.load(open(self.chat_data_path, "r"))
        if max_sample_size is not None:
            logger.info("cc3m max_sample_size: {}", max_sample_size)
            chat_data_dict = chat_data_dict[:max_sample_size]
        self.image_folder = os.path.join(data_dir, "images")
        self.chat_data_dict = chat_data_dict
        self.image_aspect_ratio = "square"
        self.is_multimodal = True

    def __len__(self):
        return len(self.chat_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.chat_data_dict[i]
        image_file = item["image"]
        image_path = os.path.join(self.image_folder, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_clip = self.visual_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        conv = conversation_lib.default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
        # print(f"conversations: {conversations}")
        inference = False
        return (
            image_path,
            image_clip,
            conversations,
            masks,
            label,
            questions,
            sampled_classes,
            inference,
        )

    @staticmethod
    def from_config(config):
        data_dir = config.get("dataset_dir")
        max_sample_size = config.get("max_sample_size")
        dataset = CC3MDataset(data_dir, max_sample_size, config.get("vision_tower"))
        return dataset
