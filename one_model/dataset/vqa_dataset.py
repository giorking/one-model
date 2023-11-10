import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from one_model.common import conversation as conversation_lib

from .utils import DEFAULT_IMAGE_TOKEN
from loguru import logger


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


class VQADataset(torch.utils.data.Dataset):
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        vqa_data="llava_instruct_150k",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        logger.info("vqa data dir {}", DATA_DIR)
        coco_dir = os.environ.get("COCO_2017_DIR")
        if coco_dir is not None:
            # self.vqa_image_root = '/opt/datasets/COCO_2017/raw/Images/train2017'
            self.vqa_image_root = coco_dir
        else:
            self.vqa_image_root = "/opt/datasets/COCO_2017/raw/Images/train2017"
            # TODO disable later
            # self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        logger.info("vqa image root {}", self.vqa_image_root)
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        print("vqa_data: ", len(self.vqa_data))

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][
            0
        ]  # preprocess image for clip

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
        return (
            image_path,
            image_clip,
            conversations,
            masks,
            label,
            questions,
            sampled_classes,
        )
