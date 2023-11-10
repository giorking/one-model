import pytest
from loguru import logger
from one_model.common.registries import DATASET_REGISTRY
from one_model.dataset import *
import transformers
from one_model.model.encoder.clip_encoder import CLIPEncoder
from addict import Dict
from one_model.dataset.vqa_dataset import VQADataset
from one_model.dataset.reason_seg_dataset import ReasonSegDataset
from one_model.dataset.sem_seg_dataset import SemSegDataset
from one_model.dataset.refer_seg_dataset import ReferSegDataset
import os
from one_model.common.config import Config
from torch.utils.data import Dataset

cur_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(cur_dir, "test_config_7B.yaml")


def test_hybrid_dataset():
    config: Config = Config(Dict(cfg_path=config_path))
    hybrid_ds_config = config.dataset_cfg.hybrid_ds
    hybrid_ds_cls = DATASET_REGISTRY.get(hybrid_ds_config.type)
    hybrid_ds: Dataset = hybrid_ds_cls.from_config(hybrid_ds_config)
    for i in range(10):
        hybrid_ds[i]


def test_hybrid_val_dataset():
    config: Config = Config(Dict(cfg_path=config_path))
    hybrid_ds_config = config.dataset_cfg.hybrid_val_ds
    hybrid_ds_cls = DATASET_REGISTRY.get(hybrid_ds_config.type)
    hybrid_ds: Dataset = hybrid_ds_cls.from_config(hybrid_ds_config)
    for i in range(10):
        hybrid_ds[i]


def test_sem_seg_dataset():
    model_name_or_path = "/opt/product/llama/llama-2-7b-chat-hf"
    clip_model_name_or_path = "/opt/product/LLaVA/checkpoints/clip-vit-large-patch14"
    base_image_dir = "/opt/product/dataset"
    dataset = SemSegDataset(
        base_image_dir, clip_model_name_or_path, sem_seg_data="ade20k"
    )
    for i in range(10):
        (
            image_path,
            image_clip,
            conversations,
            masks,
            label,
            questions,
            sampled_sents,
        ) = dataset[i]
        if len(conversations) > 1:
            logger.info(
                "conversations len {} , conversations {}",
                len(conversations),
                conversations,
            )
            logger.info("masks {}", len(masks))
    ...


def test_reason_seg_dataset():
    model_name_or_path = "/opt/product/llama/llama-2-7b-chat-hf"
    clip_model_name_or_path = "/opt/product/LLaVA/checkpoints/clip-vit-large-patch14"
    base_image_dir = "/opt/product/dataset"
    dataset = ReasonSegDataset(base_image_dir, clip_model_name_or_path)
    for i in range(10):
        (
            image_path,
            image_clip,
            conversations,
            masks,
            label,
            questions,
            sampled_sents,
        ) = dataset[i]
        if len(conversations) > 1:
            logger.info(
                "conversations len {} , conversations {}",
                len(conversations),
                conversations,
            )
            logger.info("masks {}", len(masks))
    ...


def test_refer_seg_dataset():
    model_name_or_path = "/opt/product/llama/llama-2-7b-chat-hf"
    clip_model_name_or_path = "/opt/product/LLaVA/checkpoints/clip-vit-large-patch14"
    base_image_dir = "/opt/product/dataset"
    dataset = ReferSegDataset(
        base_image_dir, clip_model_name_or_path, refer_seg_data="refcoco+"
    )
    for i in range(10):
        (
            image_path,
            image_clip,
            conversations,
            masks,
            label,
            questions,
            sampled_sents,
        ) = dataset[i]
        if len(conversations) > 1:
            logger.info(
                "conversations len {} , conversations {}",
                len(conversations),
                conversations,
            )
            logger.info("masks {}", len(masks))


def test_vqa_dataset():
    model_name_or_path = "/opt/product/llama/llama-2-7b-chat-hf"
    clip_model_name_or_path = "/opt/product/LLaVA/checkpoints/clip-vit-large-patch14"
    base_image_dir = "/opt/product/dataset"
    samples_per_epoch = 1000
    num_classes_per_sample: int = 3
    exclude_val = False
    vqa_data = "llava_instruct_150k"

    dataset = VQADataset(
        base_image_dir,
        clip_model_name_or_path,
        samples_per_epoch,
        num_classes_per_sample,
        exclude_val,
        vqa_data,
    )
    for i in range(10):
        (
            image_path,
            image_clip,
            conversations,
            masks,
            label,
            questions,
            sampled_classes,
        ) = dataset[i]
        if len(conversations) > 1:
            print(len(conversations))


def test_cc3m_ds():
    cc3m_ds = DATASET_REGISTRY.get("cc3m")
    assert cc3m_ds is not None
    max_sample_size = 100
    dataset: CC3MDataset = cc3m_ds.from_config(
        {
            "dataset_dir": "/opt/product/LLaVA/dataset/LLaVA-CC3M-Pretrain-595K",
            "max_sample_size": max_sample_size,
            "vision_tower": "/opt/product//LLaVA/checkpoints/clip-vit-large-patch14",
        }
    )

    for i in range(max_sample_size):
        data_dict = dataset[i]
        logger.info("conversations {}", data_dict[2])
        # print(data_dict.keys())
