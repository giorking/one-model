import pytest
from loguru import logger
from one_model.common.registries import DECODER_REGISTRY, ENCODER_REGISTRY
from one_model.dataset import *
from one_model.common.mm_utils import load_image
from addict import Dict
from one_model.dataset import *
from one_model.model.decoder import SamDecoder, OpenSeeDDecoder, StableDiffusionDecoder
from one_model.model.projector import *
from addict import Dict
from one_model.common.config import Config
from one_model.common.mm_utils import preprocess_image
from one_model.model.llm import *
import os
from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
import torch
import cv2
from transformers import AutoTokenizer, CLIPTextModel


cur_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(cur_dir, "test_config_7B.yaml")


def test_sam_decoder():
    config: Config = Config(Dict(cfg_path=config_path))
    sam_decoder_l_config = config.decoder_cfg.sam_decoder_l
    sam_decoder_cls = DECODER_REGISTRY.get(sam_decoder_l_config.type)
    sam_decoder: SamDecoder = sam_decoder_cls.from_config(sam_decoder_l_config)

    image_file = f"{cur_dir}/view.jpg"
    image = load_image(image_file)
    transform = ResizeLongestSide(1024)
    image = transform.apply_image(np.asarray(image))
    image_shape = image.shape[:2]
    logger.info("image shape {}", image_shape)
    image_tensor = preprocess_image(
        torch.from_numpy(image).permute(2, 0, 1).contiguous()
    ).unsqueeze(0)
    logger.info(f"image_tensor: {image_tensor.shape}")
    image_embeddings = sam_decoder.image_encoder(image_tensor)
    logger.info(f"image_embeddings: {image_embeddings.shape}")


def test_sam_decoder_forword():
    config: Config = Config(Dict(cfg_path=config_path))
    sam_decoder_l_config = config.decoder_cfg.sam_decoder_l
    sam_decoder_cls = DECODER_REGISTRY.get(sam_decoder_l_config.type)
    sam_decoder: SamDecoder = sam_decoder_cls.from_config(sam_decoder_l_config)
    sam_decoder = sam_decoder.to("cuda:0")
    image_file = f"{cur_dir}/view.jpg"

    seg_token_mask = torch.zeros(1, 381).bool().to("cuda:0")
    hidden_states = []
    hidden_states.append(torch.randn(1, 381, 256).to("cuda:0"))
    result = sam_decoder.forward(
        image_paths=[image_file],
        hidden_states=hidden_states,
        gt_masks=None,
        inference=True,
        seg_token_mask=seg_token_mask,
        offset=None,
    )
    pred_masks = result["pred_masks"]
    logger.info(f"pred_masks len : {len(pred_masks)}")


def test_openseed_decoder():
    config: Config = Config(Dict(cfg_path=config_path))
    openseed_decoder_config = config.decoder_cfg.openseed_decoder_s
    openseed_decoder_cls = DECODER_REGISTRY.get(openseed_decoder_config.type)
    openseed_decoder: OpenSeeDDecoder = openseed_decoder_cls.from_config(
        openseed_decoder_config
    )
    openseed_decoder = openseed_decoder.to("cuda:0")
    image_file = f"{cur_dir}/street.jpg"

    result = openseed_decoder.forward(
        image_paths=[image_file],
        hidden_states=None,
        gt_masks=None,
        inference=True,
        seg_token_mask=None,
        offset=None,
    )
    pred_masks = result["pred_masks"]
    seg_mask = pred_masks[0]
    cv2.imwrite(f"{cur_dir}/pano_seg_mask.png", seg_mask)


def test_stable_diffusion_decoder():
    config: Config = Config(Dict(cfg_path=config_path))
    sd_decoder_config = config.decoder_cfg.stable_diffusion_decoder
    sd_decoder_cls = DECODER_REGISTRY.get(sd_decoder_config.type)
    sd_decoder: StableDiffusionDecoder = sd_decoder_cls.from_config(sd_decoder_config)
    sd_decoder = sd_decoder.to("cuda:0")

    model_name_or_path = "/opt/product/LLaVA/checkpoints/clip-vit-large-patch14"
    model = CLIPTextModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    inputs = tokenizer(
        ["a photograph of an astronaut riding a horse"],
        padding=True,
        return_tensors="pt",
    )

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

    result = sd_decoder.forward(
        image_paths=[],
        hidden_states=[last_hidden_state],
        gt_masks=None,
        inference=True,
        seg_token_mask=None,
        offset=None,
    )
    pred_masks = result["pred_masks"]
    logger.info(f"pred_masks len : {len(pred_masks)}")
    cv2.imwrite(f"{cur_dir}/sd.png", pred_masks[0])


if __name__ == "__main__":
    # test_sam_decoder()
    # test_sam_decoder_forword()
    # test_openseed_decoder()
    test_stable_diffusion_decoder()
