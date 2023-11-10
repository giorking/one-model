import pytest
from loguru import logger
from one_model.common.registries import (
    LLM_MODEL_REGISTRY,
    TOKENIZER_REGISTRY,
    ENCODER_REGISTRY,
    PROJECTOR_REGISTRY,
    DECODER_REGISTRY,
)
from one_model.dataset import *
import transformers
from one_model.common.mm_utils import (
    get_model_name_from_path,
    load_image,
    tokenizer_image_token,
)

from addict import Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
from one_model.dataset import *
from one_model.model.encoder import *
from one_model.model.decoder import *
from one_model.model.projector import *
from one_model.model.tokenizer import *
from addict import Dict
from one_model.common.config import Config
from one_model.model.llm import *
import os
from one_model.common.conversation import conv_templates, SeparatorStyle
from one_model.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
import torch
import numpy as np
import cv2
from pathlib import Path

cur_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(cur_dir, "test_config_13B.yaml")


def test_llave_model_segment_infer():
    device = "cuda:0"
    config: Config = Config(Dict(cfg_path=config_path))
    one_model_cfg = config.model_cfg
    tokenizer_cfg = config.tokenizer_cfg[one_model_cfg.tokenizer]
    tokenizer_cls = TOKENIZER_REGISTRY.get(tokenizer_cfg.type)
    tokenizer = tokenizer_cls.from_config(tokenizer_cfg).tokenizer

    llm_config = config.llm_cfg[one_model_cfg.llm]
    llm_cls = LLM_MODEL_REGISTRY.get(llm_config.type)
    model: LlavaLlamaForCausalLM = llm_cls.from_config(llm_config, tokenizer)
    assert model is not None
    logger.info("model {}", model)

    conv_mode = "llava_llama_2"
    conv = conv_templates[conv_mode].copy()

    clip_encoder_large_config = config.encoder_cfg[one_model_cfg.encoder]
    clip_encoder_cls = ENCODER_REGISTRY.get(clip_encoder_large_config.type)
    clip_encoder: CLIPEncoder = clip_encoder_cls.from_config(clip_encoder_large_config)
    clip_encoder = clip_encoder.cuda(0)

    image_proj_13B_config = config.projector_cfg[one_model_cfg.in_projector]
    projector_cls = PROJECTOR_REGISTRY.get(image_proj_13B_config.type)
    logger.info("image_proj_13B_config {}", image_proj_13B_config)
    in_projector = projector_cls.from_config(image_proj_13B_config)
    logger.info("projector {}", in_projector)
    in_projector = in_projector.cuda(0)

    out_project_13B_config = config.projector_cfg[one_model_cfg.out_projector]
    out_projector_cls = PROJECTOR_REGISTRY.get(out_project_13B_config.type)
    out_projector = out_projector_cls.from_config(out_project_13B_config)
    out_projector = out_projector.cuda(0)
    logger.info("out_project_13B_config {}", out_project_13B_config)
    logger.info("out_projector {}", out_projector)

    sam_decoder_config = config.decoder_cfg[one_model_cfg.decoder]
    sam_decoder_cls = DECODER_REGISTRY.get(sam_decoder_config.type)
    sam_decoder: SamDecoder = sam_decoder_cls.from_config(sam_decoder_config)
    sam_decoder = sam_decoder.cuda(0)

    model.get_model().vision_tower = clip_encoder
    model.get_model().mm_projector = in_projector

    image_file = f"{cur_dir}/view.jpg"
    image_processor = clip_encoder.image_processor

    # image processor
    image = load_image(image_file)
    image_tensor = (
        image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        .half()
        .to(device)
    )

    inp = "describe the image"
    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + inp
            )
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    logger.info("prompt {}", prompt)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )
    prompt_len = len(prompt)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            max_new_tokens=512,
            num_beams=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    output_hidden_states = outputs.hidden_states[-1]
    output_ids = outputs.sequences

    text_output = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
    seg_token_mask = output_ids[:, 1:] == model.seg_token_idx
    seg_token_mask = seg_token_mask.to(device)
    # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
    seg_token_mask = torch.cat(
        [
            torch.zeros((seg_token_mask.shape[0], 255))
            .bool()
            .to(seg_token_mask.device),
            seg_token_mask,
        ],
        dim=1,
    )

    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    # text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    # logger.info("text_output {}", text_output)
    # if len(text_output) > prompt_len:
    #     text_output = text_output[prompt_len - 1 :]
    # logger.info("text_output {}", text_output)
    text_output = text_output.replace("\n", "").replace("</s>", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    hidden_states = []

    hidden_states.append(out_projector(output_hidden_states))
    logger.info("hidden_states {}", hidden_states[0].shape)
    logger.info("seg_token_mask shape {}", seg_token_mask.shape)
    decoder_result = sam_decoder.forward(
        image_paths=[image_file],
        hidden_states=hidden_states,
        gt_masks=None,
        inference=True,
        seg_token_mask=seg_token_mask,
    )

    print("\n", {"prompt": prompt, "outputs": text_output}, "\n")
    save_img = None
    pred_masks = decoder_result["pred_masks"]
    image_np = cv2.imread(image_file)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    logger.info("pred_masks {}", len(pred_masks))
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask]
    image_name = Path(image_file).name
    vis_save_path = "./vis"
    if save_img is not None:
        save_path = f"{vis_save_path}/{image_name}"
        logger.info("save segment to {}", save_path)
        cv2.imwrite(save_path, save_img[:, :, ::-1])
    logger.info("infer text out {}", text_output)


if __name__ == "__main__":
    test_llave_model_segment_infer()
