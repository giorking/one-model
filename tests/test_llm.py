import pytest
from loguru import logger
from one_model.common.registries import (
    LLM_MODEL_REGISTRY,
    TOKENIZER_REGISTRY,
    ENCODER_REGISTRY,
    PROJECTOR_REGISTRY,
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


cur_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(cur_dir, "test_config_7B.yaml")


def test_llama2_load():
    """test llama2 prompt and tokenizer function"""
    kwargs = {"device_map": "auto", "load_in_8bit": True}
    model_path = "/opt/product/LLaVA/checkpoints/llava-7b-llama-2-7b-chat"
    model_path = "/opt/product/llama/llama-2-7b-chat-hf"
    model_name = get_model_name_from_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LLAMA2ForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, **kwargs
    )
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"model name: {model_name}")
    # test tokenizer
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    output = tokenizer.decode(
        generate_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    logger.info(f"output: {output}")
    output = tokenizer.decode(generate_ids[0], skip_special_tokens=True).strip()
    logger.info(f"output: {output}")

    output = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    logger.info(f"output: {output}")


def test_llave_model_load():
    config: Config = Config(Dict(cfg_path=config_path))
    tokenizer_cfg = config.tokenizer_cfg.llama2
    tokenizer_cls = TOKENIZER_REGISTRY.get(tokenizer_cfg.type)
    tokenizer = tokenizer_cls.from_config(tokenizer_cfg)
    llm_7B_config = config.llm_cfg.llava_7b
    llm_cls = LLM_MODEL_REGISTRY.get(llm_7B_config.type)
    llm = llm_cls.from_config(llm_7B_config, tokenizer.tokenizer)
    assert llm is not None
    logger.info("llm {}", llm)


def test_llave_model_infer():
    config: Config = Config(Dict(cfg_path=config_path))
    tokenizer_cfg = config.tokenizer_cfg.llama2
    tokenizer_cls = TOKENIZER_REGISTRY.get(tokenizer_cfg.type)
    tokenizer = tokenizer_cls.from_config(tokenizer_cfg).tokenizer

    llm_7B_config = config.llm_cfg.llava_7b
    llm_cls = LLM_MODEL_REGISTRY.get(llm_7B_config.type)
    model: LlavaLlamaForCausalLM = llm_cls.from_config(llm_7B_config, tokenizer)
    assert model is not None
    logger.info("model {}", model)
    conv_mode = "llava_llama_2"
    conv = conv_templates[conv_mode].copy()

    clip_encoder_large_config = config.encoder_cfg.clip_encoder
    clip_encoder_cls = ENCODER_REGISTRY.get(clip_encoder_large_config.type)
    clip_encoder: CLIPEncoder = clip_encoder_cls.from_config(clip_encoder_large_config)

    image_proj_7B_config = config.projector_cfg.image_proj_7B
    projector_cls = PROJECTOR_REGISTRY.get(image_proj_7B_config.type)
    logger.info("proj 7B config {}", image_proj_7B_config)
    projector = projector_cls.from_config(image_proj_7B_config)
    logger.info("projector {}", projector)

    model.get_model().vision_tower = clip_encoder
    model.get_model().mm_projector = projector

    image_file = f"{cur_dir}/view.jpg"
    image_processor = clip_encoder.image_processor

    # image processor
    image = load_image(image_file)
    image_tensor = (
        image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        .half()
        .cuda()
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
        .cuda()
    )
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()

    if True:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
