import torch
import torch.nn as nn
from one_model.common.registries import DECODER_REGISTRY
from typing import Optional, Tuple
from loguru import logger
from one_model.loss import *
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPSegProcessor,
    CLIPSegForImageSegmentation,
)
from diffusers import StableDiffusionPipeline
import numpy as np
from .decoder_selector import decoder_router


@DECODER_REGISTRY.register(alias="sd")
class StableDiffusionDecoder(nn.Module):
    def __init__(self, model_name_or_path) -> None:
        super().__init__()
        # model_name_or_path = "/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name_or_path, torch_dtype=torch.half
        )
        self.device = torch.cuda.current_device()
        self.pipe = pipe.to(self.device)

    def forward(
        self,
        image_paths,
        hidden_states,
        gt_masks,
        inference,
        seg_token_mask,
        offset: torch.LongTensor,
        **kwargs,
    ):
        # TODO return image ?
        if inference:
            prompt = kwargs.get("prompt", None)
            if prompt:
                run_flag = decoder_router.need_run(prompt, "sd")
                if not run_flag:
                    logger.info("skip sd decoder infer")
                    return {}
            '''
            image = self.pipe(prompt_embeds=hidden_states[-1], guidance_scale=0).images[
                0
            ]
            '''
            image = self.pipe(prompt).images[0]
            return {"pred_masks": [np.asarray(image)]}
        else:
            # TODO return embedding ?
            ...

    @staticmethod
    def from_config(config):
        model_name_or_path = config.get(
            "model_name_or_path", "runwayml/stable-diffusion-v1-5"
        )
        return StableDiffusionDecoder(
            model_name_or_path,
        )
