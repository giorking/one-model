import torch
import torch.nn as nn
from addict import Dict
from loguru import logger

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from one_model.common.registries import ENCODER_REGISTRY


@ENCODER_REGISTRY.register(alias="clip")
class CLIPEncoder(nn.Module):
    def __init__(self, model_name_or_path, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = model_name_or_path
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = args.get("mm_vision_select_feature", "patch")
        logger.info(
            "select_layer {}, select_feature {}", self.select_layer, self.select_feature
        )

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def preprocess(self, image, torch_dtype, device):
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ].to(device=device, dtype=torch_dtype)
        return image_tensor

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @staticmethod
    def from_config(config):
        model_name_or_path = config.get("model_name_or_path")
        freeze = config.get("freeze", True)
        select_feature_layer = config.get("select_feature_layer", -2)
        return CLIPEncoder(
            model_name_or_path,
            Dict({"mm_vision_select_layer": select_feature_layer, "freeze": freeze}),
        )
