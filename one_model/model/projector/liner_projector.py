import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from one_model.common.registries import PROJECTOR_REGISTRY


@PROJECTOR_REGISTRY.register(alias="linear")
class LinerProjector(nn.Module):
    def __init__(self, in_features, out_features, freeze, ckpt_path) -> None:
        """init liner projector

        Args:
            in_features (_type_): _description_
            out_features (_type_): _description_
            freeze (_type_): _description_
            ckpt_path (_type_): _description_
        """
        super().__init__()
        self.projector = nn.Linear(in_features, out_features)
        if ckpt_path is not None:
            mm_projector_weights = torch.load(ckpt_path, map_location="cpu")
            self.projector.load_state_dict(mm_projector_weights, strict=False)
        if freeze:
            self.projector.requires_grad_(False)
        else:
            self.projector.requires_grad_(True)

    def forward(self, data):
        if torch.is_tensor(data):
            original_dtype = data.dtype
            result = self.projector(data.to(self.projector.weight.dtype))
            return result.to(original_dtype)
        return None

    @property
    def in_dim(self):
        return self.projector.in_features

    @property
    def out_dim(self):
        return self.projector.out_features

    @staticmethod
    def from_config(config: dict):
        in_features = config.get("in_features")
        out_features = config.get("out_features")
        freeze = config.get("freeze", True)
        ckpt_path = config.get("ckpt_path")
        return LinerProjector(in_features, out_features, freeze, ckpt_path)
