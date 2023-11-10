import torch
import torch.nn as nn

from one_model.common.registries import PROJECTOR_REGISTRY


@PROJECTOR_REGISTRY.register(alias="mlp")
class MlpProjector(nn.Module):
    def __init__(self, in_features, out_features, drop_out, freeze, ckpt_path) -> None:
        """init liner projector

        Args:
            in_features (_type_): _description_
            out_features (_type_): _description_
            freeze (_type_): _description_
            ckpt_path (_type_): _description_
        """
        super().__init__()
        text_fc = [
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, out_features),
            nn.Dropout(drop_out),
        ]
        self.register_buffer("in_features", torch.tensor(in_features))
        self.register_buffer("out_features", torch.tensor(out_features))
        self.projector = nn.Sequential(*text_fc)
        if ckpt_path is not None:
            self.projector.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        if freeze:
            self.projector.requires_grad_(False)
        else:
            self.projector.requires_grad_(True)

    def forward(self, data):
        if torch.is_tensor(data):
            original_dtype = data.dtype
            result = self.projector(data.to(self.projector[0].weight.dtype))
            return result.to(original_dtype)
        return None

    @property
    def in_dim(self):
        return self.get_buffer("in_features").item()

    @property
    def out_dim(self):
        return self.get_buffer("out_features").item()

    @staticmethod
    def from_config(config: dict):
        in_features = config.get("in_features")
        out_features = config.get("out_features")
        freeze = config.get("freeze", True)
        ckpt_path = config.get("ckpt_path")
        drop_out = config.get("drop_out", 0.0)
        return MlpProjector(in_features, out_features, drop_out, freeze, ckpt_path)
