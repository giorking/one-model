import torch
import torch.nn as nn

from one_model.common.registries import PROJECTOR_REGISTRY


@PROJECTOR_REGISTRY.register(alias="dummy")
class DummyProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data):
        return data

    @property
    def in_dim(self):
        return 256

    @staticmethod
    def from_config(config: dict):
        return DummyProjector()
