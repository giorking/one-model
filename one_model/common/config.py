import io
from loguru import logger
from typing import Dict

from omegaconf import OmegaConf


class Config:
    def __init__(self, args):
        self.config = {}

        self.args = args
        if "options" in args:
            user_config = self._build_opt_list(self.args.options)

        self.config = OmegaConf.load(self.args.cfg_path)

    @property
    def dataset_cfg(self):
        return self.config.dataset

    @property
    def encoder_cfg(self):
        return self.config.encoder

    @property
    def projector_cfg(self):
        return self.config.projector

    @property
    def llm_cfg(self):
        return self.config.llm

    @property
    def model_cfg(self):
        return self.config.one_model

    @property
    def tokenizer_cfg(self):
        return self.config.tokenizer

    @property
    def train_cfg(self):
        return self.config.trainer

    @property
    def decoder_cfg(self):
        return self.config.decoder

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def get_config(self):
        return self.config
