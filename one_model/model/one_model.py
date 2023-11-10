import torch
import torch.nn as nn
from loguru import logger
from one_model.common.config import Config
from one_model.common.registries import *
from one_model.dataset import *
from one_model.model.encoder import *
from one_model.model.decoder import *
from one_model.model.projector import *
from one_model.model.tokenizer import *
from one_model.model.llm import *
from one_model.common.mm_utils import (
    load_image,
    tokenizer_image_token,
)
from addict import Dict
from one_model.common.config import Config
from one_model.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from one_model.common.conversation import conv_templates, SeparatorStyle
from one_model.common import conversation as conversation_lib
from .decoder.decoder_selector import decoder_router


class OneModel(nn.Module):
    """one model to run llm train and infe

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.conv_mode = "llava_llama_2"
        self.device = torch.cuda.current_device()
        logger.info("current select device {}", self.device)

    def init_model(self, mode="infer", torch_dtype=torch.float32):
        self.torch_dtype = torch_dtype
        config = self.config
        one_model_cfg = config.model_cfg
        tokenizer_cfg = config.tokenizer_cfg[one_model_cfg.tokenizer]
        tokenizer_cls = TOKENIZER_REGISTRY.get(tokenizer_cfg.type)
        tokenizer = tokenizer_cls.from_config(tokenizer_cfg).tokenizer

        llm_config = config.llm_cfg[one_model_cfg.llm]
        llm_cls = LLM_MODEL_REGISTRY.get(llm_config.type)
        llm_model: LlavaLlamaForCausalLM = llm_cls.from_config(llm_config, tokenizer)
        # logger.info("model {}", llm_model)
        llm_model.resize_token_embeddings(len(tokenizer))
        conversation = conv_templates[self.conv_mode].copy()
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            self.conv_mode
        ]

        encoder_large_config = config.encoder_cfg[one_model_cfg.encoder]
        encoder_cls = ENCODER_REGISTRY.get(encoder_large_config.type)
        encoder = encoder_cls.from_config(encoder_large_config)
        encoder = encoder.to(dtype=torch_dtype, device=self.device)

        encoder_name_or_path = encoder_large_config.model_name_or_path
        decoder_router.init_model(encoder_name_or_path)

        image_proj_config = config.projector_cfg[one_model_cfg.in_projector]
        projector_cls = PROJECTOR_REGISTRY.get(image_proj_config.type)
        logger.info("iin_projector_config {}", image_proj_config)
        in_projector = projector_cls.from_config(image_proj_config)
        logger.info("projector {}", in_projector)
        in_projector = in_projector.to(dtype=torch_dtype, device=self.device)

        out_projectors = []
        decoders = []

        if "decoder" in one_model_cfg:
            out_projector_names = []
            if type(one_model_cfg.out_projector) == str:
                out_projector_names.append(one_model_cfg.out_projector)
            else:
                out_projector_names.extend(one_model_cfg.out_projector)
            for projector_name in out_projector_names:
                out_project_config = config.projector_cfg[projector_name]
                out_projector_cls = PROJECTOR_REGISTRY.get(out_project_config.type)
                out_projector = out_projector_cls.from_config(out_project_config)
                out_projector = out_projector.to(dtype=torch_dtype, device=self.device)
                out_projectors.append(out_projector)

            decoder_names = []
            if type(one_model_cfg.decoder) == str:
                decoder_names.append(one_model_cfg.decoder)
            else:
                decoder_names.extend(one_model_cfg.decoder)

            for decoder_name in decoder_names:
                decoder_config = config.decoder_cfg[decoder_name]
                decoder_cls = DECODER_REGISTRY.get(decoder_config.type)
                decoder = decoder_cls.from_config(decoder_config)
                if "openseed" == decoder_config.type:
                    decoder = decoder.to(device=self.device)
                else:
                    decoder = decoder.to(dtype=torch_dtype, device=self.device)
                decoders.append(decoder)

            if len(decoders) != len(out_projectors):
                raise Exception(
                    f"decoder and out projector not the same! decoder len {len(decoders)}, projector len {len(out_projectors)}"
                )

        if mode == "train":
            # train mode
            llm_model.enable_input_require_grads()
            llm_model.gradient_checkpointing_enable()

        self.conversation = conversation
        self.encoder = encoder
        self.in_projector = in_projector
        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.out_projectors = out_projectors
        self.decoders = decoders

    def forward(self, **kwargs):
        """call model train"""
        # logger.info("forward kwargs {}", kwargs.keys())
        image_paths = kwargs["image_paths"]
        # logger.info("inference {}", kwargs["inference"])
        inference = kwargs.get("inference", False)

        # preprocess image
        if inference:
            images_clip = kwargs.get("images_clip", None)
            images_clip_expand = images_clip.expand(1, -1, -1, -1).contiguous()
            image_embdings = self.encoder.forward(images_clip_expand)
            image_embdings = self.in_projector.forward(image_embdings)
            kwargs["image_embdings"] = image_embdings
            torch.cuda.empty_cache()
        else:
            images_clip = kwargs.get("images_clip", None)
            offset = kwargs.get("offset", None)
            images_clip_list = []
            # 对于多段对话的情况，图片需要复制多份， 拆分成单个的对话列表
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip_expand = torch.cat(images_clip_list, dim=0)
            # calc image embdings
            image_embdings = self.encoder.forward(images_clip_expand)
            image_embdings = self.in_projector.forward(image_embdings)
            kwargs["image_embdings"] = image_embdings
            torch.cuda.empty_cache()

        llm_output = self.llm_model.forward(**kwargs)

        if len(self.decoders) == 0:
            # maybe only train llm
            return llm_output

        seg_token_mask = llm_output["seg_token_mask"]
        output_hidden_states = llm_output["hidden_states"]

        # for train side, support only one decoder and projector
        out_projector = self.out_projectors[0]
        decoder = self.decoders[0]

        hidden_states = []
        hidden_states.append(out_projector(output_hidden_states))
        # logger.info("hidden_states {}", hidden_states[0].shape)
        # logger.info("seg_token_mask shape {}", seg_token_mask.shape)
        ce_loss = None
        if "loss" in llm_output:
            ce_loss = llm_output["loss"]
        decoder_output = decoder.forward(
            image_paths=image_paths,
            hidden_states=hidden_states,
            gt_masks=llm_output["gt_masks"],
            inference=kwargs["inference"],
            seg_token_mask=seg_token_mask,
            ce_loss=ce_loss,
            offset=kwargs["offset"],
            mode=kwargs.get("mode", "train"),
        )
        return decoder_output

    def generate(self, inputs: Dict):
        """run inference on the image

        example inputs
        {
            "image": "view.jpg",
            "prompt" : "segment the lake"
        }
        Args:
            inputs (dict): _description_
        """
        # handle input
        image_file = inputs["image"]
        device = inputs.get("device", "cuda:0")
        prompt = inputs.get("prompt", "describe the pic")

        image = load_image(image_file)
        image_tensor = self.encoder.preprocess(image, self.torch_dtype, device)
        # generate image embdings
        image_embding = self.encoder.forward(image_tensor)
        image_embding = self.in_projector.forward(image_embding)
        # generate inputs
        input_ids = self.generate_input_ids(image, prompt, device)
        self.llm_model.eval()

        # run model inference
        with torch.no_grad():
            outputs = self.llm_model.generate(
                input_ids=input_ids,
                image_embdings=image_embding,
                max_new_tokens=512,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        output_hidden_states = outputs.hidden_states[-1]
        output_ids = outputs.sequences

        # decode text output
        text_output = self.tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        text_output = (
            text_output.replace("\n", "").replace("</s>", "").replace("  ", " ")
        )
        text_output = text_output.split("ASSISTANT: ")[-1]

        # handle seg token mask
        seg_token_mask = output_ids[:, 1:] == self.llm_model.seg_token_idx
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

        pred_masks = None
        for i in range(len(self.out_projectors)):
            out_projector = self.out_projectors[i]
            decoder = self.decoders[i]
            hidden_states = []
            hidden_states.append(out_projector(output_hidden_states))
            logger.info("hidden_states {}", hidden_states[0].shape)
            logger.info("seg_token_mask shape {}", seg_token_mask.shape)

            # do decoder
            decoder_result = decoder.forward(
                image_paths=[image_file],
                hidden_states=hidden_states,
                gt_masks=None,
                inference=True,
                seg_token_mask=seg_token_mask,
                offset=None,
                prompt=prompt,
            )
            if "pred_masks" in decoder_result:
                pred_masks = decoder_result["pred_masks"]
                break
        return {
            "text_output": text_output,
            "pred_masks": pred_masks,
            "llm_embeddings": output_hidden_states,
        }

    def generate_input_ids(self, image, prompt, device):
        """根据image 和 prompt 通过tokenizer 生成对应的input_ids

        Args:
            image (_type_): _description_ 原始图片
            prompt (_type_): _description_
            device (_type_): _description_

        Returns:
            _type_: _description_
        """
        conversation = self.conversation.copy()
        conversation.messages = []
        if image is not None:
            # first message
            if self.llm_model.config.mm_use_im_start_end:
                prompt = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + prompt
                )
            else:
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conversation.append_message(conversation.roles[0], prompt)
        else:
            # later messages
            conversation.append_message(conversation.roles[0], prompt)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()

        logger.info("prompt {}", prompt)
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(device)
        )
        return input_ids

    @staticmethod
    def from_config(config: Config):
        return OneModel(config)
