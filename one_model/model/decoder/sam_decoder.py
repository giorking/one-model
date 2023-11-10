import torch
import torch.nn as nn
from one_model.common.mm_utils import load_image_as_ndarray

from one_model.common.registries import DECODER_REGISTRY
from typing import Optional, Tuple
from segment_anything import build_sam_vit_h, build_sam_vit_l
from loguru import logger
from one_model.processor import SamPreProcessor
from one_model.loss import *
import numpy as np
import cv2
from .decoder_selector import decoder_router


@DECODER_REGISTRY.register(alias="sam")
class SamDecoder(nn.Module):
    def __init__(self, model_name_or_path, model_type, train_mask_decoder) -> None:
        super().__init__()
        logger.info(
            "sam decoder init, model_name_or_path {}, model_type {}",
            model_name_or_path,
            model_type,
        )
        self.visual_model = self.build_visual_model(model_type, model_name_or_path)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        self.processor = SamPreProcessor()
        if train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        self.ce_loss_weight = 1.0
        self.dice_loss_weight = 0.5
        self.bce_loss_weight = 2.0

    def build_visual_model(self, model_type, model_name_or_path):
        if model_type == "sam_l":
            return build_sam_vit_l(model_name_or_path)
        elif model_type == "sam_h":
            return build_sam_vit_h(model_name_or_path)
        else:
            raise Exception(f"not supprt model type {model_type}")

    @property
    def image_encoder(self):
        return self.visual_model.image_encoder

    @property
    def prompt_encoder(self):
        return self.visual_model.prompt_encoder

    @property
    def mask_decoder(self):
        return self.visual_model.mask_decoder

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.image_encoder(pixel_values[i].unsqueeze(0))
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def get_img_pixel_value(self, img_path):
        image = load_image_as_ndarray(img_path)
        image_tensor, image_shape = self.processor.process(image)
        return image_tensor, image_shape, image.shape[:2]

    def get_visual_embs_img_paths(self, img_paths, device, dtype):
        # logger.info("get_visual_embs_img_paths dtype {}, device {}", dtype, device)
        resize_list = []
        original_size = []
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(len(img_paths)):
                (
                    pixel_value,
                    resize_shape,
                    original_img_shape,
                ) = self.get_img_pixel_value(img_paths[i])
                pixel_value = pixel_value.to(dtype=dtype, device=device)
                resize_list.append(resize_shape)
                original_size.append(original_img_shape)
                # logger.info("pixel_value.shape {}", pixel_value.shape)
                torch.cuda.empty_cache()
                image_embeddings = self.image_encoder(pixel_value.unsqueeze(0))
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings, resize_list, original_size

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
        """sam decoder forward, do follow steps:
        1. generate image embeddings
        2. generate prompt encoder
        3. run mask decoder, predict the masks
        """

        # check need run infer
        if inference:
            prompt = kwargs.get("prompt", None)
            if prompt:
                run_flag = decoder_router.need_run(prompt, "sam")
                if not run_flag:
                    logger.info("skip sam decoder infer")
                    return {}

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]

        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().to(seg_token_mask.device), seg_token_offset], dim=0
        )
        if offset is not None:
            seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_
        (
            image_embeddings,
            resize_list,
            original_size_list,
        ) = self.get_visual_embs_img_paths(
            image_paths, seg_token_mask.device, last_hidden_state.dtype
        )
        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, _ = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=original_size_list[i],
            )
            pred_masks.append(pred_mask[:, 0])

        if inference:
            mode = kwargs.get("mode")
            # fix sam validate stage issue
            if mode is not None and mode == "val":
                return {
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                }
            logger.info("pred_masks {}", len(pred_masks))
            image_np = cv2.imread(image_paths[0])
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            save_img = None
            for i, pred_mask in enumerate(pred_masks):
                if pred_mask.shape[0] == 0:
                    continue

                pred_mask = pred_mask.detach().cpu().numpy()[0]
                pred_mask = pred_mask > 0

                save_img = image_np.copy()
                save_img[pred_mask] = (
                    image_np * 0.5
                    + pred_mask[:, :, None].astype(np.uint8)
                    * np.array([255, 0, 0])
                    * 0.5
                )[pred_mask]
            if save_img is not None:
                return {
                    "pred_masks": [save_img],
                    "gt_masks": gt_masks,
                }
            else:
                return {}

        ce_loss = kwargs.get("ce_loss", 0)
        # logger.info("ce loss {}", ce_loss)
        ce_loss = ce_loss * self.ce_loss_weight
        loss = ce_loss
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss += mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    @staticmethod
    def from_config(config):
        model_name_or_path = config.get("model_name_or_path")
        train_mask_decoder = config.get("train_mask_decoder", True)
        model_type = config.get("model_type", "sam_l")
        return SamDecoder(
            model_name_or_path,
            model_type,
            train_mask_decoder,
        )
