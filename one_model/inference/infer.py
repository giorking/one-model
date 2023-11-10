import torch
import torch.nn as nn
import argparse
from loguru import logger
from typing import List, Optional, Tuple, Union
from one_model.common.config import Config
from one_model.model import OneModel
from addict import Dict
import cv2
from pathlib import Path
import numpy as np
import os


class Infer:
    def __init__(self, config: Config) -> None:
        # init config
        self.config = config
        self.model: OneModel = OneModel(config)

    def init_model(self):
        """init model"""
        self.model.init_model(mode="infer", torch_dtype=torch.half)
        self.model.eval()

    def predict(self, image_file, prompt, vis_save_path, **kwargs):
        """do predict"""
        inputs = Dict({"image": image_file, "prompt": prompt})
        output_dict = self.model.generate(inputs)
        text_output = output_dict["text_output"]
        pred_masks = output_dict["pred_masks"]
        # handle save logic
        logger.info("prompt {}, text output {}", prompt, text_output)
        save_img = None
        # image_np = cv2.imread(image_file)
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        save_img = None
        if pred_masks is not None and len(pred_masks) > 0:
            save_img = pred_masks[0]
        # for i, pred_mask in enumerate(pred_masks):
        #     if pred_mask.shape[0] == 0:
        #         continue

        #     pred_mask = pred_mask.detach().cpu().numpy()[0]
        #     pred_mask = pred_mask > 0

        #     save_img = image_np.copy()
        #     save_img[pred_mask] = (
        #         image_np * 0.5
        #         + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        #     )[pred_mask]
        image_name = Path(image_file).name
        os.makedirs(vis_save_path, exist_ok=True)
        save_path = None
        if save_img is not None:
            save_path = f"{vis_save_path}/{image_name}"
            logger.info("save segment to {}", save_path)
            cv2.imwrite(save_path, save_img[:, :, ::-1])
        return text_output, save_path


def parse_args():
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument("--conf", required=True, help="path to configuration file.")
    parser.add_argument(
        "--image",
        default="/opt/product/one_model/tests/view.jpg",
        help="image to describe",
    )
    parser.add_argument(
        "--prompt",
        default="describe the image",
        help="image prompt",
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config: Config = Config(Dict(cfg_path=args.conf))
    infer = Infer(config)
    infer.init_model()
    logger.info("predict image {}, prompt {}", args.image, args.prompt)
    infer.predict(args.image, args.prompt, args.vis_save_path)
