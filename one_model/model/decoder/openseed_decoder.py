import torch
import torch.nn as nn
from one_model.common.registries import DECODER_REGISTRY
from typing import Optional, Tuple
from loguru import logger
from one_model.loss import *
from addict import Dict

# Note: should set openseed path to import openseed library
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.colormap import random_color
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager
from .decoder_selector import decoder_router


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    thing_classes.append("tyre")
    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta

meta = get_metadata()

@DECODER_REGISTRY.register(alias="openseed")
class OpenSeeDDecoder(nn.Module):
    def __init__(self, model_name_or_path, config_path) -> None:
        super().__init__()
        from utils.arguments import (
            load_opt_command,
            load_config_dict_to_opt,
            load_opt_from_config_files,
        )
        from openseed.BaseModel import BaseModel
        from openseed import build_model

        opt = load_opt_from_config_files([config_path])
        overrides = ["WEIGHT", model_name_or_path]
        if overrides:
            keys = [overrides[idx * 2] for idx in range(len(overrides) // 2)]
            vals = [overrides[idx * 2 + 1] for idx in range(len(overrides) // 2)]
            vals = [
                val.replace("false", "").replace("False", "")
                if len(val.replace(" ", "")) == 5
                else val
                for val in vals
            ]

            types = []
            for key in keys:
                key = key.split(".")
                ele = opt.copy()
                while len(key) > 0:
                    ele = ele[key.pop(0)]
                types.append(type(ele))

            config_dict = {x: z(y) for x, y, z in zip(keys, vals, types)}
            load_config_dict_to_opt(opt, config_dict)

        opt["command"] = "evaluate"
        # load model
        self.model = (
            BaseModel(opt, build_model(opt)).from_pretrained(model_name_or_path)
            # todo disable
            .eval()
        )

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
        from utils.visualizer import Visualizer

        if not inference:
            # currently not support training
            raise NotImplementedError
        else:
            # check need run infer
            prompt = kwargs.get("prompt", None)
            if prompt:
                run_flag = decoder_router.need_run(prompt, "openseed")
                if not run_flag:
                    logger.info("skip openseed decoder infer")
                    return {}
            image_path = image_paths[0]
            logger.info("handle image {}", image_path)
            transform = transforms.Compose(
                [transforms.Resize(512, interpolation=Image.BICUBIC)]
            )

            # TODO remove later
            # thing_classes = ["car", "person", "traffic light", "truck", "motorcycle"]
            # stuff_classes = ["building", "sky", "street", "tree", "rock", "sidewalk"]
            thing_classes = meta["thing_classes"]
            stuff_classes = meta["stuff_classes"]
            thing_colors = [
                random_color(rgb=True, maximum=255).astype(np.int).tolist()
                for _ in range(len(thing_classes))
            ]
            stuff_colors = [
                random_color(rgb=True, maximum=255).astype(np.int).tolist()
                for _ in range(len(stuff_classes))
            ]
            thing_dataset_id_to_contiguous_id = {
                x: x for x in range(len(thing_classes))
            }
            stuff_dataset_id_to_contiguous_id = {
                x + len(thing_classes): x for x in range(len(stuff_classes))
            }

            MetadataCatalog.get("demo").set(
                thing_colors=meta["thing_colors"],
                thing_classes=thing_classes,
                thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
                stuff_colors=meta["stuff_colors"],
                stuff_classes=stuff_classes,
                stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
            )
            self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                thing_classes + stuff_classes, is_eval=False
            )
            metadata = MetadataCatalog.get("demo")
            self.model.model.metadata = metadata
            self.model.model.sem_seg_head.num_classes = len(
                thing_classes + stuff_classes
            )
            # torch_dtype = torch.float
            device = torch.cuda.current_device()
            with torch.no_grad():
                image_ori = Image.open(image_path).convert("RGB")
                width = image_ori.size[0]
                height = image_ori.size[1]
                image = transform(image_ori)
                image = np.asarray(image)
                image_ori = np.asarray(image_ori)
                images = (
                    torch.from_numpy(image.copy()).permute(2, 0, 1).to(device=device)
                )

                batch_inputs = [{"image": images, "height": height, "width": width}]
                outputs = self.model.forward(batch_inputs)
                visual = Visualizer(image_ori, metadata=metadata)

                pano_seg = outputs[-1]["panoptic_seg"][0]
                pano_seg_info = outputs[-1]["panoptic_seg"][1]

                for i in range(len(pano_seg_info)):
                    if (
                        pano_seg_info[i]["category_id"]
                        in metadata.thing_dataset_id_to_contiguous_id.keys()
                    ):
                        pano_seg_info[i][
                            "category_id"
                        ] = metadata.thing_dataset_id_to_contiguous_id[
                            pano_seg_info[i]["category_id"]
                        ]
                    else:
                        pano_seg_info[i]["isthing"] = False
                        pano_seg_info[i][
                            "category_id"
                        ] = metadata.stuff_dataset_id_to_contiguous_id[
                            pano_seg_info[i]["category_id"]
                        ]

                demo = visual.draw_panoptic_seg(
                    pano_seg.cpu(), pano_seg_info
                )  # rgb Image
                # TODO save
                demo.save("/tmp/pano.png")
                return {"pred_masks": [demo.get_image()]}

    @staticmethod
    def from_config(config):
        model_name_or_path = config.get("model_name_or_path")
        config_path = config.get("config_path")
        return OpenSeeDDecoder(
            model_name_or_path,
            config_path,
        )
