import os
from dataclasses import dataclass, field
import json
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from loguru import logger
import argparse
import os
import shutil
import time
from functools import partial
from addict import Dict
import deepspeed
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from one_model.common import conversation as conversation_lib
from one_model.model import *
from one_model.common.utils import *
from one_model.common.dist_utils import get_rank
from one_model.common.config import Config

from one_model.dataset.dataset_collate import collate_fn
from one_model.common.registries import DATASET_REGISTRY

from one_model.dataset.dataset import HybridDataset, ValDataset
from one_model.dataset.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    AverageMeter,
    ProgressMeter,
    Summary,
    dict_to_cuda,
    intersectionAndUnionGPU,
)


local_rank = None


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--conf", required=True, help="path to configuration file.")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="one-model-13b", type=str)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--ds_config_path",
        default="train_configs/zero2.json",
        help="path to deepspeed config file.",
    )
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=10, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    args = parser.parse_args()

    return args


def initialize_distributed(args):
    # args["master_ip"] = os.getenv("MASTER_ADDR", "localhost")
    # args["master_port"] = os.getenv("MASTER_PORT", "6000")
    # args["world_size"] = int(os.getenv("WORLD_SIZE", "1"))
    device = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    # deepspeed.init_distributed(dist_backend="nccl")


def setup_seeds(config: Config):
    seed = config.train_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


class Trainer:
    def __init__(self, job_id, config, args) -> None:
        self.job_id = job_id
        self.config = config
        self.args = args
        self.model: OneModel = None
        logger.info("init trainer {}", job_id)

    def build_model(self, torch_dtype):
        self.model: OneModel = OneModel(self.config)
        self.model.init_model(mode="train", torch_dtype=torch_dtype)
        return self.model.to(dtype=torch_dtype)

    def load_dataset(self):
        # start load dataset
        dataset_cfg = self.config.model_cfg.dataset
        logger.info("dataset_cfg {}", dataset_cfg)
        train_ds_name = dataset_cfg.train
        val_ds_name = dataset_cfg.val
        train_ds_cfg = self.config.dataset_cfg[train_ds_name]
        logger.info("train_ds_cfg {}", train_ds_cfg)
        train_ds_cls = DATASET_REGISTRY.get(train_ds_cfg.type)
        train_dataset: Dataset = train_ds_cls.from_config(train_ds_cfg)

        logger.info(f"Training with {len(train_dataset)} examples")

        val_dataset: Dataset = None
        if val_ds_name:
            val_ds_cfg = self.config.dataset_cfg[val_ds_name]
            logger.info("val_ds_cfg {}", val_ds_cfg)
            val_ds_cls = DATASET_REGISTRY.get(val_ds_cfg.type)
            val_dataset: Dataset = val_ds_cls.from_config(val_ds_cfg)
            logger.info(f"validating with {len(val_dataset)} examples")

        return train_dataset, val_dataset

    def train(self):
        logger.info("start train ...")
        args = self.args
        args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
        if args.local_rank == 0:
            os.makedirs(args.log_dir, exist_ok=True)
            writer = SummaryWriter(args.log_dir)
        else:
            writer = None
        train_cfg = self.config.train_cfg
        torch_dtype = torch.float32
        if train_cfg.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif train_cfg.precision == "fp16":
            torch_dtype = torch.half
        logger.info(
            "torch dtype {}, current device {}",
            torch_dtype,
            torch.cuda.current_device(),
        )
        # build model
        model = self.build_model(torch_dtype)
        tokenizer = model.tokenizer
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        conv_type = "llava_llama_2"
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            conv_type
        ]

        world_size = torch.cuda.device_count()
        args.distributed = world_size > 1

        args.precision = train_cfg.precision
        samples_per_epoch = (
            args.batch_size
            * args.grad_accumulation_steps
            * args.steps_per_epoch
            * world_size
        )
        logger.info("samples_per_epoch {}", samples_per_epoch)

        # load dataset
        train_dataset, val_dataset = self.load_dataset()

        ds_config = json.load(open(args.ds_config_path))
        # config model
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
            config=ds_config,
        )

        # resume deepspeed checkpoint
        # TODO remove
        args.auto_resume = False
        args.resume = None
        if args.auto_resume and (args.resume is None or len(args.resume) == 0):
            resume = os.path.join(args.log_dir, "ckpt_model")
            if os.path.exists(resume):
                args.resume = resume
        logger.info("resume path {}", args.resume)
        if args.resume:
            load_path, client_state = model_engine.load_checkpoint(args.resume)
            with open(os.path.join(args.resume, "latest"), "r") as f:
                ckpt_dir = f.readlines()[0].strip()
            args.start_epoch = (
                int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
            )
            print(
                "resume training from {}, start from epoch {}".format(
                    args.resume, args.start_epoch
                )
            )

        # validation dataset
        if val_dataset is not None:
            assert args.val_batch_size == 1
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    conv_type=conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    local_rank=args.local_rank,
                ),
            )

        train_iter = iter(train_loader)
        best_score, cur_ciou = 0.0, 0.0

        if args.eval_only:
            giou, ciou = self.validate(val_loader, model_engine, 0, writer, args)
            exit()

        # train model part
        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train_iter = self.train_one_epoch(
                train_loader,
                model_engine,
                epoch,
                scheduler,
                writer,
                train_iter,
                args,
            )

            if val_dataset is not None:
                giou, ciou = self.validate(
                    val_loader, model_engine, epoch, writer, args
                )
                is_best = giou > best_score
                best_score = max(giou, best_score)
                cur_ciou = ciou if is_best else cur_ciou

            if is_best:
                save_dir = os.path.join(args.log_dir, "ckpt_model")
                if args.local_rank == 0:
                    torch.save(
                        {"epoch": epoch},
                        os.path.join(
                            args.log_dir,
                            "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                                best_score, cur_ciou
                            ),
                        ),
                    )
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)
                torch.distributed.barrier()
                model_engine.save_checkpoint(save_dir)

    def train_one_epoch(
        self,
        train_loader,
        model,
        epoch,
        scheduler,
        writer,
        train_iter,
        args,
    ):
        """Main training loop."""
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        ce_losses = AverageMeter("CeLoss", ":.4f")
        mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
        mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
        mask_losses = AverageMeter("MaskLoss", ":.4f")

        progress = ProgressMeter(
            args.steps_per_epoch,
            [
                batch_time,
                losses,
                ce_losses,
                mask_losses,
                mask_bce_losses,
                mask_dice_losses,
            ],
            prefix="Epoch: [{}]".format(epoch),
        )

        # switch to train mode
        model.train()
        end = time.time()
        for global_step in range(args.steps_per_epoch):
            for i in range(args.grad_accumulation_steps):
                try:
                    input_dict = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    input_dict = next(train_iter)

                data_time.update(time.time() - end)
                input_dict = dict_to_cuda(input_dict)

                if args.precision == "fp16":
                    input_dict["images_clip"] = input_dict["images_clip"].half()
                elif args.precision == "bf16":
                    input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                else:
                    input_dict["images_clip"] = input_dict["images_clip"].float()

                output_dict = model(**input_dict)

                loss = output_dict["loss"]
                ce_loss = output_dict["ce_loss"]
                mask_bce_loss = output_dict["mask_bce_loss"]
                mask_dice_loss = output_dict["mask_dice_loss"]
                mask_loss = output_dict["mask_loss"]

                losses.update(loss.item(), input_dict["images_clip"].size(0))
                ce_losses.update(ce_loss.item(), input_dict["images_clip"].size(0))
                mask_bce_losses.update(
                    mask_bce_loss.item(), input_dict["images_clip"].size(0)
                )
                mask_dice_losses.update(
                    mask_dice_loss.item(), input_dict["images_clip"].size(0)
                )
                mask_losses.update(mask_loss.item(), input_dict["images_clip"].size(0))
                model.backward(loss)
                model.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if global_step % args.print_freq == 0:
                if args.distributed:
                    batch_time.all_reduce()
                    data_time.all_reduce()

                    losses.all_reduce()
                    ce_losses.all_reduce()
                    mask_bce_losses.all_reduce()
                    mask_dice_losses.all_reduce()
                    mask_losses.all_reduce()

                if args.local_rank == 0:
                    progress.display(global_step + 1)
                    writer.add_scalar("train/loss", losses.avg, global_step)
                    writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                    writer.add_scalar(
                        "train/mask_bce_loss", mask_bce_losses.avg, global_step
                    )
                    writer.add_scalar(
                        "train/mask_dice_loss", mask_dice_losses.avg, global_step
                    )
                    writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                    writer.add_scalar(
                        "metrics/total_secs_per_batch", batch_time.avg, global_step
                    )
                    writer.add_scalar(
                        "metrics/data_secs_per_batch", data_time.avg, global_step
                    )

                batch_time.reset()
                data_time.reset()
                losses.reset()
                ce_losses.reset()
                mask_bce_losses.reset()
                mask_dice_losses.reset()
                mask_losses.reset()

            if global_step != 0:
                curr_lr = scheduler.get_last_lr()
                if args.local_rank == 0:
                    writer.add_scalar("train/lr", curr_lr[0], global_step)

        return train_iter

    def validate(self, val_loader, model_engine, epoch, writer, args):
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
        model_engine.eval()

        for input_dict in tqdm.tqdm(val_loader):
            torch.cuda.empty_cache()

            input_dict = dict_to_cuda(input_dict)
            if args.precision == "fp16":
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images_clip"] = input_dict["images_clip"].float()

            # add infer mode
            input_dict["mode"] = "val"

            with torch.no_grad():
                # logger.info("in eval {}, keys {}", epoch, input_dict.keys())
                output_dict = model_engine(**input_dict)

            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(),
                    mask_i.contiguous(),
                    2,
                    ignore_index=255,
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection), union_meter.update(
                union
            ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]

        if args.local_rank == 0:
            writer.add_scalar("val/giou", giou, epoch)
            writer.add_scalar("val/giou", ciou, epoch)
            print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

        return giou, ciou


def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    args = parse_args()
    logger.info("args {}", args)
    config: Config = Config(Dict(cfg_path=args.conf))
    initialize_distributed(args)
    setup_seeds(config)

    train: Trainer = Trainer(job_id, config, args)
    train.train()


if __name__ == "__main__":
    main()
