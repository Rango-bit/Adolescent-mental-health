import glob
import os
import re
import shutil
import time
import json
from contextlib import nullcontext
from typing import Dict
import numpy as np
import torch

from llama_recipes.model_checkpointing.checkpoint_handler import (
    fullstate_save_policy,
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType,
)

from configs import TrainConfig
from tqdm import tqdm
from tools.utils import timestamp
from tools.ntl_loss import number_token_loss

from peft import PeftModel


SCALER_STATE_PT = "scaler_state.pt"
MODEL_STATE_PT = "model.pt"
SCHEDULER_STATE_PT = "scheduler_state.pt"
OPTIMIZER_STATE_PT = "optimizer_state.pt"

def get_steps(ckpt_dir) -> int:
    step = re.search("step-(\d+)", ckpt_dir).group(1)
    print(f"loaded step is {step}")

    return int(step)

def load_peft_model_from_checkpoint(model, ckpt_dir) -> torch.nn.Module:
    print(f"loading peft model weights from {ckpt_dir}")

    model = PeftModel.from_pretrained(model, ckpt_dir)
    # 确保 LoRA 参数为可训练状态
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    return model

def save_state_dict_to_default_directory(
    state_dict: Dict[str, torch.Tensor], cfg: TrainConfig, step: int, filename: str
):
    """Save an arbitrary state dict using a consistent path and file naming schema."""
    save_dir = cfg.make_save_folder_name(step)
    os.makedirs(save_dir, exist_ok=True)

    save_full_path = os.path.join(save_dir, filename)

    # Handle case where cfg.model_name contains a slash (i.e. uses a hf repo)
    os.makedirs(os.path.dirname(save_full_path), exist_ok=True)

    print(f"--> saving state to {save_full_path}")

    torch.save(state_dict, save_full_path)

    print(f"--> finished saving to {save_full_path}")
    return

def save_optimizer_and_scheduler_unsharded(
    model, optimizer, train_config: TrainConfig, rank: int, step: int, lr_scheduler
):
    if train_config.save_optimizer:
        optim_state = optimizer.state_dict()

        # optimizer and scheduler saving
        if rank == 0:
            assert (
                optim_state is not None
            ), f"expected optimizer state; could be unhandled case."
            print(f"--> saving optimizer state...")
            save_state_dict_to_default_directory(
                optim_state, train_config, step, OPTIMIZER_STATE_PT
            )

            print(f"--> saving scheduler state...")
            scheduler_state = lr_scheduler.state_dict()
            save_state_dict_to_default_directory(
                scheduler_state, train_config, step, SCHEDULER_STATE_PT
            )
    return

def save_model_and_optimizer_unsharded(
    model,
    optimizer,
    lr_scheduler,
    rank,
    cfg: TrainConfig,
    step: int,
):
    """Saving model via rank0 cpu streaming and full_state_dict, if FSDP is used."""

    # create save path
    save_dir = cfg.make_save_folder_name(step)
    os.makedirs(save_dir, exist_ok=True)

    # FSDP model saving
    if cfg.enable_fsdp:
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            cpu_state = model.state_dict()
            print(f"saving process: rank {rank}  done w model state_dict\n")

    if cfg.enable_fsdp and rank == 0:
        print(f"--> saving FSDP model on rank 0...")
        save_state_dict_to_default_directory(cpu_state, cfg, step, MODEL_STATE_PT)

    # non-FSDP model saving
    elif not cfg.enable_fsdp:
        print("non-FSDP training run; saving checkpoint in HF format...")
        model.save_pretrained(save_dir)
        print(f"HF model checkpoint saved for step {step} at {save_dir}\n")

    if cfg.save_optimizer:
        save_optimizer_and_scheduler_unsharded(
            model, optimizer, cfg, rank, step, lr_scheduler
        )

def save_json(file_path, loss_dict):
    # 保存到本地 JSON 文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(loss_dict, f, ensure_ascii=False, indent=2)

def save_train_state(
    train_config: TrainConfig,
    model,
    optimizer,
    lr_scheduler,
    rank,
    step: int,
    step_loss_record: dict
):
    """Save the model, optimizer, scheduler, and other info to restore the training state.

    Saving is conducted as specified in the TrainConfig (e.g. full vs. sharded state dict,
    optional saving of optimizer state).
    """
    checkpoint_start_time = time.perf_counter()
    is_main_process = rank==0

    if train_config.use_peft:
        # create save path
        save_dir = train_config.make_save_folder_name(step)
        os.makedirs(save_dir, exist_ok=True)
        # model.save_pretrained(save_dir)  # 这个地方的代码不能直接保存lora训练的模型

        # 使用DDP需要使用下方的代码保持lora参数
        # 假设 model 是 DDP 包裹的模型
        model_to_save = model.module if hasattr(model, "module") else model

        # 保存 LoRA adapter 权重（推荐，只保存 LoRA 层）
        print('-----model save path: ', save_dir)
        model_to_save.save_pretrained(save_dir)

        if is_main_process:
            print(f"PEFT modules are saved in {save_dir} directory")
        save_optimizer_and_scheduler_unsharded(
            model, optimizer, train_config, rank, step, lr_scheduler
        )
        loss_file_name = os.path.join(save_dir, "step_loss_record.json")
        save_json(loss_file_name, step_loss_record)
    else:
        save_model_and_optimizer_unsharded(
            model, optimizer, lr_scheduler, rank, train_config, step=step
        )

    # Remove checkpoints if too many have accumulated.
    if train_config.save_total_limit and is_main_process:
        save_dir = train_config.make_save_folder_name()
        ckpt_dirs = [x for x in glob.glob(os.path.join(save_dir, "*step*"))]

        # Sort oldest-first
        ckpt_dirs = sorted(ckpt_dirs, key=os.path.getmtime)

        if len(ckpt_dirs) > train_config.save_total_limit:
            num_to_remove = len(ckpt_dirs) - train_config.save_total_limit
            for dir_to_remove in ckpt_dirs[:num_to_remove]:
                shutil.rmtree(dir_to_remove)

    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
    return checkpoint_end_time


def run_steps_pretrain(
    model,
    train_dataloader,
    optimizer,
    lr_scheduler,
    train_config: TrainConfig,
    train_sampler,
    local_rank=None,
    rank=None,
    wandb_run=None,
    epochs=None,
    total_updates: int = 0,
    step: int = 0,
    ce_with_ntl_loss: bool=False
):
    """
    Trains the model on the given dataloader

    Adapted from llama-recipes/utils/train_utils.py.

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    gradient_accumulation_steps = train_config.gradient_accumulation_steps
    # Create a gradient scaler for fp16
    if train_config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    model.train()

    pbar = tqdm(
        colour="blue",
        desc=f"Train",
        initial=step,
        total=total_updates,
        dynamic_ncols=True,
    )
    step_loss_record = {}

    reach_max_steps = False
    model.train()

    step_per_epoch = len(train_dataloader)
    # 设置最大epoch
    for i in range(epochs):
        train_sampler.set_epoch(i) # 保证每个 epoch shuffle 不一样

        for idx, batch in enumerate(train_dataloader):
            start_ts = timestamp()

            batch = {"input_ids": batch["input_ids"], "labels": batch["labels"]}

            for key in batch.keys():
                if not torch.cuda.is_available():
                    pass
                else:
                    batch[key] = batch[key].to(f"cuda:{local_rank}")

            # 之前的损失
            with autocast():
                if ce_with_ntl_loss:
                    # 优化后的损失
                    outputs = model(**batch)
                    logits = outputs.logits
                    labels = batch['labels']
                    ce_loss = outputs.loss
                    ntl_loss = number_token_loss(logits, labels)
                    loss = ce_loss + train_config.ntl_weight * ntl_loss
                else:
                    loss = model(**batch).loss

            loss = loss / gradient_accumulation_steps

            # 每隔save_loss_steps个step保存一次loss值，第一个step也要记录
            if step % train_config.save_loss_steps == 0:
                if ce_with_ntl_loss:
                    step_loss_record[str(step) + '_CE'] = ce_loss.detach().cpu().item()
                    step_loss_record[str(step) + '_NTL'] = ntl_loss.detach().cpu().item()
                else:
                    step_loss_record[str(step)] = loss.detach().cpu().item()

            if train_config.use_fp16:
                # if fp16 is enabled, use gradient scaler to handle gradient update
                scaler.scale(loss).backward()
                if (
                    step + 1
                ) % gradient_accumulation_steps == 0 or step == total_updates - 1:
                    if (
                        train_config.gradient_clipping
                        and train_config.gradient_clipping_threshold > 0.0
                    ):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            train_config.gradient_clipping_threshold,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    pbar.update(1)
            else:
                # regular backpropagation when fp16 is not used
                loss.backward()
                if (
                    step + 1
                ) % gradient_accumulation_steps == 0 or step == total_updates - 1:
                    # 梯度裁剪
                    if (
                        train_config.gradient_clipping
                        and train_config.gradient_clipping_threshold > 0.0
                    ):
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            train_config.gradient_clipping_threshold,
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True) # 节省显存，避免梯度残留
                    lr_scheduler.step()
                    pbar.update(1)

            if wandb_run:
                if rank == 0:
                    batch_tokens_count = np.prod(list(batch["input_ids"].shape))
                    step_time = timestamp() - start_ts
                    wandb_run.log(
                        {
                            "train/step": step,
                            "train/loss": loss.detach().float(),
                            "train/perplexity": float(torch.exp(loss.detach().float())),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/tokens_per_batch": batch_tokens_count,
                            "train/step_time_secs": step_time,
                            "train/tokens_per_gpu_per_sec": batch_tokens_count / step_time,
                        },
                        step=step,
                    )

            if train_config.save_steps and (step + 1) % train_config.save_steps == 0:
                checkpoint_end_time = save_train_state(
                    train_config,
                    model,
                    optimizer,
                    lr_scheduler,
                    rank,
                    step,
                    step_loss_record
                )
                if wandb_run:
                    wandb_run.log({"train/checkpoint_time": checkpoint_end_time}, step=step)

            pbar.set_description(f"Step {step} loss: {loss.detach().float()}")
            step += 1

    pbar.close()

    if train_config.save_model:
        # save the state in .pt format
        checkpoint_end_time = save_train_state(
            train_config, model, optimizer, lr_scheduler, rank, step, step_loss_record
        )

    return