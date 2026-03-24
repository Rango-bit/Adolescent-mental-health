import os
import torch
import random
import dataclasses

from typing import Optional, Dict, Any
import torch.distributed as dist
import torch.optim as optim

from transformers import (
    HfArgumentParser,
    get_cosine_schedule_with_warmup,
    AutoTokenizer
)

from my_datasets.get_tokenized_dataset import DataCollatorForSupervisedDataset
from my_datasets.arguments import DataArguments
from my_datasets.dataset import DriveDataset_pretrain, DriveDataset
import configs as cfg

from llama_recipes.configs import wandb_config as WandbConfig
from llama_recipes.utils.config_utils import update_config

from models.llama_model import load_llama_2, load_llama_3_lora
from torch.nn.parallel import DistributedDataParallel as DDP

from peft import PeftModel

from tools.utils import get_latest_checkpoint
from tools.train_utils import (
    OPTIMIZER_STATE_PT,
    SCHEDULER_STATE_PT,
    load_peft_model_from_checkpoint,
    get_steps,
    run_steps_pretrain,
)

from tools.eval_utils import (
    save_eval_output_pretrain,
    save_eval_output
)


def get_config_classes(task_type: str):
    train_registry = {
        "Pretrain": cfg.TrainConfig_Pretrain,
        "Screen": cfg.TrainConfig_Screen,
        "Assessment": cfg.TrainConfig_Assessment,
    }

    lora_registry = {
        "Pretrain": cfg.LoraConfig_Pretrain,
        "Screen": cfg.LoraConfig_Screen,
        "Assessment": cfg.LoraConfig_Assessment,
    }

    if task_type not in train_registry:
        raise ValueError(f"Unknown task_type: {task_type}")

    return train_registry[task_type], lora_registry[task_type]

def build_parser(train_cls, lora_cls):
    return HfArgumentParser(
        (
            train_cls,
            lora_cls,
            DataArguments,
            cfg.ModelArguments,
            cfg.ExperimentArguments,
            cfg.SerializerConfig,
            cfg.TokenizerConfig,
        )
    )

def parse_args():
    task_parser = HfArgumentParser(cfg.TaskArguments)
    task_args, remaining_args = task_parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    TrainConfigClass, LoraConfigClass = get_config_classes(task_args.task_type)

    parser = build_parser(TrainConfigClass, LoraConfigClass)

    (
        train_config,
        lora_config,
        data_args,
        model_args,
        exp_args,
        serializer_config,
        tokenizer_config,
    ) = parser.parse_args_into_dataclasses(remaining_args)

    return (
        task_args,
        train_config,
        lora_config,
        data_args,
        model_args,
        exp_args,
        serializer_config,
        tokenizer_config,
    )

def setup_wandb(config_dict: Dict[str, Any], **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )

    wandb_config = WandbConfig()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict, config=config_dict)
    return run

def setup_distributed():
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size

def train_or_eval(
        train_config,
        lora_config,
        data_args,
        serializer_config,
        tokenizer_config,
        model_args,
        exp_args,
        task_type,
    ):
    local_rank, rank, world_size = setup_distributed()
    if local_rank == 0:
        if exp_args.test_task:
            print(f"--------{task_type}: run  testing--------")
        else:
            print(f"--------{task_type}: run training--------")

    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    wandb_run = None
    if train_config.use_wandb:
        config_dict = {
            **data_args.__dict__,
            **dataclasses.asdict(train_config),
            **dataclasses.asdict(tokenizer_config),
        }
        if rank == 0:
            wandb_run = setup_wandb(config_dict=config_dict)

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name_or_path, use_fast=False)

    model, tokenizer = load_llama_3_lora(tokenizer,
                                    serializer_config,
                                    tokenizer_config,
                                    train_config.model_name_or_path,
                                    task_type=task_type,
                                    train_config=train_config,
                                    lora_config=lora_config,
                                    pretrain_state_dict_path = model_args.pretrain_state_dict_path,
                                    screen_state_dict_path = model_args.screen_state_dict_path,
                                    test_task=exp_args.test_task,
                                    local_rank=local_rank
                                    )

    # 测试预训练后的模型
    if exp_args.test_task:
        if task_type == "Pretrain":
            model = PeftModel.from_pretrained(model, model_args.pretrain_state_dict_path)
        elif task_type == "Screen":
            model = PeftModel.from_pretrained(model, model_args.screen_state_dict_path)
        else:
            model = PeftModel.from_pretrained(model, model_args.assessment_state_dict_path)

        # 确保 LoRA 参数为可训练状态，否则后续DDP会报错
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    if train_config.resume:
        ckpt_dir = get_latest_checkpoint(train_config.resume)
        model = load_peft_model_from_checkpoint(model, ckpt_dir)

    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
                )

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer,
        use_position_ids=data_args.use_position_ids,
        task_type=task_type,
        test_task=exp_args.test_task
    )

    if not exp_args.test_task:
        '''
        由于预训练阶段与其余两个阶段的在数据处理和模型推理方面差异性很大，
        因此将DriveDataset和后续的run_steps分别分成了两部分代码
        '''
        if task_type == "Pretrain":
            train_dataset = DriveDataset_pretrain(
                train_config.data_path,
                tokenizer,
                local_rank,
                train=True,
                mask_ratio=1.0
            )
        else:
            train_dataset = DriveDataset(
                train_config.data_path,
                train_config.train_data_file,
                train_config.test_data_file,
                tokenizer,
                local_rank,
                exp_args.use_comparative_reason,
                train=True,
                template_num=exp_args.template_num
            )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank = rank,
            shuffle = True
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_config.batch_size_training,
            collate_fn=data_collator,
            num_workers=train_config.num_workers_dataloader,
            sampler=train_sampler
        )

        model._set_static_graph()

        '''
        基于epoch设置进度条
        '''
        steps_per_rank = len(train_dataloader)
        total_steps = steps_per_rank * train_config.epochs
        total_updates = total_steps // train_config.gradient_accumulation_steps

        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_config.warmup_steps,
            num_training_steps=total_updates
        )

        # 断点续训
        if train_config.resume:
            print("#" * 50)
            print(f"Resuming scheduler, optimizer, and model from {ckpt_dir}")
            optimizer_state = torch.load(
                os.path.join(ckpt_dir, OPTIMIZER_STATE_PT), map_location="cpu"
            )

            optimizer.load_state_dict(optimizer_state)
            scheduler_state = torch.load(
                os.path.join(ckpt_dir, SCHEDULER_STATE_PT), map_location="cpu"
            )
            scheduler.load_state_dict(scheduler_state)

            global_step = get_steps(ckpt_dir)
            global_step += 1
        else:
            global_step = 0

        # Start the training process
        run_steps_pretrain(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            train_config,
            train_sampler=train_sampler,
            local_rank=local_rank,
            rank=rank if torch.distributed.is_initialized() else None,
            wandb_run=wandb_run,
            epochs=train_config.epochs,
            total_updates=total_updates,
            step=global_step, # 断点续训step
            ce_with_ntl_loss=train_config.ce_with_ntl_loss
        )

    else:
        # eval
        if task_type == "Pretrain":
            test_dataset = DriveDataset_pretrain(
                data_args.data_path,
                tokenizer,
                local_rank,
                train=False,
                mask_ratio=1.0
            )
        else:
            test_dataset = DriveDataset(
                train_config.data_path,
                train_config.train_data_file,
                train_config.test_data_file,
                tokenizer,
                local_rank,
                task_type,
                exp_args.use_comparative_reason,
                train=False,
                template_num=exp_args.template_num
            )

        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank = rank,
            shuffle = False
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=train_config.batch_size_testing,
            collate_fn=data_collator,
            num_workers=train_config.num_workers_dataloader,
            sampler=test_sampler
        )
        step_value_in_state = model_args.pretrain_state_dict_path.split('-')[-1]
        if task_type == "Pretrain":
            save_eval_output_pretrain(
                model,
                tokenizer,
                test_dataloader,
                local_rank,
                world_size,
                step_value_in_state,
                exp_args.eval_results_folder,
                train_config.batch_size_split,
                exp_args.use_context_info
            )
        else:
            save_eval_output(
                model,
                tokenizer,
                test_dataloader,
                local_rank,
                world_size,
                step_value_in_state,
                exp_args.eval_results_folder,
                task_type,
                exp_args.use_comparative_reason,
                exp_args.use_context_info
            )


def main():
    (
        task_args,
        train_config,
        lora_config,
        data_args,
        model_args,
        exp_args,
        serializer_config,
        tokenizer_config
    ) = parse_args()

    train_or_eval(
        train_config,
        lora_config,
        data_args,
        serializer_config,
        tokenizer_config,
        model_args=model_args,
        exp_args=exp_args,
        task_type=task_args.task_type,
    )


def cleanup():
    if dist.is_initialized():
        dist.barrier()  # 确保所有进程同步
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

    # 关闭所有进程
    cleanup()