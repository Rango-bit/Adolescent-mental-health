import torch
from typing import Optional
from transformers import LlamaForCausalLM
from dataclasses import asdict

from my_datasets.serialization.serializers import get_serializer
from my_datasets.tokenization.text import prepare_tokenizer, sanity_check_tokenizer
from configs import SerializerConfig, TokenizerConfig
from tools.utils import freeze_transformer_layers

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel


def load_llama_2(tokenizer,
                    serializer_config: SerializerConfig,
                    tokenizer_config: TokenizerConfig,
                    model_name_or_path: str,
                    train_config
                 ):

    # Load base model
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        device_map = {"": torch.cuda.current_device()},
        attn_implementation="sdpa"
    )

    train_serializer = get_serializer(serializer_config)
    tokenizer, model = prepare_tokenizer(
        model,
        tokenizer=tokenizer,
        pretrained_model_name_or_path=model_name_or_path,
        model_max_length=train_config.context_length,
        use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
        serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
        serializer_tokens=train_serializer.special_tokens
        if tokenizer_config.add_serializer_tokens
        else None,
    )
    # sanity check for tokenizer
    sanity_check_tokenizer(tokenizer, model_name_or_path)

    # setting up FSDP if enable_fsdp is enabled
    if train_config.freeze_layers:
        print('-------------Freeze layers-------------')
        freeze_transformer_layers(model, train_config.num_freeze_layers)

    return model, tokenizer


def load_llama_3_lora(tokenizer,
                serializer_config: SerializerConfig,
                tokenizer_config: TokenizerConfig,
                model_name_or_path: str,
                task_type,
                train_config,
                lora_config,
                pretrain_state_dict_path: Optional[str] = None,
                screen_state_dict_path: Optional[str] = None,
                test_task: bool=False,
                local_rank: Optional[int]=None):

    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=True,
        attn_implementation="sdpa",
        device_map={"": torch.cuda.current_device()}
    )

    train_serializer = get_serializer(serializer_config)
    tokenizer, model = prepare_tokenizer(
        model,
        tokenizer=tokenizer,
        pretrained_model_name_or_path=model_name_or_path,
        model_max_length=train_config.context_length,
        use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
        serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
        serializer_tokens=train_serializer.special_tokens
        if tokenizer_config.add_serializer_tokens
        else None,
    )

    # sanity check for tokenizer
    sanity_check_tokenizer(tokenizer, model_name_or_path)

    if task_type != "Pretrain":
        if local_rank == 0:
            print("-----Load pretrained weights-----")
        model = PeftModel.from_pretrained(model, pretrain_state_dict_path)

        if task_type == "Assessment":
            if local_rank == 0:
                print("-----Load LoRA weights from the Screen stage-----")
            model = PeftModel.from_pretrained(model, screen_state_dict_path)

        for param in model.parameters():
            param.requires_grad = False

    if not test_task:
        if local_rank == 0:
            print("-----Initialize a new LoRA module-----")
        model.gradient_checkpointing_enable()

        # Prepare for k-bit LoRA training
        model = prepare_model_for_kbit_training(model)

        # Load LoRA config
        lora_config = LoraConfig(**asdict(lora_config))

        # Wrap model with LoRA
        model = get_peft_model(model, lora_config)

    return model, tokenizer