import os
from dataclasses import dataclass, field
from typing import Optional, Literal, List

@dataclass
class TaskArguments:
    task_type: str = field(
        default="Pretrain",
        metadata={"help": "Task type: Pretrain / Screen / Assessment"}
    )

@dataclass
class ModelArguments:
    pretrain_state_dict_path = "/state_dict/Pretrain/"
    screen_state_dict_path = "/state_dict/Screen/"
    assessment_state_dict_path = "/state_dict/Assessment/"


@dataclass
class ExperimentArguments:
    # 推理策略
    use_comparative_reason: bool = False
    template_num: int = 2

    # train / test
    test_task: bool = False

    # context 控制
    use_context_info: bool = False

    # 输出
    eval_results_folder: str = "test-"

    # 消融
    drop_class: Optional[str] = None


@dataclass
class BaseTrainConfig:
    seed: int = 42
    context_length: int = 8192
    shuffle_buffer_size: int = 10000
    use_peft: bool = True

    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 1

    batch_size_training: int = 1
    gradient_accumulation_steps: int = 1

    num_workers_dataloader: int = 2
    warmup_steps: int = 200
    warmup_ratio: float = 0.0

    ce_with_ntl_loss: bool = False
    ntl_weight: float = 0.1

    use_fp16: bool = False
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 0.1

    freeze_layers: bool = False
    num_freeze_layers: int = 0

    use_wandb: bool = False

    resume: Optional[str] = None

    save_optimizer: bool = True
    save_total_limit: int = 0
    save_model: bool = True
    save_checkpoint_root_dir: str = "checkpoints"
    run_name: str = "lora"

    @property
    def output_dir(self) -> str:
        return os.path.join(self.save_checkpoint_root_dir, self.run_name)

    def make_save_folder_name(self, model_name: str, step: Optional[int] = None) -> str:
        base_dir = (
            self.save_checkpoint_root_dir
            + "/"
            + self.run_name
            + "-"
            + model_name.split("/")[-1]
        )
        return base_dir if not step else base_dir + "/" + f"step-{step}"

@dataclass
class TrainConfig_Pretrain(BaseTrainConfig):
    data_path: str = "/data_samples/Pretrain/"
    train_data_file: str = "/data_samples/Pretrain/train.txt"
    test_data_file: str = "/data_samples/Pretrain/test.txt"

    lr: float = 1e-4
    epochs: int = 2

    batch_size_training: int = 2
    batch_size_testing: int = 1
    batch_size_split: int = 18

    gradient_accumulation_steps: int = 8

    save_steps: int = 800
    save_loss_steps: int = 10

    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B"

@dataclass
class TrainConfig_Screen(BaseTrainConfig):
    data_path: str = "/data_samples/Screen/"
    train_data_file: str = "/data_samples/Screen/train.txt"
    test_data_file: str = "/data_samples/Screen/test.txt"

    lr: float = 3e-5

    batch_size_testing: int = 8

    gradient_accumulation_steps: int = 8

    save_steps: int = 200

    model_name: str = "meta-llama/Meta-Llama-3-8B"


@dataclass
class TrainConfig_Assessment(BaseTrainConfig):
    data_path: str = "/data_samples/Assessment/"
    train_data_file: str = "/data_samples/Assessment/train.txt"
    test_data_file: str = "/data_samples/Assessment/test.txt"

    lr: float = 3e-5
    epochs: int = 10

    batch_size_training: int = 1
    batch_size_testing: int = 2

    gradient_accumulation_steps: int = 4

    save_steps: int = 200

    model_name: str = "meta-llama/Meta-Llama-3-8B"


@dataclass
class LoraConfig_Pretrain:
    r = 32
    lora_alpha = 64
    target_modules = ["q_proj", "v_proj"]
    lora_dropout = 0.05
    bias = "none"
    task_type = "CAUSAL_LM"


@dataclass
class LoraConfig_Screen:
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.1
    inference_mode: bool = False


@dataclass
class LoraConfig_Assessment:
    r: int = 4
    lora_alpha: int = 8
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.1
    inference_mode: bool = False


@dataclass
class TokenizerConfig:
    """Configuration class for tokenization and serialization."""

    # "whether to add special tokens for the serializer to the vocabulary. "
    # "If False, the tokens (i.e. <VALUE_START> for StructuredSerializer) are added"
    # " to the example text, but are not explicitly added as special tokens "
    # "to the tokenizer vocabulary."
    add_serializer_tokens: bool = True
    # "Embedding initialization method to use for serializer special tokens."
    # "Only used if add_serializer_tokens=True (ignored otherwise)"
    serializer_tokens_embed_fn: Literal["smart", "vipi", "hf"] = "smart"
    use_fast_tokenizer: bool = True


@dataclass
class SerializerConfig:
    """Configuration class for serializer."""

    serializer_cls: str = "BasicSerializerV2"
    shuffle_instance_features: bool = False
    #     default=False,
    #     metadata={
    #         "help": "If true, randomly shuffle the order of features for each instance."
    #     },
    # )
    feature_dropout: float = 0.0
    #     default=0.0,
    #     metadata={
    #         "help": "Proportion of features in each example to randomly drop out during training."
    #     },
    # )
    use_prefix: bool = True
    #     default=True,
    #     metadata={
    #         "help": "whether to use a prefix for examples. The prefix lists "
    #                 "valid choices, and describes the prediction task."
    #     },
    # )
    use_suffix: bool = True
    #     default=True,
    #     metadata={
    #         "help": "Whether to use a suffix for examples. "
    #                 "The suffix phrases the prediction tasks as a question, "
    #                 "and lists valid choices."
    #     },
    # )
    use_choices: bool = True
    #     default=True,
    #     metadata={"help": "Whether to list the class choices in the prompt."},
    # )
    choices_position: Literal["front", "back", "both"] = "both"
    max_precision: Optional[int] = None