DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

KEY_START_TOKEN = "<|KEY|>"
KEY_END_TOKEN = "<|/KEY|>"

VALUE_START_TOKEN = "<|VALUE|>"
VALUE_END_TOKEN = "<|/VALUE|>"

TRAIN_EXAMPLE_START_TOKEN = "<|EXAMPLE|>"
TRAIN_EXAMPLE_END_TOKEN = "<|/EXAMPLE|>"

CONTEXT_START_TOKEN = "<|CONTEXT|>"
CONTEXT_END_TOKEN = "<|/CONTEXT|>"

META_START_TOKEN = "<|META|>"
META_END_TOKEN = "<|/META|>"

PREFIX_START_TOKEN = "<|PREFIX|>"
PREFIX_END_TOKEN = "<|/PREFIX|>"

QUANTILE_START_TOKEN = "<|QUANTILE|>"
QUANTILE_END_TOKEN = "<|/QUANTILE|>"

SCALE_START_TOKEN = "<|SCALE|>"
SCALE_END_TOKEN = "<|/SCALE|>"

SUFFIX_START_TOKEN = "<|SUFFIX|>"
SUFFIX_END_TOKEN = "<|/SUFFIX|>"

CHOICES_START_TOKEN = "<|CHOICES|>"
CHOICES_END_TOKEN = "<|/CHOICES|>"
IGNORE_INDEX = -100

# Follow the OpenAI fine-tuning guide (and also LiFT)
# https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
EOC_TOKEN = "<|endcompletion|>"  # Indicates end of completion.

FEATURE_START_TOKEN = "<|feature|>"
FEATURE_END_TOKEN = "<|/feature|>"

SHARED_START_TOKEN = "<|sharedfeatures|>"
SHARED_END_TOKEN = "<|/sharedfeatures|>"

DIFFERENT_START_TOKEN = "<|differentfeatures|>"
DIFFERENT_END_TOKEN = "<|/differentfeatures|>"

Q_START_TOKEN = "<|question|>"
Q_END_TOKEN = "<|/question|>"

A_START_TOKEN = "<|answer|>"
A_END_TOKEN = "<|/answer|>"

ANS_CHOICES_SEP_TOKEN = "||"

MASK_TOKEN = "<|mask|>"