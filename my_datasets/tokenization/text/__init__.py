import copy
import os
import logging
from typing import Optional, Dict, Tuple, Sequence, Callable, Union

import numpy as np
import torch
import transformers
from my_datasets import special_tokens as tok
from my_datasets.special_tokens import DEFAULT_PAD_TOKEN, IGNORE_INDEX


def fetch_auth_token() -> Union[str, None]:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]

def fetch_tokenizer(
    pretrained_model_name_or_path: str,
    model_max_length: int,
    use_fast_tokenizer: bool,
    use_auth_token=None,
):
    tokenizer_kwargs = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "cache_dir": None,
        "model_max_length": model_max_length,
        "padding_side": "right",
        "use_auth_token": use_auth_token,
        "use_fast": use_fast_tokenizer,
    }

    return transformers.AutoTokenizer.from_pretrained(**tokenizer_kwargs)


def prepare_tokenizer(
    model,
    pretrained_model_name_or_path: str,
    model_max_length: int,
    use_fast_tokenizer: bool,
    serializer_tokens_embed_fn: Optional[str] = None,
    serializer_tokens: Optional[Dict[str, str]] = None,
    tokenizer=None,
) -> Tuple[transformers.PreTrainedTokenizer, transformers.AutoModelForCausalLM]:
    logging.info(f"setting up tokenizer %s" % pretrained_model_name_or_path)

    if pretrained_model_name_or_path == "yujiepan/llama-2-tiny-random":
        tokenizer = fetch_tokenizer(
            pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf",
            model_max_length=model_max_length,
            use_fast_tokenizer=use_fast_tokenizer,
            use_auth_token=fetch_auth_token(),
        )

    else:
        assert tokenizer is not None

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = model_max_length

    special_tokens_dict = {}

    if tokenizer.pad_token is None:
        logging.info("no pad token detected; adding pad token")
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    assert tokenizer.eos_token is not None
    assert tokenizer.bos_token is not None

    assert (
        serializer_tokens_embed_fn is not None
    ), f"Must provide serializer_tokens_embed_fn if is_train=True."
    tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
        other_tokens_dict=serializer_tokens,
        other_tokens_are_special_tokens=True,
        embed_fn=serializer_tokens_embed_fn,
    )

    return tokenizer, model


def unmasked_token_idxs(tokens):
    """Helper function to fetch indices of unmasked tokens."""
    return np.flatnonzero(tokens != IGNORE_INDEX)


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

'''
Assign IDs to custom special tokens:
EOC_TOKEN = "<|endcompletion|>"

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
'''
def sanity_check_tokenizer(tokenizer, model_name):
    logging.warning("sanity checking the tokenizer special tokens are in vocab...")
    if (
        "llama" in model_name.lower()
        and "2" in model_name
        and len(tokenizer) < 128254
        # and len(tokenizer.vocab) < 128_254
    ):
        # Case: this is llama 2 model.
        eoc_token_id_expected = 32000
        feature_start_token_id_expected = 32001
        feature_end_token_id_expected = 32002
        shared_start_token_id_expected = 32003
        shared_end_token_id_expected = 32004
        different_start_token_id_expected = 32005
        different_end_token_id_expected = 32006
        q_start_token_id_expected = 32007
        q_end_token_id_expected = 32008
        a_start_token_id_expected = 32009
        a_end_token_id_expected = 32010
        mask_token_id_expected = 32011
        choices_sep_token_expected = 8876  # this token is already in llama2 vocab

    elif (
        "llama" in model_name.lower() and "3" in model_name and len(tokenizer) > 128254
    ) or ("tabula-8b" in model_name.lower()):
        # Case: this is llama 3 model.
        eoc_token_id_expected = 128256
        feature_start_token_id_expected = 128257
        feature_end_token_id_expected = 128258
        shared_start_token_id_expected = 128259
        shared_end_token_id_expected = 128260
        different_start_token_id_expected = 128261
        different_end_token_id_expected = 128262
        q_start_token_id_expected = 128263
        q_end_token_id_expected = 128264
        a_start_token_id_expected = 128265
        a_end_token_id_expected = 128266
        mask_token_id_expected = 128267
        choices_sep_token_expected = 8651  # this token is already in llama3 vocab

    elif "llama" in model_name.lower() and len(tokenizer.vocab) < 128_254:
        # Case: this is llama 1 model.
        eoc_token_id_expected = 32000
        feature_start_token_id_expected = 32001
        feature_end_token_id_expected = 32002
        shared_start_token_id_expected = 32003
        shared_end_token_id_expected = 32004
        different_start_token_id_expected = 32005
        different_end_token_id_expected = 32006
        q_start_token_id_expected = 32007
        q_end_token_id_expected = 32008
        a_start_token_id_expected = 32009
        a_end_token_id_expected = 320010
        mask_token_id_expected = 128267
        choices_sep_token_expected = 8876
    else:
        raise ValueError(f"unknown model name: {model_name}")

    assert tokenizer(tok.EOC_TOKEN, add_special_tokens=False)["input_ids"] == [
        eoc_token_id_expected
    ], f"EOC token tokenizes to {tokenizer(tok.EOC_TOKEN, add_special_tokens=False)['input_ids']}"

    assert tokenizer(tok.FEATURE_START_TOKEN, add_special_tokens=False)["input_ids"] == [
        feature_start_token_id_expected
    ], f"SHARED_START token tokenizes to {tokenizer(tok.FEATURE_START_TOKEN, add_special_tokens=False)['input_ids']}"

    assert tokenizer(tok.FEATURE_END_TOKEN, add_special_tokens=False)["input_ids"] == [
        feature_end_token_id_expected
    ], f"SHARED_START token tokenizes to {tokenizer(tok.FEATURE_END_TOKEN, add_special_tokens=False)['input_ids']}"

    assert tokenizer(tok.SHARED_START_TOKEN, add_special_tokens=False)["input_ids"] == [
        shared_start_token_id_expected
    ], f"SHARED_START token tokenizes to {tokenizer(tok.SHARED_START_TOKEN, add_special_tokens=False)['input_ids']}"

    assert tokenizer(tok.SHARED_END_TOKEN, add_special_tokens=False)["input_ids"] == [
        shared_end_token_id_expected
    ], f"SHARED_END token tokenizes to {tokenizer(tok.SHARED_END_TOKEN, add_special_tokens=False)['input_ids']}"

    assert tokenizer(tok.DIFFERENT_START_TOKEN, add_special_tokens=False)["input_ids"] == [
        different_start_token_id_expected
    ], f"DIFFERENT_START token tokenizes to {tokenizer(tok.DIFFERENT_START_TOKEN, add_special_tokens=False)['input_ids']}"

    assert tokenizer(tok.DIFFERENT_END_TOKEN, add_special_tokens=False)["input_ids"] == [
        different_end_token_id_expected
    ], f"DIFFERENT_END token tokenizes to {tokenizer(tok.DIFFERENT_END_TOKEN, add_special_tokens=False)['input_ids']}"

    assert tokenizer(tok.Q_START_TOKEN, add_special_tokens=False)["input_ids"] == [
        q_start_token_id_expected
    ], f"Q_START token tokenizes to {tokenizer(tok.Q_START_TOKEN, add_special_tokens=False)['input_ids']}"

    assert tokenizer(tok.Q_END_TOKEN, add_special_tokens=False)["input_ids"] == [
        q_end_token_id_expected
    ], f"Q_END token tokenizes to {tokenizer(tok.Q_END_TOKEN, add_special_tokens=False)['input_ids']}"

    assert tokenizer(tok.A_START_TOKEN, add_special_tokens=False)["input_ids"] == [
        a_start_token_id_expected
    ], f"A_START token tokenizes to {tokenizer(tok.A_START_TOKEN, add_special_tokens=False)['input_ids']}"

    assert tokenizer(tok.A_END_TOKEN, add_special_tokens=False)["input_ids"] == [
        a_end_token_id_expected
    ], f"A_END token tokenizes to {tokenizer(tok.A_END_TOKEN, add_special_tokens=False)['input_ids']}"

    assert tokenizer(tok.ANS_CHOICES_SEP_TOKEN, add_special_tokens=False)["input_ids"] == [
        choices_sep_token_expected
    ], f"ANS_CHOICES_SEP_TOKEN token tokenizes to {tokenizer(tok.ANS_CHOICES_SEP_TOKEN, add_special_tokens=False)['input_ids']}"
    logging.warning("tokenizer sanity check passed!")

    assert tokenizer(tok.MASK_TOKEN, add_special_tokens=False)["input_ids"] == [
        mask_token_id_expected
    ], f"MASK token tokenizes to {tokenizer(tok.MASK_TOKEN, add_special_tokens=False)['input_ids']}"
    logging.warning("tokenizer sanity check passed!")

def tokenizer_and_embedding_resize(
    special_tokens_dict: Dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    other_tokens_dict: Optional[Dict[str, str]] = None,
    other_tokens_are_special_tokens: bool = True,
    embed_fn: str = "smart",
):
    """Wrapper function to perform tokenizer and embedding resizing."""
    _embedding_resize_fns: Dict[str, Callable] = {
        "vipi": vipi_tokenizer_and_embedding_resize,
        "smart": smart_tokenizer_and_embedding_resize,
        "hf": hf_init_tokenizer_and_embedding_resize,
    }
    _embedding_resize_fn = _embedding_resize_fns[embed_fn]
    return _embedding_resize_fn(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
        other_tokens_dict=other_tokens_dict,
        other_tokens_are_special_tokens=other_tokens_are_special_tokens,
    )


def hf_init_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    other_tokens_dict: Optional[Dict[str, str]] = None,
    other_tokens_are_special_tokens: bool = True,
):
    """Use the HF default method.

    Note that this is the method used by e.g. LLaVA; see https://github.com/haotian-liu/LLaVA/blob/7775b12d6b20cd69089be7a18ea02615a59621cd/llava/model/builder.py#L134
    """
    # use smart_tokenizer_and_embedding_resize for the special tokens
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    tokenizer.add_tokens(
        list(other_tokens_dict.values()), special_tokens=other_tokens_are_special_tokens
    )
    model.resize_token_embeddings(len(tokenizer))


def vipi_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    other_tokens_dict: Optional[Dict[str, str]] = None,
    other_tokens_are_special_tokens: bool = True,
):
    """A form of the VIPI tokenizer, applied only to 'other tokens dict'."""
    # use smart_tokenizer_and_embedding_resize for the special tokens
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)

    # for the 'other' tokens, use VIPI initialization
    new_tokens = [
        x for x in other_tokens_dict.values() if x not in tokenizer.get_vocab()
    ]
    new_tokens_prev_ids = [
        tokenizer(x, add_special_tokens=False).input_ids for x in new_tokens
    ]
    logging.warning(
        f"adding tokens {other_tokens_dict} to vocab (as special tokens={other_tokens_are_special_tokens}"
    )
    num_new_tokens = tokenizer.add_tokens(
        list(other_tokens_dict.values()), special_tokens=other_tokens_are_special_tokens
    )

    logging.info(f"adding {num_new_tokens} to vocab")
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        for token, prev_token_ids in zip(new_tokens, new_tokens_prev_ids):
            new_token_id = tokenizer.convert_tokens_to_ids(token)

            # Sanity check that the prev_token_ids exactly reconstruct the token
            assert tokenizer.decode(prev_token_ids) == token

            input_embeds_mean = torch.stack(
                [input_embeddings[i] for i in prev_token_ids]
            ).mean(dim=0)
            output_embeds_mean = torch.stack(
                [output_embeddings[i] for i in prev_token_ids]
            ).mean(dim=0)

            input_embeddings[new_token_id, :] = input_embeds_mean
            output_embeddings[new_token_id, :] = output_embeds_mean
    logging.debug(f"len(tokenizer) after resize is {len(tokenizer)}")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    other_tokens_dict: Optional[Dict[str, str]] = None,
    other_tokens_are_special_tokens: bool = True,
):
    """Resize tokenizer and embedding matrix, adding both special_tokens_dict and other_tokens_dict.

    :param special_tokens_dict: special tokens that can be added with tokenizer.add_special_tokens().
        Typically this only includes tokens like bos_token, eos_token, pad_token.
        See transformers.tokenization_utils method .add_special_tokens() for more info.
    :param other_tokens_dict: tokens that cannot be added with tokenizer.add_special_tokens().
        This is where most tokens should be added.
    :param tokenizer: the tokenizer to modify.
    :param model: the model to be used with the tokenizer; its embedding matrix will be resized accordinly.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    logging.debug(f"len(tokenizer) before resize is {len(tokenizer)}")
    logging.warning(f"adding special tokens {special_tokens_dict} to vocab")
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if other_tokens_dict:
        logging.warning(
            f"adding tokens {other_tokens_dict} to vocab (as special tokens={other_tokens_are_special_tokens}"
        )
        num_new_tokens += tokenizer.add_tokens(
            list(other_tokens_dict.values()),
            special_tokens=other_tokens_are_special_tokens,
        )
    logging.info(f"adding {num_new_tokens} to vocab")
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    logging.debug(f"len(tokenizer) after resize is {len(tokenizer)}")