import json
import logging
from typing import Dict, Sequence, Optional, List, Union, Any, Tuple

import random
import torch
import scipy
import transformers
import numpy as np
import webdataset as wds
from dataclasses import dataclass

from my_datasets.arguments import DataArguments
from my_datasets.special_tokens import IGNORE_INDEX


def example_ids_to_attention_mask(example_ids: List[int]) -> np.ndarray:
    """Construct a boolean attention mask from a sequence of example_ids, representing a single element in a batch.

    :param example_ids: List of length (seq_len) containing token IDs as integers.

    The output is a np.array of type bool, with lower-block-triangular entries.
    """
    assert isinstance(
        example_ids, list
    ), f"expected list of example_ids, got type {type(example_ids)}"
    max_example_id = max(example_ids)
    example_ids = torch.Tensor(example_ids)
    block_sizes = [(example_ids == i).sum() for i in range(max_example_id + 1)]
    blocks = [np.tril(np.full((x, x), True)) for x in block_sizes]
    mask = scipy.linalg.block_diag(*blocks)
    return mask


def prepare_4d_attention_mask(instances: Sequence[Dict]) -> np.ndarray:
    # each attention mask is of shape [seq_len, seq_len]
    attention_masks = [
        example_ids_to_attention_mask(x["example_ids"]) for x in instances
    ]
    try:
        attention_mask = np.stack(
            attention_masks, axis=0
        )  # shape [batch_size, seq_len, seq_len]
    except ValueError as ve:
        if "all input arrays must have the same shape" in str(ve):
            logging.warning(
                "ValueError in prepare_4d_attention_mask(); expected all attention "
                f"masks to have same shape; got shapes {[x.shape for x in attention_masks]}"
            )
            raise ve
    attention_mask = np.expand_dims(
        attention_mask, axis=1
    )  # shape [batch_size, 1, seq_len, seq_len]
    return attention_mask


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    use_position_ids: bool = True

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        output = dict(input_ids=input_ids, labels=labels)

        if any("example_ids" in x for x in instances):
            # Case: this is a packed batch. Prepare the 4d attention mask, and the position IDs.
            attention_mask = prepare_4d_attention_mask(instances)
            attention_mask = torch.from_numpy(attention_mask).to(input_ids.device)
            output.update(dict(attention_mask=attention_mask))
            if self.use_position_ids:
                position_ids = torch.LongTensor(
                    [instance["position_ids"] for instance in instances]
                )
                output.update(dict(position_ids=position_ids))

        else:
            # Case: this is not a packed batch. Prepare a standard attention mask.
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            output.update(dict(attention_mask=attention_mask))
        return output


def maybe_cast_to_tensor(x: Union[torch.Tensor, List, np.array]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, list):
        return torch.Tensor(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        raise ValueError(f"unknown dtype: {type(x)}")


def table_id_from_key(k: str) -> str:
    return k.split("__")[0]


def ids_and_lens_from_batch(
    batch: Dict[str, List[Union[torch.Tensor, str]]]
) -> List[List[int]]:
    """Return a nested list where the ith element is [i, len(example_i)]."""
    return [[i, len(ids)] for i, ids in enumerate(batch["input_ids"])]


def merge_batch_samples_by_key(
    batch: Dict[str, List[Union[torch.Tensor, str]]]
) -> List[List[int]]:
    """When samples are from the same source table, combine a sample with the one preceding it.

    Returns two lists of length len(batch["input_ids"]):
        ids: list where the iths element indicates the example id of the ith element in the batch. Elements from the
            same source table have the same ids. ids start from zero and count up consecutively.
        lens: list where the iths element indicates the length of the ith element in the batch.
    """
    ids_and_lens = ids_and_lens_from_batch(batch)
    for i in range(1, len(batch["__key__"])):
        if table_id_from_key(batch["__key__"][i]) == table_id_from_key(
            batch["__key__"][i - 1]
        ):
            # print(
            #     f"[DEBUG] merging samples with key {batch['__key__'][i]} and {batch['__key__'][i-1]}"
            # )
            ids_and_lens[i][0] = ids_and_lens[i - 1][0]
        else:
            # print(
            #     f"[DEBUG] NOT merging samples with key {batch['__key__'][i]} and {batch['__key__'][i - 1]}"
            # )
            # Ensure sample IDs are contiguous
            ids_and_lens[i][0] = ids_and_lens[i - 1][0] + 1
    return ids_and_lens


def generate_position_ids(ids_and_lens, max_len) -> List[int]:
    # Initialize an empty array for position indices
    position_ids = []

    # Track the current ID and initialize a position counter
    current_id = None
    current_position = 0

    for id_, length in ids_and_lens:
        # If the ID changes, reset the current position counter
        if id_ != current_id:
            current_id = id_
            current_position = 0

        # Append a range of numbers from current_position to current_position+length to the position_ids array
        position_ids.extend(
            np.arange(current_position, current_position + length).tolist()
        )
        current_position += length
        if len(position_ids) >= max_len:
            break

    return position_ids[:max_len]


def pack_samples(
    batch: Dict[str, Union[str, List[torch.Tensor]]],
    max_len: int,
    trim_extra_bos_tokens: bool = False,
    merge_samples_by_key: bool = True,
    bos_token_id: Optional[int] = None,
) -> Dict[str, List[torch.Tensor]]:
    """ "Pack a set of samples into a batch, discarding any extra data.

    The resulting dict has keys ['input_ids', 'labels', 'example_ids', 'position_ids'].
    """
    assert len(batch["input_ids"]) == len(
        batch["labels"]
    ), f"expected equal-length inputs and labels, got {len(batch['input_ids'])} and {len(batch['labels'])}"

    if trim_extra_bos_tokens and len(batch["input_ids"]) > 1:
        assert (
            bos_token_id is not None
        ), "bos_token_id is required to trim extra bos tokens."
        for i in range(1, len(batch["input_ids"])):
            if batch["input_ids"][i][0] == bos_token_id:
                batch["input_ids"][i] = batch["input_ids"][i][1:]
                batch["labels"][i] = batch["labels"][i][1:]

    # example_ids is a sequence where the integer at each sequence identifies the index of the sample
    # in the batch from which that token originated; this allows to construct an example-wise
    # attention matrix. Note that the attention matrix also needs to account for masking
    # (the attention matrix should mask tokens where labels != IGNORE_INDEX).
    # For example, it looks like [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, ...]
    if merge_samples_by_key:
        ids_and_lens = merge_batch_samples_by_key(batch)
    else:
        ids_and_lens = ids_and_lens_from_batch(batch)

    example_ids = [[i] * ids_len for i, ids_len in ids_and_lens]
    example_ids = [i for ids in example_ids for i in ids][:max_len]

    # position_ids gives the positional index of an element within its sequence.
    # For example, it looks like [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, ...]
    position_ids = generate_position_ids(ids_and_lens, max_len)

    input_ids = torch.cat([maybe_cast_to_tensor(x) for x in batch["input_ids"]], dim=0)
    input_ids = input_ids[:max_len]
    labels = torch.cat([maybe_cast_to_tensor(x) for x in batch["labels"]], dim=0)
    labels = labels[:max_len]

    return {
        "input_ids": [input_ids],
        "labels": [labels],
        "example_ids": [example_ids],
        "position_ids": [position_ids],
    }


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True

def load_and_tokenize_preserialized_wds(
    tokenizer,
    urls: Sequence[str],
    data_arguments: DataArguments,
    split: str,
    is_train=True,
    shuffle_shards: bool = True,
    shuffle_before_packing: bool = False,
    shuffle_after_packing: bool = True,
    shuffle_buffer_size: Optional[int] = 10_000,
    shuffle_random_seed=42,
    require_full_context_size: bool = False,
    shards_shuffle_buffer_size=100,
    label_skip_config: Tuple[int, float] = (0, 0.9),  # (skip_label, skip_prob)
) -> Dict[str, wds.WebDataset]:

    if urls[0].startswith("s3://"):
        logging.warning(f"s3 file urls detected; attempting to pipe data from s3")
        urls = [f"pipe:aws s3 cp {url} -" for url in urls]

    def _extract_json(sample) -> Dict[str, str]:
        """Fetch the {'text': ..., 'label_spans': ...} for a sample."""
        key = [x for x in sample.keys() if x.endswith("json")][0]
        json_bytes = sample[key]
        return json.loads(json_bytes.decode("utf-8"))

    def _return_preprocessed(ex):
        # 提取json文件中的内容后要转换为tensor
        input_ids, labels, example_ids, sample_type = ex["input_ids"], ex['labels'], ex['sample_ids'], ex['sample_type']

        input_ids = [torch.tensor(input_ids, dtype=torch.long)]
        labels = [torch.tensor(labels, dtype=torch.long)]
        example_ids = [example_ids]
        sample_type = [torch.tensor(sample_type, dtype=torch.long)]

        return dict(input_ids=input_ids, labels=labels, example_ids=example_ids, sample_type=sample_type, __key__=ex['__key__'])

    def _flatten_values(example: Dict[str, List[Any]]) -> Dict[str, Any]:
        return {k: v[0] for k, v in example.items()}

    def _pack_samples(x: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """Take a 'batch' of samples and pack it."""
        # We have to do some weird acrobatics with the samples here to ensure we can use the exact same preprocessing
        # functions with the HF datasets and the webdatasets.
        assert len(x) == 1, f"expected len(x)==1, got len(x)=={len(x)}"
        samples = x[0]
        # Each element of samples is a dict, with the keys ['input_ids', 'labels', '__key__'].
        # Those elements contain values of a *list* where each list has a single element
        # (i.e. the input_ids, labels, or key).
        batch = {
            "input_ids": [x["input_ids"][0] for x in samples],
            "labels": [x["labels"][0] for x in samples],
            "sample_type": [x["sample_type"] for x in samples],
            "__key__": [x["__key__"] for x in samples],
        }

        packed = pack_samples(
            batch,
            tokenizer.model_max_length,
            trim_extra_bos_tokens=data_arguments.trim_extra_bos_tokens,
            bos_token_id=tokenizer.bos_token_id,
            merge_samples_by_key=data_arguments.merge_samples_by_key,
        )
        # Returns a dict with keys ['input_ids', 'labels', 'example_ids', 'position_ids'], where we have unpacked
        # the 'HF-style' batch formatting {str: List[Tensor]} to a 'torch style' batch formatting {str: Tensor}.
        return _flatten_values(packed)

    def _filter_fn(example) -> bool:
        """Return True if example length is less than ir equal to tokenizer.model_max_length."""
        if is_train and require_full_context_size:
            # Require samples to be exactly length tokenizer.model_max_length
            length_is_ok = len(example["input_ids"]) == tokenizer.model_max_length
        elif is_train:
            # Require samples to fit in context window
            length_is_ok = len(example["input_ids"]) <= tokenizer.model_max_length
        else:
            # For non-training cases, we consider padding tokens, and require samples
            # to fit in context window.
            length_is_ok = (
                example["input_ids"].ne(tokenizer.pad_token_id).sum().item()
                + example["labels"].ne(-100).sum().item()
                <= tokenizer.model_max_length
            )
        if not length_is_ok:
            logging.warning(f"dropping sample with length {len(example['input_ids'])}")
        return length_is_ok


    def _skip_label_with_probability(sample, label_key="sample_type", skip_label=0, skip_prob=0.9):
        """Return True to keep the sample, False to skip it"""
        label = sample.get(label_key)
        if label == skip_label and random.random() < skip_prob:
            return False
        return True


    pipeline = [
        wds.SimpleShardList(urls, seed=shuffle_random_seed),
    ]

    if shuffle_shards:
        # at this point we have an iterator over all the shards
        pipeline.append(
            wds.shuffle(shards_shuffle_buffer_size),
        )

    pipeline.extend(
        [
            wds.split_by_node,
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ]
    )

    if shuffle_before_packing:
        # This will pack random/unrelated samples together if activated
        pipeline.append(wds.shuffle(shuffle_buffer_size))

    pipeline.extend(
        [
            wds.map(_extract_json),
            # The following line is used to balance data sampling
            wds.select(lambda sample: _skip_label_with_probability(
                sample, skip_label=label_skip_config[0], skip_prob=label_skip_config[1]
            )),
            wds.map(_return_preprocessed),
        ]
    )

    if data_arguments.pack_samples:
        pipeline.extend(
            [
                # Elements must be lists for batching
                wds.map(lambda x: [x]),
                wds.batched(data_arguments.pack_samples_batch_size),
                wds.map(_pack_samples),
            ]
        )
    else:
        pipeline.append(wds.map(_flatten_values))

    if data_arguments.handle_too_long == "drop":
        pipeline.append(wds.select(_filter_fn))

    if shuffle_after_packing:
        pipeline.append(wds.shuffle(shuffle_buffer_size))

    dataset = wds.DataPipeline(*pipeline)

    return {split: dataset}

def example_ids_to_attention_mask(example_ids: List[int]) -> np.ndarray:
    """Construct a boolean attention mask from a sequence of example_ids, representing a single element in a batch.

    :param example_ids: List of length (seq_len) containing token IDs as integers.

    The output is a np.array of type bool, with lower-block-triangular entries.
    """
    assert isinstance(
        example_ids, list
    ), f"expected list of example_ids, got type {type(example_ids)}"
    max_example_id = max(example_ids)
    example_ids = torch.Tensor(example_ids)
    block_sizes = [(example_ids == i).sum() for i in range(max_example_id + 1)]
    blocks = [np.tril(np.full((x, x), True)) for x in block_sizes]
    mask = scipy.linalg.block_diag(*blocks)
    return mask

def prepare_4d_attention_mask(instances: Sequence[Dict]) -> np.ndarray:
    # each attention mask is of shape [seq_len, seq_len]
    attention_masks = [
        example_ids_to_attention_mask(x["example_ids"]) for x in instances
    ]
    try:
        attention_mask = np.stack(
            attention_masks, axis=0
        )  # shape [batch_size, seq_len, seq_len]
    except ValueError as ve:
        if "all input arrays must have the same shape" in str(ve):
            logging.warning(
                "ValueError in prepare_4d_attention_mask(); expected all attention "
                f"masks to have same shape; got shapes {[x.shape for x in attention_masks]}"
            )
            raise ve
    attention_mask = np.expand_dims(
        attention_mask, axis=1
    )  # shape [batch_size, 1, seq_len, seq_len]
    return attention_mask

def build_position_ids(example_ids):

    assert isinstance(
        example_ids, list
    ), f"expected list of example_ids, got type {type(example_ids)}"

    position_ids = []
    current_id = None
    pos = 0

    for sid in example_ids:
        if sid != current_id:
            current_id = sid
            pos = 0
        position_ids.append(pos)
        pos += 1

    return position_ids

def build_2D_position_ids(instances):
    position_ids = [build_position_ids(x["example_ids"]) for x in instances]
    position_ids = np.stack(position_ids, axis=0)  # shape [batch_size, seq_len]
    return position_ids

def left_pad_sequences(sequences, padding_value, batch_first=True):
    # Reverse the sequence first
    reversed_seqs = [torch.flip(seq, dims=[0]) for seq in sequences]
    # Apply right padding (equivalent to left padding the original sequence)
    padded = torch.nn.utils.rnn.pad_sequence(
        reversed_seqs, batch_first=batch_first, padding_value=padding_value
    )
    # Reverse it back
    return torch.flip(padded, dims=[1]) if batch_first else torch.flip(padded, dims=[0])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    use_position_ids: bool = True
    task_type: str = "Pretrain",
    test_task: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        if not self.test_task:
            '''
            Use right padding during training
            '''
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )
        else:
            '''
            Use left padding during testing
            '''
            input_ids = left_pad_sequences(input_ids, self.tokenizer.pad_token_id, batch_first=True)
            labels = left_pad_sequences(labels, IGNORE_INDEX, batch_first=True)

        output = dict(input_ids=input_ids, labels=labels)

        if any("example_ids" in x for x in instances):
            # Case: this is a packed batch. Prepare the 4d attention mask, and the position IDs.
            attention_mask = prepare_4d_attention_mask(instances)
            attention_mask = torch.from_numpy(attention_mask).to(input_ids.device)
            attention_mask = attention_mask.to(torch.float32)

            if self.use_position_ids:
                position_ids = build_2D_position_ids(instances)
                position_ids = torch.LongTensor(position_ids)
                output.update(dict(position_ids=position_ids))

        else:
            # Case: this is not a packed batch. Prepare a standard attention mask.
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            attention_mask = attention_mask.to(torch.float32)

        output.update(dict(attention_mask=attention_mask))
        output.update(dict(sample_id=[x["sample_id"] for x in instances]))

        return output
