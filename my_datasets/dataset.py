import os
import json
import random
import torch
import transformers
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
from dataset_info import get_question_prompt, get_label_candidate

# https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
FEATURE_START_TOKEN = "<|feature|>"
FEATURE_END_TOKEN = "<|/feature|>"

Q_START_TOKEN = "<|question|>"
Q_END_TOKEN = "<|/question|>"

A_START_TOKEN = "<|answer|>"
A_END_TOKEN = "<|/answer|>"

SHARED_START_TOKEN = "<|sharedfeatures|>"
SHARED_END_TOKEN = "<|/sharedfeatures|>"

DIFFERENT_START_TOKEN = "<|differentfeatures|>"
DIFFERENT_END_TOKEN = "<|/differentfeatures|>"

EOC_TOKEN = "<|endcompletion|>"
MASK_TOKEN = "<|mask|>"

pretrain_question_prompt = "Predicting"
select_prompt = "Choice:"


def load_txt(txt_file: str):
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def load_json(json_file: str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def process_mask_data_pretrain(mask_data: dict, train: bool=True):
    parts = []
    for key, info in mask_data.items():
        candidates = "||".join(info["candidate"])
        # Format into the specified structure
        value = info["value"]
        part = (f"{Q_START_TOKEN}{pretrain_question_prompt} {key}. {select_prompt} ||{candidates}||{Q_END_TOKEN}"
                f"{A_START_TOKEN}{value}{A_END_TOKEN}")
        parts.append(part)

    if train:
        # Shuffle samples before concatenation during training; no shuffling is applied during testing
        random.shuffle(parts)

    mask_txt = "".join(parts) + EOC_TOKEN
    return mask_txt

def build_input_pretrain(data: dict, mask_ratio: float, train: bool=True, drop_features: str=None):
    input_data, mask_data = data['input'], data['mask']
    if drop_features is not None:
        input_data = {k: v for k, v in input_data.items() if k not in drop_features}

    feature_txt_input = ", ".join([f"{k}: {v}" for k, v in input_data.items()])
    feature_txt = FEATURE_START_TOKEN + feature_txt_input + FEATURE_END_TOKEN

    mask_txt = process_mask_data_pretrain(mask_data, train)
    return feature_txt + mask_txt

def text_tokenizer(
        text: str,
        tokenizer: transformers.PreTrainedTokenizer,
        padding="longest",
        truncation="do_not_truncate",
):
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        padding=padding,
        return_offsets_mapping=True,
        max_length=tokenizer.model_max_length,
        truncation=truncation
    )
    input_ids = tokenized.input_ids[0]

    return input_ids


def make_labels_from_ids_pretrain(input_ids, label_start_id, label_end_id):
    labels = torch.full_like(input_ids, -100)

    # Locate all start and end positions
    start_positions = (input_ids == label_start_id).nonzero(as_tuple=True)[0]
    end_positions = (input_ids == label_end_id).nonzero(as_tuple=True)[0]

    # Check whether the number of start and end tokens match
    assert len(start_positions) == len(end_positions), "Mismatch in the number of start and end tokens"

    for start, end in zip(start_positions, end_positions):
        # The label span excludes the start/end tokens themselves
        answer = input_ids[start+1:end]

        # If the label is "-", the loss at this position should be ignored
        # Skip if the answer contains only one element and its value is 12
        if answer.numel() == 1 and answer.item() == 12:
            continue
        else:
            labels[start+1:end] = answer

    return labels


class DriveDataset_pretrain(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 local_rank,
                 train=True,
                 mask_ratio: float=1.0):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.train = train
        self.mask_ratio = mask_ratio

        # ablation features
        self.drop_features = None

        self.data_file = load_txt(data_path)
        if local_rank == 0:
            print('Number of training samples: ', len(self.data_file))

        self.a_start_token_id = self.tokenizer.convert_tokens_to_ids(A_START_TOKEN)
        self.a_end_token_id = self.tokenizer.convert_tokens_to_ids(A_END_TOKEN)

    def set_mask_ratio(self, ratio: float):
        self.mask_ratio = ratio

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        # Get the file name
        sample_id = self.data_file[idx].split("/")[-1]
        data = load_json(os.path.join('make_dataset/latest_del_emotion_json', self.data_file[idx]))

        input_txt = build_input_pretrain(data, self.mask_ratio, self.train, self.drop_features)
        input_ids = text_tokenizer(input_txt, self.tokenizer)
        labels = make_labels_from_ids_pretrain(input_ids, self.a_start_token_id, self.a_end_token_id)

        return dict(input_ids=input_ids, labels=labels, sample_id=sample_id)


def sample_batch(grade_group: list, pred_sample: str, group_weight: list, train: bool=True, template_num: int=1):
    if train:
        # Since the target sample cannot be used as a template, first remove it from the group
        group_idx = grade_group.index(pred_sample)
        grade_group.remove(pred_sample)
        del group_weight[group_idx]

    group_weight = np.array(group_weight, dtype=np.float64)
    # Renormalize after removing the sample
    probs = group_weight / group_weight.sum()
    # Sample two templates based on probability
    templates = np.random.choice(grade_group, size=template_num, replace=False, p=probs)

    return templates

def merge_dicts_strict(d1, d2):
    overlap = set(d1) & set(d2)
    if overlap:
        raise ValueError(f"Duplicate keys found: {overlap}")
    return {**d1, **d2}

def merge_samples(samples, drop_features=None):
    '''
    :param samples:
    :param drop_features: During testing, remove selected features; these features should not be included in the shared part
    Additionally, remove drop_features from the test sample in the individual part

    :return:
    '''
    shared = {}
    individual = []
    labels = []

    if not samples:
        return shared, individual

    input_keys = samples[0]['input'].keys()
    mask_keys = samples[0]['mask'].keys()

    # Identify keys whose values are identical across all samples
    for key in input_keys:
        if key in drop_features:
            continue
        values = [s['input'][key] for s in samples]
        if all(v == values[0] for v in values):
            shared[key] = values[0]

    for key in mask_keys:
        if key in drop_features:
            continue
        values = [s['mask'][key]['value'] for s in samples]
        if all(v == values[0] for v in values):
            shared[key] = values[0]

    # Keep the remaining key-value pairs
    for idx, s in enumerate(samples):
        input_ind = {key: value for key, value in s['input'].items() if key not in shared}
        mask_ind = {key: value['value'] for key, value in s['mask'].items() if key not in shared}

        '''
        Remove drop_features only from the test sample, not from template samples
        '''
        if idx == len(samples) - 1 and drop_features is not None:
            for key in drop_features:
                input_ind.pop(key, None)
                mask_ind.pop(key, None)

        # merge key
        ind = merge_dicts_strict(input_ind, mask_ind)

        individual.append(ind)
        labels.append(s['label'])

    return shared, individual, labels

def safe_clean(v):
    if isinstance(v, (int, float)):
        return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
    return str(v).strip().rstrip('.')

def dict_to_text(d):
    """
    Convert the dictionary into text format: 'key1: value1, key2: value2'
    """
    return ", ".join(f"{k}: {safe_clean(v)}" for k, v in d.items())

def make_labels_from_ids(input_ids, label_start_id, label_end_id, question_prompt):
    labels = torch.full_like(input_ids, -100)

    # Locate all start and end positions
    start_positions = (input_ids == label_start_id).nonzero(as_tuple=True)[0]
    end_positions = (input_ids == label_end_id).nonzero(as_tuple=True)[0]

    # Check whether the number of start and end tokens match
    assert len(start_positions) == len(end_positions), "start/end 数量不匹配"

    # Apply masking only to the final sample; preceding samples (templates) do not require masking
    start_positions = start_positions[-len(question_prompt):]
    end_positions = end_positions[-len(question_prompt):]

    for start, end in zip(start_positions, end_positions):
        # The label span excludes the start/end tokens themselves
        answer = input_ids[start+1:end]

        # If the label is "-", the loss at this position should be ignored
        # Skip if the answer contains only one element and its value is 12
        if answer.numel() == 1 and answer.item() == 12:
            continue
        else:
            labels[start+1:end] = answer

    return labels

def process_mask_data(
        mask_data: dict,
        question_prompt,
        label_candidate,
        add_EOC_token: bool=True,
        add_candidate: bool=True
):
    parts = []
    for key, value in mask_data.items():
        candidates = "||".join(label_candidate[key])
        # Concatenate into the specified format
        if add_candidate:
            part = (f"{Q_START_TOKEN}{question_prompt[key]}. {select_prompt} ||{candidates}||{Q_END_TOKEN}"
                    f"{A_START_TOKEN}{value}{A_END_TOKEN}")
        else:
            part = (f"{Q_START_TOKEN}{question_prompt[key]}{Q_END_TOKEN}"
                    f"{A_START_TOKEN}{value}{A_END_TOKEN}")

        parts.append(part)

    # Note: Unlike the pretraining stage, no shuffling is applied before concatenating masked features
    # random.shuffle(parts)
    if add_EOC_token:
        mask_text = "".join(parts) + EOC_TOKEN
    else:
        mask_text = "".join(parts)
    return mask_text

def make_input_text(shared, individual, labels, question_prompt, label_candidate, add_candidate=True):
    shared_features = SHARED_START_TOKEN + dict_to_text(shared) + SHARED_END_TOKEN

    different_parts = ''
    for idx, data in enumerate(individual):
        input = DIFFERENT_START_TOKEN + dict_to_text(data) + DIFFERENT_END_TOKEN

        if idx == len(individual)-1:
            add_candidate = True
        mask_text = process_mask_data(
            labels[idx],
            question_prompt,
            label_candidate,
            add_EOC_token=False,
            add_candidate=add_candidate
        )

        different_parts = different_parts + input + mask_text

    input_text = shared_features + different_parts + EOC_TOKEN
    return input_text

def build_input(data: dict, question_prompt: dict, label_candidate: dict):
    input_data, mask_data, label = data['input'], data['mask'], data['re_label']
    feature_text_input = ", ".join([f"{k}: {v}" for k, v in input_data.items()])
    mask_text_input = ", ".join([f"{k}: {v['value']}" for k, v in mask_data.items()])

    feature_text = FEATURE_START_TOKEN + feature_text_input +', '+ mask_text_input + FEATURE_END_TOKEN

    mask_text = process_mask_data(label, question_prompt, label_candidate)
    return feature_text + mask_text


class DriveDataset(Dataset):
    def __init__(self,
                 data_path,
                 train_data_file,
                 test_data_file,
                 tokenizer: transformers.PreTrainedTokenizer,
                 local_rank,
                 task_type,
                 use_comparative_reason: bool = False,
                 train=True,
                 template_num: int=1):
        self.data_path = data_path
        self.train_data = load_txt(train_data_file)
        self.test_data = load_txt(test_data_file)
        self.task_type = task_type

        self.question_prompt = get_question_prompt(task_type)
        self.label_candidate = get_label_candidate(task_type)

        self.drop_features = None

        self.train_grade_file = train_data_file.replace(
            "train_", "grade_train_"
        )
        self.train_grade_label = load_txt(self.train_grade_file)
        self.train_prob_file = train_data_file.replace(
            "train_", "mean_IP_train_"
        )
        self.train_prob = load_txt(self.train_prob_file)

        self.test_grade_file = test_data_file.replace(
            "test", "grade_test"
        )
        self.test_grade_label = load_txt(self.test_grade_file)

        self.use_comparative_reason = use_comparative_reason

        self.tokenizer = tokenizer
        self.train = train
        self.template_num = template_num

        if local_rank == 0:
            if train:
                print('Number of training samples: ', len(self.train_data))
            else:
                print('Number of testing samples: ', len(self.test_data))

        # For the training set, group samples by train_grade_label using defaultdict
        grade_groups = defaultdict(list)
        weight_groups = defaultdict(list)
        valid_labels = {"Primary school", "Middle school"}

        for k, v, p in zip(self.train_data, self.train_grade_label, self.train_prob):
            if v not in valid_labels:
                raise ValueError(f"Grade label is wrong: {v}")
            grade_groups[v].append(k)
            weight_groups[v].append(p)

        self.primary_group = grade_groups["Primary school"]
        self.middle_group = grade_groups["Middle school"]
        self.primary_weight = weight_groups["Primary school"]
        self.middle_weight = weight_groups["Middle school"]

        self.a_start_token_id = self.tokenizer.convert_tokens_to_ids(A_START_TOKEN)
        self.a_end_token_id = self.tokenizer.convert_tokens_to_ids(A_END_TOKEN)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.use_comparative_reason:
            if self.train:
                pred_sample, grade_label = self.train_data[idx], self.train_grade_label[idx]
            else:
                pred_sample, grade_label = self.test_data[idx], self.test_grade_label[idx]

            if grade_label == 'Primary school':
                self.grade_group = self.primary_group
                self.group_weight = self.primary_weight
            else:
                self.grade_group = self.middle_group
                self.group_weight = self.middle_weight

            templates = sample_batch(self.grade_group, pred_sample, self.group_weight, self.train, self.template_num)

            sample_id = pred_sample.split('/')[-1]

            temp_files = list(templates)
            temp_files.append(pred_sample) # Template samples + target sample

            # Load JSON file and construct input text
            temp_data_path = [os.path.join(self.data_path, file) for file in temp_files]
            temp_data = [load_json(file) for file in temp_data_path]
            shared, individual, labels = merge_samples(temp_data, self.drop_features)
            input_text = make_input_text(shared, individual, labels, self.question_prompt, self.label_candidate)

        else:
            if self.train:
                sample = load_json(os.path.join(self.data_path, self.train_data[idx]))
                sample_id = self.train_data[idx].split('/')[-1]
            else:
                sample = load_json(os.path.join(self.data_path, self.test_data[idx]))
                sample_id = self.test_data[idx].split('/')[-1]

            # Construct input text
            input_text = build_input(sample, self.question_prompt, self.label_candidate)

        input_ids = text_tokenizer(input_text, self.tokenizer)
        labels = make_labels_from_ids(input_ids, self.a_start_token_id, self.a_end_token_id, self.question_prompt)

        return dict(input_ids=input_ids, labels=labels, sample_id=sample_id)