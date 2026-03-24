import os
import json
import math
import random
import torch
import transformers
import torch.nn.functional as F

from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessor

from my_datasets.special_tokens import (
    FEATURE_END_TOKEN,
    Q_START_TOKEN,
    Q_END_TOKEN,
    A_START_TOKEN,
    A_END_TOKEN,
    EOC_TOKEN,
    IGNORE_INDEX,
    DIFFERENT_END_TOKEN
)

from dataset_info import get_candidate_class, get_candidate_token


class RestrictTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for i in self.allowed_token_ids:
            mask[:, i] = 0
        return scores + mask


def write_dict_to_json(sample_id: int, data_dict: dict, folder_path: str):
    save_path = os.path.join(folder_path, f"{sample_id}")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False)


def extract_segments(input: torch.Tensor, start_token_id: int, end_token_id: int):
    """
    Args:
        input: 1D tensor, shape [T]
        start_token_id: int, start token id
        end_token_id: int, end token id

    Returns:
        List[torch.Tensor]: 每个 answer 区间的 input_ids，包含 start 和 end token
    """
    assert input.dim() == 1, "input_ids must be 1D tensor"

    start_indices = (input == start_token_id).nonzero(as_tuple=True)[0]
    end_indices = (input == end_token_id).nonzero(as_tuple=True)[0]

    if len(start_indices) != len(end_indices):
        raise ValueError(f"start_indices ({len(start_indices)}) != end_indices ({len(end_indices)})")

    segments = []
    for start, end in zip(start_indices.tolist(), end_indices.tolist()):
        if end < start:
            raise ValueError(f"Found end token before start token at positions {start}, {end}")
        segments.append(input[start:end + 1])

    return segments


def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def candidate_probs_single_forward(outputs_logits, candidate_token_ids):
    """
    outputs_logits: (B, L_max, V) Obtain results from a single forward pass of the model
    candidate_token_ids: list[list[int]]  Token sequence for each candidate
    """
    B, L_max, V = outputs_logits.shape
    n_cand = len(candidate_token_ids)
    cand_probs = torch.zeros(B, n_cand, device=outputs_logits.device)

    for i, cand in enumerate(candidate_token_ids):
        # Truncate to the corresponding length
        cand_len = len(cand)
        cand_logits_slice = outputs_logits[:, :cand_len, :]  # (B, cand_len, V)
        cand_log_probs = F.log_softmax(cand_logits_slice, dim=-1)

        # Gather the log-probabilities of the corresponding tokens
        token_ids = torch.tensor(cand, device=outputs_logits.device).unsqueeze(0).expand(B, -1)
        token_logp = cand_log_probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)  # (B, cand_len)

        # Accumulate the log-probabilities
        cand_probs[:, i] = token_logp.sum(dim=1)

    probs = torch.softmax(cand_probs, dim=-1)
    return probs


def candidate_probs_batch_padded(logits, batch_candidate):
    """
    logits: torch.Tensor, shape = [batch, seq_len, vocab_size]
    batch_candidate: list[list[list[int]]], Each sample has multiple candidate classes, and each class may consist of multiple token IDs
    """
    batch_size = len(batch_candidate)
    device = logits.device
    probs_list = []

    # Use the logits at the last token position (classification is typically based on the final token)
    last_logits = logits[:, -1, :]  # [batch, vocab_size]
    last_probs = F.softmax(last_logits, dim=-1)  # [batch, vocab_size]

    for b_idx in range(batch_size):
        candidate_probs = []
        for candidate in batch_candidate[b_idx]:
            # Ensure the candidate list is not empty and indices are valid
            valid_tokens = [t for t in candidate if 0 <= t < last_probs.shape[1]]
            if len(valid_tokens) == 0:
                candidate_probs.append(torch.tensor(0.0, device=device))
                continue

            # Extract probabilities of these tokens and compute their sum
            token_probs = last_probs[b_idx, valid_tokens]
            prob = torch.mean(token_probs)
            candidate_probs.append(prob)

        # Convert to a tensor and normalize
        candidate_probs = torch.stack(candidate_probs)
        candidate_probs = candidate_probs / (candidate_probs.sum() + 1e-12)
        probs_list.append(candidate_probs)

    return probs_list


def match_generated_with_candidates_probs(gen_tokens, batch_candidate, temperature=0.5):
    """
    gen_tokens: Tensor, [batch, seq_len]
    batch_candidate: list[list[list[int]]]
    temperature: float
    return: list[Tensor]
    """
    probs_list = []

    for b_idx, candidates in enumerate(batch_candidate):
        pred_seq = gen_tokens[b_idx].tolist()
        scores = []
        full_match_idx = -1  # Mark indices of exactly matched candidates

        for i, cand in enumerate(candidates):
            if pred_seq[:len(cand)] == cand and len(pred_seq) == len(cand):
                full_match_idx = i
                break

            match_len = 0
            for a, b in zip(pred_seq, cand):
                if a == b:
                    match_len += 1
                else:
                    break
            score = match_len / len(cand)
            scores.append(score)

        if full_match_idx != -1:
            probs = torch.zeros(len(candidates), dtype=torch.float32)
            probs[full_match_idx] = 1.0
        else:
            scores = torch.tensor(scores, dtype=torch.float32)
            probs = torch.softmax(scores / temperature, dim=0)

        probs_list.append(probs)

    return probs_list

def build_context_q(context_q, shuffle=True):
    questions = list(context_q.values())

    if shuffle:
        random.shuffle(questions)

    context_q = [x for sublist in questions for x in sublist]

    return context_q

def save_eval_output_pretrain(
        model,
        tokenizer: transformers.PreTrainedTokenizer,
        test_dataloader: torch.utils.data.DataLoader,
        local_rank: int,
        world_size: int,
        step_value_in_state: str,
        eval_results_folder: str,
        batch_size_split: int,
        use_context_info: bool=False
):

    model.eval()

    # Load the question and candidate from the sample
    primary_question = load_json('eval_data_process/split_data/sample_6_1_question.json')
    primary_answer = load_json('eval_data_process/split_data/sample_6_1_answer.json')
    primary_candidate = load_json('eval_data_process/split_data/sample_6_1_candidate.json')
    middle_question = load_json('eval_data_process/split_data/sample_0_0_question.json')
    middle_answer = load_json('eval_data_process/split_data/sample_0_0_answer.json')
    middle_candidate = load_json('eval_data_process/split_data/sample_0_0_candidate.json')

    if local_rank == 0:
        pbar = tqdm(
            desc=f"Eval",
            colour="green",
            initial=0,
            total=len(test_dataloader) * world_size,
            dynamic_ncols=True
        )
    else:
        pbar = None

    feature_end_token_id = tokenizer.convert_tokens_to_ids(FEATURE_END_TOKEN)
    a_start_token_id = tokenizer.convert_tokens_to_ids(A_START_TOKEN)
    a_end_token_id = tokenizer.convert_tokens_to_ids(A_END_TOKEN)
    q_start_token_id= tokenizer.convert_tokens_to_ids(Q_START_TOKEN)
    q_end_token_id = tokenizer.convert_tokens_to_ids(Q_END_TOKEN)

    if local_rank == 0:
        print('--------save path:{}--------'.format(eval_results_folder))
        os.makedirs(eval_results_folder, exist_ok=True)

    device = f"cuda:{local_rank}"
    for batch in test_dataloader:

        batch_size = batch["input_ids"].shape[0]

        # During testing, the sample batch size must be 1, as batching is constructed over multiple questions
        assert batch_size == 1, 'batch_size must be 1'

        a_start_tensor = torch.tensor([a_start_token_id], device='cuda')
        a_end_tensor = torch.tensor([a_end_token_id], device='cuda')

        # Used to store results for each sample
        answer_records = {"sample_id": batch["sample_id"][0],
                           "label": [],
                           "pred_token": [],
                           "pred_probs": []}

        input_ids = batch["input_ids"][0].to(device)

        feature_end_idx = (input_ids == feature_end_token_id).nonzero(as_tuple=True)[0].item()

        feature_part = input_ids[:feature_end_idx + 1]

        # Split only the answer for label storage
        answers = extract_segments(input_ids[feature_end_idx + 1:], a_start_token_id, a_end_token_id)
        answer_records["label"] = [ans.tolist() for ans in answers]

        # No need to split the answer;
        # directly use the number of questions to determine whether it is primary or secondary school
        question_num = (input_ids[feature_end_idx + 1:] == q_start_token_id).sum().item()
        if question_num == 33:
            questions = primary_question
            answers = primary_answer
            candidates = primary_candidate
        elif question_num == 64:
            questions = middle_question
            answers = middle_answer
            candidates = middle_candidate
        else:
            raise ValueError(f"Question num {question_num} is not supported")

        # Split the questions of a sample into batches
        question_batch = math.ceil(question_num / batch_size_split)

        for b in range(question_batch):
            batch_input_ids = []
            batch_candidate = []

            # Combine questions to construct batch inputs
            for q_idx in range(b*batch_size_split, (b+1)*batch_size_split):
                if q_idx == question_num: # 越界
                    break
                q = torch.tensor(questions[str(q_idx)], dtype=torch.long).to(device)

                if use_context_info:
                    context_q, context_a = questions.copy(), answers.copy()
                    del context_q[str(q_idx)]
                    del context_a[str(q_idx)]
                    context_q = build_context_q(context_q, shuffle=True)
                    context_q = torch.tensor(context_q, dtype=torch.long).to(device)
                    cur_input = torch.cat([feature_part.to(device), context_q, q, a_start_tensor])
                else:
                    cur_input = torch.cat([feature_part.to(device), q, a_start_tensor])

                batch_input_ids.append(cur_input)
                batch_candidate.append(candidates[str(q_idx)])

            # After extracting features and QA pairs, sequence lengths vary, so left padding is still required
            max_len = max([x.shape[0] for x in batch_input_ids])
            padded_inputs = []
            for x in batch_input_ids:
                pad_len = max_len - x.shape[0]
                if pad_len > 0:
                    padded_x = torch.cat([
                        torch.full((pad_len,), tokenizer.pad_token_id,
                                   dtype=torch.long, device=device), x])
                else:
                    padded_x = x
                padded_inputs.append(padded_x)

            input_ids_batch = torch.stack(padded_inputs, dim=0).to(device)
            attention_mask = input_ids_batch.ne(tokenizer.pad_token_id).to(torch.long).to(device)
            max_len = max(len(cand) for sample in batch_candidate for cand in sample)

            with torch.no_grad():
                generated = model.module.generate(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask,
                    max_new_tokens=max_len,
                    # logits_processor=logits_processor,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=a_end_token_id,
                )
                gen_tokens = generated[:, input_ids_batch.shape[1]:]
                probs_list = match_generated_with_candidates_probs(gen_tokens, batch_candidate)

            for b in range(len(batch_candidate)):
                pred_token = batch_candidate[b][torch.argmax(probs_list[b]).item()]
                answer_records["pred_token"].append(pred_token)
                batch_probs = {i: round(float(probs_list[b][i]), 4) for i in range(len(batch_candidate[b]))}
                answer_records["pred_probs"].append(batch_probs)

        # Save results
        write_dict_to_json(answer_records["sample_id"], answer_records, eval_results_folder)

        if local_rank == 0:
            pbar.update(world_size)

    if local_rank == 0:
        pbar.close()

def check_candidate_token(candidate_token):
    for i in range(len(candidate_token)):
        candidate = candidate_token[i]
        assert len(set(candidate)) == len(candidate), \
            'Wrong, The candidate tokens has repeated tokens!'

def make_save_path(
        step_value_in_state: str,
        eval_results_folder: str,
):
    result_folder = 'test_results'
    folder_name = eval_results_folder + step_value_in_state
    save_path = os.path.join(result_folder, folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    return save_path

def extract_segments(input: torch.Tensor, start_token_id: int, end_token_id: int):
    """
    Args:
        input: 1D tensor, shape [T]
        start_token_id: int, start token id
        end_token_id: int, end token id

    Returns:
        List[torch.Tensor]: Input IDs for each answer span, including start and end tokens
    """
    assert input.dim() == 1, "input_ids must be 1D tensor"

    start_indices = (input == start_token_id).nonzero(as_tuple=True)[0]
    end_indices = (input == end_token_id).nonzero(as_tuple=True)[0]

    if len(start_indices) != len(end_indices):
        raise ValueError(f"start_indices ({len(start_indices)}) != end_indices ({len(end_indices)})")

    segments = []
    for start, end in zip(start_indices.tolist(), end_indices.tolist()):
        if end < start:
            raise ValueError(f"Found end token before start token at positions {start}, {end}")
        segments.append(input[start:end + 1])

    return segments

def save_eval_output(
        model,
        tokenizer,
        test_dataloader: torch.utils.data.DataLoader,
        local_rank: int,
        world_size: int,
        step_value_in_state: str,
        eval_results_folder: str,
        task_type: str,
        use_comparative_reason: bool=False,
        use_context_info: bool=False
):

    model.eval()

    if local_rank == 0:
        pbar = tqdm(
            desc=f"Eval",
            colour="green",
            initial=0,
            total=len(test_dataloader) * world_size,
            dynamic_ncols=True
        )
    else:
        pbar = None

    candidate_token = get_candidate_token(task_type)
    candidate_class = get_candidate_class(task_type)

    # Check whether candidate tokens are duplicated
    check_candidate_token(candidate_token)

    if use_comparative_reason:
        # If comparative reasoning is enabled, split the input using "<|/differentfeatures|>"
        feature_end_token_id = tokenizer.convert_tokens_to_ids(DIFFERENT_END_TOKEN)
    else:
        # Otherwise, split the input using "<|/feature|>"
        feature_end_token_id = tokenizer.convert_tokens_to_ids(FEATURE_END_TOKEN)

    a_start_token_id = tokenizer.convert_tokens_to_ids(A_START_TOKEN)
    a_end_token_id = tokenizer.convert_tokens_to_ids(A_END_TOKEN)
    q_start_token_id= tokenizer.convert_tokens_to_ids(Q_START_TOKEN)
    q_end_token_id = tokenizer.convert_tokens_to_ids(Q_END_TOKEN)

    # Create the output path for saving test results
    folder_path = make_save_path(step_value_in_state, eval_results_folder)
    if local_rank == 0:
        print('--------save path:{}--------'.format(folder_path))

    device = f"cuda:{local_rank}"

    for batch in test_dataloader:
        batch_size = batch["input_ids"].shape[0]

        # Create these tokens at the batch level on the target device (to avoid device mismatch)
        a_start_tensor = torch.tensor([a_start_token_id], device='cuda')
        a_end_tensor = torch.tensor([a_end_token_id], device='cuda')

        # Used to store results for each sample
        answer_records = [{"sample_id": batch["sample_id"][b],
                           "label": [],
                           "pred_token": [],
                           "pred_probs": []} for b in range(batch_size)]

        # Extract feature_part, questions, and labels (ensure they are on the correct device)
        feature_parts, questions_list, answers_list = [], [], []
        for b in range(batch_size):
            input_ids = batch["input_ids"][b].to(device)
            if use_comparative_reason:
                # In comparative reasoning, multiple "<|/differentfeatures|>" may be present
                feature_end_idx = (input_ids == feature_end_token_id).nonzero(as_tuple=True)[0][-1].item()
            else:
                # Without comparative reasoning, there should be exactly one "<|/feature|>" (otherwise an error is raised)
                feature_end_idx = (input_ids == feature_end_token_id).nonzero(as_tuple=True)[0].item()

            feature_parts.append(input_ids[:feature_end_idx + 1])

            # Split question and answer
            questions = extract_segments(input_ids[feature_end_idx + 1:], q_start_token_id, q_end_token_id)
            answers = extract_segments(input_ids[feature_end_idx + 1:], a_start_token_id, a_end_token_id)

            questions_list.append(questions)
            answers_list.append([ans.tolist() for ans in answers])
            answer_records[b]["label"] = [ans.tolist() for ans in answers]

        num_questions = len(questions_list[0])
        prev_answers = [None] * batch_size

        for q_idx in range(num_questions):
            batch_input_ids = []

            for b in range(batch_size):
                q = questions_list[b][q_idx].to(device)

                if use_context_info:
                    if q_idx == 0:
                        cur_input = torch.cat([feature_parts[b].to(device), q, a_start_tensor])
                    else:
                        cur_input = torch.cat([feature_parts[b].to(device), prev_answers[b].to(device),
                                               a_end_tensor, q, a_start_tensor])
                else:
                    cur_input = torch.cat([feature_parts[b].to(device), q, a_start_tensor])

                # Update the original feature part by appending historical QA records
                feature_parts[b] = cur_input
                batch_input_ids.append(cur_input)

            # After extracting features and historical QA, sequence lengths vary, so left padding is still required
            max_len = max([x.shape[0] for x in batch_input_ids])
            padded_inputs = []
            for x in batch_input_ids:
                pad_len = max_len - x.shape[0]
                if pad_len > 0:
                    padded_x = torch.cat([
                        torch.full((pad_len,), tokenizer.pad_token_id,
                                   dtype=torch.long, device=device), x])
                else:
                    padded_x = x
                padded_inputs.append(padded_x)

            input_ids_batch = torch.stack(padded_inputs, dim=0).to(device)
            attention_mask = input_ids_batch.ne(tokenizer.pad_token_id).to(torch.long).to(device)

            with torch.no_grad():
                # outputs.logits: (batch_size, seq_len, vocab_size)
                outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask)

                # Take the last token → (batch_size, vocab_size)
                logits = outputs.logits[:, -1, :]

                # Keep only candidate tokens
                candidate_ids = candidate_token[q_idx]
                candidate_ids_tensor = torch.tensor(
                    candidate_ids,
                    device=logits.device,
                    dtype=torch.long
                )

                # (batch_size, n_candidates)
                candidate_logits = torch.index_select(
                    logits, dim=1, index=candidate_ids_tensor
                )
                probs = F.softmax(candidate_logits, dim=-1)

            for b in range(batch_size):
                batch_probs = {i: round(float(probs[b, i]), 4) for i in range(len(candidate_ids))}
                answer_records[b]["pred_probs"].append(batch_probs)

                # Select the token with the highest probability (returns index in candidate_ids)
                max_idx = torch.argmax(probs[b]).item()
                chosen_class_tokens = candidate_class[q_idx][max_idx]  # Expected to be list[int] or int

                # Convert the full token ID list into a tensor and move it to the correct device/dtype
                full_tokens = torch.tensor(chosen_class_tokens, device=device, dtype=torch.long)

                # Concatenate as the next input
                prev_answers[b] = full_tokens
                answer_records[b]["pred_token"].append(full_tokens.cpu().tolist())

        # Save results
        for record in answer_records:
            write_dict_to_json(record["sample_id"], record, folder_path)

        if local_rank == 0:
            pbar.update(world_size)

    if local_rank == 0:
        pbar.close()