import torch
import torch.nn.functional as F

def number_token_loss(logits, labels, token_start_id=15, token_end_id=24, reduction='mean'):
    """
    logits: Tensor, shape [B, T, V] (batch, time, vocab)
    labels: Tensor, shape [B, T] (token-level label IDs, in range 15-24)
    """
    # Step 1: mask出标签位置
    mask = (labels >= token_start_id) & (labels <= token_end_id)

    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Step 2: 获取对应位置的logits，shape [N, V]
    selected_logits = logits[mask]  # shape: [N, V]
    selected_labels = labels[mask]  # shape: [N]

    # Step 3: 仅保留数字token logits（15–24），映射到 [0–9]
    number_token_ids = list(range(token_start_id, token_end_id + 1))  # 15–24
    number_values = torch.arange(0, 10, device=logits.device, dtype=torch.float)  # 0–9
    probs = F.softmax(selected_logits[:, number_token_ids], dim=-1)  # [N, 10]

    # Step 4: 计算 soft-label 的数值期望（预测值）
    pred_values = (probs * number_values).sum(dim=-1)  # [N]
    true_values = selected_labels.float() - token_start_id  # 映射成 0–9

    # Step 5: 使用 MAE 或 MSE
    if reduction == 'mean':
        loss = F.l1_loss(pred_values, true_values, reduction='mean')  # MAE
    elif reduction == 'sum':
        loss = F.l1_loss(pred_values, true_values, reduction='sum')
    else:
        loss = torch.abs(pred_values - true_values)

    return loss