"""Contrastive denoising (DN) training group generation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_contrastive_denoising_training_group(
    targets,
    num_classes: int,
    num_queries: int,
    class_embed,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
):
    """Generate noisy positive/negative query groups for DN training.

    Returns:
        dn_meta: dict with denoising metadata
        input_query_class: [B, dn_num, C] embedded noisy labels (or None)
        input_query_bbox: [B, dn_num, 4] noisy boxes (or None)
        attn_mask: [total_q, total_q] attention mask (or None)
    """
    if num_denoising <= 0:
        return None, None, None, None

    # Count max objects in batch
    num_gts = [len(t['labels']) for t in targets]
    max_gt = max(num_gts) if num_gts else 0
    if max_gt == 0:
        return None, None, None, None

    # Number of DN groups (positive + negative pairs)
    dn_groups = num_denoising // (2 * max_gt)
    dn_groups = max(1, dn_groups)
    dn_num = dn_groups * 2 * max_gt  # total denoising queries

    device = targets[0]['labels'].device
    B = len(targets)

    # Build padded gt labels and boxes
    gt_labels = torch.zeros(B, max_gt, dtype=torch.int64, device=device)
    gt_boxes = torch.zeros(B, max_gt, 4, dtype=torch.float32, device=device)
    for i, t in enumerate(targets):
        n = num_gts[i]
        if n > 0:
            gt_labels[i, :n] = t['labels']
            gt_boxes[i, :n] = t['boxes']

    # Expand for dn_groups * 2 (positive + negative)
    # Shape: [B, dn_groups*2, max_gt, ...]
    gt_labels_expand = gt_labels.unsqueeze(1).repeat(1, dn_groups * 2, 1)  # [B, 2G, Ngt]
    gt_boxes_expand = gt_boxes.unsqueeze(1).repeat(1, dn_groups * 2, 1, 1)  # [B, 2G, Ngt, 4]

    # -- Label noise --
    # First dn_groups groups: positive (may flip label with probability label_noise_ratio)
    # Last dn_groups groups: negative (always flip label)
    noise_mask = torch.rand(B, dn_groups * 2, max_gt, device=device) < label_noise_ratio
    # For negative groups, always noisy
    noise_mask[:, dn_groups:, :] = True

    rand_labels = torch.randint(0, num_classes, (B, dn_groups * 2, max_gt), device=device)
    noisy_labels = torch.where(noise_mask, rand_labels, gt_labels_expand)

    # -- Box noise --
    box_noise = torch.rand_like(gt_boxes_expand) * 2 - 1  # [-1, 1]
    wh = gt_boxes_expand[..., 2:]  # w, h
    box_noise[..., :2] *= wh * box_noise_scale * 0.5
    box_noise[..., 2:] *= wh * box_noise_scale * 0.5
    noisy_boxes = (gt_boxes_expand + box_noise).clamp(0.0, 1.0)

    # Reshape: [B, dn_num]
    noisy_labels_flat = noisy_labels.view(B, dn_num)   # [B, 2G*Ngt]
    noisy_boxes_flat = noisy_boxes.view(B, dn_num, 4)  # [B, 2G*Ngt, 4]

    # Embed noisy labels via class_embed (same as decoder cls head)
    # class_embed: nn.Embedding(num_classes, hidden_dim) or linear
    with torch.no_grad():
        if isinstance(class_embed, nn.Embedding):
            input_query_class = class_embed(noisy_labels_flat)  # [B, dn_num, hidden]
        else:
            one_hot = F.one_hot(noisy_labels_flat, num_classes).float()
            input_query_class = class_embed(one_hot)

    input_query_bbox = noisy_boxes_flat  # [B, dn_num, 4]

    # -- Attention mask --
    # total queries = dn_num + num_queries
    total_q = dn_num + num_queries
    attn_mask = torch.zeros(total_q, total_q, dtype=torch.bool, device=device)

    # DN queries cannot attend to matching queries
    attn_mask[:dn_num, dn_num:] = True
    # Matching queries cannot attend to DN queries
    attn_mask[dn_num:, :dn_num] = True

    # Within DN: each group cannot attend to other groups
    group_size = max_gt * 2  # positive + negative per original group
    for g in range(dn_groups):
        start = g * group_size
        end = start + group_size
        # block other dn groups
        attn_mask[start:end, :start] = True
        attn_mask[start:end, end:dn_num] = True

    dn_meta = {
        'dn_num': dn_num,
        'dn_groups': dn_groups,
        'max_gt': max_gt,
        'num_gts': num_gts,
    }

    return dn_meta, input_query_class, input_query_bbox, attn_mask
