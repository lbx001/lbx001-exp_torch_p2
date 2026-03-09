"""RT-DETR loss criterion."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

__all__ = ['RTDETRCriterion']


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    """Sigmoid focal loss."""
    prob = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce_loss
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


class RTDETRCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict,
                 losses=('labels', 'boxes'), alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma

    # ------------------------------------------------------------------
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # ------------------------------------------------------------------
    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']  # [B, Q, C]
        B, Q, C = src_logits.shape
        target_classes = torch.full((B, Q), self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        idx = self._get_src_permutation_idx(indices)
        tgt_cls = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        target_classes[idx] = tgt_cls

        target_onehot = F.one_hot(target_classes.clamp(0, self.num_classes),
                                   self.num_classes + 1)[..., :self.num_classes].float()
        loss = sigmoid_focal_loss(src_logits, target_onehot,
                                   alpha=self.alpha, gamma=self.gamma, reduction='sum')
        return {'loss_cls': loss / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # [N, 4] cxcywh
        tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)])

        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction='sum') / num_boxes

        src_xyxy = box_cxcywh_to_xyxy(src_boxes).clamp(0, 1)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes).clamp(0, 1)
        loss_giou = (1 - generalized_box_iou(src_xyxy, tgt_xyxy).diag()).sum() / num_boxes

        return {'loss_bbox': loss_l1, 'loss_giou': loss_giou}

    # ------------------------------------------------------------------
    def _compute_losses(self, outputs, targets, indices, num_boxes):
        losses = {}
        for loss in self.losses:
            if loss == 'labels':
                losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            elif loss == 'boxes':
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

    # ------------------------------------------------------------------
    def forward(self, outputs, targets):
        # Main outputs (excluding aux)
        outputs_without_aux = {k: v for k, v in outputs.items()
                                if k not in ('aux_outputs', 'dn_meta', 'dn_out')}

        # Number of target boxes across batch
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = max(1, num_boxes)

        indices = self.matcher(outputs_without_aux, targets)
        losses = self._compute_losses(outputs_without_aux, targets, indices, num_boxes)

        # Auxiliary decoder outputs
        if 'aux_outputs' in outputs:
            for i, aux in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux, targets)
                aux_losses = self._compute_losses(aux, targets, aux_indices, num_boxes)
                losses.update({f'{k}_aux{i}': v for k, v in aux_losses.items()})

        # DN losses
        if 'dn_out' in outputs and outputs['dn_out'] is not None:
            dn_meta = outputs.get('dn_meta', {})
            dn_out = outputs['dn_out']
            dn_losses = self._dn_losses(dn_out, targets, dn_meta, num_boxes)
            losses.update(dn_losses)

        # Apply weights — match by longest-prefix to avoid ambiguous matches
        weighted = {}
        for k, v in losses.items():
            w = 1.0
            best_prefix_len = -1
            for wk, wv in self.weight_dict.items():
                if k == wk or k.startswith(wk + '_'):
                    if len(wk) > best_prefix_len:
                        w = wv
                        best_prefix_len = len(wk)
            weighted[k] = v * w

        return weighted

    def _dn_losses(self, dn_out, targets, dn_meta, num_boxes):
        """Compute DN losses by matching noisy queries to their GT directly."""
        if not dn_meta:
            return {}

        dn_num = dn_meta.get('dn_num', 0)
        dn_groups = dn_meta.get('dn_groups', 1)
        max_gt = dn_meta.get('max_gt', 0)
        num_gts = dn_meta.get('num_gts', [0] * len(targets))

        if dn_num == 0 or max_gt == 0:
            return {}

        B = len(targets)
        device = dn_out['pred_logits'].device
        losses = {}

        # Build direct assignment indices (no Hungarian needed)
        # Each positive group of max_gt queries is directly matched to GTs
        # (first dn_groups groups are positive, last are negative)
        dn_indices = []
        for i in range(B):
            n_gt = num_gts[i]
            if n_gt == 0:
                dn_indices.append((torch.zeros(0, dtype=torch.int64, device=device),
                                   torch.zeros(0, dtype=torch.int64, device=device)))
                continue
            src_rows, tgt_cols = [], []
            for g in range(dn_groups):
                # positive group g: queries [g*2*max_gt : g*2*max_gt + max_gt]
                offset = g * 2 * max_gt
                src_rows.append(torch.arange(offset, offset + n_gt, device=device))
                tgt_cols.append(torch.arange(n_gt, device=device))
            dn_indices.append((torch.cat(src_rows), torch.cat(tgt_cols)))

        # Compute losses
        dn_num_boxes = max(1, sum(num_gts) * dn_groups)
        losses_dn = self._compute_losses(dn_out, targets, dn_indices, dn_num_boxes)
        return {f'{k}_dn': v for k, v in losses_dn.items()}
