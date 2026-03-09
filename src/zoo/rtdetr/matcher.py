"""Hungarian matcher for RT-DETR."""
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

__all__ = ['HungarianMatcher']


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2., cost_bbox=5., cost_giou=2.,
                 use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        B, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].flatten(0, 1)   # [B*Q, C]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)    # [B*Q, 4]

        if self.use_focal_loss:
            out_prob = out_prob.sigmoid()
        else:
            out_prob = out_prob.softmax(-1)

        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_bbox = torch.cat([t['boxes'] for t in targets])

        if self.use_focal_loss:
            neg_cost = -(1 - out_prob + 1e-8).log() * (1 - self.alpha) * out_prob.pow(self.gamma)
            pos_cost = -(out_prob + 1e-8).log() * self.alpha * (1 - out_prob).pow(self.gamma)
            cost_class = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Avoid NaN when boxes are all zeros
        out_xyxy = box_cxcywh_to_xyxy(out_bbox).clamp(0, 1)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox).clamp(0, 1)
        cost_giou = -generalized_box_iou(out_xyxy, tgt_xyxy)

        C = (self.cost_class * cost_class
             + self.cost_bbox * cost_bbox
             + self.cost_giou * cost_giou)
        C = C.view(B, num_queries, -1).cpu()

        sizes = [len(t['boxes']) for t in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            row, col = linear_sum_assignment(c[i])
            indices.append((
                torch.as_tensor(row, dtype=torch.int64),
                torch.as_tensor(col, dtype=torch.int64),
            ))
        return indices
