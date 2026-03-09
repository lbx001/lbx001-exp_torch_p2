"""RT-DETR post-processor."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy

__all__ = ['RTDETRPostProcessor']


class RTDETRPostProcessor(nn.Module):
    def __init__(self, num_classes: int = 80, use_focal_loss: bool = True,
                 num_top_queries: int = 300):
        super().__init__()
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries

    @torch.no_grad()
    def forward(self, outputs, target_sizes: torch.Tensor):
        """
        outputs: dict with 'pred_logits' [B, Q, C], 'pred_boxes' [B, Q, 4] cxcywh-norm
        target_sizes: [B, 2] (H, W) original image sizes

        Returns: list of dicts, each with 'boxes' (xyxy), 'scores', 'labels'
        """
        logits = outputs['pred_logits']   # [B, Q, C]
        boxes = outputs['pred_boxes']     # [B, Q, 4]

        if self.use_focal_loss:
            scores_all = logits.sigmoid()  # [B, Q, C]
        else:
            scores_all = logits.softmax(-1)[..., :-1]  # [B, Q, C]

        # Top-k over all (query, class) combinations
        B, Q, C = scores_all.shape
        k = min(self.num_top_queries, Q * C)
        scores_flat = scores_all.flatten(1)  # [B, Q*C]
        topk_scores, topk_idx = scores_flat.topk(k, dim=1)

        topk_labels = topk_idx % C
        topk_query_idx = topk_idx // C

        # Gather boxes
        topk_boxes = torch.gather(
            boxes, 1,
            topk_query_idx.unsqueeze(-1).expand(-1, -1, 4)
        )  # [B, k, 4] cxcywh norm

        topk_boxes_xyxy = box_cxcywh_to_xyxy(topk_boxes)  # [B, k, 4] norm

        results = []
        for i in range(B):
            h, w = float(target_sizes[i, 0]), float(target_sizes[i, 1])
            scale = torch.tensor([w, h, w, h], device=boxes.device)
            scaled_boxes = topk_boxes_xyxy[i] * scale
            results.append({
                'boxes': scaled_boxes,
                'scores': topk_scores[i],
                'labels': topk_labels[i],
            })
        return results
