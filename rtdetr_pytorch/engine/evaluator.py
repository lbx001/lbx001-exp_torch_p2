from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import torch



def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = boxes.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)



def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32, device=boxes1.device)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)



def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou = box_iou(boxes1, boxes2)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return iou
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt_inter = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb_inter = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_inter = (rb_inter - lt_inter).clamp(min=0)
    inter = wh_inter[..., 0] * wh_inter[..., 1]
    union = area1[:, None] + area2 - inter
    return iou - (area - union) / area.clamp(min=1e-6)



def decode_predictions(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]], score_threshold: float = 0.05) -> List[Dict[str, torch.Tensor]]:
    probs = outputs['pred_logits'].softmax(-1)
    scores, labels = probs[..., :-1].max(-1)
    boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])
    predictions = []
    for batch_idx, target in enumerate(targets):
        height, width = target['orig_size'].tolist()
        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
        image_scores = scores[batch_idx]
        keep = image_scores >= score_threshold
        image_boxes = boxes[batch_idx][keep] * scale
        image_boxes[:, 0::2] = image_boxes[:, 0::2].clamp(min=0, max=width)
        image_boxes[:, 1::2] = image_boxes[:, 1::2].clamp(min=0, max=height)
        predictions.append(
            {
                'scores': image_scores[keep].detach().cpu(),
                'labels': labels[batch_idx][keep].detach().cpu(),
                'boxes': image_boxes.detach().cpu(),
            }
        )
    return predictions



def _compute_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for idx in range(precisions.numel() - 1, 0, -1):
        precisions[idx - 1] = torch.maximum(precisions[idx - 1], precisions[idx])
    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap.item())



def evaluate_map50(model: torch.nn.Module, data_loader, device: torch.device, score_threshold: float = 0.05) -> Dict[str, Any]:
    model.eval()
    preds_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    gts_by_image_class: dict[tuple[int, int], dict[str, Any]] = {}

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = decode_predictions(outputs, targets, score_threshold=score_threshold)
            for target, prediction in zip(targets, predictions):
                image_id = int(target['image_id'])
                gt_boxes = target['boxes_xyxy'].cpu()
                gt_labels = target['labels'].cpu()
                for class_id in gt_labels.unique().tolist() if len(gt_labels) else []:
                    class_mask = gt_labels == class_id
                    gts_by_image_class[(image_id, int(class_id))] = {
                        'boxes': gt_boxes[class_mask],
                        'matched': torch.zeros(class_mask.sum(), dtype=torch.bool),
                    }
                for score, label, box in zip(prediction['scores'], prediction['labels'], prediction['boxes']):
                    preds_by_class[int(label)].append({'image_id': image_id, 'score': float(score), 'box': box})

    aps = []
    per_class = {}
    class_ids = sorted(set(list(preds_by_class.keys()) + [key[1] for key in gts_by_image_class.keys()]))
    for class_id in class_ids:
        predictions = sorted(preds_by_class.get(class_id, []), key=lambda item: item['score'], reverse=True)
        total_gts = sum(value['boxes'].shape[0] for key, value in gts_by_image_class.items() if key[1] == class_id)
        if total_gts == 0:
            continue
        tps = torch.zeros(len(predictions))
        fps = torch.zeros(len(predictions))
        for idx, pred in enumerate(predictions):
            gt_info = gts_by_image_class.get((pred['image_id'], class_id))
            if gt_info is None or gt_info['boxes'].numel() == 0:
                fps[idx] = 1
                continue
            ious = box_iou(pred['box'].unsqueeze(0), gt_info['boxes']).squeeze(0)
            best_iou, best_idx = (ious.max(dim=0) if ious.numel() else (torch.tensor(0.0), torch.tensor(0)))
            if best_iou >= 0.5 and not gt_info['matched'][best_idx]:
                gt_info['matched'][best_idx] = True
                tps[idx] = 1
            else:
                fps[idx] = 1
        cum_tp = torch.cumsum(tps, dim=0)
        cum_fp = torch.cumsum(fps, dim=0)
        recalls = cum_tp / max(total_gts, 1)
        precisions = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-6)
        ap = _compute_ap(recalls, precisions) if len(predictions) else 0.0
        aps.append(ap)
        per_class[class_id] = {'ap50': ap, 'gt_count': total_gts, 'pred_count': len(predictions)}

    map50 = sum(aps) / len(aps) if aps else 0.0
    return {'map50': map50, 'per_class': per_class}
