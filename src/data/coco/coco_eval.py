"""COCO evaluation wrapper."""
import copy
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.misc.dist import all_gather, is_main_process


class CocoEvaluator:
    def __init__(self, coco_gt: COCO, iou_types=None):
        if iou_types is None:
            iou_types = ['bbox']
        self.coco_gt = copy.deepcopy(coco_gt)
        self.iou_types = iou_types
        self.coco_eval = {}
        self.img_ids = []
        self.eval_imgs = {t: [] for t in iou_types}
        self._predictions = {}

    # ------------------------------------------------------------------
    def update(self, predictions: dict):
        """predictions: {image_id: {'boxes': Tensor xyxy, 'scores': Tensor, 'labels': Tensor}}"""
        img_ids = list(predictions.keys())
        self.img_ids.extend(img_ids)

        for img_id, pred in predictions.items():
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            if boxes.numel() == 0:
                continue
            # Convert xyxy → xywh (COCO format)
            xywh_boxes = boxes.clone()
            xywh_boxes[:, 2] -= xywh_boxes[:, 0]
            xywh_boxes[:, 3] -= xywh_boxes[:, 1]

            for box, score, label in zip(xywh_boxes.tolist(), scores.tolist(), labels.tolist()):
                self._predictions.setdefault(img_id, []).append({
                    'image_id': img_id,
                    'category_id': int(label),
                    'bbox': [round(c, 3) for c in box],
                    'score': round(float(score), 6),
                })

    def synchronize_between_processes(self):
        all_preds = all_gather(self._predictions)
        merged = {}
        for preds in all_preds:
            merged.update(preds)
        self._predictions = merged

    def accumulate(self):
        for iou_type in self.iou_types:
            coco_dt_list = []
            for preds in self._predictions.values():
                coco_dt_list.extend(preds)

            if coco_dt_list:
                coco_dt = self.coco_gt.loadRes(coco_dt_list)
            else:
                coco_dt = COCO()

            coco_eval = COCOeval(self.coco_gt, coco_dt, iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            self.coco_eval[iou_type] = coco_eval

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f'IoU metric: {iou_type}')
            coco_eval.summarize()

    def get_stats(self) -> dict:
        stats = {}
        for iou_type, coco_eval in self.coco_eval.items():
            stats[iou_type] = coco_eval.stats.tolist()
        return stats
