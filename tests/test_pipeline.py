from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from rtdetr_pytorch.engine.trainer import evaluate_only, train_model
from rtdetr_pytorch.data.coco_prepare import prepare_dataset, processed_dataset_paths


class PipelineTestCase(unittest.TestCase):
    def _write_sample_coco(self, root: Path) -> None:
        categories = [
            {'id': 1, 'name': 'cls_a'},
            {'id': 2, 'name': 'cls_b'},
            {'id': 3, 'name': 'cls_c'},
        ]
        for split in ('train', 'valid', 'test'):
            split_dir = root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            images = []
            annotations = []
            ann_id = 1
            for idx, file_name in enumerate((f'sample_{split}_{split}.jpg', f'sample_dup.rf.{split}.jpg')):
                image_path = split_dir / file_name
                image = Image.new('RGB', (96, 64), color=(40 * (idx + 1), 20, 20))
                draw = ImageDraw.Draw(image)
                draw.rectangle((10, 10, 40, 40), outline='white', width=2)
                draw.rectangle((45, 20, 80, 50), outline='yellow', width=2)
                image.save(image_path)
                image_id = idx + 1
                images.append({'id': image_id, 'file_name': file_name, 'width': 96, 'height': 64})
                annotations.append({'id': ann_id, 'image_id': image_id, 'category_id': 1, 'bbox': [10, 10, 30, 30], 'area': 900, 'iscrowd': 0})
                ann_id += 1
                annotations.append({'id': ann_id, 'image_id': image_id, 'category_id': 2, 'bbox': [45, 20, 35, 30], 'area': 1050, 'iscrowd': 0})
                ann_id += 1
                annotations.append({'id': ann_id, 'image_id': image_id, 'category_id': 3, 'bbox': [1, 1, 3, 3], 'area': 9, 'iscrowd': 0})
                ann_id += 1
            payload = {'images': images, 'annotations': annotations, 'categories': categories}
            (split_dir / '_annotations.coco.json').write_text(json.dumps(payload), encoding='utf-8')

    def _build_config(self, temp_dir: Path) -> dict:
        return {
            'project': {'name': 'unittest', 'work_dir': str(temp_dir / 'work_dirs'), 'seed': 7},
            'dataset': {
                'input_root': str(temp_dir / 'raw'),
                'output_root': str(temp_dir / 'processed'),
                'auto_prepare': True,
                'force_rebuild': True,
                'train_ratio': 0.8,
                'split_seed': 7,
                'sample_ratio': 1.0,
                'dedup_delim': '.rf.',
                'annotation_globs': ['/_annotations.coco.json'],
                'category_id_whitelist': [],
                'category_id_blacklist': [],
                'category_name_blacklist': [],
                'max_ann_per_image': 200,
                'crop': {'enabled': True, 'size': 64, 'mode': 'center', 'seed': 7},
                'min_box_area': 25,
            },
            'model': {
                'num_classes': None,
                'backbone': 'resnet18',
                'hidden_dim': 64,
                'num_queries': 20,
                'nheads': 8,
                'num_encoder_layers': 1,
                'num_decoder_layers': 1,
                'dim_feedforward': 128,
                'dropout': 0.0,
                'use_soep': True,
                'token_pool_sizes': [4, 2, 1],
            },
            'training': {
                'epochs': 1,
                'batch_size': 1,
                'num_workers': 0,
                'amp': False,
                'eos_coef': 0.1,
                'optimizer': {'name': 'adamw', 'lr': 1e-4, 'weight_decay': 0.0},
                'scheduler': {'name': 'step', 'step_size': 1, 'gamma': 0.1},
                'matcher': {'class_cost': 1.0, 'bbox_cost': 1.0, 'giou_cost': 1.0},
                'loss_weights': {'loss_ce': 1.0, 'loss_bbox': 1.0, 'loss_giou': 1.0},
                'early_stopping': {'patience': 1},
                'augmentation': {'enabled': True, 'horizontal_flip_prob': 0.0, 'color_jitter': {}},
            },
            'evaluation': {'batch_size': 1, 'num_workers': 0, 'score_threshold': 0.01},
            'inference': {'max_detections': 20},
        }

    def test_dataset_prepare_and_training_eval(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            self._write_sample_coco(temp_dir / 'raw')
            config = self._build_config(temp_dir)
            metadata = prepare_dataset(config['dataset'])
            processed = processed_dataset_paths(config['dataset'])
            self.assertTrue(metadata['metadata'].exists())
            train_payload = json.loads(processed['train_annotations'].read_text(encoding='utf-8'))
            val_payload = json.loads(processed['val_annotations'].read_text(encoding='utf-8'))
            self.assertEqual(len(train_payload['categories']), 3)
            self.assertTrue(all(ann['area'] >= 25 for ann in train_payload['annotations']))
            self.assertTrue(all(image['width'] == 64 and image['height'] == 64 for image in train_payload['images']))
            self.assertEqual(len(train_payload['images']) + len(val_payload['images']), 4)

            config_path = temp_dir / 'config.yaml'
            config_path.write_text('project: {}\n', encoding='utf-8')
            summary = train_model(config, config_path)
            self.assertGreaterEqual(summary['best_epoch'], 1)
            run_dir = Path(summary['run_dir'])
            self.assertTrue((run_dir / 'best_map50.pt').exists())
            self.assertTrue((run_dir / 'resolved_config.yaml').exists())

            eval_config = dict(config)
            eval_config['checkpoint'] = {'path': str(run_dir / 'best_map50.pt')}
            metrics = evaluate_only(eval_config, config_path)
            self.assertIn('map50', metrics['metrics'])


if __name__ == '__main__':
    unittest.main()
