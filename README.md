# lbx001-exp_torch_p2

基于 PyTorch 的轻量 RT-DETR 风格训练脚手架，包含：
- COCO 数据集合并 / 去重 / 抽样 / 切分 / 裁剪 / 小框过滤
- 带 SOEP 模块的 RT-DETR 风格检测模型
- `python train.py` / `python eval.py` 配置化训练与评估入口
- work_dir 自动保存日志、原始配置副本、resolved 配置和最佳权重

默认配置文件位于仓库根目录：
- `/home/runner/work/lbx001-exp_torch_p2/lbx001-exp_torch_p2/config.yaml`
- `/home/runner/work/lbx001-exp_torch_p2/lbx001-exp_torch_p2/eval_config.yaml`

如需在本地临时切换配置，可使用环境变量：
- `RTDETR_CONFIG_PATH=/abs/path/to/config.yaml python train.py`
- `RTDETR_EVAL_CONFIG_PATH=/abs/path/to/eval_config.yaml python eval.py`
