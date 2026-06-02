# BONAI MMDetection 3.3 迁移说明

本仓库在 MMDetection 3.3.0 上移植了 BONAI/LOFT 的主要训练、测试和评估流程，测试结果与原文指标相符。

移植代码使用codex辅助进行

## 已移植内容

- BONAI 数据集适配：`mmdet/datasets/bonai.py`
- BONAI 数据增强与打包：
  - `LoadBONAIAnnotations`
  - `BONAIRandomFlip`
  - `PackBONAIInputs`
- LOFT/offset 分支相关模型：
  - `mmdet/models/detectors/loft.py`
  - `mmdet/models/roi_heads/loft_roi_head.py`
  - `mmdet/models/roi_heads/attribute_heads/offset_head.py`
  - `mmdet/models/roi_heads/attribute_heads/offset_head_expand_feature.py`
  - `mmdet/models/task_modules/coders/delta_xy_offset_coder.py`
  - `mmdet/models/task_modules/coders/delta_polar_offset_coder.py`
- BONAI 在线评估：
  - `mmdet/evaluation/metrics/bonai_metric.py`
  - `mmdet/evaluation/metrics/bonai_rle_metric.py`
- BONAI-DINO 实验模型：
  - `projects/BONAIDINO/`
- 原版风格离线评估脚本：
  - `tools/bonai/offline_evaluation.py`

## 数据目录

默认配置假设数据放在：

```text
data/BONAI/
  coco/
    bonai_shanghai_trainval.json
    bonai_beijing_trainval.json
    bonai_jinan_trainval.json
    bonai_haerbin_trainval.json
    bonai_chengdu_trainval.json
    bonai_shanghai_xian_test.json
  trainval/images/
  test/images/
  csv/
    shanghai_xian_v3_merge_val_roof_crop1024_gt_minarea500.csv
    shanghai_xian_v3_merge_val_footprint_crop1024_gt_minarea500.csv
```

COCO JSON 中需要包含 BONAI 字段，例如 `building_bbox`、`footprint_bbox`、`roof_mask`、`footprint_mask`、`offset` 等。当前新框架在线评估直接使用 COCO JSON 中的标签；`csv/` 目录只用于复刻 BONAI 原版离线评估流程。

## 配置文件

LOFT 配置：

```text
configs/bonai/loft_foa_r50_fpn_2x_bonai.py
```


LOFT 配置默认训练五个 trainval 城市，并使用 `bonai_shanghai_xian_test.json` 作为测试集。

## 训练

训练 LOFT：

```bash
python tools/train.py \
  configs/bonai/loft_foa_r50_fpn_2x_bonai.py \
  --work-dir work_dirs/loft_foa_r50_fpn_2x_bonai
```



如果显存不足，可以先减小配置里的 `train_dataloader.batch_size` 或 `num_workers`。

## 测试与在线评估

测试 LOFT epoch 24，并把 BONAI RLE 中间结果和 summary 写到本次测试目录：

```bash
python tools/test.py \
  configs/bonai/loft_foa_r50_fpn_2x_bonai.py \
  work_dirs/loft_foa_r50_fpn_2x_bonai/epoch_24.pth \
  --work-dir work_dirs/loft_foa_r50_fpn_2x_bonai/test_epoch24 \
  --cfg-options \
  test_evaluator.0.outfile_prefix=work_dirs/loft_foa_r50_fpn_2x_bonai/test_epoch24/bonai_test_rle \
  test_evaluator.0.summary_file=work_dirs/loft_foa_r50_fpn_2x_bonai/test_epoch24/bonai_online_summary.csv
```

`BONAIRLEMetric` 会输出：

- `roof_F1_score`
- `roof_Precision`
- `roof_Recall`
- `roof_TP`
- `roof_FN`
- `roof_FP`
- `footprint_F1_score`
- `footprint_Precision`
- `footprint_Recall`
- `footprint_TP`
- `footprint_FN`
- `footprint_FP`
- `aEPE`
- `aAE`

同时会保存：

```text
work_dirs/loft_foa_r50_fpn_2x_bonai/test_epoch24/bonai_test_rle.bonai_rle.pkl
work_dirs/loft_foa_r50_fpn_2x_bonai/test_epoch24/bonai_online_summary.csv
```

## 只保存 pkl，不跑在线评估

如果只想保存测试结果，避免在线评估耗时，可以设置 `format_only=True`：

```bash
python tools/test.py \
  configs/bonai/loft_foa_r50_fpn_2x_bonai.py \
  work_dirs/loft_foa_r50_fpn_2x_bonai/epoch_24.pth \
  --work-dir work_dirs/loft_foa_r50_fpn_2x_bonai/test_epoch24 \
  --cfg-options \
  test_evaluator.0.format_only=True \
  test_evaluator.0.outfile_prefix=work_dirs/loft_foa_r50_fpn_2x_bonai/test_epoch24/bonai_test_rle
```

这种方式只保存：

```text
work_dirs/loft_foa_r50_fpn_2x_bonai/test_epoch24/bonai_test_rle.bonai_rle.pkl
```

## 原版风格离线评估

原版 BONAI 的离线评估流程是：

```text
预测 pkl -> roof/footprint 预测 CSV -> 与 GT CSV 做 polygon IoU 匹配 -> 输出 F1/Precision/Recall
```

迁移版提供了兼容脚本：

```bash
python tools/bonai/offline_evaluation.py \
  --result work_dirs/loft_foa_r50_fpn_2x_bonai/test_epoch24/bonai_test_rle.bonai_rle.pkl \
  --out-dir work_dirs/loft_foa_r50_fpn_2x_bonai/test_epoch24/offline_eval
```

默认读取：

```text
data/BONAI/csv/shanghai_xian_v3_merge_val_roof_crop1024_gt_minarea500.csv
data/BONAI/csv/shanghai_xian_v3_merge_val_footprint_crop1024_gt_minarea500.csv
```

输出目录包含：

```text
pred_roof.csv
pred_footprint.csv
bonai_offline_summary.csv
bonai_offline_results.json
```

如果已经有预测 CSV，也可以直接评估 CSV：

```bash
python tools/bonai/offline_evaluation.py \
  --pred-roof-csv path/to/pred_roof.csv \
  --pred-footprint-csv path/to/pred_footprint.csv \
  --out-dir work_dirs/bonai_offline_eval
```

## 在线评估与离线评估的区别

在线评估：

- 直接使用 COCO JSON 中的 GT。
- 直接使用预测 RLE。
- 不需要 GT CSV。
- 适合训练和常规测试。

离线评估：

- 使用 BONAI 原版的 GT CSV。
- 会将预测结果转成 roof/footprint CSV。
- 更接近原版 `BONAI-master/tools/bonai/bonai_evaluation.py` 的工作流。
- 适合和原论文/原工程结果对齐。

两者的目标指标是一致的：roof/footprint 的 F1、Precision、Recall、TP、FN、FP，以及 offset 的 aEPE、aAE。

## 常见问题

### pkl 保存很久没有反应

`*.bonai_rle.pkl` 保存通常只需要几秒。测试结束后如果还在等待，多数情况下是在继续跑在线评估，不是卡在保存 pkl。只需要 pkl 时使用：

```bash
--cfg-options test_evaluator.0.format_only=True
```

### pkl 默认写到了 work_dir 外面

配置里的 `test_evaluator.0.outfile_prefix` 是固定路径。如果不覆盖，它会写到配置默认目录。建议测试时显式指定：

```bash
--cfg-options test_evaluator.0.outfile_prefix=你的测试目录/bonai_test_rle
```

### 为什么数据集里有 CSV

CSV 不是新框架在线评估必须的标签文件。测试标签已经在 COCO JSON 里。CSV 是为了复刻 BONAI 原版离线评估：原版会把 GT 和预测都表示成 polygon CSV，再做实例级 IoU 匹配。


## 依赖提示

常规 MMDetection 3.3 环境需要安装 `mmcv`、`mmengine`、`mmdet` 依赖。BONAI 评估还依赖 `pycocotools`。离线 CSV polygon 评估会优先使用 `shapely`；没有安装时会退回 raster 后端，但从 `*.bonai_rle.pkl` 生成 CSV 仍需要 `pycocotools`。

