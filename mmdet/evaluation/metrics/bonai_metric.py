import csv
import os.path as osp
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist
from pycocotools import mask as maskUtils

from mmdet.registry import METRICS


@METRICS.register_module()
class BONAIMetric(BaseMetric):
    """BONAI roof/footprint segmentation F1 metric.

    This is the online MMDetection 3.x counterpart of BONAI's offline
    ``tools/bonai/bonai_evaluation.py`` segmentation summary. It evaluates
    roof masks directly and footprint masks by translating predicted roof masks
    with the predicted offset. The counting rules intentionally follow the
    offline script: masks under ``min_area`` are ignored, images with empty GT
    or prediction lists are skipped by default, and IoU is computed as
    ``intersection / (union + 1)`` before thresholding.
    """

    default_prefix: Optional[str] = 'bonai'

    def __init__(self,
                 iou_thr: float = 0.5,
                 score_thr: float = 0.4,
                 min_area: int = 500,
                 mask_types: Sequence[str] = ('roof', 'footprint'),
                 skip_empty: bool = True,
                 with_offset_error: bool = True,
                 summary_file: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = iou_thr
        self.score_thr = score_thr
        self.min_area = min_area
        self.mask_types = tuple(mask_types)
        self.skip_empty = skip_empty
        self.with_offset_error = with_offset_error
        self.summary_file = summary_file

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            ori_shape = data_sample['ori_shape']
            img_h, img_w = ori_shape[:2]
            raw_instances = data_sample.get('instances', [])
            mask_types = set(self.mask_types)

            gt_masks_by_type = {
                mask_type: []
                for mask_type in self.mask_types
            }
            pred_masks_by_type = {
                mask_type: []
                for mask_type in self.mask_types
            }
            gt_footprint_offsets = []
            for instance in raw_instances:
                if instance.get('ignore_flag', 0):
                    continue
                roof_mask_ann = instance.get('roof_mask',
                                             instance.get('mask', None))
                footprint_mask_ann = instance.get('footprint_mask', None)
                if 'roof' in mask_types and roof_mask_ann is not None:
                    roof_rle = self._ann_to_rle(roof_mask_ann, img_h, img_w)
                    if self._rle_area(roof_rle) >= self.min_area:
                        gt_masks_by_type['roof'].append(roof_rle)
                if ('footprint' in mask_types
                        and footprint_mask_ann is not None):
                    footprint_rle = self._ann_to_rle(footprint_mask_ann,
                                                      img_h, img_w)
                    if self._rle_area(footprint_rle) >= self.min_area:
                        gt_masks_by_type['footprint'].append(footprint_rle)
                        gt_footprint_offsets.append(
                            instance.get('offset', [0., 0.]))

            pred = data_sample['pred_instances']
            pred_scores = self._to_numpy(pred.get('scores', []))
            pred_mask_array = self._to_numpy(pred.get('masks', []))
            pred_offsets = self._to_numpy(pred.get('offsets', []))
            pred_offsets_for_metric = []

            for idx, score in enumerate(pred_scores):
                if score < self.score_thr:
                    continue
                pred_mask = pred_mask_array[idx].astype(bool)
                if pred_mask.sum() < self.min_area:
                    continue
                if 'roof' in mask_types:
                    pred_masks_by_type['roof'].append(
                        self._mask_to_rle(pred_mask))
                if 'footprint' in mask_types:
                    if len(pred_offsets) > idx:
                        offset = pred_offsets[idx]
                    else:
                        offset = np.zeros(2, dtype=np.float32)
                    pred_offsets_for_metric.append(offset)
                    pred_masks_by_type['footprint'].append(
                        self._mask_to_rle(
                            self._translate_mask(pred_mask, -offset[0],
                                                 -offset[1])))

            image_result = {}
            for mask_type in self.mask_types:
                cur_tp, cur_fp, cur_fn, matched = self._count_image_confusion(
                    gt_masks_by_type[mask_type],
                    pred_masks_by_type[mask_type])
                image_result[f'{mask_type}_TP'] = cur_tp
                image_result[f'{mask_type}_FP'] = cur_fp
                image_result[f'{mask_type}_FN'] = cur_fn
                if (self.with_offset_error and mask_type == 'footprint'
                        and matched.size > 0):
                    offset_result = self._offset_error_sum(
                        gt_footprint_offsets, pred_offsets_for_metric,
                        matched)
                    image_result.update(offset_result)
            self.results.append(image_result)

    def compute_metrics(self, results: list) -> dict:
        logger = MMLogger.get_current_instance()
        eval_results = OrderedDict()
        summary = OrderedDict()

        for mask_type in self.mask_types:
            tp = sum(result[f'{mask_type}_TP'] for result in results)
            fp = sum(result[f'{mask_type}_FP'] for result in results)
            fn = sum(result[f'{mask_type}_FN'] for result in results)

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1_score = (2 * precision * recall / (precision + recall)
                        if precision + recall > 0 else 0.0)

            summary[mask_type] = dict(
                F1_score=f1_score,
                Precision=precision,
                Recall=recall,
                TP=tp,
                FN=fn,
                FP=fp)
            eval_results[f'{mask_type}_F1_score'] = f1_score
            eval_results[f'{mask_type}_Precision'] = precision
            eval_results[f'{mask_type}_Recall'] = recall
            eval_results[f'{mask_type}_TP'] = tp
            eval_results[f'{mask_type}_FN'] = fn
            eval_results[f'{mask_type}_FP'] = fp

        if self.with_offset_error:
            offset_count = sum(result.get('offset_count', 0)
                               for result in results)
            if offset_count > 0:
                offset_aEPE = sum(
                    result.get('offset_EPE_sum', 0.0)
                    for result in results) / offset_count
                offset_aAE = sum(
                    result.get('offset_AE_sum', 0.0)
                    for result in results) / offset_count
            else:
                offset_aEPE = 0.0
                offset_aAE = 0.0
            summary['offset'] = dict(
                aEPE=offset_aEPE,
                aAE=offset_aAE)
            eval_results['aEPE'] = offset_aEPE
            eval_results['aAE'] = offset_aAE

        self._log_summary(summary, logger)
        if self.summary_file:
            self._write_summary_csv(summary)
        return eval_results

    def _count_image_confusion(self, gt_masks, pred_masks):
        num_gts = len(gt_masks)
        num_preds = len(pred_masks)
        if self.skip_empty and (num_gts == 0 or num_preds == 0):
            return 0, 0, 0, np.empty((0, 2), dtype=np.int64)
        if num_gts == 0:
            return 0, num_preds, 0, np.empty((0, 2), dtype=np.int64)
        if num_preds == 0:
            return 0, 0, num_gts, np.empty((0, 2), dtype=np.int64)

        ious = self._bonai_iou(pred_masks, gt_masks)

        matched = np.argwhere(ious >= self.iou_thr)
        if matched.size == 0:
            return 0, num_preds, num_gts, matched

        pred_tp_indexes = matched[:, 0].tolist()
        gt_tp_indexes = matched[:, 1].tolist()
        tp = len(gt_tp_indexes)
        fp = len(set(range(num_preds)) - set(pred_tp_indexes))
        fn = len(set(range(num_gts)) - set(gt_tp_indexes))
        return tp, fp, fn, matched

    @staticmethod
    def _offset_error_sum(gt_offsets, pred_offsets, matched):
        gt_offsets = np.asarray(gt_offsets, dtype=np.float32)
        pred_offsets = np.asarray(pred_offsets, dtype=np.float32)
        matched_gt_offsets = gt_offsets[matched[:, 1]]
        matched_pred_offsets = pred_offsets[matched[:, 0]]

        error_vectors = matched_gt_offsets - matched_pred_offsets
        epe = np.sqrt(error_vectors[:, 0]**2 + error_vectors[:, 1]**2)

        gt_angle = np.arctan2(matched_gt_offsets[:, 1],
                              matched_gt_offsets[:, 0])
        pred_angle = np.arctan2(matched_pred_offsets[:, 1],
                                matched_pred_offsets[:, 0])
        angle_error = np.abs(gt_angle - pred_angle)

        return dict(
            offset_EPE_sum=float(epe.sum()),
            offset_AE_sum=float(angle_error.sum()),
            offset_count=int(matched.shape[0]))

    @staticmethod
    def _ann_to_rle(mask_ann, img_h: int, img_w: int) -> dict:
        if isinstance(mask_ann, dict):
            if isinstance(mask_ann.get('counts'), list):
                rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            else:
                rle = mask_ann
        else:
            polygons = mask_ann
            if len(polygons) > 0 and isinstance(polygons[0], (int, float)):
                polygons = [polygons]
            polygons = [
                polygon for polygon in polygons
                if len(polygon) >= 6 and len(polygon) % 2 == 0
            ]
            if len(polygons) == 0:
                return BONAIMetric._mask_to_rle(
                    np.zeros((img_h, img_w), dtype=bool))
            rle = maskUtils.merge(maskUtils.frPyObjects(polygons, img_h,
                                                        img_w))
        return rle

    @staticmethod
    def _mask_to_rle(mask: np.ndarray) -> dict:
        return maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))

    @staticmethod
    def _ensure_rle_counts_bytes(rle: dict) -> dict:
        rle = dict(rle)
        if isinstance(rle.get('counts'), str):
            rle['counts'] = rle['counts'].encode()
        return rle

    @staticmethod
    def _rle_area(rle: dict) -> float:
        return float(maskUtils.area(BONAIMetric._ensure_rle_counts_bytes(rle)))

    @staticmethod
    def _bonai_iou(pred_masks, gt_masks) -> np.ndarray:
        """Compute the same IoU denominator used by bonai_evaluation.py."""
        ious = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
        if len(pred_masks) == 0 or len(gt_masks) == 0:
            return ious

        pred_rles = [
            BONAIMetric._ensure_rle_counts_bytes(rle) for rle in pred_masks
        ]
        gt_rles = [
            BONAIMetric._ensure_rle_counts_bytes(rle) for rle in gt_masks
        ]
        pred_arrays = [maskUtils.decode(rle).astype(bool) for rle in pred_rles]
        gt_arrays = [maskUtils.decode(rle).astype(bool) for rle in gt_rles]
        pred_areas = [float(mask.sum()) for mask in pred_arrays]
        gt_areas = [float(mask.sum()) for mask in gt_arrays]

        for pred_idx, pred_mask in enumerate(pred_arrays):
            for gt_idx, gt_mask in enumerate(gt_arrays):
                intersection = float(np.logical_and(pred_mask, gt_mask).sum())
                union = pred_areas[pred_idx] + gt_areas[gt_idx] - intersection
                ious[pred_idx, gt_idx] = intersection / (union + 1.0)
        return ious

    @staticmethod
    def _translate_mask(mask: np.ndarray, x_shift, y_shift) -> np.ndarray:
        x_shift = int(np.rint(float(x_shift)))
        y_shift = int(np.rint(float(y_shift)))
        height, width = mask.shape[:2]
        shifted = np.zeros_like(mask, dtype=bool)

        src_x1 = max(0, -x_shift)
        src_x2 = min(width, width - x_shift)
        dst_x1 = max(0, x_shift)
        dst_x2 = min(width, width + x_shift)
        src_y1 = max(0, -y_shift)
        src_y2 = min(height, height - y_shift)
        dst_y1 = max(0, y_shift)
        dst_y2 = min(height, height + y_shift)

        if src_x1 < src_x2 and src_y1 < src_y2:
            shifted[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2,
                                                         src_x1:src_x2]
        return shifted

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)

    @staticmethod
    def _log_summary(summary, logger) -> None:
        for mask_type, result in summary.items():
            if mask_type == 'offset':
                logger.info(
                    'BONAI offset: '
                    f"aEPE={result['aEPE']:.4f}, "
                    f"aAE={result['aAE']:.4f}")
                continue
            logger.info(
                f'BONAI {mask_type}: '
                f"F1={result['F1_score']:.4f}, "
                f"Precision={result['Precision']:.4f}, "
                f"Recall={result['Recall']:.4f}, "
                f"TP={result['TP']}, FP={result['FP']}, FN={result['FN']}")

    def _write_summary_csv(self, summary) -> None:
        out_dir = osp.dirname(self.summary_file)
        if out_dir:
            mkdir_or_exist(out_dir)
        with open(self.summary_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Meta Info'])
            writer.writerow(['iou_thr', self.iou_thr])
            writer.writerow(['score_thr', self.score_thr])
            writer.writerow(['min_area', self.min_area])
            writer.writerow(['skip_empty', self.skip_empty])
            writer.writerow(['with_offset_error', self.with_offset_error])
            writer.writerow([])
            for mask_type, result in summary.items():
                if mask_type == 'offset':
                    continue
                writer.writerow([mask_type])
                writer.writerow([result])
                writer.writerow(['F1 Score', result['F1_score']])
                writer.writerow(['Precision', result['Precision']])
                writer.writerow(['Recall', result['Recall']])
                writer.writerow(['True Positive', result['TP']])
                writer.writerow(['False Positive', result['FP']])
                writer.writerow(['False Negative', result['FN']])
                writer.writerow([])
