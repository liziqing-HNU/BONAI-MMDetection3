import csv
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist
from pycocotools import mask as maskUtils

from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results


@METRICS.register_module()
class BONAIRLEMetric(BaseMetric):
    """BONAI metric using COCO-style RLE intermediate results.

    ``process`` only converts predictions and ground truth masks to an
    intermediate RLE representation. TP/FP/FN and offset errors are computed
    later in ``compute_metrics`` after all image results have been collected.
    The evaluation output follows BONAI's offline
    ``tools/bonai/bonai_evaluation.py``: roof/footprint segmentation metrics
    are reported as F1/Precision/Recall/TP/FN/FP, and offset evaluation reports
    only aEPE and aAE.
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
                 outfile_prefix: Optional[str] = None,
                 format_only: bool = False,
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
        self.outfile_prefix = outfile_prefix
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, (
                'outfile_prefix must be set when format_only=True.')

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            img_h, img_w = data_sample['ori_shape'][:2]
            gt = self._collect_gt(data_sample, img_h, img_w)
            pred = self._collect_pred(data_sample)
            self.results.append((gt, pred))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()
        gts, preds = zip(*results) if results else ([], [])

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        rle_file = self.results2rle(results, outfile_prefix)
        logger.info(f'BONAI RLE results have been saved to {rle_file}.')

        eval_results = OrderedDict()
        if self.format_only:
            if tmp_dir is not None:
                tmp_dir.cleanup()
            return eval_results

        summary = self._evaluate_rle(gts, preds)
        self._fill_eval_results(summary, eval_results)
        self._log_summary(summary, logger)
        if self.summary_file:
            self._write_summary_csv(summary)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def results2rle(self, results: Sequence[tuple], outfile_prefix: str) -> str:
        """Dump collected RLE intermediates to a pickle file."""
        out_file = f'{outfile_prefix}.bonai_rle.pkl'
        dump(
            dict(
                meta=dict(
                    iou_thr=self.iou_thr,
                    score_thr=self.score_thr,
                    min_area=self.min_area,
                    mask_types=self.mask_types,
                    skip_empty=self.skip_empty,
                    with_offset_error=self.with_offset_error),
                results=list(results)),
            out_file)
        return out_file

    def _collect_gt(self, data_sample: dict, img_h: int, img_w: int) -> dict:
        gt = dict(
            img_id=data_sample.get('img_id'),
            img_path=data_sample.get('img_path'),
            width=img_w,
            height=img_h,
            roof=[],
            footprint=[],
            footprint_offsets=[])

        for instance in data_sample.get('instances', []):
            if instance.get('ignore_flag', 0):
                continue
            roof_mask_ann = instance.get('roof_mask', instance.get('mask'))
            footprint_mask_ann = instance.get('footprint_mask')
            if roof_mask_ann is not None:
                roof_rle = self._ann_to_rle(roof_mask_ann, img_h, img_w)
                if self._rle_area(roof_rle) >= self.min_area:
                    gt['roof'].append(roof_rle)
            if footprint_mask_ann is not None:
                footprint_rle = self._ann_to_rle(footprint_mask_ann, img_h,
                                                  img_w)
                if self._rle_area(footprint_rle) >= self.min_area:
                    gt['footprint'].append(footprint_rle)
                    gt['footprint_offsets'].append(
                        np.asarray(instance.get('offset', [0., 0.]),
                                   dtype=np.float32))
        return gt

    def _collect_pred(self, data_sample: dict) -> dict:
        pred_instances = data_sample['pred_instances']
        pred = dict(
            img_id=data_sample.get('img_id'),
            img_path=data_sample.get('img_path'),
            scores=self._to_numpy(pred_instances.get('scores', [])),
            labels=self._to_numpy(pred_instances.get('labels', [])),
            bboxes=self._to_numpy(pred_instances.get('bboxes', [])),
            offsets=self._to_numpy(pred_instances.get('offsets', [])),
            roof=[],
            footprint=[],
            roof_areas=[])

        roof_rles = self._encode_pred_masks(pred_instances.get('masks', []))
        pred['roof'] = roof_rles
        pred['roof_areas'] = [float(maskUtils.area(rle)) for rle in roof_rles]

        pred['footprint'] = []
        for idx, roof_rle in enumerate(roof_rles):
            offset = (pred['offsets'][idx]
                      if len(pred['offsets']) > idx else np.zeros(2))
            roof_mask = maskUtils.decode(roof_rle).astype(bool)
            footprint_mask = self._translate_mask(roof_mask, -offset[0],
                                                  -offset[1])
            pred['footprint'].append(self._mask_to_rle(footprint_mask))
        return pred

    def _evaluate_rle(self, gts: Sequence[dict],
                      preds: Sequence[dict]) -> OrderedDict:
        summary = OrderedDict()
        totals = {
            mask_type: dict(TP=0, FP=0, FN=0)
            for mask_type in self.mask_types
        }
        offset_sums = dict(
            offset_EPE_sum=0.0,
            offset_AE_sum=0.0,
            offset_count=0)

        for gt, pred in zip(gts, preds):
            valid_pred_indexes = self._valid_pred_indexes(pred)
            for mask_type in self.mask_types:
                gt_masks = gt[mask_type]
                pred_masks = [
                    pred[mask_type][idx] for idx in valid_pred_indexes
                ]
                tp, fp, fn, matched = self._count_image_confusion(
                    gt_masks, pred_masks)
                totals[mask_type]['TP'] += tp
                totals[mask_type]['FP'] += fp
                totals[mask_type]['FN'] += fn
                if (self.with_offset_error and mask_type == 'footprint'
                        and matched.size > 0):
                    pred_offsets = [
                        pred['offsets'][idx] if len(pred['offsets']) > idx
                        else np.zeros(2, dtype=np.float32)
                        for idx in valid_pred_indexes
                    ]
                    offset_result = self._offset_error_sum(
                        gt['footprint_offsets'], pred_offsets, matched)
                    for key, value in offset_result.items():
                        offset_sums[key] += value

        for mask_type in self.mask_types:
            tp = totals[mask_type]['TP']
            fp = totals[mask_type]['FP']
            fn = totals[mask_type]['FN']
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

        if self.with_offset_error:
            count = offset_sums['offset_count']
            if count > 0:
                offset_aEPE = offset_sums['offset_EPE_sum'] / count
                offset_aAE = offset_sums['offset_AE_sum'] / count
            else:
                offset_aEPE = 0.0
                offset_aAE = 0.0
            summary['offset'] = dict(
                aEPE=offset_aEPE,
                aAE=offset_aAE)
        return summary

    def _valid_pred_indexes(self, pred: dict) -> list:
        valid_indexes = []
        for idx, score in enumerate(pred['scores']):
            if score < self.score_thr:
                continue
            if pred['roof_areas'][idx] < self.min_area:
                continue
            valid_indexes.append(idx)
        return valid_indexes

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
                return BONAIRLEMetric._mask_to_rle(
                    np.zeros((img_h, img_w), dtype=bool))
            rle = maskUtils.merge(maskUtils.frPyObjects(polygons, img_h,
                                                        img_w))
        return BONAIRLEMetric._ensure_rle_counts_bytes(rle)

    @staticmethod
    def _encode_pred_masks(masks):
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        elif hasattr(masks, 'masks'):
            masks = masks.masks
        if len(masks) == 0:
            return []
        first_mask = masks[0]
        if isinstance(first_mask, dict):
            return [
                BONAIRLEMetric._ensure_rle_counts_bytes(mask) for mask in masks
            ]
        return [
            BONAIRLEMetric._ensure_rle_counts_bytes(rle)
            for rle in encode_mask_results(masks)
        ]

    @staticmethod
    def _mask_to_rle(mask: np.ndarray) -> dict:
        return maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))

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
    def _ensure_rle_counts_bytes(rle: dict) -> dict:
        rle = dict(rle)
        if isinstance(rle.get('counts'), str):
            rle['counts'] = rle['counts'].encode()
        return rle

    @staticmethod
    def _rle_area(rle: dict) -> float:
        return float(
            maskUtils.area(BONAIRLEMetric._ensure_rle_counts_bytes(rle)))

    @staticmethod
    def _bonai_iou(pred_masks, gt_masks) -> np.ndarray:
        """Compute IoU with BONAI's ``union + 1`` denominator."""
        ious = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
        if len(pred_masks) == 0 or len(gt_masks) == 0:
            return ious

        pred_rles = [
            BONAIRLEMetric._ensure_rle_counts_bytes(rle) for rle in pred_masks
        ]
        gt_rles = [
            BONAIRLEMetric._ensure_rle_counts_bytes(rle) for rle in gt_masks
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
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)

    @staticmethod
    def _fill_eval_results(summary, eval_results) -> None:
        for mask_type, result in summary.items():
            if mask_type == 'offset':
                eval_results['aEPE'] = result['aEPE']
                eval_results['aAE'] = result['aAE']
                continue
            eval_results[f'{mask_type}_F1_score'] = result['F1_score']
            eval_results[f'{mask_type}_Precision'] = result['Precision']
            eval_results[f'{mask_type}_Recall'] = result['Recall']
            eval_results[f'{mask_type}_TP'] = result['TP']
            eval_results[f'{mask_type}_FN'] = result['FN']
            eval_results[f'{mask_type}_FP'] = result['FP']

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
            writer.writerow(['intermediate_format', 'RLE'])
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
