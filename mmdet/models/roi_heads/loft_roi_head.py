from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class LoftRoIHead(StandardRoIHead):
    """Standard RoI head with BONAI's offset regression branch."""

    def __init__(self,
                 offset_roi_extractor: Optional[ConfigType] = None,
                 offset_head: Optional[ConfigType] = None,
                 **kwargs) -> None:
        assert offset_head is not None
        super().__init__(**kwargs)
        self.init_offset_head(offset_roi_extractor, offset_head)

    @property
    def with_offset(self) -> bool:
        return hasattr(self, 'offset_head') and self.offset_head is not None

    def init_offset_head(self, offset_roi_extractor: Optional[ConfigType],
                         offset_head: ConfigType) -> None:
        if offset_roi_extractor is not None:
            self.offset_roi_extractor = MODELS.build(offset_roi_extractor)
        else:
            self.offset_roi_extractor = self.bbox_roi_extractor
        self.offset_head = MODELS.build(offset_head)

    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None) -> tuple:
        results = super().forward(x, rpn_results_list, batch_data_samples)
        if self.with_offset:
            proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
            rois = bbox2roi(proposals)
            offset_results = self._offset_forward(x, rois)
            results = results + (offset_results['offset_pred'], )
        return results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        if self.with_offset:
            offset_results = self.offset_loss(x, sampling_results,
                                              batch_gt_instances)
            losses.update(offset_results['loss_offset'])

        return losses

    def offset_loss(self, x: Tuple[Tensor], sampling_results,
                    batch_gt_instances: InstanceList) -> dict:
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        offset_results = self._offset_forward(x, pos_rois)

        offset_loss_and_target = self.offset_head.loss_and_target(
            offset_pred=offset_results['offset_pred'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)
        offset_results.update(
            loss_offset=offset_loss_and_target['loss_offset'],
            offset_targets=offset_loss_and_target['offset_targets'])
        return offset_results

    def _offset_forward(self,
                        x: Tuple[Tensor],
                        rois: Tensor = None,
                        pos_inds: Optional[Tensor] = None,
                        bbox_feats: Optional[Tensor] = None) -> dict:
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            offset_feats = self.offset_roi_extractor(
                x[:self.offset_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                offset_feats = self.shared_head(offset_feats)
        else:
            assert bbox_feats is not None
            offset_feats = bbox_feats[pos_inds]

        offset_pred = self.offset_head(offset_feats)
        return dict(offset_pred=offset_pred, offset_feats=offset_feats)

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=False)

        if self.with_offset:
            results_list = self.predict_offset(
                x, batch_img_metas, results_list, rescale=rescale)

        if self.with_mask:
            results_list = self.predict_mask(
                x, batch_img_metas, results_list, rescale=rescale)
        elif rescale:
            self._rescale_bboxes(results_list, batch_img_metas)

        return results_list

    def predict_offset(self,
                       x: Tuple[Tensor],
                       batch_img_metas: List[dict],
                       results_list: InstanceList,
                       rescale: bool = False) -> InstanceList:
        bboxes = [res.bboxes for res in results_list]
        offset_rois = bbox2roi(bboxes)
        num_rois_per_img = [len(res) for res in results_list]

        if offset_rois.shape[0] == 0:
            for results in results_list:
                results.offsets = offset_rois.new_zeros((0, 2))
            return results_list

        offset_results = self._offset_forward(x, offset_rois)
        offset_preds = self._split_offset_preds(offset_results['offset_pred'],
                                                num_rois_per_img)

        for img_id, results in enumerate(results_list):
            offsets = self.offset_head.get_offsets(
                offset_preds[img_id],
                results.bboxes,
                img_shape=batch_img_metas[img_id]['img_shape'])
            if rescale and offsets.numel() > 0:
                scale_factor = offsets.new_tensor(
                    batch_img_metas[img_id]['scale_factor'])
                if scale_factor.numel() == 4:
                    scale_factor = scale_factor[:2]
                offsets = offsets / scale_factor
            results.offsets = offsets
        return results_list

    def _split_offset_preds(self, offset_pred: Tensor,
                            num_rois_per_img: List[int]) -> List[Tensor]:
        expand_feature_num = getattr(self.offset_head, 'expand_feature_num', 1)
        if expand_feature_num == 1:
            return list(offset_pred.split(num_rois_per_img, 0))

        total_rois = sum(num_rois_per_img)
        branch_preds = offset_pred.split(total_rois, 0)
        branch_splits = [
            branch_pred.split(num_rois_per_img, 0)
            for branch_pred in branch_preds
        ]
        offset_preds = []
        for img_id in range(len(num_rois_per_img)):
            offset_preds.append(
                torch.cat(
                    [branch[img_id] for branch in branch_splits], dim=0))
        return offset_preds

    @staticmethod
    def _rescale_bboxes(results_list: InstanceList,
                        batch_img_metas: List[dict]) -> None:
        for img_id, results in enumerate(results_list):
            if results.bboxes.numel() == 0:
                continue
            scale_factor = results.bboxes.new_tensor(
                batch_img_metas[img_id]['scale_factor'])
            if scale_factor.numel() == 2:
                scale_factor = scale_factor.repeat(2)
            results.bboxes /= scale_factor
