import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList, kaiming_init, normal_init
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import get_box_tensor


@MODELS.register_module()
class OffsetHead(BaseModule):
    """BONAI offset regression head for MMDetection 3.x."""

    def __init__(self,
                 roi_feat_size=7,
                 in_channels=256,
                 num_convs=4,
                 num_fcs=2,
                 reg_num=2,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 offset_coordinate='rectangle',
                 offset_coder=dict(
                     type='DeltaXYOffsetCoder',
                     target_means=[0.0, 0.0],
                     target_stds=[0.5, 0.5]),
                 reg_decoded_offset=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_offset=dict(type='MSELoss', loss_weight=1.0),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.offset_coordinate = offset_coordinate
        self.reg_decoded_offset = reg_decoded_offset
        self.reg_num = reg_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.offset_coder = TASK_UTILS.build(offset_coder)
        self.loss_offset = MODELS.build(loss_offset)

        self.convs = ModuleList()
        for i in range(num_convs):
            conv_in_channels = self.in_channels if i == 0 else \
                self.conv_out_channels
            self.convs.append(
                nn.Conv2d(
                    conv_in_channels,
                    self.conv_out_channels,
                    kernel_size=3,
                    padding=1))

        roi_feat_size = _pair(roi_feat_size)
        roi_feat_area = roi_feat_size[0] * roi_feat_size[1]
        self.fcs = ModuleList()
        for i in range(num_fcs):
            fc_in_channels = self.conv_out_channels * roi_feat_area if i == 0 \
                else self.fc_out_channels
            self.fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))

        self.fc_offset = nn.Linear(self.fc_out_channels, self.reg_num)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self) -> None:
        super().init_weights()
        for conv in self.convs:
            kaiming_init(conv)
        for fc in self.fcs:
            kaiming_init(
                fc,
                a=1,
                mode='fan_in',
                nonlinearity='leaky_relu',
                distribution='uniform')
        normal_init(self.fc_offset, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(0) == 0:
            return x.new_empty((0, self.reg_num))
        for conv in self.convs:
            x = self.relu(conv(x))

        self.vis_featuremap = x.clone()
        x = torch.flatten(x, 1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        return self.fc_offset(x)

    def loss(self, offset_pred: Tensor, offset_targets: Tensor) -> dict:
        if offset_pred.size(0) == 0:
            loss_offset = offset_pred.sum() * 0
        else:
            loss_offset = self.loss_offset(
                offset_pred.float(), offset_targets.float())
        return dict(loss_offset=loss_offset)

    def loss_and_target(self, offset_pred, sampling_results,
                        batch_gt_instances, rcnn_train_cfg) -> dict:
        offset_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)
        return dict(
            loss_offset=self.loss(offset_pred, offset_targets),
            offset_targets=offset_targets)

    def _offset_target_single(self, pos_proposals, pos_assigned_gt_inds,
                              gt_offsets, cfg):
        pos_proposals_tensor = get_box_tensor(pos_proposals)
        num_pos = pos_proposals_tensor.size(0)

        if num_pos > 0:
            pos_gt_offsets = gt_offsets[pos_assigned_gt_inds.long()].to(
                pos_proposals_tensor.device).float()
            if not self.reg_decoded_offset:
                offset_targets = self.offset_coder.encode(
                    pos_proposals, pos_gt_offsets)
            else:
                offset_targets = pos_gt_offsets
        else:
            offset_targets = pos_proposals_tensor.new_zeros((0, self.reg_num))

        return offset_targets, offset_targets

    def get_targets(self,
                    sampling_results,
                    batch_gt_instances,
                    rcnn_train_cfg,
                    concat=True):
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_offsets = [
            gt_instances.offsets for gt_instances in batch_gt_instances
        ]
        offset_targets, _ = multi_apply(
            self._offset_target_single,
            pos_proposals,
            pos_assigned_gt_inds,
            gt_offsets,
            cfg=rcnn_train_cfg)

        if concat:
            offset_targets = torch.cat(offset_targets, 0)

        if self.reg_num == 2:
            return offset_targets
        if self.reg_num == 3:
            length = offset_targets[:, 0]
            angle = offset_targets[:, 1]
            return torch.stack(
                [length, torch.cos(angle),
                 torch.sin(angle)], dim=-1)
        raise RuntimeError(f'Unsupported reg_num={self.reg_num}')

    def _decode_offset_pred(self, offset_pred, det_bboxes, img_shape):
        if self.reg_num == 2:
            pred = offset_pred
        elif self.reg_num == 3:
            length = offset_pred[:, 0]
            angle = torch.atan2(offset_pred[:, 2], offset_pred[:, 1])
            pred = torch.stack([length, angle], dim=-1)
        else:
            raise RuntimeError(f'Unsupported reg_num={self.reg_num}')

        return self.offset_coder.decode(
            det_bboxes, pred, max_shape=img_shape).float()

    def get_offsets(self,
                    offset_pred,
                    det_bboxes,
                    scale_factor=None,
                    rescale=False,
                    img_shape=(1024, 1024)):
        if offset_pred is None:
            offsets = det_bboxes.new_zeros((det_bboxes.size(0), self.reg_num))
        else:
            offsets = self._decode_offset_pred(offset_pred, det_bboxes,
                                               img_shape)

        if self.offset_coordinate == 'rectangle':
            return offsets
        if self.offset_coordinate == 'polar':
            length, angle = offsets[:, 0], offsets[:, 1]
            offset_x = length * torch.cos(angle)
            offset_y = length * torch.sin(angle)
            return torch.stack([offset_x, offset_y], dim=-1)
        raise RuntimeError(
            f'Unsupported offset_coordinate={self.offset_coordinate}')

    def get_roof_footprint_bbox_offsets(self,
                                        offset_pred,
                                        det_bboxes,
                                        img_shape=(1024, 1024)):
        if offset_pred is None:
            return det_bboxes.new_zeros((det_bboxes.size(0), self.reg_num))
        return self.offset_coder.decode(
            det_bboxes, offset_pred, max_shape=img_shape)

