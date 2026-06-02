import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule, ModuleList, kaiming_init, normal_init
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import get_box_tensor
from .offset_head import OffsetHead


@MODELS.register_module()
class OffsetHeadExpandFeature(OffsetHead):
    """Feature-orientation augmentation offset head used by BONAI LOFT."""

    def __init__(self,
                 roi_feat_size=7,
                 in_channels=256,
                 num_convs=4,
                 num_fcs=2,
                 reg_num=2,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 expand_feature_num=4,
                 share_expand_fc=False,
                 rotations=(0, 90, 180, 270),
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
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.offset_coordinate = offset_coordinate
        self.reg_decoded_offset = reg_decoded_offset
        self.reg_num = reg_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.expand_feature_num = expand_feature_num
        self.share_expand_fc = share_expand_fc
        self.rotations = list(rotations)

        self.offset_coder = TASK_UTILS.build(offset_coder)
        self.loss_offset = MODELS.build(loss_offset)

        self.expand_convs = ModuleList()
        for _ in range(self.expand_feature_num):
            convs = ModuleList()
            for i in range(num_convs):
                conv_in_channels = self.in_channels if i == 0 else \
                    self.conv_out_channels
                convs.append(
                    nn.Conv2d(
                        conv_in_channels,
                        self.conv_out_channels,
                        kernel_size=3,
                        padding=1))
            self.expand_convs.append(convs)

        roi_feat_size = _pair(roi_feat_size)
        roi_feat_area = roi_feat_size[0] * roi_feat_size[1]
        if not self.share_expand_fc:
            self.expand_fcs = ModuleList()
            self.expand_fc_offsets = ModuleList()
            for _ in range(self.expand_feature_num):
                fcs = ModuleList()
                for i in range(num_fcs):
                    fc_in_channels = self.conv_out_channels * roi_feat_area \
                        if i == 0 else self.fc_out_channels
                    fcs.append(nn.Linear(fc_in_channels,
                                         self.fc_out_channels))
                self.expand_fcs.append(fcs)
                self.expand_fc_offsets.append(
                    nn.Linear(self.fc_out_channels, self.reg_num))
        else:
            self.fcs = ModuleList()
            for i in range(num_fcs):
                fc_in_channels = self.conv_out_channels * roi_feat_area \
                    if i == 0 else self.fc_out_channels
                self.fcs.append(nn.Linear(fc_in_channels,
                                          self.fc_out_channels))
            self.fc_offset = nn.Linear(self.fc_out_channels, self.reg_num)

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self) -> None:
        BaseModule.init_weights(self)
        for convs in self.expand_convs:
            for conv in convs:
                kaiming_init(conv)
        if not self.share_expand_fc:
            for fcs in self.expand_fcs:
                for fc in fcs:
                    kaiming_init(
                        fc,
                        a=1,
                        mode='fan_in',
                        nonlinearity='leaky_relu',
                        distribution='uniform')
            for fc_offset in self.expand_fc_offsets:
                normal_init(fc_offset, std=0.01)
        else:
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

        input_feature = x
        offsets = []
        for idx in range(self.expand_feature_num):
            branch_x = self.expand_feature(input_feature, idx)
            for conv in self.expand_convs[idx]:
                branch_x = self.relu(conv(branch_x))

            branch_x = torch.flatten(branch_x, 1)
            if not self.share_expand_fc:
                for fc in self.expand_fcs[idx]:
                    branch_x = self.relu(fc(branch_x))
                offset = self.expand_fc_offsets[idx](branch_x)
            else:
                for fc in self.fcs:
                    branch_x = self.relu(fc(branch_x))
                offset = self.fc_offset(branch_x)
            offsets.append(offset)

        return torch.cat(offsets, dim=0)

    def expand_feature(self, feature: Tensor, operation_idx: int) -> Tensor:
        if operation_idx >= 4:
            raise NotImplementedError

        rotate_angle = self.rotations[operation_idx]
        angle = rotate_angle * math.pi / 180.0
        theta = feature.new_zeros((feature.size(0), 2, 3))
        theta[:, 0, 0] = math.cos(angle)
        theta[:, 0, 1] = math.sin(-angle)
        theta[:, 1, 0] = math.sin(angle)
        theta[:, 1, 1] = math.cos(angle)

        grid = F.affine_grid(theta, feature.size(), align_corners=False)
        return F.grid_sample(feature, grid, align_corners=False)

    def expand_gt_offset(self, gt_offset: Tensor,
                         operation_idx: int) -> Tensor:
        if operation_idx >= 4:
            raise NotImplementedError

        rotate_angle = self.rotations[operation_idx]
        angle = rotate_angle * math.pi / 180.0
        offset_x = gt_offset[:, 0]
        offset_y = gt_offset[:, 1]
        transformed_x = offset_x * math.cos(angle) + offset_y * math.sin(
            angle)
        transformed_y = -offset_x * math.sin(angle) + offset_y * math.cos(
            angle)
        return torch.stack([transformed_x, transformed_y], dim=-1)

    def _offset_target_single(self, pos_proposals, pos_assigned_gt_inds,
                              gt_offsets, cfg, operation_idx):
        pos_proposals_tensor = get_box_tensor(pos_proposals)
        num_pos = pos_proposals_tensor.size(0)

        if num_pos > 0:
            pos_gt_offsets = gt_offsets[pos_assigned_gt_inds.long()].to(
                pos_proposals_tensor.device).float()
            pos_gt_offsets = self.expand_gt_offset(pos_gt_offsets,
                                                   operation_idx)

            if not self.reg_decoded_offset:
                if self.rotations[operation_idx] in (90, 270):
                    offset_targets = self.offset_coder.encode(
                        pos_proposals, pos_gt_offsets[:, [1, 0]])
                    offset_targets = offset_targets[:, [1, 0]]
                else:
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

        expand_offset_targets = []
        for idx in range(self.expand_feature_num):
            offset_targets, _ = multi_apply(
                self._offset_target_single,
                pos_proposals,
                pos_assigned_gt_inds,
                gt_offsets,
                cfg=rcnn_train_cfg,
                operation_idx=idx)
            if concat:
                offset_targets = torch.cat(offset_targets, 0)
            expand_offset_targets.append(offset_targets)

        return torch.cat(expand_offset_targets, 0)

    def offset_fusion(self, offset_pred: Tensor, model='max') -> Tensor:
        split_offsets = offset_pred.split(
            int(offset_pred.shape[0] / self.expand_feature_num), dim=0)
        main_offsets = split_offsets[0]
        if model == 'mean':
            offset_values = 0
            for idx in range(self.expand_feature_num):
                if self.rotations[idx] in (90, 270):
                    current_offsets = split_offsets[idx][:, [1, 0]]
                elif self.rotations[idx] in (0, 180):
                    current_offsets = split_offsets[idx]
                else:
                    raise NotImplementedError(
                        f'rotation angle: {self.rotations[idx]}')
                offset_values += torch.abs(current_offsets)
            offset_values /= 1
        elif model == 'max':
            if self.expand_feature_num == 2 and self.rotations == [0, 180]:
                offset_value_x = torch.stack(
                    [split_offsets[0][:, 0], split_offsets[1][:, 0]], dim=1)
                offset_value_y = torch.stack(
                    [split_offsets[0][:, 1], split_offsets[1][:, 1]], dim=1)
            elif self.expand_feature_num == 2 and self.rotations == [0, 90]:
                offset_value_x = torch.stack(
                    [split_offsets[0][:, 0], split_offsets[1][:, 1]], dim=1)
                offset_value_y = torch.stack(
                    [split_offsets[0][:, 1], split_offsets[1][:, 0]], dim=1)
            elif self.expand_feature_num == 3 and self.rotations == [
                    0, 90, 180
            ]:
                offset_value_x = torch.stack([
                    split_offsets[0][:, 0], split_offsets[1][:, 1],
                    split_offsets[2][:, 0]
                ],
                                             dim=1)
                offset_value_y = torch.stack([
                    split_offsets[0][:, 1], split_offsets[1][:, 0],
                    split_offsets[2][:, 1]
                ],
                                             dim=1)
            elif self.expand_feature_num == 4:
                offset_value_x = torch.stack([
                    split_offsets[0][:, 0], split_offsets[1][:, 1],
                    split_offsets[2][:, 0], split_offsets[3][:, 1]
                ],
                                             dim=1)
                offset_value_y = torch.stack([
                    split_offsets[0][:, 1], split_offsets[1][:, 0],
                    split_offsets[2][:, 1], split_offsets[3][:, 0]
                ],
                                             dim=1)
            else:
                raise NotImplementedError

            offset_values = torch.stack([
                torch.max(torch.abs(offset_value_x), dim=1)[0],
                torch.max(torch.abs(offset_value_y), dim=1)[0]
            ],
                                        dim=1)
        else:
            raise NotImplementedError

        offset_polarity = torch.zeros_like(main_offsets)
        offset_polarity[main_offsets > 0] = 1
        offset_polarity[main_offsets <= 0] = -1
        return offset_values * offset_polarity

    def get_offsets(self,
                    offset_pred,
                    det_bboxes,
                    scale_factor=None,
                    rescale=False,
                    img_shape=(1024, 1024)):
        if offset_pred is None:
            offsets = det_bboxes.new_zeros((det_bboxes.size(0), self.reg_num))
        else:
            fused_pred = self.offset_fusion(offset_pred)
            offsets = self.offset_coder.decode(
                det_bboxes, fused_pred, max_shape=img_shape).float()

        if self.offset_coordinate == 'rectangle':
            return offsets
        if self.offset_coordinate == 'polar':
            length, angle = offsets[:, 0], offsets[:, 1]
            offset_x = length * torch.cos(angle)
            offset_y = length * torch.sin(angle)
            return torch.stack([offset_x, offset_y], dim=-1)
        raise RuntimeError(
            f'Unsupported offset_coordinate={self.offset_coordinate}')
