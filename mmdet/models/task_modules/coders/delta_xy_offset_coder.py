from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, get_box_tensor
from .base_bbox_coder import BaseBBoxCoder


@TASK_UTILS.register_module()
class DeltaXYOffsetCoder(BaseBBoxCoder):
    """Encode roof-to-footprint offsets in x/y coordinates."""

    encode_size = 2

    def __init__(self,
                 target_means: Sequence[float] = (0., 0.),
                 target_stds: Sequence[float] = (0.5, 0.5),
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes: Union[Tensor, BaseBoxes],
               gt_offsets: Tensor) -> Tensor:
        bboxes = get_box_tensor(bboxes)
        assert bboxes.size(0) == gt_offsets.size(0)
        assert gt_offsets.size(-1) == 2
        return offset2delta(bboxes, gt_offsets, self.means, self.stds)

    def decode(self,
               bboxes: Union[Tensor, BaseBoxes],
               pred_offsets: Tensor,
               max_shape: Optional[Union[Sequence[int], Tensor]] = None,
               wh_ratio_clip: float = 16 / 1000) -> Tensor:
        bboxes = get_box_tensor(bboxes)
        assert pred_offsets.size(0) == bboxes.size(0)
        return delta2offset(bboxes, pred_offsets, self.means, self.stds,
                            max_shape, wh_ratio_clip)


def offset2delta(proposals: Tensor,
                 gt: Tensor,
                 means: Sequence[float] = (0., 0.),
                 stds: Sequence[float] = (0.5, 0.5)) -> Tensor:
    proposals = proposals.float()
    gt = gt.float()

    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    dx = gt[..., 0] / pw
    dy = gt[..., 1] / ph
    deltas = torch.stack([dx, dy], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    return deltas.sub(means).div(stds)


def delta2offset(rois: Tensor,
                 deltas: Tensor,
                 means: Sequence[float] = (0., 0.),
                 stds: Sequence[float] = (1., 1.),
                 max_shape: Optional[Union[Sequence[int], Tensor]] = None,
                 wh_ratio_clip: float = 16 / 1000) -> Tensor:
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 2)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 2)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::2]
    dy = denorm_deltas[:, 1::2]

    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dx)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dy)
    gx = pw * dx
    gy = ph * dy

    if max_shape is not None:
        max_h, max_w = max_shape[:2]
        gx = gx.clamp(min=-max_w, max=max_w)
        gy = gy.clamp(min=-max_h, max=max_h)

    return torch.stack([gx, gy], dim=-1).view_as(deltas)
