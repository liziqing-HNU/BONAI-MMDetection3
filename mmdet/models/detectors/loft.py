import math

import numpy as np

from mmdet.registry import MODELS
from .two_stage import TwoStageDetector


@MODELS.register_module()
class LOFT(TwoStageDetector):
    """LOFT detector wrapper for MMDetection 3.x.

    The training and prediction behavior is provided by ``TwoStageDetector`` and
    the custom ``LoftRoIHead``. The helper methods are kept for compatibility
    with BONAI's offset post-processing utilities.
    """

    @staticmethod
    def offset_coordinate_transform(offset, transform_flag='xy2la'):
        if transform_flag == 'xy2la':
            offset_x, offset_y = offset
            length = math.sqrt(offset_x**2 + offset_y**2)
            angle = math.atan2(offset_y, offset_x)
            return [length, angle]
        if transform_flag == 'la2xy':
            length, angle = offset
            offset_x = length * np.cos(angle)
            offset_y = length * np.sin(angle)
            return [offset_x, offset_y]
        raise NotImplementedError

    def offset_rotate(self, offsets, rotate_angle):
        offsets = [
            self.offset_coordinate_transform(offset, transform_flag='xy2la')
            for offset in offsets
        ]
        offsets = [[offset[0], offset[1] + rotate_angle * np.pi / 180.0]
                   for offset in offsets]
        offsets = [
            self.offset_coordinate_transform(offset, transform_flag='la2xy')
            for offset in offsets
        ]
        return np.array(offsets, dtype=np.float32)
