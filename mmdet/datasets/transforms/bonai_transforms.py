import math

import numpy as np

from mmdet.registry import TRANSFORMS
from .formatting import PackDetInputs
from .loading import LoadAnnotations
from .transforms import RandomFlip


@TRANSFORMS.register_module()
class LoadBONAIAnnotations(LoadAnnotations):
    """Load standard detection annotations plus BONAI offset annotations."""

    def __init__(self, *args, with_offset=False, **kwargs) -> None:
        self.with_offset = with_offset
        super().__init__(*args, **kwargs)

    def _load_offsets(self, results: dict) -> None:
        gt_offsets = [
            instance.get('offset', [0., 0.])
            for instance in results.get('instances', [])
        ]
        results['gt_offsets'] = np.array(
            gt_offsets, dtype=np.float32).reshape((-1, 2))

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if self.with_offset:
            self._load_offsets(results)
        return results


@TRANSFORMS.register_module()
class BONAIRandomFlip(RandomFlip):
    """RandomFlip variant that also flips BONAI offset vectors."""

    def __init__(self,
                 prob=None,
                 direction='horizontal',
                 legacy_direction_choice=False,
                 **kwargs) -> None:
        # BONAI's MMDetection 2.x RandomFlip chose one direction at pipeline
        # construction when a direction list was supplied.
        if legacy_direction_choice and isinstance(direction, list):
            direction = str(np.random.choice(direction))
        self.legacy_direction_choice = legacy_direction_choice
        super().__init__(prob=prob, direction=direction, **kwargs)

    @staticmethod
    def _flip_xy_offsets(offsets: np.ndarray, direction: str) -> np.ndarray:
        flipped = offsets.copy()
        if direction in ('horizontal', 'diagonal'):
            flipped[:, 0] = -flipped[:, 0]
        if direction in ('vertical', 'diagonal'):
            flipped[:, 1] = -flipped[:, 1]
        if direction not in ('horizontal', 'vertical', 'diagonal'):
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    @staticmethod
    def _flip_polar_offsets(offsets: np.ndarray, direction: str) -> np.ndarray:
        flipped = offsets.copy()
        angles = flipped[:, 1]
        if direction == 'horizontal':
            flipped[:, 1] = math.pi - angles
        elif direction == 'vertical':
            flipped[:, 1] = -angles
        elif direction == 'diagonal':
            flipped[:, 1] = angles + math.pi
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if results.get('flip', False) and results.get('gt_offsets',
                                                      None) is not None:
            coordinate = results.get('offset_coordinate', 'rectangle')
            if coordinate == 'rectangle':
                results['gt_offsets'] = self._flip_xy_offsets(
                    results['gt_offsets'], results['flip_direction'])
            elif coordinate == 'polar':
                results['gt_offsets'] = self._flip_polar_offsets(
                    results['gt_offsets'], results['flip_direction'])
            else:
                raise RuntimeError(
                    f'Unsupported offset_coordinate={coordinate}')
        return results


@TRANSFORMS.register_module()
class PackBONAIInputs(PackDetInputs):
    """Pack detector inputs and keep ``gt_offsets`` with gt instances."""

    mapping_table = PackDetInputs.mapping_table.copy()
    mapping_table.update({'gt_offsets': 'offsets'})
