import math
import os.path as osp
from typing import List, Union

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module(name='BONAI')
@DATASETS.register_module()
class BONAIDataset(CocoDataset):
    """BONAI dataset adapter for MMDetection 3.x.

    The parser keeps BONAI's roof/building/footprint bbox switch and exposes
    per-instance ``offset`` values so ``LoadBONAIAnnotations`` can pack them
    into ``gt_instances.offsets``.
    """

    METAINFO = {
        'classes': ('building', ),
        'palette': [(220, 20, 60)]
    }

    def __init__(self,
                 *args,
                 gt_footprint_csv_file=None,
                 bbox_type='roof',
                 mask_type='roof',
                 offset_coordinate='rectangle',
                 resolution=0.6,
                 ignore_buildings=True,
                 **kwargs) -> None:
        self.gt_footprint_csv_file = gt_footprint_csv_file
        self.bbox_type = bbox_type
        self.mask_type = mask_type
        self.offset_coordinate = offset_coordinate
        self.resolution = resolution
        self.ignore_buildings = ignore_buildings
        super().__init__(*args, **kwargs)

    def _get_bbox(self, ann: dict) -> List[float]:
        if self.bbox_type == 'roof':
            return ann['bbox']
        if self.bbox_type == 'building':
            return ann['building_bbox']
        if self.bbox_type == 'footprint':
            return ann['footprint_bbox']
        raise TypeError(f"Unsupported bbox_type={self.bbox_type}")

    def _get_mask(self, ann: dict, only_footprint: bool):
        if only_footprint:
            return [ann['footprint_mask']]
        if self.mask_type == 'roof':
            if ann.get('roof_mask', None) is not None:
                return [ann['roof_mask']]
            return ann['segmentation']
        if self.mask_type == 'footprint':
            return [ann['footprint_mask']]
        raise TypeError(f"Unsupported mask_type={self.mask_type}")

    def _get_offset(self, ann: dict) -> List[float]:
        offset_x, offset_y = ann.get('offset', [0, 0])
        if self.offset_coordinate == 'rectangle':
            return [offset_x, offset_y]
        if self.offset_coordinate == 'polar':
            length = math.sqrt(offset_x**2 + offset_y**2)
            angle = math.atan2(offset_y, offset_x)
            return [length, angle]
        raise RuntimeError(
            f'Unsupported offset_coordinate={self.offset_coordinate}')

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        file_name = img_info.get('file_name', img_info.get('filename'))
        data_info = dict(
            img_path=osp.join(self.data_prefix['img'], file_name),
            img_id=img_info['img_id'],
            seg_map_path=None,
            height=img_info['height'],
            width=img_info['width'],
            offset_coordinate=self.offset_coordinate)

        if self.data_prefix.get('seg', None):
            data_info['seg_map_path'] = osp.join(
                self.data_prefix['seg'],
                file_name.rsplit('.', 1)[0] + self.seg_map_suffix)

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for ann in ann_info:
            if ann.get('ignore', False):
                continue

            x1, y1, w, h = self._get_bbox(ann)
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann.get('area', w * h) <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue

            ignore_flag = int(bool(ann.get('iscrowd', False))
                              and self.ignore_buildings)
            only_footprint = bool(ann.get('only_footprint', 0) == 1)

            instance = dict(
                bbox=[x1, y1, x1 + w, y1 + h],
                bbox_label=self.cat2label[ann['category_id']],
                ignore_flag=ignore_flag,
                offset=self._get_offset(ann))

            if ann.get('segmentation', None) is not None:
                instance['mask'] = self._get_mask(ann, only_footprint)
                instance['roof_mask'] = ann.get('roof_mask',
                                                ann['segmentation'])
            if ann.get('footprint_mask', None) is not None:
                instance['footprint_mask'] = [ann['footprint_mask']]
            if 'building_height' in ann:
                instance['building_height'] = ann['building_height']
            if 'roof_bbox' in ann:
                rx, ry, rw, rh = ann['roof_bbox']
                instance['roof_bbox'] = [rx, ry, rx + rw, ry + rh]
            if 'footprint_bbox' in ann:
                fx, fy, fw, fh = ann['footprint_bbox']
                instance['footprint_bbox'] = [fx, fy, fx + fw, fy + fh]
            instances.append(instance)

        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter images with the same empty/all-crowd rule as BONAI 2.x."""
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        valid_data_infos = []
        for data_info in self.data_list:
            instances = data_info.get('instances', [])
            if filter_empty_gt:
                if len(instances) == 0:
                    continue
                if all(instance.get('ignore_flag', 0)
                       for instance in instances):
                    continue
            if min(data_info['width'], data_info['height']) >= min_size:
                valid_data_infos.append(data_info)
        return valid_data_infos
