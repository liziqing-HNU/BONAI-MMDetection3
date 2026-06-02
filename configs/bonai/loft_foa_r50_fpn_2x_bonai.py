_base_ = ['../_base_/default_runtime.py']

dataset_type = 'BONAI'
data_root = 'data/BONAI/'
backend_args = None

train_cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
val_cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']

model = dict(
    type='LOFT',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='LoftRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        offset_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        offset_head=dict(
            type='OffsetHeadExpandFeature',
            expand_feature_num=4,
            share_expand_fc=True,
            rotations=[0, 90, 180, 270],
            num_fcs=2,
            fc_out_channels=1024,
            num_convs=10,
            loss_offset=dict(type='SmoothL1Loss', loss_weight=16.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                gpu_assign_thr=512),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1,
                gpu_assign_thr=512),
            sampler=dict(
                type='RandomSampler',
                num=1024,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=2000,
            mask_thr_binary=0.5)))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadBONAIAnnotations',
        with_bbox=True,
        with_mask=True,
        with_offset=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='BONAIRandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical'],
        legacy_direction_choice=True),
    dict(type='PackBONAIInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadBONAIAnnotations',
        with_bbox=True,
        with_mask=True,
        with_offset=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='PackBONAIInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]

train_datasets = [
    dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f'coco/bonai_{city}_trainval.json',
        data_prefix=dict(img='trainval/images/'),
        bbox_type='building',
        mask_type='roof',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args) for city in train_cities
]

test_datasets = [
    dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f'coco/bonai_{city}_trainval.json',
        data_prefix=dict(img='trainval/images/'),
        bbox_type='building',
        mask_type='roof',
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args) for city in val_cities
]

test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='coco/bonai_shanghai_xian_test.json',
    data_prefix=dict(img='test/images/'),
    bbox_type='building',
    mask_type='roof',
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=train_datasets))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='ConcatDataset', datasets=test_datasets))

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=None,
        metric=['bbox', 'segm'],
        format_only=False,
        backend_args=backend_args)
]
test_evaluator = [
    dict(
        type='BONAIRLEMetric',
        score_thr=0.4,
        min_area=500,
        iou_thr=0.5,
        skip_empty=True,
        with_offset_error=True,
        outfile_prefix='work_dirs/loft_foa_r50_fpn_2x_bonai/'
        'bonai_test_rle',
        summary_file='work_dirs/loft_foa_r50_fpn_2x_bonai/'
        'bonai_test_summary.csv')
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=24)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=300),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

auto_scale_lr = dict(enable=False, base_batch_size=8)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
