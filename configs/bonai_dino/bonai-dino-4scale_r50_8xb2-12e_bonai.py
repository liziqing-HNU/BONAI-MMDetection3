_base_ = ['../_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.BONAIDINO'], allow_failed_imports=False)

dataset_type = 'BONAI'
data_root = 'data/BONAI/'
backend_args = None
train_cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
val_cities = ['jinan']

model = dict(
    type='BONAIDINO',
    mask_feature_level=0,
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='BONAIDINOHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        mask_feature_channels=256,
        mask_thr_binary=0.5,
        offset_head=dict(
            type='DINOFOAQueryOffsetHead',
            expand_feature_num=4,
            share_expand_fc=True,
            rotations=(0, 90, 180, 270),
            offset_coordinate='rectangle',
            offset_coder=dict(
                type='DeltaXYOffsetCoder',
                target_means=[0.0, 0.0],
                target_stds=[0.5, 0.5]),
            loss_offset=dict(type='SmoothL1Loss', loss_weight=16.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            loss_weight=1.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

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
        direction=['horizontal', 'vertical']),
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
        offset_coordinate='rectangle',
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
        offset_coordinate='rectangle',
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args) for city in val_cities
]

test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='coco/bonai_shanghai_xian_test.json',
    data_prefix=dict(img='test/'),
    bbox_type='building',
    mask_type='roof',
    offset_coordinate='rectangle',
    test_mode=True,
    pipeline=test_pipeline,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
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
        outfile_prefix='work_dirs/bonai-dino-4scale_r50_8xb2-12e_bonai/'
        'bonai_test_rle',
        summary_file='work_dirs/bonai-dino-4scale_r50_8xb2-12e_bonai/'
        'bonai_test_summary.csv')
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00125, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

max_epochs = 24
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

auto_scale_lr = dict(enable=False, base_batch_size=2)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
