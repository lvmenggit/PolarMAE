############################### default runtime #################################
import sys
sys.path.append('')
default_scope = 'mmseg'

custom_imports = dict(
    imports=['fetal.vit_timm', 'fetal.fetal'],
    allow_failed_imports=False
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=1.0
)

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False


############################### dataset #################################
dataset_type = 'FetalHeadBiometryDataset'
data_root = ''

crop_size = (448, 448)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomResize', scale=(448, 448), ratio_range=(1.0, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    #dict(type='Pad', size=(448, 448), pad_val=0, seg_pad_val=255),
    dict(type='Pad', size=(448, 448), pad_val=dict(img=0, seg=255)),
    dict(type='LoadAnnotations', reduce_zero_label=True),

    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/masks_remap_by_table'
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/masks_remap_by_table'
        ),
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/images',
            seg_map_path='test/masks_remap_by_table'
        ),
        pipeline=test_pipeline
    )
)

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    ignore_index=255,
    format_only=False
)

test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    ignore_index=255,
    format_only=False
)


############################### hooks #################################
custom_hooks = [
    dict(type='MaskIgnoreForVisHook', ignore_index=255, priority='HIGHEST'),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_best='mIoU',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=200),
)


############################### running schedule #################################
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    constructor='LayerDecayOptimizerConstructor_ViT',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-4, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=0.0, T_max=2000, begin=500, end=2500, by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=2500, val_interval=500)


val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


############################### model #################################
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='VisionTransformer_timm',
        img_size=448,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=[2, 5, 8, 11],
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        use_checkpoint=False,
        pretrained=None,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        num_classes=2,
        ignore_index=255,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.8,
                avg_non_ignore=True,
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                loss_weight=1.2,
                ignore_index=255,
            ),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        stride=(336, 336),
        crop_size=(448, 448)
    ),
)
