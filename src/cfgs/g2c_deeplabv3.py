_base_ = [
	'./c2c_deeplabv3.py',
]

# dataset settings
crop_size = (512, 512)
data_folder = "/data" # ! CHANGE for dataset path
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1280,720), # ! from daformer settings
        # ratio_range=(0.71, 0.74), # ! use 0.5, 2.0 before but would cause bug
        # ratio_range=(0.57,0.59),
        ratio_range=(0.5,2.0),
        keep_ratio=True),
    # dict(type='Resize', scale=(1280, 720)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5), # ! ? why this is no normization
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'), # ! TODO do we need to keep as train size
    dict(type='Resize', scale=(1024, 512), keep_ratio=True), # ! why, the origin is 2048, 1024
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type="GTADataset",
        data_root=data_folder+"/GTA5",
        data_prefix=dict(
            img_path='images', seg_map_path='labels'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="CityscapesDataset",
        data_root=data_folder+"/Cityscapes/",
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator