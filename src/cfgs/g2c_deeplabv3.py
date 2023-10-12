_base_ = [
	'./c2c_deeplabv3.py',
]

# dataset settings
crop_size = (512, 512)
data_folder = "/data" 
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1280,720), # ! from daformer settings
		# scale=(2560,1440),
        ratio_range=(0.5,2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    dataset=dict(
        type="GTADataset",
        data_root=data_folder+"/GTA5",
        data_prefix=dict(
            img_path='images', seg_map_path='labels'),
        pipeline=train_pipeline))
