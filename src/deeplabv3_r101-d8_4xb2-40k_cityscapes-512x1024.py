_base_ = [
    # '../configs/_base_/models/deeplabv3_r101-d8.py', 
	'../configs/deeplabv3/deeplabv3_r101-d8_4xb2-40k_cityscapes-512x1024.py'
    # '../configs/_base_/models/pspnet_r50-d8.py',
    # '../configs/_base_/models/daformer_aspp_mitb5.py',
    # '../configs/_base_/datasets/cityscapes.py',
    # '../configs/_base_/default_runtime.py', 
    # '../configs/_base_/schedules/schedule_40k.py'
]

data_folder = '/data'
train_dataloader = dict(
    dataset=dict(
        data_root=data_folder+"/GTA5",
    ))
val_dataloader = dict(
    dataset=dict(
        data_root=data_folder+"/Cityscapes/",
    ))