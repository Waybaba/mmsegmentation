# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# This is the same as SegFormer but with 256 embed_dims
# SegF. with C_e=256 in Tab. 7


_base_ = [
	"./daformer_conv1_mitb0.py"
]

model = dict(
    # pretrained='/data/models/SegFormer/pretrained_models/mit_b5.pth',
    backbone=dict(
        embed_dims=64, 
        num_heads=[1, 2, 5, 8], 
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
    )
)


