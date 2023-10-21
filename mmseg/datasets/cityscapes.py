# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from mmengine.structures import PixelData
import torch

# @DATASETS.register_module()
# class CityscapesDataset(BaseSegDataset):
#     """Cityscapes dataset.

#     The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
#     fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
#     """
#     METAINFO = dict(
#         classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
#                  'traffic light', 'traffic sign', 'vegetation', 'terrain',
#                  'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
#                  'motorcycle', 'bicycle'),
#         palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
#                  [190, 153, 153], [153, 153, 153], [250, 170,
#                                                     30], [220, 220, 0],
#                  [107, 142, 35], [152, 251, 152], [70, 130, 180],
#                  [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
#                  [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

#     def __init__(self,
#                  img_suffix='_leftImg8bit.png',
#                  seg_map_suffix='_gtFine_labelTrainIds.png',
#                  **kwargs) -> None:
#         super().__init__(
#             img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

import numpy as np

@DATASETS.register_module()
class CityscapesDataset(BaseSegDataset):
	"""Cityscapes dataset.

	The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
	fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
	"""
	METAINFO = dict(
		classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
				 'traffic light', 'traffic sign', 'vegetation', 'terrain',
				 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
				 'motorcycle', 'bicycle'),
		palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
				 [190, 153, 153], [153, 153, 153], [250, 170,
													30], [220, 220, 0],
				 [107, 142, 35], [152, 251, 152], [70, 130, 180],
				 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
				 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

	def __init__(self,
				 img_suffix='_leftImg8bit.png',
				 seg_map_suffix='_gtFine_labelTrainIds.png',
				 **kwargs) -> None:
		super().__init__(
			img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

	def __getitem__(self, idx: int) -> dict:
		res = super().__getitem__(idx)
		res = self.add_automask(res)
		return res

	def add_automask(self, res):
		"""
		read mask from path, where each pixel is a int value
		input: 
			info: {...}
		output: 
			ori_data: {
				img_metas: {
					filename: "/data/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_012009_leftImg8bit.png"
				}
			}
			info_new: {
				...,
				"automask": np.array,
			}
		"""
		# TODO sam extend for other datasets
		path = self.data_root + "/sam_val/automask_bwh/" + \
			res["data_samples"].img_path \
			.split("/")[-1].replace(".png", ".npy")
		seg_data = np.load(path, allow_pickle=True) # {"segmentation": np.array, "area": int}
		automask = seg_data
		automask = torch.tensor(automask, dtype=torch.uint8)
		res["data_samples"].automask = PixelData()
		res["data_samples"].automask.data = automask.unsqueeze(0)
		return res