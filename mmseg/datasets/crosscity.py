# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS
from mmengine.fileio import join_path, list_from_file, load
from mmengine.logging import print_log
import logging

@DATASETS.register_module()
class CrosscityDataset(BaseSegDataset):
    METAINFO = dict(
        classes= ['bg', 'bg', 'bg', 'bg', 'bg', 'bg', 'bg', 'road', 'sidewalk', 'bg', 'bg', 'building', 'wall', 'fence', 'bg', 'bg', 'bg', 'pole', 'bg', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'bg', 'bg', 'train', 'motorcycle', 'bicycle'],
        # classes= ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        #             'traffic light', 'traffic sign', 'vegetation', 'terrain',
        #             'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        #             'motorcycle', 'bicycle'], # cityscape classes
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_eval.png',
                 reduce_zero_label=False,
                 metainfo={
                     "classes": ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                    'traffic light', 'traffic sign', 'vegetation', 'terrain',
                    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                    'motorcycle', 'bicycle'], # cityscape classes
                    "palette": [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                            [190, 153, 153], [153, 153, 153], [250, 170,
                                                                30], [220, 220, 0],
                            [107, 142, 35], [152, 251, 152], [70, 130, 180],
                            [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                            [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
                 },
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            metainfo=metainfo,
            **kwargs)
        # see LoadAnnotations for details

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')
