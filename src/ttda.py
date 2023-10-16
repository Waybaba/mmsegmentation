from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
import mmcv
import mmengine.fileio as fileio

from mmengine.config import Config, DictAction
from copy import deepcopy
from mmseg.registry import HOOKS
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
							is_model_wrapper, revert_sync_batchnorm)
from mmengine.model.efficient_conv_bn_eval import \
	turn_on_efficient_conv_bn_eval
import torch.nn.functional as F
import cv2
from mmengine.structures import BaseDataElement, PixelData

@HOOKS.register_module()
class TTDAHook(Hook):
	"""Check invalid loss hook.

	This hook will regularly check whether the loss is valid
	during training.

	Args:
		interval (int): Checking interval (every k iterations).
			Default: 50.
	"""

	def __init__(self, **kwargs) -> None:
		self.kwargs = Config(kwargs)

	def fake_train_init(self, runner):
		""" Since we need to init something before test
		the original mmseg does not do this since test does not optimizer ...
		we here fetch some train code to init
		mmengine/runner/runner.py -> Runner.train()
		"""
		if is_model_wrapper(runner.model):
			ori_model = runner.model.module
		else:
			ori_model = runner.model
		assert hasattr(ori_model, 'train_step'), (
			'If you want to train your model, please make sure your model '
			'has implemented `train_step`.')

		if runner._val_loop is not None:
			assert hasattr(ori_model, 'val_step'), (
				'If you want to validate your model, please make sure your '
				'model has implemented `val_step`.')

		if runner._train_loop is None:
			raise RuntimeError(
				'`self._train_loop` should not be None when calling train '
				'method. Please provide `train_dataloader`, `train_cfg`, '
				'`optimizer` and `param_scheduler` arguments when '
				'initializing runner.')

		# runner._train_loop = runner.build_train_loop(
		#     runner._train_loop)  # type: ignore

		# `build_optimizer` should be called before `build_param_scheduler`
		#  because the latter depends on the former
		runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
		# Automatically scaling lr by linear scaling rule
		runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)

		if runner.param_schedulers is not None:
			runner.param_schedulers = runner.build_param_scheduler(  # type: ignore
				runner.param_schedulers)  # type: ignore

		if runner._val_loop is not None:
			runner._val_loop = runner.build_val_loop(
				runner._val_loop)  # type: ignore
		# TODO: add a contextmanager to avoid calling `before_run` many times
		# runner.call_hook('before_run')

		# initialize the model weights
		runner._init_model_weights() # ! note the process when train multiple times

		# try to enable efficient_conv_bn_eval feature
		modules = runner.cfg.get('efficient_conv_bn_eval', None)
		if modules is not None:
			runner.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
							 f' for sub-modules: {modules}')
			turn_on_efficient_conv_bn_eval(ori_model, modules)

		# make sure checkpoint-related hooks are triggered after `before_run`
		runner.load_or_resume()

		# Initiate inner count of `optim_wrapper`.
		# runner.optim_wrapper.initialize_count_status( # ! TODO how to handle this?
		#     runner.model,
		#     runner._train_loop.iter,  # type: ignore
		#     runner._train_loop.max_iters)  # type: ignore

		# Maybe compile the model according to options in self.cfg.compile
		# This must be called **AFTER** model has been wrapped.
		# runner._maybe_compile('train_step')

		# model = runner.train_loop.run()  # type: ignore
		# runner.call_hook('after_run')
		return None

	def ttda_before_test(self, runner, batch_idx, batch):
		"""
		input:
			inputs: 
				inputs [(CH, 512, 1024)], data_samples [SegDataSample]
				{
					gt_sem_seg: (512, 1024)
					img_path: 
					metainfo:
						scale_factor:
						ori_shape:
						img_shape:
						reduce_zero_label:
					img_shape:
					ori_shape:
					pred_sem_seg: # would have value after model.test_step
					seg_logits: # would have value after model.test_step
					scale_factor:
				}
		"""
		# init
		if not self.kwargs.turn_on_adapt: return
		use_pseudo_label = self.kwargs.use_pseudo_label
		slide_adapt = self.kwargs.slide_adapt
		assert len(batch["inputs"]) == 1, "only support batch_size=1"
		inputs, data_samples = batch["inputs"], batch["data_samples"]

		# inference label
		batch_pseudoed = runner.model.test_step(batch) # data_samples_[0].pred_sem_seg.shape = (512, 1024)

		# top-p mask
		# TODO each img should have some weight
		# TODO if buffer is to long, half it by only keep the even index'
		# TODO for cases where no class in buffer

		class SegMasker:
			""" 
			Mask out low confidence pseudo-labels in seg
			Detail: 
				maintain a buffer of confidence of pixels for each class,
				for each seg, set pixels to 255 if its confidence is not 
				in top-p percent.
			"""
			def __init__(self):
				self.buffer = {}

			def add_to_buffer(self, seg_conf, seg_pred):
				""" 
				seg_conf: (H, W) with each pixel value as confidence (0-1)
				seg_pred: (H, W) with each pixel value as predicted class
				save confidence of each pixel for each class
				"""
				for cls in seg_pred.unique():
					if cls.item() == 255:  # Skip the reserved value
						continue
					if cls.item() not in self.buffer:
						self.buffer[cls.item()] = []
					# Get confidences for pixels where the prediction matches the current class
					cls_confs = seg_conf[seg_pred == cls].flatten().tolist()
					self.buffer[cls.item()].extend(cls_confs)
				
			def cal_mask(self, seg_conf, top_p):
				"""
				return a mask shape as seg with True for high confidence,
				cal top_p percent
				"""
				mask = torch.zeros_like(seg_conf, dtype=torch.bool)
				for cls, confs in self.buffer.items():
					if not confs:  # if no confidences stored for this class, skip
						continue
					all_confs = torch.tensor(confs).flatten()
					threshold = all_confs.quantile(top_p)  # Notice that it's now `top_p` directly as we're marking high confidences
					mask[(seg_conf == cls) & (seg_conf >= threshold)] = True  # Only change the mask where condition is met
				return mask
		
		if self.kwargs.high_conf_mask.turn_on:
			if batch_idx == 0: 
				assert not hasattr(self, "seg_masker"), "seg_masker should not be init twice"
				self.seg_masker = SegMasker()
			seg_conf = F.softmax(batch_pseudoed[0].seg_logits.data, dim=0).max(0)[0]
			self.seg_masker.add_to_buffer(seg_conf, batch_pseudoed[0].pred_sem_seg.data[0])
			hign_conf_mask = self.seg_masker.cal_mask(
				seg_conf,
				self.kwargs.high_conf_mask.top_p
			)
			sem_seg_ = batch_pseudoed[0].pred_sem_seg.data[0]
			sem_seg_[~hign_conf_mask] = 255
			batch_pseudoed[0].pred_sem_seg = PixelData()
			batch_pseudoed[0].pred_sem_seg.data = sem_seg_.unsqueeze(0)

		# set data_batch_for_adapt for train

		###
		# def slide_crop(data_batch): return data_batch with {inputs: [n * (crop_size], ...} with predict set as gt
		class SegValueTrans:
			"""
			turn img size value to seg for crop
			"""
			def __init__(self, img_size, seg_size):
				self.img_size = img_size  # (h, w)
				self.seg_size = seg_size  # (h, w)

			def trans(self, v, type='w'):
				"""
				type:
					w: width
					h: height
					x: x pos
					y: y pos
				"""
				if type == 'h':
					return int(v * (self.seg_size[0] / self.img_size[0]))
				elif type == 'w':
					return int(v * (self.seg_size[1] / self.img_size[1]))
				elif type == 'y':
					# Assuming y pos translation maintains the same ratio
					return int(v * (self.seg_size[0] / self.img_size[0]))
				elif type == 'x':
					# Assuming x pos translation maintains the same ratio
					return int(v * (self.seg_size[1] / self.img_size[1]))
				else:
					raise ValueError("Invalid type provided.")
		
		batch_pseudoed_slided = {"inputs": [], "data_samples": []}
		batch_slided = {"inputs": [], "data_samples": []}
		# assert ori shape == seg shape
		st = SegValueTrans(data_samples[0].img_shape, data_samples[0].ori_shape)
		data_samples_tp = deepcopy(data_samples[0])
		h_stride, w_stride = runner.model.test_cfg.stride
		h_crop, w_crop = runner.model.test_cfg.crop_size
		_, h_img, w_img = inputs[0].size()
		h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
		w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
		for h_idx in range(h_grids):
			for w_idx in range(w_grids):
				y1 = h_idx * h_stride
				x1 = w_idx * w_stride
				y2 = min(y1 + h_crop, h_img)
				x2 = min(x1 + w_crop, w_img)
				y1 = max(y2 - h_crop, 0)
				x1 = max(x2 - w_crop, 0)
				crop_img = inputs[0][:, y1:y2, x1:x2]
				# change the image shape to patch shape
				data_sample_this = deepcopy(data_samples_tp)
				# TODO change size, ... except for value
				# data_samples_this.ori_shape = crop_img.shape[2:]

				meta_info_ = data_sample_this.metainfo
				meta_info_["img_shape"] = crop_img.shape[1:]
				meta_info_["ori_shape"] = (st.trans(h_crop, 'h'), st.trans(w_crop, 'w'))
				data_sample_this.set_metainfo(meta_info_)
				
				meta_this = deepcopy(data_sample_this)
				meta_pseudoed_this = deepcopy(data_sample_this)
				meta_this.gt_sem_seg = batch_pseudoed[0].gt_sem_seg[
					st.trans(y1,'y'):st.trans(y2,'y'),
					st.trans(x1,'x'):st.trans(x2,'x'),
				]
				meta_pseudoed_this.gt_sem_seg = batch_pseudoed[0].pred_sem_seg[
					st.trans(y1,'y'):st.trans(y2,'y'),
					st.trans(x1,'x'):st.trans(x2,'x'),
				]

				batch_slided['inputs'].append(crop_img) # 
				batch_slided['data_samples'].append(meta_this) # 
				batch_pseudoed_slided['inputs'].append(crop_img)
				batch_pseudoed_slided['data_samples'].append(meta_pseudoed_this)
		
		### choose adapt
		if slide_adapt:
			data_batch_for_adapt = deepcopy(batch_pseudoed_slided) \
				if use_pseudo_label else deepcopy(batch_slided)
		else:
			data_batch_for_adapt = deepcopy(batch)
			if use_pseudo_label:
				for i in range(len(data_samples)):
					data_batch_for_adapt["data_samples"][i].gt_sem_seg = batch_pseudoed[i].pred_sem_seg
		data_batch_for_adapt_bak = deepcopy(data_batch_for_adapt)
		### adapt - data_batch_for_adapt
		with torch.enable_grad():
			optim_wrapper = runner.optim_wrapper
			model = runner.model
			with optim_wrapper.optim_context(model):
				data_batch_for_adapt = model.data_preprocessor(data_batch_for_adapt, True)
				losses = model._run_forward(data_batch_for_adapt, mode='loss')  # type: ignore
			parsed_losses, log_vars = model.parse_losses(losses)  # type: ignore
			optim_wrapper.update_params(parsed_losses)

		# draw train sample
		if self.every_n_inner_iters(batch_idx, self.kwargs.adapt_img_vis_freq):
			for i, output in enumerate(data_batch_for_adapt["data_samples"]):
				# img_path = output.img_path
				# img_bytes = fileio.get(
				# 	img_path, backend_args=None) # runner.backend_args?
				# img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
				size_ = output.gt_sem_seg.shape # (1024, 1024) # ! note w,h order
				img = data_batch_for_adapt_bak["inputs"][i] # tensor (3, H, W)
				# interpolate to same size_
				# TODO

				img = img.float()
				img = F.interpolate(img.unsqueeze(0), size=(size_[0], size_[1]), mode='bilinear', align_corners=True)
				img = (img).byte()
				img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)

				if img.shape[-1] == 3:
					img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
				runner.visualizer.add_datasample(
					f"adapt/train_samples_{i}",
					img,
					data_sample=output,
					show=False,
					step=runner.iter,
					draw_pred=False
					)
		
		runner.visualizer.add_scalars({
			"adapt/"+k: v.item() for k, v in log_vars.items()
		},step=runner.iter)
		dict_log = {k: "{:.2f}".format(v.item()) for k, v in log_vars.items()}
		runner.logger.info(f"log_vars: {dict_log}")

	### hooks
	
	def before_run(self, runner) -> None:
		if not self.kwargs.turn_on_adapt: return
		# if self.mode == "test":
		runner.logger.info('fake train init')
		self.fake_train_init(runner)
		runner.logger.info('fake train init done')
		# else:
		#     raise ValueError(f"mode {self.mode} not supported, only 'test' for TTDA ")

	def before_test_iter(self,
						 runner: Runner,
						 batch_idx: int,
						 data_batch: Optional[dict] = None) -> None:
		"""Regularly check whether the loss is valid every n iterations.

		Args:
			runner (:obj:`Runner`): The runner of the training process.
			batch_idx (int): The index of the current batch in the train loop.
			data_batch (dict, Optional): Data from dataloader.
				Defaults to None.
			outputs (dict, Optional): Outputs from model. Defaults to None.
		"""
		self.ttda_before_test(runner, batch_idx, data_batch)

	def after_test_epoch(self, runner, metrics):
		# runner.logger.info('after test epoch')
		# revert_sync_batchnorm(runner.model) # ! ??? shoud we handle?
		# runner.logger.info('after test epoch done')
		runner.visualizer.add_scalars({
			"test/"+k: v for k, v in metrics.items()
		})

	def after_test_iter(self, runner, batch_idx, data_batch, outputs):
		# metrics = runner.test_evaluator.evaluate(len(runner.test_dataloader.dataset))
		# runner.visualizer.add_scalars({
		#     "test_step/"+k: v for k, v in metrics.items()
		# })
		# runner.logger.info(f"test iter {batch_idx} done, metrics: {metrics}")
		pass