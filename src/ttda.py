from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
import mmcv
import mmengine.fileio as fileio
import torch.nn as nn
from collections.abc import Iterable

from mmengine.config import Config, DictAction
from copy import deepcopy
from mmseg.registry import HOOKS, METRICS
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
							is_model_wrapper, revert_sync_batchnorm)
from mmengine.model.efficient_conv_bn_eval import \
	turn_on_efficient_conv_bn_eval
import torch.nn.functional as F
from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw, resize
from functools import partial
import utils
from mmseg.utils.misc import add_prefix
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prettytable import PrettyTable
logger: MMLogger = MMLogger.get_current_instance()
from matplotlib.colors import ListedColormap
from mmengine.model import BaseTTAModel
import cv2
from mmengine.structures import BaseDataElement, PixelData
from mmengine.registry import MODELS
from mmseg.models.backbones import MixVisionTransformer
from mmseg.evaluation.metrics import IoUMetric
from mmseg.models import EncoderDecoder
import torch.utils.checkpoint as cp
import math
from mmengine.registry import LOOPS
from scipy.ndimage import label, find_objects
from mmengine.runner.loops import TestLoop
from mmengine.runner.amp import autocast
from mmengine.optim import build_optim_wrapper

EPS = 1e-10


### utils ###

def is_deeplab(model):
	""" TODO not sure if this is the best way to check
	"""
	m = model.module if hasattr(model, "module") else model
	if m.backbone.__class__.__name__ == "MixVisionTransformerTPT":
		return False
	elif m.backbone.__class__.__name__ == "MixVisionTransformerTPT":
		raise NotImplementedError(f"unknown backbone {m.backbone.__class__.__name__}")
	else:
		raise NotImplementedError(f"unknown backbone {m.backbone.__class__.__name__}")

def is_tta(data):
	""" check if the data is tta format
	"""
	if isinstance(data['inputs'][0], tuple):
		return True
	else:
		return False

def _scaled_dot_product_attention_llama_adapter(
	q,
	k,
	v,
	attn_mask = None,
	dropout_p = 0.0,
	tpt_num = None,
	gate = None,
):
	B, Nt, E = q.shape
	q = q / math.sqrt(E)
	# (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
	if attn_mask is not None:
		attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
	else:
		attn = torch.bmm(q, k.transpose(-2, -1))

	### @waybaba
	if tpt_num is not None:
		# attn = F.softmax(attn, dim=-1)
		attn_ori = attn[:, :, :-tpt_num] # [B, num_tokens_out, num_tokens_in]
		attn_extra = attn[:, :, -tpt_num:]
		attn_ori = torch.softmax(attn_ori, dim=-1)
		attn_extra = torch.softmax(attn_extra, dim=-1) * torch.tanh(gate)
		attn = torch.cat([attn_ori, attn_extra], dim=-1)
	else:
		attn = F.softmax(attn, dim=-1)
	### @waybaba
	if dropout_p > 0.0:
		attn = F.dropout(attn, p=dropout_p)
	# (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
	output = torch.bmm(attn, v)
	return output, attn

def quadratic_function(x, alpha):
	"""
	Computes the value of a quadratic function that passes through points (0,0) and (1,1).

	This function is defined as: f(x) = ax^2 + (1-a)x
	The shape of the curve (concave or convex) is determined by the parameter 'a'.
	
	Parameters:
	- x (float): The input value for which the function value needs to be computed. 
				 Expected to be in the range [0, 1].
	- a (float): The parameter that controls the concavity or convexity of the function.
				 When a > 0, the function is convex (upwards facing).
				 When a < 0, the function is concave (downwards facing).
				 When a = 0, the function is a straight line.

	Returns:
	- float: The computed value of the function for the given 'x'.
	"""
	return x ** alpha

def sam_feats_proto_predict(sam_feats, logits, cfg):
	""" 
	use logits to calculate the confidence of each pixel for each class
	use it as weight to calculate the prototype of each class with sam_feats space
	then use the prototype to predict
		sam_feats: (Ch, w_, h_) small
		logits: (C, w, h) big 
	return:
		pred: (w, h)
	"""
	# If size is not provided, default to spatial dimensions of logits
	sam_feats, logits = sam_feats.clone(), logits.clone()
	sam_feats = F.normalize(sam_feats, dim=0)
	size_ori = logits.shape[1:]
	size_sm = sam_feats.shape[1:]

	logits = F.interpolate(logits.unsqueeze(0), size=size_sm, mode='bilinear', align_corners=False).squeeze(0)
	
	pred_weights = F.softmax(logits / cfg.tau, dim=0)
	weighted_feats = (sam_feats.unsqueeze(0) * pred_weights.unsqueeze(1)).sum(dim=-1).sum(dim=-1)
	prototypes = F.normalize(weighted_feats, dim=1)
	similarity = torch.matmul(prototypes, sam_feats.view(sam_feats.size(0), -1)).view(prototypes.size(0), sam_feats.size(1), sam_feats.size(2))
	# similarity = torch.einsum('chw,cd->dhw', sam_feats, prototypes)
	similarity = F.interpolate(similarity.unsqueeze(0), size=size_ori, mode='bilinear', align_corners=False).squeeze(0)
	pred = similarity.argmax(dim=0)
	return pred

def automask_consistency_loss(sam_automask, feats, outputs, cfg):
	"""
	Calculate the consistency of the same color
		input:
			sam_automask: (B, w, h)
				different color is indicated by different int
			feats: (B, d, w, h)
			outputs: (B, C, w, h)
	"""
	sam_automask, feats, outputs = sam_automask.unsqueeze(1), feats.unsqueeze(1), outputs.unsqueeze(1)
	batch_size = sam_automask.shape[0]
	loss = 0.0

	pixel_total = feats.shape[2] * feats.shape[3]
	
	for b in range(batch_size):
		# flatten the automask and feats for convenience
		sam_automask_flat = sam_automask[b].view(-1)
		feats_flat = feats[b].view(feats.shape[1], -1)
		outputs_flat = outputs[b].view(outputs.shape[1], -1)

		# get unique segments in the mask
		segments = torch.unique(sam_automask_flat)

		if cfg.strategy == "min_variance":
			for segment in segments:
				# get indices of current segment
				segment_indices = torch.where(sam_automask_flat == segment)[0]

				# get features of current segment
				segment_feats = feats_flat[:, segment_indices]

				# calculate variance along the feature dimension
				segment_variance = torch.var(segment_feats, dim=1, unbiased=False)

				# count number of pixels in the segment
				pixel_count = segment_indices.numel()

				# add mean of variances to total loss, weighted by inverse of pixel count
				loss += torch.mean(segment_variance) * (pixel_count / pixel_total)

		elif cfg.strategy == "close_to_confident":
			"""
				use the top-{confidence_selected_ratio} confident pixel in the segment as the anchor
				minimize the distance between the anchor and other pixels in the segment
				outputs_flat: (C, N) which is before softmax
				# cfg.confidence_type: "confidence" or "entropy"
				# cfg.confidence_selected_ratio: 0.1
			"""
			for segment in segments:
				# get indices of current segment
				segment_indices = torch.where(sam_automask_flat == segment)[0]

				# get outputs of current segment
				segment_outputs = outputs_flat[:, segment_indices]

				# get the confidence scores of the segment
				if cfg.confidence_type == "confidence":
					segment_confidence = torch.max(torch.softmax(segment_outputs, dim=0), dim=0)[0]
					confidence_threshold = torch.topk(segment_confidence, max(1, int(cfg.confidence_selected_ratio * segment_confidence.shape[0])), largest=True)[0][-1]
					anchor_indices = torch.where(segment_confidence >= confidence_threshold)[0]
				elif cfg.confidence_type == "entropy":
					segment_confidence = torch.sum(-torch.softmax(segment_outputs, dim=0) * torch.log(torch.softmax(segment_outputs, dim=0) + 1e-8), dim=0)
					confidence_threshold = torch.topk(segment_confidence, max(1, int(cfg.confidence_selected_ratio * segment_confidence.shape[0])), largest=False)[0][-1]
					anchor_indices = torch.where(segment_confidence <= confidence_threshold)[0]
				else:
					raise ValueError(f"Unknown confidence type: {cfg.confidence_type}")

				# select the top-{confidence_selected_ratio} confident pixel in the segment as the anchor
				anchor_feats = segment_outputs[:, anchor_indices]
				anchor_feats = anchor_feats.mean(dim=1).unsqueeze(1).detach()

				# calculate distance between the anchor and other pixels in the segment
				distance = torch.norm(segment_outputs[:, :, None] - anchor_feats[:, None, :], dim=0)

				# add mean distance to total loss, weighted by inverse of pixel count
				pixel_count = segment_indices.numel()
				loss += torch.mean(distance) * (pixel_count / pixel_total)

		else: raise NotImplementedError

	return loss / batch_size  # Normalize by batch size

def adjust_with_sam(logits, automask, cfg):
	"""
	Adjust logits based on automask value.

	Parameters:
	- logits: Tensor of logits. Shape: (C, H, W)
	- automask: Tensor of shape (1, H, W) representing automask values.
	- sam_ratio: A scalar controlling the strength of adjustment. float

	Returns:
	- Adjusted logits.
	"""
	sam_ratio = cfg.sam_ratio
	if sam_ratio == 0.: return logits
	conf = torch.softmax(logits, dim=0).max(dim=0)[0]
	unique_masks = torch.unique(automask)
	adjusted_logits = logits.clone()
	if cfg.use_prob: adjusted_logits = torch.softmax(adjusted_logits, dim=0)

	for mask in unique_masks:
		mask_indices = (automask.squeeze(0) == mask)  # Shape: (H, W)
		mean_logits_per_channel = torch.mean(logits[:, mask_indices], dim=1, keepdim=True)  # Shape: (C, 1)
		
		# Use broadcasting to update the adjusted logits for the current mask
		if cfg.sam_ratio_mul_conf:
			this_conf = conf[mask_indices]
			sam_ratio_matrix = sam_ratio * quadratic_function(1-this_conf, cfg.sam_ratio_conf_adjust)
		else:
			sam_ratio_matrix = torch.ones_like(mean_logits_per_channel) * sam_ratio

		adjusted_logits[:, mask_indices] = (1 - sam_ratio_matrix) * logits[:, mask_indices] + sam_ratio_matrix * mean_logits_per_channel

	return adjusted_logits

def param_migrate(base, tgt, rho):
	"""
	rho = 1 means copy all params from src to tgt
	rho = 0 means copy nothing
	others, final = (1-rho) * base + rho * tgt
	"""
	# Store base and target state_dict for efficiency
	base_state_dict = base.state_dict()
	tgt_state_dict = tgt.state_dict()

	# Ensure all keys are present in both models
	assert base_state_dict.keys() == tgt_state_dict.keys(), "Models have different parameters"

	# Loop through the parameters
	for name in base_state_dict:
		base_param = base_state_dict[name]
		tgt_param = tgt_state_dict[name]
		# Update the parameter based on rho
		base_param.data.copy_((1 - rho) * base_param.data + rho * tgt_param.data)

	return base

def aug_to_first(data):
	""" get one from augs
	we need aug_test to give variants of data, but we only use one for inference
	first one is ori size, no flip
	ps. the original implementation would use a tta_model wrapper to handle this and merge all preds,
	but this would put another wrapper on the model class, I dont want it, so I just modify the test_step functions.
	"""
	if isinstance(data['inputs'][0], tuple):
		data = {k: v[0] for k, v in data.items()}
	return data

def make_grid(num=32):
	"""
	Creates a grid of points in the range (0, 1), not including 0 and 1.
	The grid is of shape (num*num, 2), where each row represents the (x, y)
	coordinates of a point.
	
	:param num: Number of points in each dimension, excluding the boundaries.
	:return: A NumPy array of shape (num*num, 2) with the grid points.
	"""
	points = np.linspace(0, 1, num + 2, endpoint=False)[1:]
	x, y = np.meshgrid(points, points)
	grid = np.stack((x, y), axis=-1).reshape(-1, 2)
	return grid

def inflation(pred, inflation=0):
	"""
	inflate each class, if it expand to 255 areas, make the pixel the class.
	if it expand to other classes, do not change the pixel
	if multiple class expand to the same 255 areas, keep it 255
	ps. first calculate for each class, then merge into one
	pred: (h,w) tensor with int 0~c-1, and 255 for empty
	inflation: number of pixels inflation
	return:
		pred after inflation
	"""
	pass

class ProtoClassifier:
	"""
		cfg: 
			rho: 0.5 # for updating prototype, history_proto = (1-rho) * history_proto + rho * proto_this_img
			lambda_: count # for using prototype, final_proto = lambda_ * history_proto + (1 - lambda_) * proto_this_img. if "count", use count as lambda
	"""
	def __init__(self, cfg):
		self.protos_mean = {}  # Dictionary to store means for each class: {class_label: tensor of shape [C, H, W]}
		self.protos_count = {}  # Dictionary to store count of pixels for each class: {class_label: int}
		self.cfg = cfg # use by self.cfg.rho and self.cfg.lambda_

	def reset(self):
		"""
		Resets the learned prototypes.
		"""
		self.protos_mean = {}
		self.protos_count = {}

	def process_one(self, feature_map, logit_map, prediction):
		"""
		Process a single image's semantic segmentation.

		Args:
			feature_map (Tensor): The feature map with shape [C, H, W].
			logit_map (Tensor): The logit map with shape [C, H, W].
			prediction (Tensor): The semantic segmentation prediction with shape [H, W].

		Returns:
			Tensor: Refined semantic segmentation prediction.
		"""
		# Calculate confidence and entropy based on logit_map
		feature_map, logit_map, prediction = feature_map.cuda(), logit_map.cuda(), prediction.cuda()
		confidence = torch.softmax(logit_map, dim=0).max(dim=0)[0]
		entropy = -torch.sum(torch.log_softmax(logit_map, dim=0) * torch.softmax(logit_map, dim=0), dim=0)
		if self.cfg.norm_feats_mean: # (Ch, num)
			feature_map = F.normalize(feature_map, dim=0) # TODO move to class

		# Mask calculation and application
		if self.cfg.proto_mask_metric is not None:
			if self.cfg.proto_mask_metric not in ["confidence", "entropy"]:
				raise ValueError(f"cfg.proto_mask_metric should be confidence or entropy, but got {self.cfg.proto_mask_metric}")

			seg_metric = confidence if self.cfg.proto_mask_metric == "confidence" else -entropy
			
			if not hasattr(self, "seg_masker"):
				self.seg_masker = SegMasker(use_history=self.cfg.proto_mask_use_history)
				
			self.seg_masker.add_to_buffer(seg_metric, prediction)
			high_conf_mask = self.seg_masker.cal_mask(seg_metric, prediction, self.cfg.proto_mask_top_p)
			masked_prediction = prediction.clone()
			masked_prediction[~high_conf_mask] = 255
		else:
			masked_prediction = prediction.clone()

		# Extract and combine protos
		C, H, W = feature_map.shape
		unique_labels = [label for label in masked_prediction.unique().tolist() if label != 255]
		combined_protos = []
		for label in unique_labels:
			class_features = feature_map[:, masked_prediction == label]
			proto_current_img = class_features.mean(dim=1)
			
			if label in self.protos_mean:
				combined_protos.append(self.cfg.lambda_ * self.protos_mean[label] + (1 - self.cfg.lambda_) * proto_current_img)
			else:
				combined_protos.append(proto_current_img)
		proto_tensor = torch.stack(combined_protos, dim=0) # [cls, ch]

		# Predict using protos
		reshaped_features = feature_map.view(C, -1).transpose(0, 1) # [ch, N]
		if self.cfg.norm_feats_sim == True:
			reshaped_features_normalized = F.normalize(reshaped_features, p=2, dim=1)
			proto_tensor_normalized = F.normalize(proto_tensor, p=2, dim=1)
			distances = 1 - torch.mm(reshaped_features_normalized, proto_tensor_normalized.t())
		else:
			distances = torch.cdist(reshaped_features, proto_tensor)
		_, pred_class_indices = distances.min(dim=1)
		pred_labels = torch.tensor(unique_labels, dtype=torch.long).to(feature_map.device)
		refined_prediction = pred_labels[pred_class_indices].view(H, W)

		# Update buffer with new protos
		for label in unique_labels:
			class_features = feature_map[:, masked_prediction == label]
			if label not in self.protos_mean:
				self.protos_mean[label] = class_features.mean(dim=1)
				self.protos_count[label] = class_features.size(1)
			else:
				new_count = class_features.size(1)
				new_mean = class_features.mean(dim=1)
				if self.cfg.rho == "count":
					updated_mean = (self.protos_mean[label] * self.protos_count[label] + new_mean * new_count) / (self.protos_count[label] + new_count)
					self.protos_count[label] += new_count
				elif isinstance(self.cfg.rho, (float, int)):
					updated_mean = self.protos_mean[label] * (1 - self.cfg.rho) + new_mean * self.cfg.rho
					self.protos_count[label] += new_count
				self.protos_mean[label] = updated_mean

		return refined_prediction

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

class SegMasker:
	"""
	TODO improve buffer size
	Mask out low pseudo-labels in seg
	Detail: 
		maintain a buffer of confidence of pixels for each class,
		for each seg, set pixels to 255 if its confidence is not 
		in top-p percent.
		# TODO each img should have some weight
		# TODO if buffer is to long, half it by only keep the even index'
		# TODO for cases where no class in buffer
	"""
	def __init__(self, use_history, class_specific):
		self.buffer = {}
		self.use_history = use_history
		self.class_specific = class_specific

	def add_to_buffer(self, seg_metric, seg_pred):
		""" 
		seg_conf: (H, W) with each pixel value as confidence (0-1)
		seg_pred: (H, W) with each pixel value as predicted class
		save confidence of each pixel for each class
		TODO here, we hope the buffer can be running, so that the new conf would be used
			but if we use fixed size for each class, then the buffer would never be updated for rare class
			if use time as indication, the rare class would be updated too frequently
		"""
		SIZE_RATIO = 500
		if self.class_specific:
			for cls in seg_pred.unique():
				if cls.item() == 255:  # Skip the reserved value
					continue
				if cls.item() not in self.buffer:
					self.buffer[cls.item()] = []
				# Get confidences for pixels where the prediction matches the current class
				cls_confs = seg_metric[seg_pred == cls].flatten().tolist()
				### TODO partial
				while len(cls_confs) > 1000: cls_confs = cls_confs[::2]
				self.buffer[cls.item()].extend(cls_confs)
				while len(self.buffer[cls.item()]) > 1000*SIZE_RATIO: 
					self.buffer[cls.item()] = self.buffer[cls.item()][::2]
		else:
			# Global buffer handling
			confs = seg_metric.flatten().tolist()
			while len(confs) > 1000: confs = confs[::2]
			self.buffer.setdefault("global", []).extend(confs)
			while len(self.buffer["global"]) > 1000 * SIZE_RATIO: 
				self.buffer["global"] = self.buffer["global"][::2]
		
	def cal_mask(self, seg_metric, seg_pred, top_p):
		"""
		return a mask shape as seg with True for high confidence,
		cal top_p percent
		"""
		mask = torch.zeros_like(seg_metric, dtype=torch.bool)
		if self.class_specific:
			mask = torch.zeros_like(seg_metric, dtype=torch.bool)
			for cls, confs in self.buffer.items():
				if not confs:  # if no confidences stored for this class, skip
					continue
				if self.use_history:
					all_confs = torch.tensor(confs).flatten()
				else:
					all_confs = seg_metric[seg_pred == cls].flatten()
				if all_confs.numel() == 0: continue
				threshold = all_confs.quantile(1-top_p)  # Notice that it's now `top_p` directly as we're marking high confidences
				mask[(seg_pred == cls) & (seg_metric >= threshold)] = True  # Only change the mask where condition is met
		else:
			# Global mask computation
			if self.use_history:
				all_confs = torch.tensor(self.buffer.get("global", [])).flatten()
			else:
				all_confs = seg_metric.flatten()
			if all_confs.numel() > 0:
				threshold = all_confs.quantile(1 - top_p)
				mask[seg_metric >= threshold] = True

		return mask


### wrappers ###
@HOOKS.register_module()
class TTDABeforeRunInitHook(Hook):
	""" Why this? -> init train stuff
	The only reason for this function is doing some train_init stuff
	since when only "test=true", mmengine won't init optimizer
	and, to init it, we need to do it "before_run"
	"""
	def before_run(self, runner) -> None:
		# adapt init (e.g. optimizer)
		runner.logger.info('fake train init')
		self.fake_train_init(runner)
		runner.logger.info('fake train init done')

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
		runner.optim_wrapper_cfg = deepcopy(runner.optim_wrapper)
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

@METRICS.register_module()
class IoUMetricWrapper(IoUMetric):
	def compute_metrics(self, results: list):
		if self.format_only:
			logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
			return OrderedDict()
		# convert list of tuples to tuple of lists, e.g.
		# [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
		# ([A_1, ..., A_n], ..., [D_1, ..., D_n])
		results = tuple(zip(*results))
		assert len(results) == 4

		total_area_intersect = sum(results[0])
		total_area_union = sum(results[1])
		total_area_pred_label = sum(results[2])
		total_area_label = sum(results[3])
		ret_metrics = self.total_area_to_metrics(
			total_area_intersect, total_area_union, total_area_pred_label,
			total_area_label, self.metrics, self.nan_to_num, self.beta)

		class_names = self.dataset_meta['classes']

		# summary table
		ret_metrics_summary = OrderedDict({
			ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
			for ret_metric, ret_metric_value in ret_metrics.items()
		})
		metrics = dict()
		for key, val in ret_metrics_summary.items():
			if key == 'aAcc':
				metrics[key] = val
			else:
				metrics['m' + key] = val

		# each class table
		ret_metrics.pop('aAcc', None)
		ret_metrics_class = OrderedDict({
			ret_metric: np.round(ret_metric_value * 100, 2)
			for ret_metric, ret_metric_value in ret_metrics.items()
		})
		ret_metrics_class.update({'Class': class_names})
		ret_metrics_class.move_to_end('Class', last=False)
		class_table_data = PrettyTable()
		for key, val in ret_metrics_class.items():
			class_table_data.add_column(key, val)

		print_log('per class results:', logger)
		print_log('\n' + class_table_data.get_string(), logger=logger)

		# @waybaba add class IoU
		baseline_iou = {
			"road": 73.5,
			"sidewalk": 18.78,
			"building": 84.06,
			"wall": 39.81,
			"fence": 28.92,
			"pole": 26.07,
			"traffic light": 40.09,
			"traffic sign": 17.66,
			"vegetation": 86.02,
			"terrain": 42.96,
			"sky": 89.88,
			"person": 63.22,
			"rider": 26.73,
			"car": 85.04,
			"truck": 37.27,
			"bus": 38.49,
			"train": 36.26,
			"motorcycle": 22.87,
			"bicycle": 20.25
		}
		baseline_acc = {
			"road": 79.34,
			"sidewalk": 45.29,
			"building": 94.0,
			"wall": 51.7,
			"fence": 40.82,
			"pole": 28.89,
			"traffic light": 43.85,
			"traffic sign": 17.82,
			"vegetation": 95.32,
			"terrain": 60.95,
			"sky": 96.08,
			"person": 79.99,
			"rider": 35.72,
			"car": 91.19,
			"truck": 75.47,
			"bus": 43.87,
			"train": 37.89,
			"motorcycle": 53.46,
			"bicycle": 20.94
		}
		segmentation_categories = {
			"stuff": ["road", "sidewalk", "building", "wall", "vegetation", "terrain", "sky"],
			"things": ["fence", "pole", "traffic light", "traffic sign", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"],
			"smooth_surfaces": ["road", "car", "bus", "train", "bicycle", "motorcycle", "traffic light", "traffic sign"],
			"rough_surfaces": ["sidewalk", "building", "wall", "fence", "terrain", "vegetation"],
			"high_frequency_moving": ["person", "rider", "car", "truck", "bus", "bicycle", "motorcycle"],
			"low_or_static": ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "train"],
			"simple_structures": ["road", "sidewalk", "sky", "terrain"],
			"medium_structures": ["wall", "fence", "pole", "traffic light", "traffic sign", "car", "truck", "bus", "train"],
			"complex_structures": ["building", "vegetation", "person", "rider", "motorcycle", "bicycle"],
			"large_scale": ["road", "building", "sky", "terrain", "truck", "bus", "train"],
			"medium_scale": ["car", "vegetation", "wall"],
			"small_scale": ["sidewalk", "fence", "pole", "traffic light", "traffic sign", "person", "rider", "motorcycle", "bicycle"],
			"regular_shapes": ["road", "sidewalk", "building", "wall", "fence", "traffic light", "traffic sign", "car", "bus", "truck", "train"],
			"irregular_shapes": ["vegetation", "terrain", "sky", "pole", "person", "rider", "motorcycle", "bicycle"]
		}
		for i, name in enumerate(class_names):
			metrics[f"cls_{name}"] = ret_metrics_class["IoU"][i] - baseline_iou[name] \
			if not np.isnan(ret_metrics_class["IoU"][i]) else 0.
		for cls_mtd, clss in segmentation_categories.items():
			metrics[f"group_{cls_mtd}"] = np.nanmean([metrics[f"cls_{cls}"] for cls in clss])
		return metrics

@MODELS.register_module()
class EncoderDecoderWrapper(EncoderDecoder):
	""" Modifications: to get last layer feature
		as in paper https://github.com/matsuolab/T3A
	"""
	def train_step(self, data, optim_wrapper):
		# Enable automatic mixed precision training context.
		with optim_wrapper.optim_context(self):
			data = self.data_preprocessor(data, True)
			losses = self._run_forward(data, mode='loss')  # type: ignore
		parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
		optim_wrapper.update_params(parsed_losses)
		return log_vars

	def test_step(self, data):
		"""``BaseModel`` implements ``test_step`` the same as ``val_step``.

		Args:
			data (dict or tuple or list): Data sampled from dataset.

		Returns:
			list: The predictions of given data.
		"""
		data = aug_to_first(data)
		data = self.data_preprocessor(data, False)
		return self._run_forward(data, mode='predict')  # type: ignore

	def _run_forward(self, data, mode):
		"""Unpacks data for :meth:`forward`

		Args:
			data (dict or tuple or list): Data sampled from dataset.
			mode (str): Mode of forward.

		Returns:
			dict or list: Results of training or testing mode.
		"""
		if isinstance(data, dict):
			results = self(**data, mode=mode)
		elif isinstance(data, (list, tuple)):
			results = self(*data, mode=mode)
		else:
			raise TypeError('Output of `data_preprocessor` should be '
							f'list, tuple or dict, but got {type(data)}')
		return results

	def forward(self, inputs, data_samples = None, mode = 'tensor'):
		"""The unified entry for a forward process in both training and test.

		The method should accept three modes: "tensor", "predict" and "loss":

		- "tensor": Forward the whole network and return tensor or tuple of
		tensor without any post-processing, same as a common nn.Module.
		- "predict": Forward and return the predictions, which are fully
		processed to a list of :obj:`SegDataSample`.
		- "loss": Forward and return a dict of losses according to the given
		inputs and data samples.

		Note that this method doesn't handle neither back propagation nor
		optimizer updating, which are done in the :meth:`train_step`.

		Args:
			inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
				general.
			data_samples (list[:obj:`SegDataSample`]): The seg data samples.
				It usually includes information such as `metainfo` and
				`gt_sem_seg`. Default to None.
			mode (str): Return what kind of value. Defaults to 'tensor'.

		Returns:
			The return type depends on ``mode``.

			- If ``mode="tensor"``, return a tensor or a tuple of tensor.
			- If ``mode="predict"``, return a list of :obj:`DetDataSample`.
			- If ``mode="loss"``, return a dict of tensor.
		"""
		if mode == 'loss':
			return self.loss(inputs, data_samples)
		elif mode == 'predict':
			return self.predict(inputs, data_samples)
		elif mode == 'tensor':
			return self._forward(inputs, data_samples)
		else:
			raise RuntimeError(f'Invalid mode "{mode}". '
							   'Only supports loss, predict and tensor mode')

	def loss(self, inputs, data_samples):
		"""Calculate losses from a batch of inputs and data samples.

		Args:
			inputs (Tensor): Input images.
			data_samples (list[:obj:`SegDataSample`]): The seg data samples.
				It usually includes information such as `metainfo` and
				`gt_sem_seg`.

		Returns:
			dict[str, Tensor]: a dictionary of loss components
		"""

		x = self.extract_feat(inputs)

		losses = dict()

		loss_decode = self._decode_head_forward_train(x, data_samples)
		losses.update(loss_decode)

		if self.with_auxiliary_head:
			loss_aux = self._auxiliary_head_forward_train(x, data_samples)
			losses.update(loss_aux)

		return losses

	def predict(self, inputs, data_samples):
		"""Predict results from a batch of inputs and data samples with post-
		processing.

		Args:
			inputs (Tensor): Inputs with shape (N, C, H, W).
			data_samples (List[:obj:`SegDataSample`], optional): The seg data
				samples. It usually includes information such as `metainfo`
				and `gt_sem_seg`.

		Returns:
			list[:obj:`SegDataSample`]: Segmentation results of the
			input images. Each SegDataSample usually contain:

			- ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
			- ``seg_logits``(PixelData): Predicted logits of semantic
				segmentation before normalization.
		"""
		if data_samples is not None:
			batch_img_metas = [
				data_sample.metainfo for data_sample in data_samples
			]
		else:
			batch_img_metas = [
				dict(
					ori_shape=inputs.shape[2:],
					img_shape=inputs.shape[2:],
					pad_shape=inputs.shape[2:],
					padding_size=[0, 0, 0, 0])
			] * inputs.shape[0]

		seg_logits = self.inference(inputs, batch_img_metas)

		return self.postprocess_result(seg_logits, data_samples)

	# more
	def test_step_proto_predict(self, data, cfg=None):
		"""
		TODO remember to reset buffer
		TODO 255 class hwo to handle
		here, cfg is set by partial at init stage of runner
		"""
		data = aug_to_first(data)
		data = self.data_preprocessor(data, False)
		res = self._run_forward_plus(data, mode='predict')  # type: ignore
		if cfg.turn_on:
			if not hasattr(self, 'protos_classifier'):
				self.protos_classifier = ProtoClassifier(cfg)
			for i, d in enumerate(res):
				res_pred = self.protos_classifier.process_one(
					res[i].feats.data.clone() if cfg.debug_feats == "feats" \
					else res[i].seg_logits.data.clone() if cfg.debug_feats == "logits" \
					else F.softmax(res[i].seg_logits.data.clone()) if cfg.debug_feats == "prob" \
					else NotImplementedError(f"Unknown debug_feats: {cfg.debug_feats}"),
					res[i].seg_logits.data.clone(),
					res[i].pred_sem_seg.data[0].clone(),
				)
				res[i].pred_sem_seg.data = res_pred
		return res

	def test_step_sam_predict(self, data, cfg=None):
		"""
		TODO remember to reset buffer
		TODO 255 class hwo to handle
		here, cfg is set by partial at init stage of runner
		"""
		data = aug_to_first(data)
		data = self.data_preprocessor(data, False)
		res = self._run_forward_plus(data, mode='predict')  # type: ignore
		for i, d in enumerate(res):
			feats = res[i].feats.data
			sam_feats = res[i].sam_feats.data
			logits = res[i].seg_logits.data
			automask = res[i].automask.data
			if cfg.type == "sam_feats_proto":
				res[i].pred_sem_seg.data = sam_feats_proto_predict(sam_feats, logits, cfg)
			elif cfg.type == "logits_mask_adjust":
				# TODO add confidence threshold
				logits_ = adjust_with_sam(logits, automask, cfg)
				res[i].pred_sem_seg.data = logits_.argmax(0)
			else:
				raise NotImplementedError(f"Unknown sam type: {cfg.type}")
		return res

	def test_step_sam_model_predict(self, data, mask_generator=None, cfg=None):
		""" use sam_model to generate automask with indication points
		"""
		data = aug_to_first(data)
		imgs = deepcopy(data['inputs'])
		data = self.data_preprocessor(data, False)
		self.mask_generator = mask_generator
		res = self._run_forward_plus(data, mode='predict')  # type: ignore
		for i, d in enumerate(res):
			img = imgs[i]
			batch = data
			logits = res[i].seg_logits.data
			seg_logits = logits
			y = res[i].gt_sem_seg.data[0]
			pred_raw = res[i].pred_sem_seg.data[0]
			size_ori = y.shape
			# reshape logits
			seg_logits = seg_logits.unsqueeze(0)  # tensor (1, cls, w_, h_)
			seg_logits = F.interpolate(seg_logits, size=img.shape[1:], mode='bilinear', align_corners=True)
			y = F.interpolate(y.unsqueeze(0).unsqueeze(0).float(), size=img.shape[1:], mode='nearest').long().squeeze(0).squeeze(0)
			pred_raw = F.interpolate(pred_raw.unsqueeze(0).unsqueeze(0).float(), size=img.shape[1:], mode='nearest').long().squeeze(0).squeeze(0)
			seg_logits = seg_logits.squeeze(0)  # tensor (cls, w, h)
			seg_logits = seg_logits.cpu()  # Move seg_logits to CPU
			img = img.cpu()  # Move img to CPU
			probs = seg_logits.softmax(0)  # tensor (cls, 512, 1024)
			points = []
			# find the max prob point for each class and add to points
			for ii in range(probs.shape[0]):  # iterate over each class
				_, max_index = probs[i].view(-1).max(0)  # get the index of the max probability
				max_point = np.unravel_index(max_index, probs[ii].shape)
				points.append(max_point)  # add to the list of points

			# turn to (N, 2) array and scale the value to (0, 1)
			def get_points_from_gt(mask):
				""" get points from ground truth mask
				for each patch, give the middle as the point. Note that one class could have multiple points
				mask: (H, W) with each pixel value as class index
				return: 
					(N, 2) array with each row as (x, y) point
				"""
				from scipy.ndimage import label, distance_transform_edt
				points = []
				classes = np.unique(mask)
				if 0 in classes:  # Assume 0 is the background class
					classes = classes[1:]

				for cls in classes:
					class_mask = (mask == cls)
					labeled_patches, num_features = label(class_mask)

					for i in range(1, num_features + 1):  # Start from 1 to ignore the background
						patch_mask = (labeled_patches == i)
						# 1. if self.kwargs.sam_model.use_center
						# Perform distance transform
						# distances = distance_transform_edt(patch_mask)
						# # Find the position of the maximum distance (center of the largest inscribed circle)
						# max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)
						# points.append(max_dist_idx[::-1])  # Append (y, x) to points list (need to reverse indices)
						# 2. random select multiple based on patch size
						PIXELS_PER_POINT = 1000
						# count patch size (num of non-zero)
						patch_size = np.count_nonzero(patch_mask)
						num_points = max(patch_size // PIXELS_PER_POINT, 2)
						# randomly select num_points
						indices = np.where(patch_mask == 1)
						indices = np.array(indices).T
						np.random.shuffle(indices)
						indices = indices[:num_points]
						# change x,y order
						indices = indices[:, ::-1]
						points.extend(indices)


				return np.array(points)
			
			# points = get_points_from_gt(y.cpu().numpy()) # 512, 1024 w, h
			# points = np.array(points) / np.array([h, w]) # ! check if point ar right
			points = make_grid(10)
			# get first half
			# points = points[:6]
			
			# points -> sam pred
			self.mask_generator.point_grids[0] = points
			img_np = imgs[i].cpu()
			img_np = img_np.permute(1,2,0).numpy()
			sam_res = self.mask_generator.generate(img_np) ### ! TODO DEBUG
			automask_bwh = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
			for ii, mask in enumerate(sam_res):
				indices = np.where(mask["segmentation"] == 1)
				automask_bwh[indices] = ii + 1
			automask_bwh = torch.from_numpy(automask_bwh).long()

			# automask to pred
			def automask_to_pred(sam_mask, ref_mask):  
				"""  
				Make a pred_mask based on sam_mask and ref_mask.  
				sam_mask only includes patches without category information, so we need ref_mask.  
				The logic is to assign the category with the largest iou with the sam_mask.  
				sam_mask: (H, W) with each pixel value as patch index
				ref_mask: (H, W) with each pixel value as category index
				"""  
				# Get unique non-zero values from sam_mask  
				unique_masks = torch.unique(sam_mask)[1:]  
				
				# Initialize pred_mask with the same shape as ref_mask  
				pred_mask = ref_mask.clone()  
				
				# Iterate over each unique mask in sam_mask  
				for unique_mask in unique_masks:  
					# Find the IoU between the current unique mask and the reference mask  
					iou = torch.sum((sam_mask == unique_mask) & (ref_mask == ref_mask), dim=(0,1)) / torch.sum((sam_mask == unique_mask), dim=(0,1))  
					
					# Assign the category with the maximum IoU to the corresponding pixels in pred_mask  
					pred_mask[sam_mask == unique_mask] = torch.argmax(iou)  
				
				return pred_mask
			
			# sam_pred = automask_to_pred(automask_bwh.clone().to(y.device), y) # use gt
			sam_pred = automask_to_pred(automask_bwh.clone().to(y.device), pred_raw) # use pred
			res[i].pred_sem_seg.data = F.interpolate(sam_pred.unsqueeze(0).unsqueeze(0).float(), size=size_ori, mode='nearest').long().squeeze(0)

			# plot img, pseudo label mask, and points
			def mask_to_color(mask):
				"""
				Convert a class mask tensor to an RGB color mask using a colormap,
				with colors represented as [0, 255] integers. 
				ps. different class is indicated by different uint, but may not be continuous

				Parameters:
				- mask: PyTorch tensor of shape (H, W), where each entry represents a class index.

				Returns:
				- A NumPy array of shape (H, W, 3), representing RGB colors for each pixel
				with integers in the range [0, 255].
				"""
				# Get unique class indices from the mask and sort them
				mask = mask.clone().detach()
				unique_classes = torch.unique(mask)
				unique_classes = unique_classes[unique_classes != 255]  # remove 255 if it's used for a special purpose like ignore_index

				# Map each unique class index to a position in the range [0, number of unique classes - 1]
				index_mapping = {int(class_idx): idx for idx, class_idx in enumerate(unique_classes)}

				# Create an array of shape (num_unique_classes, 3) for RGB colors, taking modulo with a colormap size
				colormap_size = 20  # Number of distinct colors in the colormap
				colors = plt.cm.tab20(np.linspace(0, 1, colormap_size))

				# If mask is on GPU, move to CPU and convert to numpy array
				mask_array = mask.cpu().numpy() if mask.is_cuda else mask.numpy()

				# Create an empty RGB array
				height, width = mask_array.shape
				color_mask = np.zeros((height, width, 3), dtype=np.uint8)

				# Map each class index to its color
				for class_idx in unique_classes:
					mapped_index = index_mapping[int(class_idx)]
					color = (colors[mapped_index % colormap_size][:3] * 255).astype(np.uint8)
					color_mask[mask_array == class_idx.item()] = color

				return color_mask
			
			def seg_plot(img, mask=None, points=None, save_path=None):
				"""
				Plot an image with an optional mask and points, and save to save_path if provided.

				Parameters:
				- img: A PyTorch tensor of shape (C, H, W) representing the image.
				- mask: A PyTorch tensor of shape (H, W), where each entry represents a class index.
				- points: A list of tuples/lists with points to plot (e.g., [(x1, y1), (x2, y2), ...]).
				- save_path: A string representing the file path to save the image.
				"""
				fig, ax = plt.subplots()
				ax.imshow(img.permute(1, 2, 0).cpu().numpy(), aspect='auto')
				w, h = img.shape[1:]

				if mask is not None:
					color_mask = mask_to_color(mask)
					ax.imshow(color_mask, aspect='auto', alpha=0.5)

				if points is not None:
					for point in points:
						# ax.scatter(point[0], point[1], s=10, c='r')
						ax.scatter(int(point[0]*h), int(point[1]*w), s=2, c='b')
						ax.scatter(int(point[0]*h), int(point[1]*w), s=1, c='w')

				if save_path:
					plt.savefig(save_path, dpi=300)
				plt.close(fig)

			seg_plot(imgs[i], mask=y.cpu(), points=points, save_path=f"./debug/_gt.png")
			seg_plot(imgs[i], mask=seg_logits.max(0)[1].cpu(), points=points, save_path=f"./debug/_pred.png")
			seg_plot(imgs[i], mask=automask_bwh, points=points, save_path=f"./debug/_sam_automask.png")
			seg_plot(imgs[i], mask=sam_pred, points=points, save_path=f"./debug/_sam_pred.png")
			

		return res

	def test_step_plus(self, data):
		"""``BaseModel`` implements ``test_step`` the same as ``val_step``.

		Args:
			data (dict or tuple or list): Data sampled from dataset.

		Returns:
			list: The predictions of given data.
		"""
		data = aug_to_first(data)
		data = self.data_preprocessor(data, False)
		return self._run_forward_plus(data, mode='predict')  # type: ignore

	def _run_forward_plus(self, data, mode):
		"""Unpacks data for :meth:`forward`

		Args:
			data (dict or tuple or list): Data sampled from dataset.
			mode (str): Mode of forward.

		Returns:
			dict or list: Results of training or testing mode.
		"""
		if isinstance(data, dict):
			results = self.forward_plus(**data, mode=mode)
		elif isinstance(data, (list, tuple)):
			results = self.forward_plus(*data, mode=mode)
		else:
			raise TypeError('Output of `data_preprocessor` should be '
							f'list, tuple or dict, but got {type(data)}')
		return results

	def forward_plus(self, inputs, data_samples = None, mode = 'tensor'):
		"""The unified entry for a forward process in both training and test.

		The method should accept three modes: "tensor", "predict" and "loss":

		- "tensor": Forward the whole network and return tensor or tuple of
		tensor without any post-processing, same as a common nn.Module.
		- "predict": Forward and return the predictions, which are fully
		processed to a list of :obj:`SegDataSample`.
		- "loss": Forward and return a dict of losses according to the given
		inputs and data samples.

		Note that this method doesn't handle neither back propagation nor
		optimizer updating, which are done in the :meth:`train_step`.

		Args:
			inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
				general.
			data_samples (list[:obj:`SegDataSample`]): The seg data samples.
				It usually includes information such as `metainfo` and
				`gt_sem_seg`. Default to None.
			mode (str): Return what kind of value. Defaults to 'tensor'.

		Returns:
			The return type depends on ``mode``.

			- If ``mode="tensor"``, return a tensor or a tuple of tensor.
			- If ``mode="predict"``, return a list of :obj:`DetDataSample`.
			- If ``mode="loss"``, return a dict of tensor.
		"""
		if mode == 'loss':
			return self.loss_plus(inputs, data_samples)
		elif mode == 'predict':
			return self.predict_plus(inputs, data_samples)
		elif mode == 'tensor':
			return self._forward(inputs, data_samples)
		else:
			raise RuntimeError(f'Invalid mode "{mode}". '
							   'Only supports loss, predict and tensor mode')

	def loss_plus(self, inputs, data_samples):
		"""Calculate losses from a batch of inputs and data samples.

		Args:
			inputs (Tensor): Input images.
			data_samples (list[:obj:`SegDataSample`]): The seg data samples.
				It usually includes information such as `metainfo` and
				`gt_sem_seg`.

		Returns:
			dict[str, Tensor]: a dictionary of loss components
		"""

		x = self.extract_feat(inputs)

		losses = dict()

		loss_decode = self._decode_head_forward_train(x, data_samples)
		losses.update(loss_decode)

		if self.with_auxiliary_head:
			loss_aux = self._auxiliary_head_forward_train(x, data_samples)
			losses.update(loss_aux)

		return losses

	def predict_plus(self, inputs, data_samples):
		"""Predict results from a batch of inputs and data samples with post-
		processing.

		Args:
			inputs (Tensor): Inputs with shape (N, C, H, W).
			data_samples (List[:obj:`SegDataSample`], optional): The seg data
				samples. It usually includes information such as `metainfo`
				and `gt_sem_seg`.

		Returns:
			list[:obj:`SegDataSample`]: Segmentation results of the
			input images. Each SegDataSample usually contain:

			- ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
			- ``seg_logits``(PixelData): Predicted logits of semantic
				segmentation before normalization.
		"""
		if data_samples is not None:
			batch_img_metas = [
				data_sample.metainfo for data_sample in data_samples
			]
		else:
			batch_img_metas = [
				dict(
					ori_shape=inputs.shape[2:],
					img_shape=inputs.shape[2:],
					pad_shape=inputs.shape[2:],
					padding_size=[0, 0, 0, 0])
			] * inputs.shape[0]

		seg_logits, feats = self.inference_plus(inputs, batch_img_metas)

		res = self.postprocess_result(seg_logits, data_samples)

		# @waybaba add feat into TODO
		# copy code from self.postprocess_result
		batch_size, C, H, W = feats.shape
		for i in range(len(res)):
			
			img_meta = data_samples[i].metainfo
			# remove padding area
			if 'img_padding_size' not in img_meta:
				padding_size = img_meta.get('padding_size', [0] * 4)
			else:
				padding_size = img_meta['img_padding_size']
			padding_left, padding_right, padding_top, padding_bottom =\
				padding_size
			# i_feats shape is 1, C, H, W after remove padding
			i_feats = feats[i:i + 1, :,
										padding_top:H - padding_bottom,
										padding_left:W - padding_right]

			flip = img_meta.get('flip', None)
			if flip:
				flip_direction = img_meta.get('flip_direction', None)
				assert flip_direction in ['horizontal', 'vertical']
				if flip_direction == 'horizontal':
					i_feats = i_feats.flip(dims=(3, ))
				else:
					i_feats = i_feats.flip(dims=(2, ))
			i_feats = i_feats.squeeze(0)
			# resize as original shape
			# i_feats = resize( # ! this should be uncomment to make related code works
			# 	i_feats.unsqueeze(0)
			# 	size=img_meta['ori_shape'],
			# 	mode='bilinear',
			# 	align_corners=self.align_corners,
			# 	warning=False).squeeze(0)
			res[i].feats = PixelData()
			res[i].feats.data = i_feats

		return res

	def inference_plus(self, inputs, batch_img_metas):
		"""Inference with slide/whole style.

		Args:
			inputs (Tensor): The input image of shape (N, 3, H, W).
			batch_img_metas (List[dict]): List of image metainfo where each may
				also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
				'ori_shape', 'pad_shape', and 'padding_size'.
				For details on the values of these keys see
				`mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

		Returns:
			Tensor: The segmentation results, seg_logits from model of each
				input image.
		"""
		assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
			f'Only "slide" or "whole" test mode are supported, but got ' \
			f'{self.test_cfg["mode"]}.'
		ori_shape = batch_img_metas[0]['ori_shape']
		if self.test_cfg.mode == 'slide':
			assert NotImplementedError("slide inference not implemented")
			seg_logit = self.slide_inference(inputs, batch_img_metas)
		else:
			seg_logit, feats = self.whole_inference_plus(inputs, batch_img_metas)

		return seg_logit, feats

	def whole_inference_plus(self, inputs, batch_img_metas):
	
		# seg_logits, feats= self.encode_decode(inputs, batch_img_metas)
		seg_logits, feats= self.encode_decode_with_feats_plus(inputs, batch_img_metas)

		return seg_logits, feats

	def encode_decode_with_feats_plus(self, inputs, batch_img_metas):
		encode_out = self.extract_feat(inputs)
		# seg_logits = self.decode_head.predict(x, batch_img_metas,
		# 									  self.test_cfg) # @waybaba replace by the following two
		# seg_logits = self.decode_head.forward(encode_out) # @waybaba replace by the folowing
		# seg_logits = self.predict_by_feat(seg_logits, batch_img_metas) # @waybaba move the end of this function



		### from models.xxx TODO self.decode_head.forward(encode_out)
		inputs = self.decode_head._transform_inputs(encode_out)
		outs = []
		for idx in range(len(inputs)):
			x = inputs[idx]
			conv = self.decode_head.convs[idx]
			outs.append(
				resize(
					input=conv(x),
					size=inputs[0].shape[2:],
					mode=self.decode_head.interpolate_mode,
					align_corners=self.decode_head.align_corners))

		feats = self.decode_head.fusion_conv(torch.cat(outs, dim=1))

		seg_logits = self.decode_head.cls_seg(feats)
		### 

		# resize to imgsize as in self.predict_by_feat(seg_logits, batch_img_metas)
		feats = self.decode_head.predict_by_feat(feats, batch_img_metas)
		seg_logits = self.decode_head.predict_by_feat(seg_logits, batch_img_metas)

		return seg_logits, feats

	# utils

# ! TODO when use normal init, would cause pseudo label become unaccurate
@MODELS.register_module()
class MixVisionTransformerTPT(MixVisionTransformer):
	""" Modifications: add TPT token to original model
		tpt_cfg:
			num_tokens
			weight_init_type
		## make token_prompt parameters - len first (num_tokens, 1, embed_dims)
		if int, then use the same number of tokens for all layers
		elif float, then use the same percentage of tokens for all layers
		elif list
			assert len(list) == len(layers)
			assert does not contains float or int at the same time (0 does not count)
			e.g. [0,0,10,0],[10,0,0,0], [0.1,0,0,0]
			ps. 0 means no token
	"""
	def __init__(self,
				 in_channels=3,
				 embed_dims=64,
				 num_stages=4,
				 num_layers=[3, 4, 6, 3],
				 num_heads=[1, 2, 4, 8],
				 patch_sizes=[7, 3, 3, 3],
				 strides=[4, 2, 2, 2],
				 sr_ratios=[8, 4, 2, 1],
				 out_indices=(0, 1, 2, 3),
				 mlp_ratio=4,
				 qkv_bias=True,
				 drop_rate=0.,
				 attn_drop_rate=0.,
				 drop_path_rate=0.,
				 act_cfg=dict(type='GELU'),
				 norm_cfg=dict(type='LN', eps=1e-6),
				 pretrained=None,
				 init_cfg=None,
				 with_cp=False,
				 tpt_cfg=None,
				 vpt_cfg=None,
				 ):
		# super().__init__(init_cfg=init_cfg)
		super().__init__(
			in_channels=in_channels,
			embed_dims=embed_dims,
			num_stages=num_stages,
			num_layers=num_layers,
			num_heads=num_heads,
			patch_sizes=patch_sizes,
			strides=strides,
			sr_ratios=sr_ratios,
			out_indices=out_indices,
			mlp_ratio=mlp_ratio,
			qkv_bias=qkv_bias,
			drop_rate=drop_rate,
			attn_drop_rate=attn_drop_rate,
			drop_path_rate=drop_path_rate,
			act_cfg=act_cfg,
			norm_cfg=norm_cfg,
			pretrained=pretrained,
			init_cfg=init_cfg,
			with_cp=with_cp)
		# vpt and tpt should not be used at the same time but should be set
		assert vpt_cfg is not None, "vpt_cfg should not be None"
		assert tpt_cfg is not None, "tpt_cfg should not be None"
		assert not (vpt_cfg.turn_on and tpt_cfg.turn_on), "vpt and tpt should not be used at the same time"
		self.tpt_cfg, self.vpt_cfg = tpt_cfg, vpt_cfg
		
		if tpt_cfg.turn_on:
			self.token_prompts = []
			### make token_prompt parameters - len first (num_tokens, 1, embed_dims)
			# if int, then use the same number of tokens for all layers
			# elif float, then use the same percentage of tokens for all layers
			# elif list
			#     assert len(list) == len(layers)
			#     assert does not contains float or int at the same time (0 does not count)
			#     e.g. [0,0,10,0],[10,0,0,0], [0.1,0,0,0]
			#     ps. 0 means no token
			if isinstance(tpt_cfg.num_tokens, int):
				self.num_tokens = [tpt_cfg.num_tokens] * len(self.layers)
			elif isinstance(tpt_cfg.num_tokens, float):
				raise NotImplementedError("can not use float for num_tokens")
			elif isinstance(tpt_cfg.num_tokens, str):
				raise TypeError("num_tokens can not be str")
			elif isinstance(tpt_cfg.num_tokens, Iterable):
				assert len(tpt_cfg.num_tokens) == len(self.layers)
				assert all([isinstance(x, int) or x is None for x in tpt_cfg.num_tokens])
				self.num_tokens = tpt_cfg.num_tokens
			else:
				raise NotImplementedError("num_tokens must be int, float or list")
			if tpt_cfg.weight_init_type == "zero":
				init_func_q = nn.init.zeros_
				init_func_kv = nn.init.zeros_
			elif tpt_cfg.weight_init_type == "normal":
				init_func_q = nn.init.normal_
				init_func_kv = nn.init.normal_
			elif tpt_cfg.weight_init_type == "kv_normal":
				init_func_q = nn.init.zeros_
				init_func_kv = nn.init.normal_
			elif tpt_cfg.weight_init_type == "q_normal":
				init_func_q = nn.init.normal_
				init_func_kv = nn.init.zeros_
			else:
				raise NotImplementedError(f"weight_init_type {tpt_cfg.weight_init_type} is not implemented")
			# generate random variables for token prompts
			for i, layer in enumerate(self.layers):
				if self.num_tokens[i] is not None:	
					w_q = torch.zeros(self.num_tokens[i], 1, self.embed_dims*self.num_heads[i])
					w_kv = torch.zeros(self.num_tokens[i], 1, self.embed_dims*self.num_heads[i])
					self.token_prompts.append({
						"q": nn.Parameter(init_func_q(w_q)),
						"kv": nn.Parameter(init_func_kv(w_kv)),
					})
				else:
					self.token_prompts.append(None)
			# register parameters
			for i, layer in enumerate(self.layers):
				if self.token_prompts[i] is not None:
					self.register_parameter(f"token_prompt_q_{i}", self.token_prompts[i]["q"])
					self.register_parameter(f"token_prompt_kv_{i}", self.token_prompts[i]["kv"])
			tpt_gates = nn.Parameter(torch.zeros(len(self.layers)))
			self.register_parameter("tpt_gates", tpt_gates)
		
		if vpt_cfg.turn_on:
			self.visual_prompt_module = utils.VisutalPrompter(
				**{k: v  for k, v in vpt_cfg.items() if k not in ["turn_on"]},
				dynamic_cfg={
					"image_size": (1024,512) # ! dynamic for other dataset
				}
			)

	def forward(self, x):
		if self.vpt_cfg.turn_on:
			self.visual_prompt_module(x)
		
		outs = []

		for i, layer in enumerate(self.layers):
			x, hw_shape = layer[0](x)
			for block in layer[1]:
				# x = block(x, hw_shape) # ! origin from mmseg
				x = self.TEL_forward(
					block, x, hw_shape, 
					self.token_prompts[i] if self.tpt_cfg.turn_on else None,
					self.tpt_gates[i] if self.tpt_cfg.turn_on else None
				)
			x = layer[2](x)
			x = nlc_to_nchw(x, hw_shape)
			if i in self.out_indices:
				outs.append(x)

		return outs	

	def TEL_forward(self, model, x, hw_shape, token_prompt=None, gate=None):

		def _inner_forward(x):
			# x = model.attn(model.norm1(x), hw_shape, identity=x)
			x = self.EMHA_forward(model.attn, model.norm1(x), hw_shape, identity=x, token_prompt=token_prompt, gate=gate)
			# x = model.ffn(model.norm2(x), hw_shape, identity=x)
			x = self.MixFFN_forward(model.ffn, model.norm2(x), hw_shape, identity=x, token_prompt=token_prompt)
			return x

		if model.with_cp and x.requires_grad:
			x = cp.checkpoint(_inner_forward, x)
		else:
			x = _inner_forward(x)
		return x

	def EMHA_forward(self, model, x, hw_shape, identity=None, token_prompt=None, gate=None):
		x_q = x
		if model.sr_ratio > 1:
			x_kv = nlc_to_nchw(x, hw_shape)
			x_kv = model.sr(x_kv)
			x_kv = nchw_to_nlc(x_kv)
			x_kv = model.norm(x_kv)
		else:
			x_kv = x

		if identity is None:
			identity = x_q

		# Because the dataflow('key', 'query', 'value') of
		# ``torch.nn.MultiheadAttention`` is (num_query, batch,
		# embed_dims), We should adjust the shape of dataflow from
		# batch_first (batch, num_query, embed_dims) to num_query_first
		# (num_query ,batch, embed_dims), and recover ``attn_output``
		# from num_query_first to batch_first.
		if model.batch_first:
			x_q = x_q.transpose(0, 1)
			x_kv = x_kv.transpose(0, 1)
		##### ! token prompt - start
		# add token prompt
		# x_q,x_kv (len, batch, dim)  token_prompt (token_num, 1, dim)
		tpt_num = None
		if token_prompt is not None:
			token_prompt["q"] = token_prompt["q"].to(x_q.device)
			token_prompt_q = token_prompt["q"].expand(-1, x_q.shape[1], -1)
			x_q = torch.cat([x_q, token_prompt_q], dim=0)
			token_prompt["kv"] = token_prompt["kv"].to(x_kv.device)
			token_prompt_kv = token_prompt["kv"].expand(-1, x_kv.shape[1], -1)
			x_kv = torch.cat([x_kv, token_prompt_kv], dim=0)
			tpt_num = token_prompt_kv.shape[0]
		##### ! token prompt - end
		# out = model.attn(query=x_q, key=x_kv, value=x_kv)[0] # replace to ->
		# out, _ = F.multi_head_attention_forward(
		# 				x_q, x_kv, x_kv, model.attn.embed_dim, model.attn.num_heads,
		# 				model.attn.in_proj_weight, model.attn.in_proj_bias,
		# 				None, None, model.attn.add_zero_attn,
		# 				model.attn.dropout, model.attn.out_proj.weight, model.attn.out_proj.bias,
		# 				training=model.attn.training) # ! training?
		# def multi_head_attention_forward(
		#     query: Tensor,
		#     key: Tensor,
		#     value: Tensor,
		#     embed_dim_to_check: int,
		#     num_heads: int,
		#     in_proj_weight: Optional[Tensor],
		#     in_proj_bias: Optional[Tensor],
		#     bias_k: Optional[Tensor],
		#     bias_v: Optional[Tensor],
		#     add_zero_attn: bool,
		#     dropout_p: float,
		#     out_proj_weight: Tensor,
		#     out_proj_bias: Optional[Tensor],
		#     training: bool = True,
		#     key_padding_mask: Optional[Tensor] = None,
		#     need_weights: bool = True,
		#     attn_mask: Optional[Tensor] = None,
		#     use_separate_proj_weight: bool = False,
		#     q_proj_weight: Optional[Tensor] = None,
		#     k_proj_weight: Optional[Tensor] = None,
		#     v_proj_weight: Optional[Tensor] = None,
		#     static_k: Optional[Tensor] = None,
		#     static_v: Optional[Tensor] = None,
		#     average_attn_weights: bool = True,
		# ) -> Tuple[Tensor, Optional[Tensor]]:
		num_heads = model.attn.num_heads
		embed_dim = model.attn.embed_dim
		query, key, value = x_q, x_kv, x_kv
		in_proj_weight, in_proj_bias = model.attn.in_proj_weight, model.attn.in_proj_bias
		bias_k, bias_v = model.attn.bias_k, model.attn.bias_v
		add_zero_attn = model.attn.add_zero_attn
		out_proj_weight, out_proj_bias = model.attn.out_proj.weight, model.attn.out_proj.bias
		training = model.attn.training
		dropout_p = model.attn.dropout
		if not model.training:
			dropout_p = 0.0
		
		attn_mask = None

		tgt_len, bsz, embed_dim = query.shape
		src_len, _, _ = key.shape
		if isinstance(embed_dim, torch.Tensor):
			# embed_dim can be a tensor when JIT tracing
			head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
		else:
			head_dim = embed_dim // num_heads
		assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
		q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
		q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
		k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
		v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
		src_len = k.size(1)
		attn_output, attn_output_weights = _scaled_dot_product_attention_llama_adapter(
			q, k, v, attn_mask, 
			tpt_num=tpt_num,
			gate=gate, # TODO !
		)
		attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
		attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
		attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
		attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
		attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
		attn_output, attn_output_weights
		out = attn_output


		##### ! token prompt - start
		# remove token prompt from output
		if token_prompt is not None:
			out = out[:-token_prompt["q"].shape[0]]
		##### ! token prompt - end


		if model.batch_first:
			out = out.transpose(0, 1)

		return identity + model.dropout_layer(model.proj_drop(out))

	def MixFFN_forward(self, model, x, hw_shape, identity=None, token_prompt=None):
		out = nlc_to_nchw(x, hw_shape)
		out = model.layers(out)
		out = nchw_to_nlc(out)
		if identity is None:
			identity = x
		return identity + model.dropout_layer(out)


### MAIN ###
@LOOPS.register_module()
class TestLoopWrapper(TestLoop):
	def __init__(self, **kwargs) -> None:
		super().__init__(
			kwargs["runner"],
			kwargs["dataloader"],
			kwargs["evaluator"],
			kwargs["fp16"] if "fp16" in kwargs else False,
		)
		# remove some args
		for _ in ["runner", "dataloader", "evaluator", "fp16"]:
			if _ in kwargs:
				del kwargs[_]
		self.kwargs = Config(kwargs)

	### hooks
	def after_test_epoch(self, runner, metrics):
		# runner.logger.info('after test epoch')
		# revert_sync_batchnorm(runner.model) # ! ??? shoud we handle?
		# runner.logger.info('after test epoch done')
		runner.visualizer.add_scalars({
			"test/"+k: v for k, v in metrics.items()
		})
		import wandb
		# columns=[c[4:] for c in metrics.keys() if "IoU_" in c]
		# wandb_table = wandb.Table(data={c[4:]: v for c, v in metrics.items() if "IoU_" in c})
		wandb_table = wandb.Table(columns=[c[4:] for c in metrics.keys() if "IoU_" in c], data=[[v for c, v in metrics.items() if "IoU_" in c]])
		runner.visualizer.get_backend("WandbVisBackend").experiment.log({
			"IoU_Table": wandb_table
		})

	def before_test(self, runner):
		# predict mode switch
		assert sum([self.kwargs.proto_predict.turn_on, self.kwargs.sam_predict.turn_on, self.kwargs.sam_model.turn_on]) <= 1, \
			"only one of proto_predict, sam_predict, sam_model should be on"
		if self.kwargs.proto_predict.turn_on:
			raise ValueError("I change the feats dimmention to original, need to recover it to make this work")
			runner.model.test_step = partial(runner.model.test_step_proto_predict, cfg=self.kwargs.proto_predict)
		elif self.kwargs.sam_predict.turn_on:
			runner.model.test_step = partial(runner.model.test_step_sam_predict, cfg=self.kwargs.sam_predict)
		elif self.kwargs.sam_model.turn_on:
			from sam.scripts.pure_model import SegmentAnythingModelWrapper
			from sam.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
			args = self.kwargs.sam_model
			sam_model = sam_model_registry[args.sam.model_type](checkpoint=args.sam.checkpoint)
			self.mask_generator = SamAutomaticMaskGenerator(sam_model)
			if isinstance(runner.model, SegTTAModelWrapper):
				runner.model.module.test_step = partial(runner.model.module.test_step_sam_model_predict, mask_generator=self.mask_generator, cfg=self.kwargs.sam_model)
			elif isinstance(runner.model, EncoderDecoderWrapper):
				runner.model.test_step = partial(runner.model.test_step_sam_model_predict, mask_generator=self.mask_generator, cfg=self.kwargs.sam_model)
			else: raise NotImplementedError("only support SegTTAModelWrapper and EncoderDecoderWrapper")

	### utils 

	### main
	def run(self) -> dict:
		"""Launch test."""
		self.before_test(self.runner) # @waybaba
		self.runner.call_hook('before_test')
		self.runner.call_hook('before_test_epoch')
		self.runner.model.eval()
		for idx, data_batch in enumerate(self.dataloader):
			self.run_iter(idx, data_batch)

		# compute metrics
		metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
		self.runner.call_hook('after_test_epoch', metrics=metrics)
		self.after_test_epoch(self.runner, metrics) # @waybaba
		self.runner.call_hook('after_test')
		return metrics

	@torch.no_grad()
	def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
		"""Iterate one mini-batch.

		Args:
			data_batch (Sequence[dict]): Batch of data from dataloader.
		"""
		self.runner.call_hook(
			'before_test_iter', batch_idx=idx, data_batch=data_batch)

		runner, batch, batch_idx = self.runner, data_batch, idx
		IS_DEEPLAB = is_deeplab(runner.model)
		IS_TTA = is_tta(batch)
		
		### init
		to_logs = {}
		# inputs, data_samples = batch["inputs"], batch["data_samples"]
		if not hasattr(self, "model_ori"): # source model for pseudo
			assert batch_idx == 0, "model_ori should be init only once"
			self.model_ori = deepcopy(runner.model)
		batch_noaug = deepcopy(aug_to_first(batch))

		### ema model
		# 1. update ema
		# 2. take model back to ema
		if self.kwargs.ema.turn_on:
			if not hasattr(self, "model_ema"): # init self.model_ema
				self.model_ema = deepcopy(runner.model)
				for param in self.model_ema.parameters():
					param.detach_()
					param.requires_grad = False
			self.model_ema = param_migrate(self.model_ema, runner.model, self.kwargs.ema.rho) # update ema
			if self.kwargs.ema.cur_recover:
				runner.model = param_migrate(runner.model, self.model_ema, 1.0) # model to ema
				if not hasattr(self, "optim_state_dict_first"): # reset optimizer state to avoid mistake Adam ...
					self.optim_state_dict_first = deepcopy(runner.optim_wrapper.state_dict()) 
				else:
					runner.optim_wrapper.load_state_dict(self.optim_state_dict_first)
			if self.kwargs.ema.mid_pred:
				assert not self.kwargs.ema.cur_recover, f"ema.cur_recover and ema.mid_pred can not be True at the same time"
				if not hasattr(self, "model_mid"): # init self.model_ema
					self.model_mid = deepcopy(runner.model)
					self.optim_wrapper_mid = build_optim_wrapper(self.model_mid, runner.optim_wrapper_cfg)
					for group in self.optim_wrapper_mid.optimizer.param_groups:
						group['lr'] = group['lr'] * self.kwargs.ema.mid_lr_scale
				self.model_mid = param_migrate(self.model_mid, self.model_ema, 1.0) # update ema

		### pseudo label
		# -> batch_pseudoed
		# -> hign_conf_mask (then mask to batch_pseudoed)
		if True:
			if self.kwargs.pseudo_label.type == "ori":
				batch_pseudoed_model = self.model_ori
			elif self.kwargs.pseudo_label.type == "cur":
				batch_pseudoed_model = runner.model
			elif self.kwargs.pseudo_label.type == "ema":
				batch_pseudoed_model = self.model_ema
			else:
				raise ValueError(f"unknown pseudo_label type {self.kwargs.pseudo_label.type}")
			with torch.no_grad():
				batch_pseudoed = batch_pseudoed_model.test_step(batch_noaug) if IS_DEEPLAB else batch_pseudoed_model.test_step_plus(batch_noaug)
			if self.kwargs.high_conf_mask.turn_on:
				if batch_idx == 0: 
					assert not hasattr(self, "seg_masker"), "seg_masker should not be init twice"
					self.seg_masker = SegMasker(use_history=self.kwargs.high_conf_mask.use_history, class_specific=self.kwargs.high_conf_mask.class_specific)
				if self.kwargs.high_conf_mask.metric == "confidence":
					seg_conf = F.softmax(batch_pseudoed[0].seg_logits.data, dim=0).max(0)[0]
					self.seg_masker.add_to_buffer(seg_conf, batch_pseudoed[0].pred_sem_seg.data[0])
					hign_conf_mask = self.seg_masker.cal_mask(
						seg_conf,
						batch_pseudoed[0].pred_sem_seg.data[0],
						self.kwargs.high_conf_mask.top_p,
					)
				elif self.kwargs.high_conf_mask.metric == "uncertainty":
					assert IS_TTA, "uncertainty only support tta"
					augmented_data = batch
					def compute_var_mask(batch, batch_pseudoed_model, self_kwargs):
						"""
						Computes the variance mask (var_mask) for augmented data, either globally or class-specific.

						Parameters:
						- batch: The input batch data.
						- batch_pseudoed_model: The model used for predictions.
						- self_kwargs: Additional arguments, including 'high_conf_mask.top_p'.

						Returns:
						- var_mask: The computed variance mask.
						"""
						import gc
						with torch.no_grad():
							# Data preparation
							augmented_data = batch
							num_augmentations = len(augmented_data[next(iter(augmented_data))])
							aug_data_list = [{k: v[idx] for k, v in augmented_data.items()} for idx in range(num_augmentations)]

							# Predictions
							predictions = [batch_pseudoed_model.test_step_plus(data) for data in aug_data_list]
							softmax_probs = [F.softmax(pred[0].seg_logits.data, dim=0) for pred in predictions]  # [(cls,h,w)]
							softmax_probs = torch.stack(softmax_probs)  # (num_aug, cls, h, w)

							# Compute variance & mask
							if self_kwargs.high_conf_mask.class_specific:
								var_masks = []
								for c in range(softmax_probs.size(1)):  # Iterate over each class
									class_variance = softmax_probs[:, c].var(dim=0)  # Variance for each class
									class_specific_threshold = class_variance.quantile(self_kwargs.high_conf_mask.top_p)
									class_var_mask = class_variance < class_specific_threshold
									var_masks.append(class_var_mask)
								var_mask = torch.stack(var_masks).any(dim=0)  # Combine class-specific masks
							else:
								variance = softmax_probs.var(dim=0).sum(dim=0)  # Global variance
								var_mask = variance < variance.quantile(self_kwargs.high_conf_mask.top_p)

							# Clean up
							gc.collect()
							torch.cuda.empty_cache()

							return var_mask
					
					hign_conf_mask = compute_var_mask(batch, batch_pseudoed_model, self.kwargs)
					# conf_mask = var_mask & (batch_pseudoed[0].gt_sem_seg.data[0] != 255)
					# right_mask = (batch_pseudoed[0].gt_sem_seg.data[0] == batch_pseudoed[0].pred_sem_seg.data[0]) & (batch_pseudoed[0].gt_sem_seg.data[0] != 255)
					# to_logs["conf_var_iou"] = (conf_mask & right_mask).sum() / (conf_mask | right_mask).sum()
					# to_logs["conf_var_TP"] = (conf_mask & right_mask).sum() / conf_mask.sum()
				else:
					raise ValueError(f"unknown high_conf_mask type {self.kwargs.high_conf_mask.type}")
				conf_mask = hign_conf_mask & (batch_pseudoed[0].gt_sem_seg.data[0] != 255)
				right_mask = (batch_pseudoed[0].gt_sem_seg.data[0] == batch_pseudoed[0].pred_sem_seg.data[0]) & (batch_pseudoed[0].gt_sem_seg.data[0] != 255)
				to_logs["conf_iou"] = (conf_mask & right_mask).sum() / (conf_mask | right_mask).sum()
				to_logs["conf_TP"] = (conf_mask & right_mask).sum() / conf_mask.sum()
				### DAT variance
				# apply mask
				sem_seg_ = batch_pseudoed[0].pred_sem_seg.data[0]
				sem_seg_[~hign_conf_mask] = 255
				if self.kwargs.high_conf_mask.inflation:
					sem_seg_ = inflation(sem_seg_, self.kwargs.high_conf_mask.inflation)
				batch_pseudoed[0].pred_sem_seg = PixelData()
				batch_pseudoed[0].pred_sem_seg.data = sem_seg_.unsqueeze(0)
		### choose adapt
		# set data_batch_for_adapt
		if self.kwargs.slide_adapt:
			raise ValueError(f"slide_adapt is not implemented")
			batch_pseudoed_slided = {"inputs": [], "data_samples": []}
			batch_slided = {"inputs": [], "data_samples": []}
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
		if self.kwargs.slide_adapt:
			data_batch_for_adapt = deepcopy(batch_pseudoed_slided) \
				if self.kwargs.pseudo_label.turn_on else deepcopy(batch_slided)
		else:
			data_batch_for_adapt = batch_noaug
			if self.kwargs.pseudo_label.turn_on:
				for i in range(len(data_batch_for_adapt["data_samples"])):
					data_batch_for_adapt["data_samples"][i].gt_sem_seg = batch_pseudoed[i].pred_sem_seg
		data_batch_for_adapt_bak = deepcopy(data_batch_for_adapt)

		### adapt - data_batch_for_adapt
		with torch.enable_grad():
			model = runner.model
			with runner.optim_wrapper.optim_context(model):
				data_batch_for_adapt = model.data_preprocessor(data_batch_for_adapt, True)
				if self.kwargs.debug.use_mmseg_pseudo_loss:
					losses = model._run_forward(data_batch_for_adapt, mode='loss')  # type: ignore
				else:
					batch["data_samples"][0].feats = batch_pseudoed[0].feats # TODO for more than 1
					# x = model.extract_feat(data_batch_for_adapt["inputs"])
					# seg_logits = model.decode_head.forward(x)
					batch_img_metas = [
									data_sample.metainfo for data_sample in data_batch_for_adapt["data_samples"]
								]
					seg_logits, feats= model.encode_decode_with_feats_plus(
						data_batch_for_adapt["inputs"], 
						batch_img_metas,
					)
					# loss_decode = model.decode_head.loss(x, data_batch_for_adapt["data_samples"],
					# 									model.train_cfg)
					# loss_decode = model.decode_head.loss_by_feat(seg_logits, data_batch_for_adapt["data_samples"])
					seg_label = model.decode_head._stack_batch_gt(data_batch_for_adapt["data_samples"])
					seg_logits = resize(
						input=seg_logits,
						size=seg_label.shape[2:],
						mode='bilinear',
						align_corners=model.decode_head.align_corners)
					seg_label = seg_label.squeeze(1)
					prob = F.softmax(seg_logits, dim=1)
					losses = dict()
					# pseudo label
					if self.kwargs.pseudo_label_loss.ratio:
						# losses[model.decode_head.loss_decode.loss_name] = \
						# 	model.decode_head.loss_decode(
						# 	seg_logits,
						# 	seg_label,
						# 	weight=None,
						# 	ignore_index=model.decode_head.ignore_index) * \
						# self.kwargs.pseudo_label_loss.ratio
						loss_ = \
							F.cross_entropy(
								seg_logits,
								seg_label,
								weight=None,
								reduction='none',
								ignore_index=model.decode_head.ignore_index)  # B,w,h
						if self.kwargs.pseudo_label_loss.conf_weight_tau != 0.0:
							with torch.no_grad():
								weight_ = prob.max(1)[0].detach() # (B,w,h)(0-1)
								tau = self.kwargs.pseudo_label_loss.conf_weight_tau
								weight_ = quadratic_function(weight_, tau).detach()
							loss_ = (loss_ * weight_)
						loss_ = loss_.mean()
						losses[model.decode_head.loss_decode.loss_name] = loss_ * self.kwargs.pseudo_label_loss.ratio
					# entropy 
					if self.kwargs.entropy_loss.ratio:
						prob_ = F.softmax(seg_logits/self.kwargs.entropy_loss.tau, dim=1)
						entropy = -prob_ * torch.log(prob_+EPS)
						entropy = torch.sum(entropy, dim=1)
						# entropy[entropy != entropy] = 1 # nan to 1
						# if self.kwargs.high_conf_mask.turn_on:
						# 	entropy = entropy[:,hign_conf_mask] # need support batch > 1
						entropy = entropy.mean()
						losses["loss_en"] = entropy * self.kwargs.entropy_loss.ratio
					# mean entropy
					if self.kwargs.diverse_loss.ratio:
						# Calculate global entropy
						# if self.kwargs.high_conf_mask.turn_on:
						# 	entropy_global = prob[:, :, hign_conf_mask]  # (B, C, H, W) -> (B, C, H*W)
						# 	entropy_global = entropy_global.mean(-1).squeeze(0)  # (C)
						# else:
						entropy_global = prob.mean(-1).mean(-1).squeeze(0)  # (C)

						# Compute entropy loss with added epsilon
						entropy_loss = torch.sum(-entropy_global * torch.log(entropy_global + EPS), dim=-1)
						losses["loss_englobal"] = -entropy_loss * self.kwargs.diverse_loss.ratio
					# sam loss
					if self.kwargs.sam_loss.ratio:
						losses["loss_sam"] = automask_consistency_loss(
							batch["data_samples"][0].automask.data,
							feats[0].data,
							seg_logits,
							self.kwargs.sam_loss
						) * self.kwargs.sam_loss.ratio
					
					losses = add_prefix(losses, 'decode')

				if model.with_auxiliary_head: assert NotImplementedError("see the class function for this branch")
				if losses:
					parsed_losses, log_vars = model.parse_losses(losses)  # sum all element with loss in
					to_logs.update(log_vars)
					runner.optim_wrapper.update_params(parsed_losses)
			if self.kwargs.ema.turn_on and self.kwargs.ema.mid_pred:
				for _ in range(self.kwargs.ema.mid_update_times):
					with self.optim_wrapper_mid.optim_context(self.model_mid):
						losses_mid = self.model_mid._run_forward(data_batch_for_adapt, mode='loss')  # type: ignore
						parsed_losses, log_vars = model.parse_losses(losses_mid)  # sum all element with loss in
						self.optim_wrapper_mid.update_params(parsed_losses)
				to_logs.update(add_prefix(log_vars, "mid_"))


		### draw
		# if self.every_n_inner_iters(batch_idx, self.kwargs.adapt_img_vis_freq):
		if batch_idx % self.kwargs.adapt_img_vis_freq == 0:
			# train sample
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
			# sam
			if hasattr(data_batch_for_adapt["data_samples"][0], "automask"):
				for i, output in enumerate(data_batch_for_adapt["data_samples"]):
					size_ = output.gt_sem_seg.shape # (1024, 1024) # ! note w,h order
					img = data_batch_for_adapt_bak["inputs"][i] # tensor (3, H, W)
					img = img.float()
					img = F.interpolate(img.unsqueeze(0), size=(size_[0], size_[1]), mode='bilinear', align_corners=True)
					img = (img).byte()
					img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)

					output_ = deepcopy(output)
					output_.gt_sem_seg.data = output.automask.data

					if img.shape[-1] == 3:
						img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
					runner.visualizer.add_datasample(
						f"adapt/automask_sam_{i}",
						img,
						data_sample=output_,
						show=False,
						step=runner.iter,
						draw_pred=False
						)
			# draw tsne
			for i, output in enumerate(data_batch_for_adapt["data_samples"]):
				size_ = output.gt_sem_seg.shape
				feats = None

		
		# sam_model
		if self.kwargs.sam_model.turn_on:
			img = batch['inputs'][0].data  # tensor (CH, w, h)
			y = batch_pseudoed[0].gt_sem_seg.data[0] # tensor (w, h)
			seg_logits = batch_pseudoed[0].seg_logits.data  # tensor (cls, w_, h_)
			w, h = img.shape[1:]

			# reshape logits
			seg_logits = seg_logits.unsqueeze(0)  # tensor (1, cls, w_, h_)
			seg_logits = F.interpolate(seg_logits, size=img.shape[1:], mode='bilinear', align_corners=True)
			y = F.interpolate(y.unsqueeze(0).unsqueeze(0).float(), size=img.shape[1:], mode='nearest').long().squeeze(0).squeeze(0)
			seg_logits = seg_logits.squeeze(0)  # tensor (cls, w, h)
			seg_logits = seg_logits.cpu()  # Move seg_logits to CPU
			img = img.cpu()  # Move img to CPU
			probs = seg_logits.softmax(0)  # tensor (cls, 512, 1024)
			points = []
			# find the max prob point for each class and add to points
			for i in range(probs.shape[0]):  # iterate over each class
				_, max_index = probs[i].view(-1).max(0)  # get the index of the max probability
				max_point = np.unravel_index(max_index, probs[i].shape)
				points.append(max_point)  # add to the list of points

			# turn to (N, 2) array and scale the value to (0, 1)
			def get_points_from_gt(mask):
				""" get points from ground truth mask
				for each patch, give the middle as the point. Note that one class could have multiple points
				mask: (H, W) with each pixel value as class index
				return: 
					(N, 2) array with each row as (x, y) point
				"""
				from scipy.ndimage import label, distance_transform_edt
				points = []
				classes = np.unique(mask)
				if 0 in classes:  # Assume 0 is the background class
					classes = classes[1:]

				for cls in classes:
					class_mask = (mask == cls)
					labeled_patches, num_features = label(class_mask)

					for i in range(1, num_features + 1):  # Start from 1 to ignore the background
						patch_mask = (labeled_patches == i)
						# 1. if self.kwargs.sam_model.use_center
						# Perform distance transform
						# distances = distance_transform_edt(patch_mask)
						# # Find the position of the maximum distance (center of the largest inscribed circle)
						# max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)
						# points.append(max_dist_idx[::-1])  # Append (y, x) to points list (need to reverse indices)
						# 2. random select multiple based on patch size
						PIXELS_PER_POINT = 1000
						# count patch size (num of non-zero)
						patch_size = np.count_nonzero(patch_mask)
						num_points = max(patch_size // PIXELS_PER_POINT, 2)
						# randomly select num_points
						indices = np.where(patch_mask == 1)
						indices = np.array(indices).T
						np.random.shuffle(indices)
						indices = indices[:num_points]
						# change x,y order
						indices = indices[:, ::-1]
						points.extend(indices)


				return np.array(points)
			
			# points = get_points_from_gt(y.cpu().numpy()) # 512, 1024 w, h
			# points = np.array(points) / np.array([h, w]) # ! check if point ar right
			points = make_grid(10)
			# get first half
			# points = points[:6]
			
			# points -> sam pred
			self.mask_generator.point_grids[0] = points
			img_np = batch['inputs'][0]
			img_np = img_np.permute(1,2,0).numpy()
			sam_res = self.mask_generator.generate(img_np) ### ! TODO DEBUG
			automask_bwh = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
			for i, mask in enumerate(sam_res):
				indices = np.where(mask["segmentation"] == 1)
				automask_bwh[indices] = i + 1
			automask_bwh = torch.from_numpy(automask_bwh).long()

			### plot
			def mask_to_color(mask):
				"""
				Convert a class mask tensor to an RGB color mask using a colormap,
				with colors represented as [0, 255] integers. 
				ps. different class is indicated by different uint, but may not be continuous

				Parameters:
				- mask: PyTorch tensor of shape (H, W), where each entry represents a class index.

				Returns:
				- A NumPy array of shape (H, W, 3), representing RGB colors for each pixel
				with integers in the range [0, 255].
				"""
				# Get unique class indices from the mask and sort them
				unique_classes = torch.unique(mask)
				unique_classes = unique_classes[unique_classes != 255]  # remove 255 if it's used for a special purpose like ignore_index

				# Map each unique class index to a position in the range [0, number of unique classes - 1]
				index_mapping = {int(class_idx): idx for idx, class_idx in enumerate(unique_classes)}

				# Create an array of shape (num_unique_classes, 3) for RGB colors, taking modulo with a colormap size
				colormap_size = 20  # Number of distinct colors in the colormap
				colors = plt.cm.tab20(np.linspace(0, 1, colormap_size))

				# If mask is on GPU, move to CPU and convert to numpy array
				mask_array = mask.cpu().numpy() if mask.is_cuda else mask.numpy()

				# Create an empty RGB array
				height, width = mask_array.shape
				color_mask = np.zeros((height, width, 3), dtype=np.uint8)

				# Map each class index to its color
				for class_idx in unique_classes:
					mapped_index = index_mapping[int(class_idx)]
					color = (colors[mapped_index % colormap_size][:3] * 255).astype(np.uint8)
					color_mask[mask_array == class_idx.item()] = color

				return color_mask
			
			def seg_plot(img, mask=None, points=None, save_path=None):
				"""
				Plot an image with an optional mask and points, and save to save_path if provided.

				Parameters:
				- img: A PyTorch tensor of shape (C, H, W) representing the image.
				- mask: A PyTorch tensor of shape (H, W), where each entry represents a class index.
				- points: A list of tuples/lists with points to plot (e.g., [(x1, y1), (x2, y2), ...]).
				- save_path: A string representing the file path to save the image.
				"""
				fig, ax = plt.subplots()
				ax.imshow(img.permute(1, 2, 0).numpy(), aspect='auto')
				w, h = img.shape[1:]

				if mask is not None:
					color_mask = mask_to_color(mask)
					ax.imshow(color_mask, aspect='auto', alpha=0.5)

				if points is not None:
					for point in points:
						# ax.scatter(point[0], point[1], s=10, c='r')
						ax.scatter(int(point[0]*h), int(point[1]*w), s=2, c='b')
						ax.scatter(int(point[0]*h), int(point[1]*w), s=1, c='w')

				if save_path:
					plt.savefig(save_path, dpi=300)
				plt.close(fig)

			# plot img, pseudo label mask, and points

			seg_plot(batch['inputs'][0], mask=y, points=points, save_path=f"./debug/{batch_idx}_gt.png")
			seg_plot(batch['inputs'][0], mask=seg_logits.max(0)[1], points=points, save_path=f"./debug/{batch_idx}_pred.png")
			seg_plot(batch['inputs'][0], mask=automask_bwh, points=points, save_path=f"./debug/{batch_idx}_sam.png")

		
		runner.visualizer.add_scalars({
			"adapt/"+k: v.item() for k, v in to_logs.items()
		},step=runner.iter)
		dict_log = {k: "{:.2f}".format(v.item()) for k, v in to_logs.items()}
		runner.logger.info(f"log_vars: {dict_log}")

		# predictions should be sequence of BaseDataElement
		with autocast(enabled=self.fp16):
			if self.kwargs.ema.turn_on:
				# ema.ema_pred and ema.mid_pred can not be true at the same time
				assert not self.kwargs.ema.ema_pred or not self.kwargs.ema.mid_pred, "ema_pred and mid_pred can not be true at the same time"
				if self.kwargs.ema.ema_pred:
					outputs_test = self.model_ema.test_step(data_batch)
				elif self.kwargs.ema.mid_pred:
					outputs_test = self.model_mid.test_step(data_batch)
			else:
				outputs_test = self.runner.model.test_step(data_batch)
		self.evaluator.process(data_samples=outputs_test, data_batch=data_batch)
		self.runner.call_hook(
			'after_test_iter',
			batch_idx=idx,
			data_batch=data_batch,
			outputs=outputs_test)


