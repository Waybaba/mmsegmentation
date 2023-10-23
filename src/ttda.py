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
from mmseg.registry import HOOKS
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
							is_model_wrapper, revert_sync_batchnorm)
from mmengine.model.efficient_conv_bn_eval import \
	turn_on_efficient_conv_bn_eval
import torch.nn.functional as F
from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw, resize
from functools import partial
import utils
from mmseg.utils.misc import add_prefix


import cv2
from mmengine.structures import BaseDataElement, PixelData
from mmengine.registry import MODELS
from mmseg.models.backbones import MixVisionTransformer
from mmseg.models import EncoderDecoder
import torch.utils.checkpoint as cp
import math

@torch.no_grad()
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
	size_ori = logits.shape[1:]
	size_sm = sam_feats.shape[1:]

	# intepolate sam feats
	# sam_feats = F.interpolate(sam_feats.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)
	logits = F.interpolate(logits.unsqueeze(0), size=size_sm, mode='bilinear', align_corners=False).squeeze(0)
	
	# Ensure logits are of the same spatial size as interpolated sam_feats
	# logits = F.interpolate(logits.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

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

def adjust_with_sam(logits, automask, sam_ratio):
	"""
	Adjust logits based on automask value.

	Parameters:
	- logits: Tensor of logits. Shape: (C, H, W)
	- automask: Tensor of shape (1, H, W) representing automask values.
	- sam_ratio: A scalar controlling the strength of adjustment. float

	Returns:
	- Adjusted logits.
	"""
	if sam_ratio == 0.: return logits
	unique_masks = torch.unique(automask)
	adjusted_logits = logits.clone()

	for mask in unique_masks:
		mask_indices = (automask.squeeze(0) == mask)  # Shape: (H, W)
		mean_logits_per_channel = torch.mean(logits[:, mask_indices], dim=1, keepdim=True)  # Shape: (C, 1)
		
		# Use broadcasting to update the adjusted logits for the current mask
		adjusted_logits[:, mask_indices] = (1 - sam_ratio) * logits[:, mask_indices] + sam_ratio * mean_logits_per_channel

	return adjusted_logits

def param_migrate(base, tgt, rho):
	"""
	rho = 1 means copy all params from src to tgt
	rho = 0 means copy nothing
	others, final = (1-rho) * base + rho * tgt
	"""
	# check all keys the same
	for name, param in base.named_parameters():
		assert name in tgt.state_dict().keys(), f"{name} not in tgt"
	for name, param in tgt.named_parameters():
		base.state_dict()[name].data.copy_(
			(1-rho) * base.state_dict()[name].data \
				+ rho * param.data
			)
	return base

# TODO check model.train in ttda mode
# TODO check multi head ! I only have one head
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
		data = self.data_preprocessor(data, False)
		res = self._run_forward_plus(data, mode='predict')  # type: ignore
		if cfg.turn_on:
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

				def add_sample(self, feats, logits, pred):
					"""
					feats: C, H, W
					pred_sem_seg: 1, H, W 
					note case that some classes are missing in history,
					or this is the first time to see this class
					"""
					confidence = torch.softmax(logits, dim=0).max(dim=0)[0]  # shape: [1, H, W]
					entropy = -torch.sum(torch.log_softmax(logits, dim=0) * torch.softmax(logits, dim=0), dim=0)  # shape: [1, H, W]

					# proto_mask_metric
					if cfg.proto_mask_metric is not None:
						assert cfg.proto_mask_metric in ["confidence", "entropy"], f"cfg.proto_mask_metric should be confidence or entropy, but got {cfg.proto_mask_metric}"
						seg_metric = confidence if cfg.proto_mask_metric == "confidence" \
							else -entropy # use neg since seg_mask only keep higher
						if not hasattr(self, "seg_masker"):
							self.seg_masker = SegMasker(use_history=cfg.proto_mask_use_history)
						self.seg_masker.add_to_buffer(seg_metric, pred[0])
						hign_conf_mask = self.seg_masker.cal_mask(
							seg_metric,
							pred[0],
							cfg.proto_mask_top_p
						)
						pred[0][~hign_conf_mask] = 255 # to 255 and skip in the following

					# Iterate over the unique classes in pred_sem_seg
					for class_label in torch.unique(pred):
						class_label = class_label.item()
						if class_label == 255: 
							continue # 
						class_feats = feats[:, pred[0] == class_label]  # Extract features corresponding to this class
						if class_label not in self.protos_mean:
							# If the class is not seen before, simply set the mean and count
							self.protos_mean[class_label] = class_feats.mean(dim=1, keepdim=False)
							self.protos_count[class_label] = class_feats.size(1)
						else:
							# Update mean and count using the provided formula
							new_count = class_feats.size(1)
							new_mean = class_feats.mean(dim=1, keepdim=False)
							if self.cfg.rho == "count":
								updated_mean = (self.protos_mean[class_label] * self.protos_count[class_label] + new_mean * new_count) / (self.protos_count[class_label] + new_count)
							elif isinstance(self.cfg.rho, (float,int)):
								updated_mean = (self.protos_mean[class_label] *  (1 - self.cfg.rho) + new_mean * self.cfg.rho)
							self.protos_mean[class_label] = updated_mean
							self.protos_count[class_label] += new_count

				def predict(self, feats):
					"""
					feats: C, H, W
					return pred_sem_seg data: 1, H, W
					"""
					C, H, W = feats.shape
					pred_sem_seg = torch.zeros(1, H, W).long().to(feats.device)

					# Extract the prototypes from the current image
					proto_this_img_list = []
					class_labels_list = list(self.protos_mean.keys())
					for class_label in class_labels_list:
						class_feats = feats[:, pred_sem_seg[0] == class_label]  # Extract features corresponding to this class
						if class_feats.size(1) != 0:  # Check if the class exists in this image
							proto_this_img = class_feats.mean(dim=1, keepdim=False)
							proto_this_img_list.append(proto_this_img)
						else:
							proto_this_img_list.append(self.protos_mean[class_label])  # If class not in image, use the historical prototype

					# Combine historical prototypes and the prototypes from the current image
					combined_protos = [self.cfg.lambda_ * history_proto + (1 - self.cfg.lambda_) * proto_this_img
									for history_proto, proto_this_img in zip(self.protos_mean.values(), proto_this_img_list)]

					# Convert combined_protos to tensor
					proto_tensor = torch.stack(combined_protos, dim=0).to(feats.device)  # shape: [num_classes, C]

					# Reshape feats for distance computation
					feats_reshaped = feats.view(C, -1).transpose(0, 1)  # shape: [H*W, C]

					# Compute distances
					distances = torch.cdist(feats_reshaped, proto_tensor)  # shape: [H*W, num_classes]

					# Get the class labels for the minimum distances
					_, pred_class_indices = distances.min(dim=1)

					# Map the indices back to actual class labels
					pred_labels = torch.tensor(class_labels_list, dtype=torch.long).to(feats.device)
					pred_sem_seg[0] = pred_labels[pred_class_indices].view(H, W)

					return pred_sem_seg

				def reset(self):
					"""
					Resets the learned prototypes.
					"""
					self.protos_mean = {}
					self.protos_count = {}

			if not hasattr(self, 'protos_classifier'):
				self.protos_classifier = ProtoClassifier(cfg)
			for i, d in enumerate(res):
				# TODO add confidence threshold
				feats_data = res[i].feats.data
				if cfg.norm_feats: # (Ch, num)
					# norm to make each vector has norm 1, but keep the direction
					feats_data = F.normalize(feats_data, dim=0)
				self.protos_classifier.add_sample(feats_data, res[i].seg_logits.data, res[i].pred_sem_seg.data)
				pred = self.protos_classifier.predict(feats_data)
				res[i].pred_sem_seg.data = pred
		return res

	def test_step_sam_predict(self, data, cfg=None):
		"""
		TODO remember to reset buffer
		TODO 255 class hwo to handle
		here, cfg is set by partial at init stage of runner
		"""
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
				logits_ = adjust_with_sam(logits, automask, cfg.sam_ratio)
				res[i].pred_sem_seg.data = logits_.argmax(0)
			else:
				raise NotImplementedError(f"Unknown sam type: {cfg.type}")
		return res

	def test_step_plus(self, data):
		"""``BaseModel`` implements ``test_step`` the same as ``val_step``.

		Args:
			data (dict or tuple or list): Data sampled from dataset.

		Returns:
			list: The predictions of given data.
		"""
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

			# resize as original shape
			i_feats = resize(
				i_feats,
				size=img_meta['ori_shape'],
				mode='bilinear',
				align_corners=self.align_corners,
				warning=False).squeeze(0)
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
		seg_logits, feats= self.encode_decode_with_feats(inputs, batch_img_metas)

		return seg_logits, feats

	def encode_decode_with_feats(self, inputs, batch_img_metas):
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
	def __init__(self, use_history):
		self.buffer = {}
		self.use_history = use_history

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
		
	def cal_mask(self, seg_metric, seg_pred, top_p):
		"""
		return a mask shape as seg with True for high confidence,
		cal top_p percent
		"""
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
		return mask


# main

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

	def ttda_adapt(self, runner, batch_idx, batch):
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
		assert len(batch["inputs"]) == 1, "only support batch_size=1"
		inputs, data_samples = batch["inputs"], batch["data_samples"]
		if not hasattr(self, "model_ori"): # source model for pseudo
			assert batch_idx == 0, "model_ori should be init only once"
			self.model_ori = deepcopy(runner.model)
		if self.kwargs.ema.turn_on:
			if not hasattr(self, "model_ema"): # target model for pseudo
				self.model_ema = deepcopy(runner.model)
				for param in self.model_ema.parameters():
					param.detach_()
					param.requires_grad = False
			# back to ema
			runner.model = param_migrate(runner.model, self.model_ema, 1.0)

		# inference label
		
		batch_pseudoed_model = self.model_ori if self.kwargs.pseudo_use_ori \
			else runner.model
		batch_pseudoed = batch_pseudoed_model.test_step_plus(batch)
		batch["data_samples"][0].feats = batch_pseudoed[0].feats # TODO for more than 1
		if self.kwargs.high_conf_mask.turn_on:
			if batch_idx == 0: 
				assert not hasattr(self, "seg_masker"), "seg_masker should not be init twice"
				self.seg_masker = SegMasker(use_history=self.kwargs.high_conf_mask.use_history)
			seg_conf = F.softmax(batch_pseudoed[0].seg_logits.data, dim=0).max(0)[0]
			self.seg_masker.add_to_buffer(seg_conf, batch_pseudoed[0].pred_sem_seg.data[0])
			hign_conf_mask = self.seg_masker.cal_mask(
				seg_conf,
				batch_pseudoed[0].pred_sem_seg.data[0],
				self.kwargs.high_conf_mask.top_p
			)
			sem_seg_ = batch_pseudoed[0].pred_sem_seg.data[0]
			sem_seg_[~hign_conf_mask] = 255
			batch_pseudoed[0].pred_sem_seg = PixelData()
			batch_pseudoed[0].pred_sem_seg.data = sem_seg_.unsqueeze(0)

		# set data_batch_for_adapt for train
		if self.kwargs.slide_adapt:
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
		
		### choose adapt
		if self.kwargs.slide_adapt:
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
				if self.kwargs.debug.use_mmseg_pseudo_loss:
					losses = model._run_forward(data_batch_for_adapt, mode='loss')  # type: ignore
				else:
					# x = model.extract_feat(data_batch_for_adapt["inputs"])
					# seg_logits = model.decode_head.forward(x)
					batch_img_metas = [
									data_sample.metainfo for data_sample in data_batch_for_adapt["data_samples"]
								]
					seg_logits, feats= model.encode_decode_with_feats(
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
						losses[model.decode_head.loss_decode.loss_name] = \
							model.decode_head.loss_decode(
							seg_logits,
							seg_label,
							weight=None,
							ignore_index=model.decode_head.ignore_index) * \
						self.kwargs.pseudo_label_loss.ratio
					# entropy
					if self.kwargs.entropy_loss.ratio:
						entropy = -prob * torch.log(prob)
						entropy = torch.sum(entropy, dim=1)
						entropy[entropy != entropy] = 1 # nan to 1
						# if self.kwargs.high_conf_mask.turn_on:
						# 	entropy = entropy[:,hign_conf_mask] # need support batch > 1
						entropy = entropy.mean()
						losses["loss_en"] = entropy * self.kwargs.entropy_loss.ratio
					# mean entropy
					if self.kwargs.diverse_loss.ratio:
						if self.kwargs.high_conf_mask.turn_on:
							entropy_global = prob[:,:,hign_conf_mask]
						entropy_global = prob.mean()
						entropy_global = torch.sum(-entropy_global * torch.log(entropy_global), dim=-1).mean()
						losses["loss_englobal"] = - entropy_global * self.kwargs.diverse_loss.ratio
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

			parsed_losses, log_vars = model.parse_losses(losses)  # sum all element with loss in
			optim_wrapper.update_params(parsed_losses)

		# update ema 
		if self.kwargs.ema.turn_on:
			self.model_ema = param_migrate(self.model_ema, runner.model, self.kwargs.ema.rho)

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
		# adapt init (e.g. optimizer)
		if self.kwargs.turn_on_adapt:
			runner.logger.info('fake train init')
			self.fake_train_init(runner)
			runner.logger.info('fake train init done')
		
		# predict mode switch
		assert not (self.kwargs.proto_predict.turn_on and self.kwargs.sam_predict.turn_on), \
			"proto_predict and sam_predict should not be on at the same time"
		if self.kwargs.proto_predict.turn_on:
			runner.model.test_step = partial(runner.model.test_step_proto_predict, cfg=self.kwargs.proto_predict)
		elif self.kwargs.sam_predict.turn_on:
			runner.model.test_step = partial(runner.model.test_step_sam_predict, cfg=self.kwargs.sam_predict)

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
		self.ttda_adapt(runner, batch_idx, data_batch)

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
		# draw train sample
		# if self.every_n_inner_iters(batch_idx, self.kwargs.adapt_img_vis_freq_test):
		# 	for i, output in enumerate(outputs):
		# 		# img_path = output.img_path
		# 		# img_bytes = fileio.get(
		# 		# 	img_path, backend_args=None) # runner.backend_args?
		# 		# img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
		# 		size_ = output.gt_sem_seg.shape # (1024, 1024) # ! note w,h order
		# 		img = data_batch["inputs"][i] # tensor (3, H, W)
		# 		# interpolate to same size_
		# 		# TODO

		# 		img = img.float()
		# 		img = F.interpolate(img.unsqueeze(0), size=(size_[0], size_[1]), mode='bilinear', align_corners=True)
		# 		img = (img).byte()
		# 		img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)

		# 		if img.shape[-1] == 3:
		# 			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		# 		runner.visualizer.add_datasample(
		# 			f"adapt/test_samples_{i}",
		# 			img,
		# 			data_sample=output,
		# 			show=False,
		# 			step=runner.iter,
		# 			draw_pred=True
		# 			)
		pass