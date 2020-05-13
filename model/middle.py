import os
import pdb
import math
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch_layer_normalization import LayerNormalization

import pdb
import utils


class MidTemporalConv(nn.Module):
	def __init__(self, opt):
		super(MidTemporalConv, self).__init__()
		self.opt = opt
		self.inputDim = 512
		self.backend_conv1 = nn.Sequential(
			nn.Conv1d(self.inputDim, 2*self.inputDim, 5, 2, 0, bias=False),
			nn.BatchNorm1d(2*self.inputDim),
			nn.ReLU(True),
			nn.MaxPool1d(2, 2),
			nn.Conv1d(2*self.inputDim, 4*self.inputDim, 5, 2, 0, bias=False),
			nn.BatchNorm1d(4*self.inputDim),
			nn.ReLU(True),
		)
		self.backend_conv2 = nn.Sequential(
			nn.Linear(4*self.inputDim, self.inputDim),
			nn.BatchNorm1d(self.inputDim),
			nn.ReLU(True),
			nn.Linear(self.inputDim, opt.out_channel)
		)

	def forward(self, feat):
		feat = feat.transpose(1, 2)
		feat = self.backend_conv1(feat)
		feat = torch.mean(feat, 2)
		feat = self.backend_conv2(feat)

		return feat


class MultiHeadAttention(nn.Module):
	def __init__(self, opt, in_features=512, head_num=8, bias=True, activation=F.relu):
		super(MultiHeadAttention, self).__init__()
		self.opt = opt
		self.in_features = in_features
		self.head_num = head_num
		self.activation = activation
		self.bias = bias
		self.linear_q = nn.Linear(in_features, in_features, bias)
		self.linear_k = nn.Linear(in_features, in_features, bias)
		self.linear_v = nn.Linear(in_features, in_features, bias)
		self.layer_norm = LayerNormalization(512)
		self.drop = nn.Dropout(self.opt.dropout_attention)

	def forward(self, q, k, v, query_mask=None, key_mask=None, causality=False):
		_q, _k, _v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
		if self.activation is not None:
			_q = self.activation(_q)
			_k = self.activation(_k)
			_v = self.activation(_v)

		_q = self._reshape_to_batches(_q)
		_k = self._reshape_to_batches(_k)
		_v = self._reshape_to_batches(_v)

		# mask
		key_mask = key_mask.repeat(self.head_num, 1).unsqueeze(1)
		key_mask = key_mask.repeat(1, _q.size(1), 1)
		query_mask = query_mask.repeat(self.head_num, 1).unsqueeze(-1)
		query_mask = query_mask.repeat(1, 1, k.size(1))

		y = self.ScaledDotProductAttention(
			_q, _k, _v, key_mask, query_mask, causality)

		y = self._reshape_from_batches(y)

		y = y + q

		y = self.layer_norm(y)

		return y

	def ScaledDotProductAttention(self, query, key, value, key_mask=None, query_mask=None, causality=False):
		dk = query.size()[-1]
		scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
		if key_mask is not None:
			scores = scores.masked_fill(key_mask == 0, -1e9)

		if causality:
			diag_vals = torch.ones_like(scores[0, :, :])
			diag_vals[np.triu_indices(diag_vals.size(-1), 1)] = 0
			diag_vals = diag_vals.unsqueeze(0).repeat(scores.size(0), 1, 1)

			scores = scores.masked_fill(diag_vals == 0, -1e9)

		attention = F.softmax(scores, dim=-1)
		if query_mask is not None:
			attention = attention * query_mask

		attention = self.drop(attention)
		return attention.matmul(value)

	def _reshape_to_batches(self, x):
		batch_size, seq_len, in_feature = x.size()
		sub_dim = in_feature // self.head_num
		return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
				.permute(0, 2, 1, 3)\
				.reshape(batch_size * self.head_num, seq_len, sub_dim)

	def _reshape_from_batches(self, x):
		batch_size, seq_len, in_feature = x.size()
		batch_size //= self.head_num
		out_dim = in_feature * self.head_num
		return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
				.permute(0, 2, 1, 3)\
				.reshape(batch_size, seq_len, out_dim)


class MultiAtten(nn.Module):
	def __init__(self, opt, causality=False):
		super(MultiAtten, self).__init__()
		self.opt = opt
		self.multiAtten = MultiHeadAttention(opt)
		self.causality = causality

	def forward(self, feat, dec=None, query_mask=None, key_mask=None):
		if dec is None:
			key_mask = torch.zeros_like(query_mask)
			key_mask[query_mask == 1] = 1
			feat = self.multiAtten(
				feat, feat, feat, query_mask=query_mask, key_mask=key_mask, causality=self.causality)
		else:
			feat = self.multiAtten(
				feat, dec, dec, query_mask=query_mask, key_mask=key_mask, causality=self.causality)

		return feat


class PositionEmbedding(nn.Module):
	def __init__(self, opt, num_embeddings, embedding_dim):
		super(PositionEmbedding, self).__init__()
		self.opt = opt
		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.weight = nn.Embedding(num_embeddings, embedding_dim)
		if opt.s2s_embed_x_init:
			self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.xavier_normal_(self.weight)

	def forward(self, x):
		embeddings = self.weight(x)
		return embeddings


class MultiForward(nn.Module):
	def __init__(self, opt):
		super(MultiForward, self).__init__()
		self.opt = opt
		self.feed_f = nn.Sequential(
			nn.Conv1d(512, 2048, 1),
			nn.ReLU(True),
			# nn.BatchNorm1d(num_features=2048),
			nn.Conv1d(2048, 512, 1),
			nn.ReLU(True),
			# nn.BatchNorm1d(num_features=512),
		)
		self.layer_norm = LayerNormalization(512)

	def forward(self, feat, dec=None):
		feat2 = self.feed_f(feat.transpose(1, 2)).transpose(1, 2)
		feat3 = feat+feat2
		feat3 = self.layer_norm(feat3)

		return feat3


class EncodeAttention(nn.Module):
	def __init__(self, opt):
		super(EncodeAttention, self).__init__()
		self.opt = opt
		self.position_embed = PositionEmbedding(
			opt, num_embeddings=opt.min_frame, embedding_dim=512)
		self.atten_list = nn.ModuleList()
		for _ in range(self.opt.block_num):
			self.atten_list.append(MultiAtten(opt, causality=False))
			self.atten_list.append(MultiForward(opt))

	def forward(self, feat):
		pos_embed = torch.arange(0, feat.size(1), 1).unsqueeze(
			0).repeat(feat.size(0), 1).long()
		if self.opt.gpu:
			pos_embed = pos_embed.cuda()
		query_mask = torch.abs(feat.sum(-1))
		query_mask[query_mask > 0] = 1
		feat = feat + self.position_embed(pos_embed)
		for i in range(len(self.atten_list)):
			if i % 2 == 0:
				feat = self.atten_list[i](feat, query_mask=query_mask)
			else:
				feat = self.atten_list[i](feat)

		return feat, query_mask


class Pass(nn.Module):
	def __init__(self, opt):
		super(Pass, self).__init__()
		self.opt = opt

	def forward(self, feat):

		return feat
