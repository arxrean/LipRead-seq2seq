import torch
import torch.nn as nn
import random
import pdb
import sys

import torch.nn.functional as F
sys.path.append('./')
import utils
from model.middle import PositionEmbedding
from model.middle import MultiAtten
from model.middle import MultiForward

class DecodeAtten(nn.Module):
    def __init__(self, opt):
        super(DecodeAtten, self).__init__()
        self.opt = opt
        self.decode_embed = PositionEmbedding(opt, num_embeddings=opt.out_channel+1, embedding_dim=512)
        self.position_embed = PositionEmbedding(opt, num_embeddings=opt.min_frame, embedding_dim=512)
        self.atten_list = nn.ModuleList()
        for _ in range(self.opt.block_num):
            self.atten_list.append(MultiAtten(opt, causality=True))
            self.atten_list.append(MultiAtten(opt, causality=False))
            self.atten_list.append(MultiForward(opt))

        self.fc = nn.Linear(512, opt.out_channel)
        self.drop = nn.Dropout(self.opt.dropout_embed)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, feat, align, align_gt, enc_query_mask, pad_index=0):
        query_masks_dec = torch.abs(align)
        query_masks_dec[query_masks_dec>0] = 1

        align_embed = self.decode_embed(align.long())
        align_pos_embed = torch.arange(0, align_embed.size(1), 1).unsqueeze(
            0).repeat(feat.size(0), 1).long()
        if self.opt.gpu:
            align_pos_embed = align_pos_embed.cuda()
        align_embed = align_embed + self.position_embed(align_pos_embed)
        align_embed = self.drop(align_embed)
        for i in range(len(self.atten_list)):
            if i % 3 == 1:
                align_embed = self.atten_list[i](align_embed, feat, query_mask=query_masks_dec.float(), key_mask=enc_query_mask)
            elif i % 3 == 0:
                align_embed = self.atten_list[i](align_embed, query_mask=query_masks_dec.float())
            else:
                align_embed = self.atten_list[i](align_embed)

        b = align_embed.size(0)
        align_embed = self.fc(align_embed)

        is_target = align_gt != pad_index
        smoothed_one_hot = utils.one_hot(self.opt, align_gt.reshape(-1), self.opt.out_channel)
        smoothed_one_hot = smoothed_one_hot * \
                (1 - self.opt.loss_smooth) + (1 - smoothed_one_hot) * \
                self.opt.loss_smooth / (self.opt.out_channel - 1)
        align_embed = align_embed.reshape(-1, align_embed.size(-1))
        log_prb = F.log_softmax(align_embed, dim=1)
        loss = -(smoothed_one_hot * log_prb).sum(dim=1)
        final_loss = loss.mean()

        mean_loss = loss.sum() / torch.sum(is_target)

        return final_loss, mean_loss, align_embed

    def forward_infer(self, feat, align, enc_query_mask, pad_index=0):
        query_masks_dec = torch.abs(align)
        query_masks_dec[query_masks_dec>0] = 1
        # pdb.set_trace()
        align_embed = self.decode_embed(align.type(torch.cuda.LongTensor))
        align_pos_embed = torch.arange(0, align_embed.size(1), 1).unsqueeze(
            0).repeat(feat.size(0), 1).long().cuda()
        align_embed = align_embed + self.position_embed(align_pos_embed)
        for i in range(len(self.atten_list)):
            if i % 3 == 1:
                align_embed = self.atten_list[i](align_embed, feat, query_mask=query_masks_dec.float(), key_mask=enc_query_mask)
            elif i % 3 == 0:
                align_embed = self.atten_list[i](align_embed, query_mask=query_masks_dec.float())
            else:
                align_embed = self.atten_list[i](align_embed)

        b = align_embed.size(0)
        align_embed = align_embed.view(-1, align_embed.size(-1))
        align_embed = self.fc(align_embed)
        align_embed = align_embed.view(b, -1, align_embed.size(-1))

        return align_embed