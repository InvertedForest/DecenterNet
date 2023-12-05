# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# Modified by Tao Wang (wangtao@bupt.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
from einops import repeat

logger = logging.getLogger(__name__)


class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2) * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return loss


class OffsetsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def smooth_l1_loss(self, pred, gt, beta=1. / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5*l1_loss**2/beta, l1_loss-0.5*beta)
        return loss
    # sum对于维度顺序感觉也有影响啊
    # scale预测的是offset通道的的17个关键点置信度
    def forward(self, pred, gt, _weights, loc ,num_joints):
        loss = 0
        weights = _weights[:,:num_joints*2]
        loc_w = _weights[:,num_joints*2:]
        num_pos = torch.nonzero(weights > 0).size()[0]
        l1_loss = self.smooth_l1_loss(pred, gt) * weights
        # loss = self.smooth_l1_loss(pred, gt) * weights * torch.exp(loc) #- 0.01*loss_re
        for i in  range(num_joints):
            loss += (l1_loss * torch.exp(loc[:, i, None]) * loc_w[:, i, None]).sum()
        if num_pos == 0:
            num_pos = 1.
        loss = loss / num_pos
        return loss
 
class LocLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, gt):
        # 我们的radius为1/4, 0.1->0.02
        loss = ((pred - 0)**2)*(gt==0.1)*0.1  + (pred - 1)**2*(gt==1)*1
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return loss


class MultiLossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.heatmap_loss = HeatmapLoss()
        self.heatmap_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR

        self.loc_loss = LocLoss()
        self.loc_loss_factor = cfg.LOSS.LOC_LOSS_FACTOR
        
        self.offset_loss = OffsetsLoss()
        self.offset_loss_factor = cfg.LOSS.OFFSETS_LOSS_FACTOR
        self.bone_loss = HeatmapLoss()
        self.bone_loss_factor = cfg.LOSS.BONE_LOSS_FACTOR
        
    def forward(self, pheatmap, poffset, heatmap, mask, offset, offset_w, pbone, bone, bone_w, loc_map):
        
        heatmap_loss = self.heatmap_loss(pheatmap, heatmap, mask[:, :self.num_joints])
        heatmap_loss = heatmap_loss * self.heatmap_loss_factor
    
        offset_loss = self.offset_loss(poffset, offset, offset_w, loc_map, self.num_joints)
        offset_loss = offset_loss * self.offset_loss_factor
        
        bone_loss = self.bone_loss(pbone, bone, bone_w)
        bone_loss = bone_loss * self.bone_loss_factor
        # 选点
        loc_loss = self.loc_loss(loc_map, mask[:, self.num_joints:])
        loc_loss = loc_loss * self.loc_loss_factor

        return heatmap_loss, offset_loss, bone_loss, loc_loss
 