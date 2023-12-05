# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# Modified by Tao Wang (wangtao@bupt.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dis import dis

import torch
import numpy as np
from einops import repeat, rearrange
from visual import *

def get_heat_value(pose_coord, heatmap):
    _, h, w = heatmap.shape
    heatmap_nocenter = heatmap.flatten(1,2).transpose(0,1)

    y_b = torch.clamp(torch.floor(pose_coord[:,:,1]), 0, h-1).long()
    x_l = torch.clamp(torch.floor(pose_coord[:,:,0]), 0, w-1).long()
    heatval = torch.gather(heatmap_nocenter, 0, y_b * w + x_l).unsqueeze(-1)
    return heatval


def cal_area_2_torch(v):
    w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
    h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
    return w * w + h * h


def soft_nms_core(cfg, pose_coord, heat_score):
    num_people, num_joints, _ = pose_coord.shape
    pose_area = cal_area_2_torch(pose_coord)[:,None].repeat(1,num_people*num_joints)
    pose_area = pose_area.reshape(num_people,num_people,num_joints)
    pose_thre = cfg.TEST.NMS_THRE * torch.sqrt(pose_area) + 0.1 
    
    ignored_pose_inds = set()
    keep_pose_inds = []
    
    for i in range(num_people):
        pose_dist = (pose_coord[i] - pose_coord).pow(2).sum(-1).sqrt()
        pose_dist = (pose_dist < pose_thre[i]).sum(-1) # [136]
        keep_inds = list((pose_dist > cfg.TEST.NMS_NUM_THRE).nonzero()[...,0].cpu().numpy())
        ind = torch.argmax(heat_score[keep_inds])
        keep_ind = keep_inds[ind]
        if keep_ind in ignored_pose_inds:
            continue
        else:
            keep_pose_inds += [keep_ind]
        # exp(-d^2/σ)
        ignored_pose_inds.add(keep_ind)
        keep_inds = list(set(keep_inds) - ignored_pose_inds)
        distance = pose_dist[keep_inds] - cfg.TEST.NMS_NUM_THRE
        weight = (-distance.pow(2)/cfg.TEST.nms_sigma).exp()
        heat_score[keep_inds] = heat_score[keep_inds] * weight
        
        ignore = (heat_score < cfg.TEST.nms_heat_th).nonzero()[..., 0].cpu().numpy()
        ignored_pose_inds.update(ignore)

    return keep_pose_inds, heat_score

def nms_core(cfg, pose_coord, heat_score, heatval):
    num_people, num_joints, _ = pose_coord.shape
    pose_area = cal_area_2_torch(pose_coord)[:,None].repeat(1,num_people*num_joints)
    pose_area = pose_area.reshape(num_people,num_people,num_joints)
    
    pose_diff = pose_coord[:, None, :, :] - pose_coord
    pose_diff.pow_(2)
    pose_dist = pose_diff.sum(3)
    pose_dist.sqrt_()
    pose_thre = cfg.TEST.NMS_THRE * torch.sqrt(pose_area) + 0.1
    pose_dist = (pose_dist < pose_thre).sum(2)
    nms_pose = pose_dist > cfg.TEST.NMS_NUM_THRE

    ignored_pose_inds = []
    keep_pose_inds = []
    refine_inds = []
    all_nms_inds = []
    nms_poses_num = []
    for i in range(nms_pose.shape[0]):
        if i in ignored_pose_inds:
            continue
        keep_inds = nms_pose[i].nonzero().cpu().numpy()
        keep_inds = [list(kind)[0] for kind in keep_inds]
        
        # for better coord
        k_div = [[] for _ in range(num_joints)]
        keep_k_ind = [[] for _ in range(num_joints)]
        # for j in keep_inds:
        #     k_div[int(j//cfg.MODEL.GET_LOC.TOP_K)].append(j)
        for k in range(num_joints):
            if k_div[k].__len__() > 0:
                keep_scores = heat_score[k_div[k]]
                # keep_scores = heatval[k_div[k], k, 0]
                ind = torch.argmax(keep_scores)
                keep_k_ind[k].append(k_div[k][ind])
            
        # for whole person
        keep_scores = heat_score[keep_inds]
        ind = torch.argmax(keep_scores)
        keep_ind = keep_inds[ind]
        
        if keep_ind in ignored_pose_inds:
            continue
        keep_pose_inds += [keep_ind]
        refine_inds += [keep_k_ind]
        all_nms_inds += [k_div]
        nms_poses_num += [len(keep_inds)]
        ignored_pose_inds += list(set(keep_inds)-set(ignored_pose_inds))

    return keep_pose_inds, refine_inds, all_nms_inds, np.array(nms_poses_num)


def pose_nms(cfg, heatmap_avg, poses, posemap, image_resized, batch_idx):
    """
    NMS for the regressed poses results.

    Args:
        heatmap_avg (Tensor): Avg of the heatmaps at all scales (1, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """
    scale1_index = sorted(cfg.TEST.SCALE_FACTOR, reverse=True).index(1.0)
    pose_norm = poses[scale1_index]
    max_score = pose_norm[:,:,2].max() if pose_norm.shape[0] else 1

    for i, pose in enumerate(poses):
        if i != scale1_index:
            max_score_scale = pose[:,:,2].max() if pose.shape[0] else 1
            pose[:,:,2] = pose[:,:,2]/max_score_scale*max_score*cfg.TEST.DECREASE

    pose_score = torch.cat([pose[:,:,2:] for pose in poses], dim=0)
    pose_coord = torch.cat([pose[:,:,:2] for pose in poses], dim=0)

    if pose_coord.shape[0] == 0:
        return [], []

    num_people, num_joints, _ = pose_coord.shape
    heatval = get_heat_value(pose_coord, heatmap_avg[0])
    heat_score = (torch.sum(heatval, dim=1)/num_joints)[:,0]
    
    keep_pose_inds, refine_inds, all_nms_inds, nms_poses_num = nms_core(cfg, pose_coord, heat_score, heatval)
    poses = pose_coord[keep_pose_inds]
    heatval = get_heat_value(poses, heatmap_avg[0]) 
    pose_score = heat_score[keep_pose_inds][..., None, None] + torch.zeros_like(poses[...,-1:])
    pose_score = pose_score * heatval
    poses = torch.cat([poses, pose_score], dim=2)
    heat_score = (torch.sum(heatval, dim=1)/num_joints)[:,0]
    
    if len(keep_pose_inds) > cfg.DATASET.MAX_NUM_PEOPLE:
        heat_score, topk_inds = torch.topk(heat_score,
                                            cfg.DATASET.MAX_NUM_PEOPLE)
        poses = poses[topk_inds]
     
    poses = [poses.cpu().numpy()]
    scores = np.array([i[:, 2].mean()*np.sqrt(area_numpy(i)) for i in poses[0]])


    return poses, scores

def area_numpy(pose):
    w = pose[:,0].max() - pose[:,0].min()
    h = pose[:,1].max() - pose[:,1].min()
    return w*w + h*h

def same_pose(cfg, pose1, pose2):
    '''
        determine if pose1&2 are the same pose
    args:
        cfg: cfg
        pose1: torch.Size([17, 2])
        pose2: torch.Size([17, 2])
    returns:
        nms_pose.item(): bool
    '''
    pose_area = area_numpy(pose1)
    pose_thre = cfg.TEST.NMS_THRE * torch.sqrt(pose_area) + 0.1
    pose_dist = (pose1 - pose2).pow(2).sum(-1).sqrt()
    pose_dist = (pose_dist < pose_thre).sum()
    nms_pose = pose_dist > cfg.TEST.NMS_NUM_THRE
    return nms_pose.item()

def diffusion(poses, posemap, heatval=None):
    '''
        calculate pose distance for each keypoint in a pose
    args:
        poses: torch.Size([k, 17, 2 or 3])
        posemap: torch.Size([128, 192, 17, 2])
    return:
        diffusion_weight: numpy.ndarray([18])
    '''
    if heatval is None:
        heatval = poses[...,-1, None]
    
    posemap = posemap.permute(0,2,3,1,4)[0]
    h = posemap.size(0) - 1
    w = posemap.size(1) - 1
    dis_pose = (poses[...,:2]/4).to(torch.long)
    dis_pose = torch.where(dis_pose < 0, 0, dis_pose)
    dis_pose[...,0] = torch.where(dis_pose[...,0] > w, w, dis_pose[...,0])
    dis_pose[...,1] = torch.where(dis_pose[...,1] > h, h, dis_pose[...,1])
    diffusion_pose = posemap[dis_pose[...,1], dis_pose[...,0], ...]
    diffusion_distance = (poses[..., None, :2] - diffusion_pose).pow(2).sum(-1).sqrt().sum([1,2]).cpu().numpy()

    poses1 = [poses.cpu().numpy()]
    area = np.array([area_numpy(i) for i in poses1[0]])
    diffusion_weight = area/diffusion_distance
    
    return diffusion_weight

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    '''
    from https://github.com/DongPoLI/NMS_SoftNMS/blob/main/soft_nms.py
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    '''
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

    """
    :param boxes: [N, 4],  此处传进来的框, 是经过筛选(选取的得分TopK)之后的
    :param scores: [N]
    :param iou_threshold: 0.7
    :param soft_threshold soft nms 过滤掉得分太低的框 (手动设置)
    :param weight_method 权重方法 1. 线性 2. 高斯
    :return:
    """
    keep = []
    idxs = scores.argsort()
    while idxs.numel() > 0: 
        idxs = scores.argsort()

        if idxs.size(0) == 1: 
            keep.append(idxs[-1])  
            break
        keep_len = len(keep)
        max_score_index = idxs[-(keep_len + 1)]
        max_score_box = boxes[max_score_index][None, :] 
        idxs = idxs[:-(keep_len + 1)]
        other_boxes = boxes[idxs]
        keep.append(max_score_index) 
        ious = box_iou(max_score_box, other_boxes)
        if weight_method == 1: 
            ge_threshod_bool = ious[0] >= iou_threshold
            ge_threshod_idxs = idxs[ge_threshod_bool]
            scores[ge_threshod_idxs] *= (1. - ious[0][ge_threshod_bool])
        elif weight_method == 2:
            scores[idxs] *= torch.exp(-(ious[0] * ious[0]) / sigma) # 权重(0, 1]
            # idxs = idxs[scores[idxs] >= soft_threshold]
        # else:  # NMS
        #     idxs = idxs[ious[0] <= iou_threshold]

    # keep = scores[scores > soft_threshold].int()
    keep = idxs.new(keep)  # Tensor
    keep = keep[scores[keep] > soft_threshold]  # 最后处理阈值
    return keep, scores