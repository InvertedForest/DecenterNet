# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# Modified by Tao Wang (wangtao@bupt.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class HeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints
        self.num_joints_with_center = num_joints+1
        self.scale_aware_sigma = False
        self.radius = 2
        
    def get_heat_val(self, sigma, x, y, x0, y0):

        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        return g

    def __call__(self, joints, sgm, ct_sgm, bg_weight=1.0):
        assert self.num_joints_with_center == joints.shape[1], \
            'the number of joints should be %d' % self.num_joints_with_center
            
        hms = np.zeros((self.num_joints*2, self.output_res, self.output_res),
                       dtype=np.float32)
        ignored_hms = 2*np.ones((self.num_joints*3, self.output_res, self.output_res),
                                    dtype=np.float32)

        hms_list = [hms, ignored_hms]
        

        for p in joints:
            for idx, pt in enumerate(p[:-1]):
                if self.scale_aware_sigma:
                    sigma = pt[3]
                else:
                    sigma = sgm
                if pt[2] > 0:
                    x, y, v = pt[0], pt[1], pt[2]
                    if x < 0 or y < 0 or \
                            x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.floor(x - 3 * sigma - 1)
                            ), int(np.floor(y - 3 * sigma - 1))
                    br = int(np.ceil(x + 3 * sigma + 2)
                            ), int(np.ceil(y + 3 * sigma + 2))

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)

                    joint_rg = np.zeros((bb-aa, dd-cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy-aa, sx -
                                    cc] = self.get_heat_val(sigma, sx, sy, x, y)
                    if v == 2:
                        hms_list[0][2*idx, aa:bb, cc:dd] = np.maximum(
                            hms_list[0][2*idx, aa:bb, cc:dd], joint_rg)
                        hms_list[1][2*idx, aa:bb, cc:dd] = 1.
                        cc = max(int(x + 1 - self.radius), 0)
                        aa = max(int(y + 1 - self.radius), 0)
                        dd = min(int(x + 1 + self.radius), self.output_res)
                        bb = min(int(y + 1 + self.radius), self.output_res)

                        hms_list[1][self.num_joints*2 + idx, aa+1:bb-1, cc:dd] = 1.
                        hms_list[1][self.num_joints*2 + idx, aa:bb, cc+1:dd-1] = 1.
                        
                    elif v == 1:
                        hms_list[0][2*idx+1, aa:bb, cc:dd] = np.maximum(
                            hms_list[0][2*idx+1, aa:bb, cc:dd], joint_rg)
                        hms_list[1][2*idx+1, aa:bb, cc:dd] = 1.
                        

        hms_list[1][hms_list[1] == 2] = bg_weight

        return hms_list


class OffsetGenerator():
    def __init__(self, output_h, output_w, num_joints, radius):
        self.num_joints = num_joints
        self.num_joints_with_center = num_joints+1
        self.output_w = output_w
        self.output_h = output_h
        self.radius = radius

    def __call__(self, joints, area):
        assert joints.shape[1] == self.num_joints_with_center, \
            'the number of joints should be 18, 17 keypoints + 1 center joint.'

        offset_map = np.zeros((self.num_joints*2, self.output_h, self.output_w),
                              dtype=np.float32)
        weight_map = np.zeros((self.num_joints*3, self.output_h, self.output_w),
                              dtype=np.float32)
        area_map = np.zeros((self.output_h, self.output_w),
                            dtype=np.float32)

        for person_id, p in enumerate(joints):
            ct_x = int(p[-1, 0])
            ct_y = int(p[-1, 1])
            ct_v = int(p[-1, 2])
            if ct_v < 1 or ct_x < 0 or ct_y < 0 \
                    or ct_x >= self.output_w or ct_y >= self.output_h:
                continue
            weight = 1. / (np.power(area[person_id], 1/4) * 30)
            for idx, pt in enumerate(p[:-1]):
                if pt[2] > 1:
                    if pt[0] < 0 or pt[1] < 0 or \
                            pt[0] >= self.output_w or pt[1] >= self.output_h:
                        continue
                    # the center of kp, +1 for int
                    ct_x = pt[0] + 1
                    ct_y = pt[1] + 1
                    start_x = max(int(ct_x - self.radius), 0)
                    start_y = max(int(ct_y - self.radius), 0)
                    end_x = min(int(ct_x + self.radius), self.output_w)
                    end_y = min(int(ct_y + self.radius), self.output_h)

                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            if (pos_x == start_x or pos_x == (end_x - 1))  and \
                               (pos_y == start_y or pos_y == (end_y - 1)):
                               continue
                            for _idx, _pt in enumerate(p[:-1]):
                                if _pt[2] > 0:
                                    x, y = _pt[0], _pt[1]
                                    if x < 0 or y < 0 or \
                                            x >= self.output_w or y >= self.output_h:
                                        continue
                                                                  
                                    offset_x = pos_x - x
                                    offset_y = pos_y - y
                                    if offset_map[_idx*2, pos_y, pos_x] != 0 \
                                            or offset_map[_idx*2+1, pos_y, pos_x] != 0:
                                        if area_map[pos_y, pos_x] < area[person_id]:
                                            continue
                                    offset_map[_idx*2, pos_y, pos_x] = offset_x
                                    offset_map[_idx*2+1, pos_y, pos_x] = offset_y
                                    weight_map[_idx*2, pos_y, pos_x] = weight
                                    weight_map[_idx*2+1, pos_y, pos_x] = weight
                                    area_map[pos_y, pos_x] = area[person_id]
                            weight_map[self.num_joints*2 + idx, pos_y, pos_x] = int(weight_map[:, pos_y, pos_x].max() != 0)
        return offset_map, weight_map

class BoneGenerator():
    def __init__(self, output_h, output_w):
        self.output_w = output_w
        self.output_h = output_h
        # [[idx(start, end)]]
        self.bone = np.array(
                     [[5,7],# left upper arm
                      [6,8],# right upper arm
                      [7,9],# left lower arm
                      [8,10], #  right lower arm
                      [11,13], # left thigh
                      [12,14], # right thigh
                      [13,15], # left calf
                      [14,16]] # right calf
                     )
        self.bone_num = self.bone.shape[0]

    def __call__(self, joints, area):

        bone_map = np.zeros((2*self.bone_num, self.output_h, self.output_w),
                              dtype=np.float32)
        weight_map = 2 * np.ones((2*self.bone_num, self.output_h, self.output_w),
                              dtype=np.float32)
        degree = np.array([1., 1.], dtype=bone_map.dtype)
        for person_id, p in enumerate(joints):
            ct_x = int(p[-1, 0])
            ct_y = int(p[-1, 1])
            ct_v = int(p[-1, 2])
            if ct_v < 1 or ct_x < 0 or ct_y < 0 \
                    or ct_x >= self.output_w or ct_y >= self.output_h:
                continue
            for b_idx in range(self.bone_num):
                if p[self.bone[b_idx, 0], 2] !=0 and p[self.bone[b_idx, 1], 2] != 0:
                    points = p[self.bone[b_idx], :2].copy()
                    vector = points[1] - points[0]
                    distance = ((vector**2).sum())**0.5
                    if distance < 2:
                        continue
                    degree[0] = vector[0] / distance # x
                    degree[1] = vector[1] / distance # y
                    interval = int(distance) * 2 
                    for it in range(interval):
                        k = it/interval
                        point = k*points[0] + (1-k)*points[1]
                        point = (point+0.5).astype(np.int)
                        if point.min() < 0 or point.max() > self.output_h - 1:
                            continue
                        if (bone_map[2*b_idx:2*(b_idx+1), point[1], point[0]] == 0).all():
                            bone_map[2*b_idx:2*(b_idx+1), point[1], point[0]] = degree
                            weight_map[2*b_idx:2*(b_idx+1), point[1], point[0]] = 1.
                        elif  (bone_map[2*b_idx:2*(b_idx+1), point[1], point[0]] == degree).all():
                            pass
                        else:
                            weight_map[2*b_idx:2*(b_idx+1), point[1], point[0]] = 2
        weight_map = np.where(weight_map == 2, 0., weight_map)
        return bone_map, weight_map
