from lib.core.inference import get_multi_stage_outputs
from lib.core.inference import aggregate_results
from lib.core.inference import offset_to_pose
from lib.core.nms import pose_nms
from lib.core.match import match_pose_to_heatmap
from lib.dataset import make_test_dataloader
from lib.utils.utils import create_logger
from lib.utils.transforms import resize_align_multi_scale
from lib.utils.transforms import get_final_preds
from lib.utils.transforms import get_multi_scale_size
from lib.utils.rescore import rescore_valid
from visual import *
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import torchvision.ops as ops

def nms(heatmaps, nms_kernel, nms_padding):
    nms_pool = torch.nn.MaxPool2d(nms_kernel, 1, nms_padding)
    maxm = nms_pool(heatmaps)
    maxm = torch.eq(maxm, heatmaps).float()
    heatmaps = heatmaps * maxm
    return heatmaps


def get_locations(output_h, output_w, device):
    shifts_x = torch.arange(
        0, output_w, step=1,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, output_h, step=1,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)

    return locations.reshape(output_h, output_w, 2)


def get_reg_poses(offsets, num_joints):
    _, _, h, w = offsets.shape
    # offset = rearrange(offset, 'b (k n) h w -> b (h w) k n', n=2)
    # # offset = offset.permute(1, 2, 0).reshape(h*w, num_joints, 2)
    offsets = rearrange(offsets, 'b (k n) w h -> b w h k n', n=2).shape
    locations = get_locations(h, w, offsets.device) #(h, w, 2)
    poses = locations - offsets

    return poses


def offset_to_pose(offsets, flip=True, flip_index=None):
    _, _, h, w = offsets.shape
    offsets = rearrange(offsets, 'b (k n) w h -> b k w h n', n=2)
    locations = get_locations(h, w, offsets.device) #(h, w, 2)
    poses = locations - offsets

    return poses #torch.Size([8, 17, 128, 128, 2])

# train refine
def offset_to_pose_train(offsets, locations):
    offsets = rearrange(offsets, 'b (k n) w h -> b k w h n', n=2)
    poses = locations - offsets
    return poses #torch.Size([8, 17, 128, 128, 2])


def bilinear_value(map: torch.Tensor,
                   points: torch.Tensor,
                   mode='bilinear',
                   padding_mode='zeros',
                   align_corners=True):
    '''
    funtion:
        get the value of points from the map by bilinearity
    
    args:
        map: [batch_size, in_channels, h, w]
        points: [batch_size, h', w', (x, y)], (x, y) corresponds to (w, h)
                 
    return:
        [batch_size, h', w', value]
    '''
    _, _, h, w = map.shape
    # scale grid to range [-1, 1]
    grid = torch.cat((2 * points[..., 0, None] / (w - 1) - 1,
                      2 * points[..., 1, None] / (h - 1) - 1),
                     dim=-1)
    grid = grid.to(map.dtype)
    # [b, c, h, w], as grid h=n=1, w=2
    out = F.grid_sample(map,
                        grid,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners)
    return rearrange(out, 'b c h w -> b h w c')

def get_loc(cfg, heatmaps, offsets, gt_offsets, offset_w, locations, affine=None): #offsets [8, 34, 128, 128]
    # heatmap [b,18,w,h] -> [b,18,k]
    _, _, w, h = heatmaps.shape
    device = heatmaps.device
    nms_heatmaps = nms(heatmaps,
                       nms_kernel=cfg.NMS_KERNEL,
                       nms_padding=cfg.NMS_PADDING)
    posemap = offset_to_pose_train(offsets, locations)
    if affine is not None:
        posemap = posemap * affine[0] + affine[1]
    N, K, H, W = nms_heatmaps.size()

    nms_heatmaps = nms_heatmaps.flatten(-2)
    val_k, ind = nms_heatmaps.topk(cfg.TOP_K, dim=2)
    
    x = ind % W
    y = ind // W
    # Add Batch index as an additional coordinate
    b_size = ind.size(0)
    batch_ix = torch.arange(b_size, device=ind.device).view(b_size, 1, 1)
    batch_ix = batch_ix.expand(-1, ind.size(1), ind.size(2))

    ind_k = torch.stack((batch_ix, x, y), dim=3)
    # val_k[8, 18, 30], ind_k [8, 18, 30, 3]选出了heatmap前n个极大值

    # [  1,               17,           30,               17,                2     ]
    # [batch,   heatmap keypint type,   top k,   posmap keypoint type,   coordinate]
    offset_loc = posemap[ind_k[..., 0], :, ind_k[..., 2],
                         ind_k[..., 1],:]


    return {
            'loc_score': val_k.flatten(1,2), # torch.Size([b, 17 8])
            'offset_loc': offset_loc.flatten(1,2), #  torch.Size([b, 17 8, 17, 2])
            'posemap': posemap, # torch.Size([b, 17, 128, 192, 2])
            }