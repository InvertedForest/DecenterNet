import torch
import torch.nn as nn
from torch.optim import Optimizer
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Callback
from lib.dataset import make_dataloader, make_test_dataloader

from lib.core.inference import my_get_multi_stage_outputs
from lib.core.inference import my_aggregate_results
from lib.core.nms import pose_nms
from lib.dataset import make_test_dataloader
from lib.utils.transforms import resize_align_multi_scale
from lib.utils.transforms import get_final_preds
from lib.utils.transforms import get_multi_scale_size
from .get_loc import get_loc, get_loc, get_locations
from lib.core.loss import MultiLossFactory
import numpy as np
from visual import *
import sys
import os
from mmpose.apis.test import collect_results_gpu
import random
from lib.models.decenternet import get_pose_net
class MyModel(LightningModule):
    def __init__(self, cfg, v_dataset=None, modelPath=None):
        super().__init__()
        self.cfg = cfg
        self.dekr_model = get_pose_net(cfg, is_train=True)
        if self.cfg.IFPRETRAINED:
            model_CKPT = torch.load("model/pose_coco/pose_dekr_hrnetw32_coco.pth",
                                    map_location='cpu')
            self.dekr_model.load_state_dict(model_CKPT)
            print('loading DEHR weight!')

        if self.cfg.IFFREEZE:
            self.dekr_model.eval()
            self.dekr_model.requires_grad_(False)
        
        self.valmodel = None
        if modelPath is not None:
            if 'ckpt' in modelPath:
                self.valmodel = modelPath
                modelPath = os.path.dirname(os.path.dirname(modelPath))
            self.modelPath = modelPath
            self.logPath = os.path.join(self.modelPath, 'AP.log')
            self.final_output_dir = os.path.join(os.path.join(self.modelPath, 'json_out'), str(random.randint(10,30)))
        
        self.loss = MultiLossFactory(cfg)
        self.v_dataset = v_dataset
        self.transformer = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                )
            ])
        self.save_hyperparameters('cfg')

    def forward(self, x):
        return self.training_step(x,0)


    def training_step(self, batch, batch_idx=0):
        # batch
        #[b 3 512 512] [b 18 128 128] [b 18 128 128] [b 34 128 128] [b 34 128 128]
        [image, heatmap, mask, offset, offset_w, bone, bone_w]=batch
        pheatmap, poffset, pbone, ploc = self.dekr_model(image)

        heatmap_loss, offset_loss, bone_loss, loc_loss = \
            self.loss(pheatmap, poffset, heatmap, mask, offset, offset_w, pbone, bone, bone_w, ploc)
        all_loss = heatmap_loss + offset_loss + loc_loss + bone_loss

        self.log("heatmap_loss", heatmap_loss, on_step=True, on_epoch=True, logger=True)
        self.log("offset_loss", offset_loss, on_step=True, on_epoch=True, logger=True)
        self.log("bone_loss", bone_loss, on_step=True, on_epoch=True, logger=True)
        self.log("loc_loss", loc_loss, on_step=True, on_epoch=True, logger=True)
        self.log("all_loss", all_loss, on_step=True, on_epoch=True, logger=True)
        return all_loss


    def configure_optimizers(self):
        optim_class = getattr(torch.optim, self.cfg.TRAIN.OPTIMIZER)
        optimizer = optim_class([
                    {'params': self.dekr_model.parameters(),
                        'lr': self.cfg.TRAIN.DEKR_LR},
                        ] )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.cfg.TRAIN.LR_STEP, self.cfg.TRAIN.LR_FACTOR
        )
        return [optimizer], [lr_scheduler]
    
    def optimizer_step(
        self,
        epoch: int,
        batch_idx,
        optimizer,
        optimizer_idx: int,
        optimizer_closure,
        on_tpu: bool,
        using_native_amp: bool,
        using_lbfgs: bool,
    ) :
        if self.trainer.global_step < self.cfg.TRAIN.warmup.warmup_iters:
            k = (1 - self.trainer.global_step / self.cfg.TRAIN.warmup.warmup_iters) * (1 - self.cfg.TRAIN.warmup.warmup_ratio)

            for _, pg in enumerate(optimizer.param_groups):
                if _ == 0:
                    pg['lr'] = (1 - k) * self.cfg.TRAIN.DEKR_LR
                elif _ == 1:
                    pg['lr'] = (1 - k) * self.cfg.TRAIN.LR
                    
        super().optimizer_step(
                epoch,
                batch_idx,
                optimizer,
                optimizer_idx,
                optimizer_closure,
                on_tpu,
                using_native_amp,
                using_lbfgs
        )
       
    
    def validation_step(self, batch, batch_idx=0):
        assert 1 == batch.size(0), 'Test batch size should be 1'
        image=batch[0].cpu().numpy()
        # import cv2
        # image = cv2.imread(
        #         "/data/jupyter/DEKR-refine/t.jpg",
        #         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        #     )
        cfg = self.cfg
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )

        heatmap_sum = 0
        for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, cfg.DATASET.INPUT_SIZE, scale, 1.0
            )

            image_resized = self.transformer(image_resized)
            image_resized = image_resized.unsqueeze(0).to(batch.device)
# 21 44 51
            heatmap, offset, locmap = my_get_multi_stage_outputs(
                cfg, self.dekr_model, image_resized, cfg.TEST.FLIP_TEST)

            # pos_output = self.pos.forward(self.cfg, heatmap)
            locations = get_locations(heatmap.shape[2],
                                      heatmap.shape[3],
                                        heatmap.device)
            
            # # 128

            loc_res = get_loc(cfg = self.cfg.MODEL.GET_LOC,
                              heatmaps = locmap,
                              offsets = offset,
                              gt_offsets = offset,
                              offset_w = offset,
                              locations = locations)
            final_loc = loc_res['offset_loc']*cfg.DATASET.INPUT_SIZE*1.0/cfg.DATASET.OUTPUT_SIZE/scale
            posemap = loc_res['posemap']*cfg.DATASET.INPUT_SIZE*1.0/cfg.DATASET.OUTPUT_SIZE/scale

            ct_score = loc_res['loc_score'][..., None, None] + torch.zeros_like(final_loc)[..., 0, None]

            final_loc = torch.cat((final_loc, ct_score), dim=-1)

            # heatmap * 4
            heatmap_sum = my_aggregate_results(
                cfg, heatmap_sum, heatmap[:,::2]*locmap, scale
            )
            heatmap_avg = heatmap_sum/len(cfg.TEST.SCALE_FACTOR)
            heatmap_avg = torch.clamp(heatmap_avg, min=0.0)
            poses, scores = pose_nms(self.cfg, heatmap_avg, [final_loc[0]], posemap, image_resized, batch_idx)
            final_poses = get_final_preds(poses, center, scale_resized, base_size)

        return [final_poses, scores]


    def validation_epoch_end(self, outputs):

        all_out = collect_results_gpu(outputs, size=None)
        if self.global_rank == 0:
            cfg = self.cfg
            preds = []
            scores = []
            for i in all_out:
                preds.append(i[0])
                scores.append(i[1])
            
            # coco eval
            stdout_ori = sys.stdout
            sys.stdout = None
            if cfg.RESCORE.GET_DATA:
                self.v_dataset.evaluate(
                    cfg, preds, scores, self.final_output_dir, str(self.global_rank))
                print('Generating dataset for rescorenet successfully')
                sys.stdout = stdout_ori
            else:
                name_values, _ = self.v_dataset.evaluate(
                    cfg, preds, scores, self.final_output_dir, str(self.global_rank))
                sys.stdout = stdout_ori
                
                if isinstance(name_values, list):
                    for name_value in name_values:
                        self.add_log( name_value, cfg.MODEL.NAME)
                else:
                    self.add_log( name_values, cfg.MODEL.NAME)
            self.log('AP', name_values['AP'], rank_zero_only=True)

        self.trainer.strategy.barrier()
        return None

    def add_log(self, name_value, full_arch_name):
        a = sys.stdout
        # if self.trainer.training:
        if True:
            sys.stdout = open(self.logPath, mode='a', encoding='utf-8')
        self._print_name_value(name_value, full_arch_name)
        sys.stdout = a
        # self._print_name_value(name_value, full_arch_name)

    def _print_name_value(self, name_value, full_arch_name):
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        print("\n\n")
        if self.valmodel is not None:
            print(self.valmodel)
        else:
            print("epoch: " + str(self.current_epoch))
        # print(self.cfg.TEST)
        print('| Arch ' + ' '.join(['| {}'.format(name)
                                        for name in names]) + ' |')
        print('|---' * (num_values + 1) + '|')

        if len(full_arch_name) > 15:
            full_arch_name = full_arch_name[:8] + '...'
        print('| ' + full_arch_name + ' ' +
                    ' '.join(['| {:.3f}'.format(value)
                            for value in values]) + ' |')

class MyDataModule_val(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.v_dataloader, self.v_dataset = make_test_dataloader(self.cfg)
    def val_dataloader(self):
        return self.v_dataloader

class MyDataModule_train(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.t_dataloader = make_dataloader(self.cfg)
        # self.v_dataloader, self.v_dataset = make_test_dataloader(self.cfg)
    def train_dataloader(self):
        return self.t_dataloader

class MyCallback(Callback):
    def __init__(self) -> None:
        pass
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx) -> None:
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                print(name)
