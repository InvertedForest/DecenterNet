# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from cgitb import enable
from gc import callbacks
import os
import pprint
import shutil
import warnings
import sys

sys.path.append('/data/jupyter/DEKR-refine/lib/models')
sys.path.append('/data/jupyter/DEKR-refine/')

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


from lib.config import cfg
from lib.config import update_config
from lib.models.refine import MyModel, MyDataModule_train, MyDataModule_val, MyCallback

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor, DeviceStatsMonitor

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--versionid',
                        default=-1,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--train', # 1 train 0 val
                        default=1,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--model_path', # 1 t 0 v
                        default='model path is not specified',
                        type=str,
                        help='model path')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    cfg.defrost()
    cfg.RANK = args.rank
    # git
    version_id = args.versionid
    plan_name = [i[2:-1] for i in os.popen('git branch') if i[0] == '*'][0]
    logger = TensorBoardLogger("tb_logs", name=plan_name, default_hp_metric=False, version=version_id)
    
    # Data loading code
    if args.train:
        datamodule = MyDataModule_train(cfg)
    else:
        datamodule = MyDataModule_val(cfg)

    unused_para_callback = MyCallback()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    gpu_state = DeviceStatsMonitor()
    # rich_bar = RichProgressBar()
    if args.train:
        if os.path.exists(logger.log_dir): raise ValueError('version exists!')
        mymodel = MyModel(cfg, modelPath=logger.log_dir)
        print(logger.log_dir)
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            monitor="all_loss_epoch",
            mode='min',
            save_top_k=100,
            every_n_epochs=cfg.TRAIN.CKPT_VAL,
        )
        trainer = pl.Trainer(
            gpus=[2,3], # gpu id
            strategy=DDPStrategy(find_unused_parameters=False),
            gradient_clip_val=1,
            # num_sanity_val_steps=10,
            # limit_train_batches=2,
            # limit_val_batches=10, 
            max_epochs = cfg.TRAIN.MAX_EPOCHS,
            # check_val_every_n_epoch=cfg.TRAIN.CKPT_VAL,
            precision=16,
            # log_every_n_steps = 1,
            callbacks=[
                        checkpoint_callback,
                        lr_monitor,
                        # unused_para_callback,
                        # gpu_state,
                       ],
            logger=logger,
        )

        trainer.fit(mymodel, datamodule,
                    ckpt_path=None
                    )
    else:
        mymodel = MyModel.load_from_checkpoint(args.model_path,
                                                cfg=cfg,
                                                modelPath=args.model_path,
                                                v_dataset=datamodule.v_dataset)
        print('load model weights!')
        trainer = pl.Trainer(
            gpus=1,
            strategy=DDPStrategy(find_unused_parameters=False),
            # limit_val_batches=100,
            precision=32,
            )

        trainer.validate(mymodel, datamodule)

if __name__ == '__main__':
    main()