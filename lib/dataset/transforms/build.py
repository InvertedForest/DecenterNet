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

from . import transforms as T


FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
    'COCO_34_OFFSET': [
        0, 1, 4, 5, 2, 3, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 20, 21, 18, 19, 24, 25, 22, 23, 28, 29, 26, 27, 32, 33, 30, 31
    ],
    
    'CROWDPOSE_WITH_CENTER': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
    ],
    'CROWDPOSE_14': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
    ],
    'CROWDPOSE_28': [
        2,3, 0,1, 6,7, 4,5, 10,11, 8,9, 14,15, 12,13, 18,19, 16,17, 22,23, 20,21, 24,25, 26,27
    ]
}


def build_transforms(cfg, is_train=True):
    assert is_train is True, 'Please only use build_transforms for training.'

    if 'coco' in cfg.DATASET.DATASET:
        dataset_name = 'COCO'
    elif 'crowd_pose' in cfg.DATASET.DATASET:
        dataset_name = 'CROWDPOSE'
    else:
        raise ValueError('Please implement flip_index \
            for new dataset: %s.' % cfg.DATASET.DATASET)
    coco_flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER']

    transforms = T.Compose(
        [
            T.RandomAffineTransform(
                cfg.DATASET.INPUT_SIZE,
                cfg.DATASET.OUTPUT_SIZE,
                cfg.DATASET.MAX_ROTATION,
                cfg.DATASET.MIN_SCALE,
                cfg.DATASET.MAX_SCALE,
                cfg.DATASET.SCALE_TYPE,
                cfg.DATASET.MAX_TRANSLATE
            ),
            T.RandomHorizontalFlip(
                coco_flip_index, cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.FLIP),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    return transforms
