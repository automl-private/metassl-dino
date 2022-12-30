# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
import neps
import logging
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

from configspaces import get_pipeline_space
from groupaugment import get_groupaugment_transformation
from eval_linear import eval_linear

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from functools import partial


class DataAugmentationDINO(object):
    def __init__(self, dataset, global_crops_scale, local_crops_scale, local_crops_number, is_neps_run, use_fixed_DA_hypers, hyperparameters=None, config_space=None):
        if is_neps_run and config_space == "data_augmentation":
            crops_scale_boundary = hyperparameters["crops_scale_boundary"]
            global_crops_scale = (crops_scale_boundary, global_crops_scale[1])
            local_crops_scale = (local_crops_scale[0], crops_scale_boundary)
            local_crops_number = hyperparameters["local_crops_number"]

            p_horizontal_crop_1 = hyperparameters["p_horizontal_crop_1"]
            p_colorjitter_crop_1 = hyperparameters["p_colorjitter_crop_1"]
            p_grayscale_crop_1 = hyperparameters["p_grayscale_crop_1"]
            p_gaussianblur_crop_1 = hyperparameters["p_gaussianblur_crop_1"]
            p_solarize_crop_1 = hyperparameters["p_solarize_crop_1"]
            
            p_horizontal_crop_2 = hyperparameters["p_horizontal_crop_2"]
            p_colorjitter_crop_2 = hyperparameters["p_colorjitter_crop_2"]
            p_grayscale_crop_2 = hyperparameters["p_grayscale_crop_2"]
            p_gaussianblur_crop_2 = hyperparameters["p_gaussianblur_crop_2"]
            p_solarize_crop_2 = hyperparameters["p_solarize_crop_2"]
    
            p_horizontal_crop_3 = hyperparameters["p_horizontal_crop_3"]
            p_colorjitter_crop_3 = hyperparameters["p_colorjitter_crop_3"]
            p_grayscale_crop_3 = hyperparameters["p_grayscale_crop_3"]
            p_gaussianblur_crop_3 = hyperparameters["p_gaussianblur_crop_3"]
            p_solarize_crop_3 = hyperparameters["p_solarize_crop_3"]

            print("\n\nNEPS DATA AUGMENTATION HYPERPARAMETERS:\n")
            print(f"global_crops_scale: {global_crops_scale}")
            print(f"local_crops_scale: {local_crops_scale}")
            print(f"local_crops_number: {local_crops_number}")
            print(f"p_horizontal_crop_1: {p_horizontal_crop_1}")
            print(f"p_colorjitter_crop_1: {p_colorjitter_crop_1}")
            print(f"p_grayscale_crop_1: {p_grayscale_crop_1}")
            print(f"p_gaussianblur_crop_1: {p_gaussianblur_crop_1}")
            print(f"p_solarize_crop_1: {p_solarize_crop_1}")
            print(f"p_horizontal_crop_2: {p_horizontal_crop_2}")
            print(f"p_colorjitter_crop_2: {p_colorjitter_crop_2}")
            print(f"p_grayscale_crop_2: {p_grayscale_crop_2}")
            print(f"p_gaussianblur_crop_2: {p_gaussianblur_crop_2}")
            print(f"p_solarize_crop_2: {p_solarize_crop_2}")
            print(f"p_horizontal_crop_3: {p_horizontal_crop_3}")
            print(f"p_colorjitter_crop_3: {p_colorjitter_crop_3}")
            print(f"p_grayscale_crop_3: {p_grayscale_crop_3}")
            print(f"p_gaussianblur_crop_3: {p_gaussianblur_crop_3}")
            print(f"p_solarize_crop_3: {p_solarize_crop_3}")
        else:
            if use_fixed_DA_hypers:
                if dataset == "ImageNet":
                    raise NotImplementedError
                elif dataset == "CIFAR-10":
                    crops_scale_boundary = 0.35
                    global_crops_scale = (crops_scale_boundary, global_crops_scale[1])
                    local_crops_scale = (local_crops_scale[0], crops_scale_boundary)
                    local_crops_number = 5

                    p_horizontal_crop_1, p_colorjitter_crop_1, p_grayscale_crop_1, p_gaussianblur_crop_1, p_solarize_crop_1 = 0.76, 0.89, 0.07, 0.90, 0.33
                    p_horizontal_crop_2, p_colorjitter_crop_2, p_grayscale_crop_2, p_gaussianblur_crop_2, p_solarize_crop_2 = 0.01, 0.91, 0.59, 0.11, 0.17
                    p_horizontal_crop_3, p_colorjitter_crop_3, p_grayscale_crop_3, p_gaussianblur_crop_3, p_solarize_crop_3 = 0.75, 0.63, 0.00, 0.17, 0.27
                elif dataset == "CIFAR-100":
                    crops_scale_boundary = 0.38
                    global_crops_scale = (crops_scale_boundary, global_crops_scale[1])
                    local_crops_scale = (local_crops_scale[0], crops_scale_boundary)
                    local_crops_number = 8

                    p_horizontal_crop_1, p_colorjitter_crop_1, p_grayscale_crop_1, p_gaussianblur_crop_1, p_solarize_crop_1 = 0.43, 0.78, 0.05, 0.90, 0.11
                    p_horizontal_crop_2, p_colorjitter_crop_2, p_grayscale_crop_2, p_gaussianblur_crop_2, p_solarize_crop_2 = 0.35, 0.65, 0.31, 0.09, 0.17
                    p_horizontal_crop_3, p_colorjitter_crop_3, p_grayscale_crop_3, p_gaussianblur_crop_3, p_solarize_crop_3 = 0.44, 0.62, 0.45, 0.19, 0.04
                else:
                    raise NotImplementedError
            else:
                p_horizontal_crop_1, p_colorjitter_crop_1, p_grayscale_crop_1, p_gaussianblur_crop_1, p_solarize_crop_1 = 0.5, 0.8, 0.2, 1.0, 0.0
                p_horizontal_crop_2, p_colorjitter_crop_2, p_grayscale_crop_2, p_gaussianblur_crop_2, p_solarize_crop_2 = 0.5, 0.8, 0.2, 0.1, 0.2
                p_horizontal_crop_3, p_colorjitter_crop_3, p_grayscale_crop_3, p_gaussianblur_crop_3, p_solarize_crop_3 = 0.5, 0.8, 0.2, 0.5, 0.0

        if dataset == "ImageNet":
            global_crop_size = 224
            local_crop_size = 96
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif dataset == "CIFAR-10":
            global_crop_size = 32
            local_crop_size = 16
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        elif dataset == "CIFAR-100":
            global_crop_size = 32
            local_crop_size = 16
            normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        else:
            raise NotImplementedError(f"Dataset '{args.dataset}' not implemented yet!")

        normalize = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            
            transforms.RandomHorizontalFlip(p=p_horizontal_crop_1),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_colorjitter_crop_1
                ),
            transforms.RandomGrayscale(p=p_grayscale_crop_1),
            
            utils.GaussianBlur(p=p_gaussianblur_crop_1),  # default: 1.0
            utils.Solarization(p=p_solarize_crop_1),  # default: 0.0
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
    
            transforms.RandomHorizontalFlip(p=p_horizontal_crop_2),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_colorjitter_crop_2
                ),
            transforms.RandomGrayscale(p=p_grayscale_crop_2),
            
            utils.GaussianBlur(p=p_gaussianblur_crop_2),  # default: 0.1
            utils.Solarization(p=p_solarize_crop_2),  # default: 0.2
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
    
            transforms.RandomHorizontalFlip(p=p_horizontal_crop_3),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_colorjitter_crop_3
                ),
            transforms.RandomGrayscale(p=p_grayscale_crop_3),
            
            utils.GaussianBlur(p=p_gaussianblur_crop_3),  # default: 0.5
            utils.Solarization(p=p_solarize_crop_3),  # default: 0.0
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------


class GroupAugmentDataAugmentationDINO(object):
    def __init__(self, dataset, global_crops_scale, local_crops_scale, local_crops_number, is_neps_run, use_fixed_DA_hypers, hyperparameters=None, config_space=None):
        if is_neps_run and config_space == "groupaugment":
            p_color_transformations_crop_1 = hyperparameters["p_color_transformations_crop_1"]
            p_geometric_transformations_crop_1 = hyperparameters["p_geometric_transformations_crop_1"]
            p_non_rigid_transformations_crop_1 = hyperparameters["p_non_rigid_transformations_crop_1"]
            p_quality_transformations_crop_1 = hyperparameters["p_quality_transformations_crop_1"]
            p_exotic_transformations_crop_1 = hyperparameters["p_exotic_transformations_crop_1"]
            
            p_color_transformations_crop_2 = hyperparameters["p_color_transformations_crop_2"]
            p_geometric_transformations_crop_2 = hyperparameters["p_geometric_transformations_crop_2"]
            p_non_rigid_transformations_crop_2 = hyperparameters["p_non_rigid_transformations_crop_2"]
            p_quality_transformations_crop_2 = hyperparameters["p_quality_transformations_crop_2"]
            p_exotic_transformations_crop_2 = hyperparameters["p_exotic_transformations_crop_2"]

            p_color_transformations_crop_3 = hyperparameters["p_color_transformations_crop_3"]
            p_geometric_transformations_crop_3 = hyperparameters["p_geometric_transformations_crop_3"]
            p_non_rigid_transformations_crop_3 = hyperparameters["p_non_rigid_transformations_crop_3"]
            p_quality_transformations_crop_3 = hyperparameters["p_quality_transformations_crop_3"]
            p_exotic_transformations_crop_3 = hyperparameters["p_exotic_transformations_crop_3"]
            
            n_color_transformations = hyperparameters["n_color_transformations"]
            n_geometric_transformations = hyperparameters["n_geometric_transformations"]
            n_non_rigid_transformations = hyperparameters["n_non_rigid_transformations"]
            n_quality_transformations = hyperparameters["n_quality_transformations"]
            n_exotic_transformations = hyperparameters["n_exotic_transformations"]
            n_total = hyperparameters["n_total"]
            
        else:
            if use_fixed_DA_hypers:
                if dataset == "ImageNet":
                    raise NotImplementedError
                elif dataset == "CIFAR-10":
                    raise NotImplementedError
                elif dataset == "CIFAR-100":
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                p_color_transformations_crop_1 = 0.5
                p_geometric_transformations_crop_1 = 0.5
                p_non_rigid_transformations_crop_1 = 0
                p_quality_transformations_crop_1 = 0
                p_exotic_transformations_crop_1 = 0

                p_color_transformations_crop_2 = 0.5
                p_geometric_transformations_crop_2 = 0.5
                p_non_rigid_transformations_crop_2 = 0
                p_quality_transformations_crop_2 = 0
                p_exotic_transformations_crop_2 = 0

                p_color_transformations_crop_3 = 0.5
                p_geometric_transformations_crop_3 = 0.5
                p_non_rigid_transformations_crop_3 = 0
                p_quality_transformations_crop_3 = 0
                p_exotic_transformations_crop_3 = 0

                n_color_transformations = 1 
                n_geometric_transformations = 1
                n_non_rigid_transformations = 1 
                n_quality_transformations = 1
                n_exotic_transformations = 1
                n_total = 1 
        
        print("\n\nDATA AUGMENTATION HYPERPARAMETERS:\n")
        print(f"p_color_transformations_crop_1: {p_color_transformations_crop_1}")
        print(f"p_geometric_transformations_crop_1: {p_geometric_transformations_crop_1}")
        print(f"p_non_rigid_transformations_crop_1: {p_non_rigid_transformations_crop_1}")
        print(f"p_quality_transformations_crop_1: {p_quality_transformations_crop_1}")
        print(f"p_exotic_transformations_crop_1: {p_exotic_transformations_crop_1}")
        print("-----")
        print(f"p_color_transformations_crop_2: {p_color_transformations_crop_2}")
        print(f"p_geometric_transformations_crop_2: {p_geometric_transformations_crop_2}")
        print(f"p_non_rigid_transformations_crop_2: {p_non_rigid_transformations_crop_2}")
        print(f"p_quality_transformations_crop_2: {p_quality_transformations_crop_2}")
        print(f"p_exotic_transformations_crop_2: {p_exotic_transformations_crop_2}")
        print("-----")
        print(f"p_color_transformations_crop_3: {p_color_transformations_crop_3}")
        print(f"p_geometric_transformations_crop_3: {p_geometric_transformations_crop_3}")
        print(f"p_non_rigid_transformations_crop_3: {p_non_rigid_transformations_crop_3}")
        print(f"p_quality_transformations_crop_3: {p_quality_transformations_crop_3}")
        print(f"p_exotic_transformations_crop_3: {p_exotic_transformations_crop_3}")
        print("-----")
        print(f"n_color_transformations: {n_color_transformations}")
        print(f"n_geometric_transformations: {n_geometric_transformations}")
        print(f"n_non_rigid_transformations: {n_non_rigid_transformations}")
        print(f"n_quality_transformations: {n_quality_transformations}")
        print(f"n_exotic_transformations: {n_exotic_transformations}")
        print(f"n_total: {n_total}")


        if dataset == "ImageNet":
            global_crop_size = 224
            local_crop_size = 96
            normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]  # mean, std
        elif dataset == "CIFAR-10":
            global_crop_size = 32
            local_crop_size = 16
            normalize = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]  # mean, std
        elif dataset == "CIFAR-100":
            global_crop_size = 32
            local_crop_size = 16
            normalize = [[0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]]  # mean, std
        else:
            raise NotImplementedError(f"Dataset '{args.dataset}' not implemented yet!")
        
        if False:
            # Test default DINO data augmentation with albumentations
            import cv2
            import torchvision.transforms as transforms
            from albumentations import (
                ChannelShuffle,
                ColorJitter,
                Compose,
                Cutout,
                ElasticTransform,
                Equalize,
                GaussianBlur,
                GaussNoise,
                GridDistortion,
                HorizontalFlip,
                Normalize,
                OpticalDistortion,
                RandomGridShuffle,
                RandomResizedCrop,
                Solarize,
                SomeOf,
                ToGray,
            )
            from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
            from albumentations.pytorch.transforms import ToTensorV2
            
            # first global crop
            self.global_transfo1 = Compose(
                [
                    RandomResizedCrop(height=global_crop_size, width=global_crop_size, scale=(0.4, 1.0), interpolation=cv2.INTER_CUBIC),
                    
                    HorizontalFlip(p=0.5),
                    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
                    ToGray(p=0.2),
                    GaussianBlur(p=1.0),
                    Solarize(p=0.0),

                    Normalize(normalize[0], normalize[1]),
                    ToTensorV2(),

                ],
                p=1,
            )
            
            # second global crop
            self.global_transfo2 = Compose(
                [
                    RandomResizedCrop(height=global_crop_size, width=global_crop_size, scale=(0.4, 1.0), interpolation=cv2.INTER_CUBIC),

                    HorizontalFlip(p=0.5),
                    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
                    ToGray(p=0.2),
                    GaussianBlur(p=0.1),
                    Solarize(p=0.2),

                    Normalize(normalize[0], normalize[1]),
                    ToTensorV2(),

                ],
                p=1,
            )
            
            # local crops
            self.local_crops_number = local_crops_number
            self.local_transfo = Compose(
                [
                    RandomResizedCrop(height=local_crop_size, width=local_crop_size, scale=(0.05, 0.4), interpolation=cv2.INTER_CUBIC),

                    HorizontalFlip(p=0.5),
                    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
                    ToGray(p=0.2),
                    GaussianBlur(p=0.5),
                    Solarize(p=0.0),

                    Normalize(normalize[0], normalize[1]),
                    ToTensorV2(),

                ],
                p=1,
            )

        else:
            print("\n\nGroupAugment\n\n")
            # first global crop
            self.global_transfo1 = get_groupaugment_transformation(
                p_color_transformations_crop_1,
                p_geometric_transformations_crop_1,
                p_non_rigid_transformations_crop_1,
                p_quality_transformations_crop_1,
                p_exotic_transformations_crop_1,
                n_color_transformations,
                n_geometric_transformations,
                n_non_rigid_transformations,
                n_quality_transformations,
                n_exotic_transformations,
                n_total,
                normalize,
                global_crop_size,
                global_crops_scale
            )

            # second global crop
            self.global_transfo2 = get_groupaugment_transformation(
                p_color_transformations_crop_2,
                p_geometric_transformations_crop_2,
                p_non_rigid_transformations_crop_2,
                p_quality_transformations_crop_2,
                p_exotic_transformations_crop_2,
                n_color_transformations,
                n_geometric_transformations,
                n_non_rigid_transformations,
                n_quality_transformations,
                n_exotic_transformations,
                n_total,
                normalize,
                global_crop_size,
                global_crops_scale
            )

            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = get_groupaugment_transformation(
                p_color_transformations_crop_3,
                p_geometric_transformations_crop_3,
                p_non_rigid_transformations_crop_3,
                p_quality_transformations_crop_3,
                p_exotic_transformations_crop_3,
                n_color_transformations,
                n_geometric_transformations,
                n_non_rigid_transformations,
                n_quality_transformations,
                n_exotic_transformations,
                n_total,
                normalize,
                local_crop_size,
                local_crops_scale
            )
    def __call__(self, image):
        crops = []
        image = np.array(image)
        crops.append(self.global_transfo1(image=image)["image"])
        crops.append(self.global_transfo2(image=image)["image"])
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image=image)["image"])
        return crops
