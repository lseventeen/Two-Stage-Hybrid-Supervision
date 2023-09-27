from dataloading.data_loader import Flare_DataLoader,Flare_semi_supervised_DataLoader
from dataloading.dataset import Flare_Dataset_two_folder, Flare_Dataset
from base_trainer import base_trainer
import os
import sys
from logger.logger import Logger
from time import time, sleep
from typing import Union, Tuple, List
import wandb
from datetime import datetime
import numpy as np
import torch
from loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from dataloading.utils import unpack_dataset, get_case_identifiers
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch._dynamo import OptimizedModule
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from data_augmentation.data_augmentation import get_training_transforms,get_validation_transforms
from utilities.helpers import empty_cache, dummy_context
from lr_scheduler.polylr import PolyLRScheduler
from loss.compound_losses import DC_and_CE_loss,part_DC_and_CE_loss
from loss.deep_supervision import DeepSupervisionWrapper
from utilities.collate_outputs import collate_outputs
from data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from data_augmentation.compute_initial_patch_size import get_patch_size
from create_network import create_network
from torch import autocast, nn
from sklearn.model_selection import KFold
import inspect
ANISO_THRESHOLD = 3


class full_supervised_tumour_trainer(base_trainer):
    
    def __init__(self, experiment_id, fold, num_process,config):
        super().__init__(experiment_id, fold, num_process,config)

        self.class_weight = torch.tensor(config['Loss']['classes_weight']).to(self.device, non_blocking=True).float()
    def _build_loss(self):

        loss = DC_and_CE_loss({'batch_dice': self.batch_dice,
                               'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'weight':self.class_weight}, weight_ce=self.weight_ce, weight_dice=self.weight_dice,
                              ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)
       

        deep_supervision_scales = self._get_deep_supervision_scales(self.pool_op_kernel_sizes)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i)
                           for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
