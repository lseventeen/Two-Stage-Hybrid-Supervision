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
from full_supervised_tumour_trainer import full_supervised_tumour_trainer
from load_pretrained_weights import load_pretrained_weights
ANISO_THRESHOLD = 3


class hybird_supervised_tumour_trainer(base_trainer):
    def __init__(self, experiment_id, fold, num_process,config):
        super().__init__(experiment_id, fold, num_process,config)

        assert self.batch_size%2 == 0
        self.sub_batch_size = self.batch_size//2
        
        self.unlabel_data_folder = config['Data_path']['preprocess_unlabel_data_dir']
        self.unlabel_data_case = get_case_identifiers(self.unlabel_data_folder)

        self.label_loss_weight = config['Loss']['label_loss_weight']
        self.pseudo_label_loss_weight = config['Loss']['pseudo_label_loss_weight']
        self.pretrained_weights_file = config['Trainer']['pretrained_weights_file'] 
    def initialize(self):
        if not self.was_initialized:

            self.network = create_network(
                self.conv_kernel_sizes,
                self.UNet_class_name,
                self.num_classes,
                self.n_conv_per_stage_decoder,
                self.n_conv_per_stage_encoder,
                self.UNet_base_num_features,
                self.unet_max_num_features,
                self.pool_op_kernel_sizes,
                self.dropout_p,
                self.num_input_channels,
                self.deep_supervision,
                self.is_NB
            ).to(self.device)
            print(self.network)
            self.teacher_network = create_network(
                self.conv_kernel_sizes,
                self.UNet_class_name,
                self.num_classes,
                self.n_conv_per_stage_decoder,
                self.n_conv_per_stage_encoder,
                self.UNet_base_num_features,
                self.unet_max_num_features,
                self.pool_op_kernel_sizes,
                self.dropout_p,
                self.num_input_channels,
                self.deep_supervision,
                self.is_NB
            ).to(self.device)
            for param in self.teacher_network.parameters():
                param.detach_()   # ema_model set
            # compile network for free speedup
            if self.compile:
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network)
                self.teacher_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.teacher_network)
                self.network = DDP(self.network, device_ids=[self.local_rank])
                self.teacher_network = DDP(self.teacher_network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
            if self.pretrained_weights_file is not None:
                load_pretrained_weights(self.network, self.pretrained_weights_file, verbose=True)
                load_pretrained_weights(self.teacher_network, self.pretrained_weights_file, verbose=True)
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
    def _build_loss(self):

        loss = DC_and_CE_loss({'batch_dice': self.batch_dice,
                               'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=self.weight_ce, weight_dice=self.weight_dice,
                              ignore_label=-1, dice_class=MemoryEfficientSoftDiceLoss)
       

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
    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        tr_keys, val_keys = self.do_split()
        label_dataset_tr = Flare_Dataset(self.label_data_folder,tr_keys)
        unlabel_dataset_tr = Flare_Dataset(self.unlabel_data_folder,self.unlabel_data_case,load_seg=False)
        dataset_val = Flare_Dataset(self.label_data_folder,val_keys)

        dl_tr = Flare_semi_supervised_DataLoader(label_dataset_tr, unlabel_dataset_tr,self.batch_size, self.patch_size)
        dl_val = Flare_DataLoader(dataset_val, self.batch_size, self.patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales(self.pool_op_kernel_sizes)

        rotation_for_DA, _, _, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # training pipeline
        tr_transforms = get_training_transforms(
            self.patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, self.do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,removeLabel=False
        )
        val_transforms = get_validation_transforms(deep_supervision_scales)
        if self.allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)

        else:
            mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=self.allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, self.allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)

        return mt_gen_train, mt_gen_val
    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)
        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(True)

        empty_cache(self.device)

        if self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.label_data_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(self.allowed_num_processes// 2)))
            unpack_dataset(self.unlabel_data_folder, unpack_segmentation=False, overwrite_existing=False,
                           num_processes=max(1, round(self.allowed_num_processes// 2)))
            self.print_to_log_file('unpacking done...')
        # maybe unpack

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train,self.dataloader_val = self.get_dataloaders()

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            predict = self.teacher_network(data)

            pseudo_gt = [torch.argmax(torch.softmax(o, dim=1),dim=1, keepdim=True).float() for o in predict]
            # pseudo_gt = torch.argmax(torch.softmax(predict, dim=1),dim=1, keepdim=False)
        
            # pseudo_gt[target==1]=1
            # pseudo_gt = [ pseudo_gt[i][target[i]==1] = 1  ]
            for i in range(len(pseudo_gt)):
                pseudo_gt[i][:self.sub_batch_size][target[i][:self.sub_batch_size]==1] = 1 

            label_loss = self.loss(output[:self.sub_batch_size], target[:self.sub_batch_size])
            pseudo_label_loss = self.loss(output, pseudo_gt)
            # l = fs_loss + ss_loss
            l = self.label_loss_weight*label_loss + self.pseudo_label_loss_weight * pseudo_label_loss
            # l = self.loss(output, target)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            if self.CGN:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            if self.CGN:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        self.update_ema_variables(self.network, self.teacher_network, 0.99)
        return {'loss': l.detach().cpu().numpy()}

    def on_train_epoch_start(self):
        self.network.train()
        self.teacher_network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log(
            'lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
    @torch.no_grad()
    def update_ema_variables(self,model, ema_model, alpha=0.99):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            # l = self.loss(output, target)
            l = self.loss(output[:self.sub_batch_size], target[:self.sub_batch_size])

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        
        # no need for softmax
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

        # if self.label_manager.has_ignore_label:
        #     if not self.label_manager.has_regions:
        #         mask = (target != self.label_manager.ignore_label).float()
        #         # CAREFUL that you don't rely on target after this line!
        #         target[target == self.label_manager.ignore_label] = 0
        #     else:
        #         mask = 1 - target[:, -1:]
        #         # CAREFUL that you don't rely on target after this line!
        #         target = target[:, :-1]
        # else:
        mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        # if not self.label_manager.has_regions:
        #     # if we train with regions all segmentation heads predict some kind of foreground. In conventional
        #     # (softmax training) there needs tobe one output for the background. We are not interested in the
        #     # background Dice
        #     # [1:] in order to remove background
        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}