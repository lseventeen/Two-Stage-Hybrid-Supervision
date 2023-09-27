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


class abdomen_local_trainer(base_trainer):
    def __init__(self, experiment_id, fold, num_process,config):
        super().__init__(experiment_id, fold, num_process,config)
        self.main_loss_weight = config['Loss']['main_loss_weight']
        self.auxiliary_loss_weight = config['Loss']['auxiliary_loss_weight']
        self.label_data_folder1 = config['Data_path']['preprocess_val_data_dir']
        self.label_data_folder2 = config["Data_path"]["preprocess_label_data_dir"]
        self.data_case1 = get_case_identifiers(self.label_data_folder1)
        self.data_case2 = get_case_identifiers(self.label_data_folder2)
        assert self.batch_size%2 == 0
        self.sub_batch_size = self.batch_size//2






    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            # case_identifiers = get_case_identifiers(self.label_data_folder)

          
            tr_keys = self.data_case1
            val_keys = tr_keys
            
        else:
            # splits_file = join(self.label_data_folder, "splits_final.json")
            # dataset = Flare_Dataset(self.label_data_folder, case_identifiers=None)
            # if the split file does not exist we need to create it
            # if not isfile(splits_file):
            self.print_to_log_file("Creating new 5-fold cross-validation split...")
            splits = []
            all_keys_sorted = np.sort(self.data_case1)
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append({})
                splits[-1]['train'] = list(train_keys)
                splits[-1]['val'] = list(test_keys)
                # save_json(splits, splits_file)

            # else:
            #     self.print_to_log_file("Using splits from existing split file:", splits_file)
            #     splits = load_json(splits_file)
            #     self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(self.case_identifiers1)
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        tr_keys, val_keys = self.do_split()
        label_dataset_tr1 = Flare_Dataset(self.label_data_folder1,tr_keys)
        # unlabel_dataset_tr = Flare_Dataset(self.unlabel_data_folder,self.unlabel_data_case1)
        label_dataset_tr2 = Flare_Dataset_two_folder(self.label_data_folder1,self.label_data_folder2,self.data_case1,self.data_case2,load_seg=True)
        dataset_val = Flare_Dataset(self.label_data_folder1,val_keys)

        dl_tr = Flare_semi_supervised_DataLoader(label_dataset_tr1, label_dataset_tr2,self.batch_size, self.patch_size)
        dl_val = Flare_DataLoader(dataset_val, self.batch_size, self.patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales(self.pool_op_kernel_sizes)

        rotation_for_DA, _, _, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # training pipeline
        tr_transforms = get_training_transforms(
            self.patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, self.do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1
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
        # self.set_deep_supervision_enabled(True)s

        empty_cache(self.device)

        if self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.label_data_folder1, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(self.allowed_num_processes// 2)))
            unpack_dataset(self.label_data_folder2, unpack_segmentation=True, overwrite_existing=False,
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
            output1 = [t[:self.sub_batch_size] for t in output]
            output2 = [t[self.sub_batch_size:] for t in output]

            target1 = [t[:self.sub_batch_size] for t in target]
            target2 = [t[self.sub_batch_size:] for t in target]

         

            loss1 = self.loss(output1, target1)
            loss2 = self.loss(output2, target2)
            # l = fs_loss + ss_loss
            l = self.main_loss_weight*loss1 + self.auxiliary_loss_weight * loss2
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
        
        return {'loss': l.detach().cpu().numpy()}

   

   