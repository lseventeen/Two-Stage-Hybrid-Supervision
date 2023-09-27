from dataloading.data_loader import Flare_DataLoader,Flare_hybrid_supervised_DataLoader
from dataloading.dataset import Flare_Dataset,Flare_Dataset_two_folder
from base_trainer import base_trainer
import os
import numpy as np
import torch
from loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from dataloading.utils import unpack_dataset, get_case_identifiers
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from data_augmentation.data_augmentation import get_training_transforms,get_validation_transforms
from utilities.helpers import empty_cache, dummy_context
from loss.compound_losses import DC_and_CE_loss,part_DC_and_CE_loss
from loss.deep_supervision import DeepSupervisionWrapper
from data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from data_augmentation.compute_initial_patch_size import get_patch_size
from create_network import create_network
from sklearn.model_selection import KFold
from torch import autocast, nn
from load_pretrained_weights import load_pretrained_weights
ANISO_THRESHOLD = 3


class ema_hybrid_supervised_abdomen_trainer(base_trainer):
    def __init__(self, experiment_id, fold, num_process,config):
        super().__init__(experiment_id, fold, num_process,config)
        self.full_abdomen_label_case = load_json(join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                        'data_preprocess','analyze_result','full_abdomen_label_case.json'))
        # self.part_abdomen_label_case = load_json(join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        #                                   'data_preprocess','analyze_result','part_abdomen_label_case.json'))
        self.unlabel_data_case1 = load_json(join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                          'data_preprocess','analyze_result','not_abdomen_label_case.json'))
       
        self.part_label_data_folder = config['Data_path']['preprocess_part_label_data_with_seg_dir']
        self.not_full_abdomen_data_case = get_case_identifiers(self.part_label_data_folder)
        self.part_abdomen_label_case = [i for i in self.not_full_abdomen_data_case if (i not in self.unlabel_data_case1)]
        self.unlabel_data_folder = config['Data_path']['preprocess_unlabel_data_dir']
        self.unlabel_data_case2 = get_case_identifiers(self.unlabel_data_folder)
        assert self.batch_size%3 == 0
        self.sub_batch_size = self.batch_size//3

        self.label_loss_weight = config['Loss']['label_loss_weight']
        self.part_label_loss_weight = config['Loss']['part_label_loss_weight']
        self.unlabel_loss_weight = config['Loss']['unlabel_loss_weight']
        self.pretrained_weights_file = config['Trainer']['pretrained_weights_file'] 
        
    # def initialize(self):
    #     if not self.was_initialized:

    #         self.network = create_network(
    #             self.conv_kernel_sizes,
    #             self.UNet_class_name,
    #             self.num_classes,
    #             self.n_conv_per_stage_decoder,
    #             self.n_conv_per_stage_encoder,
    #             self.UNet_base_num_features,
    #             self.unet_max_num_features,
    #             self.pool_op_kernel_sizes,
    #             self.num_input_channels,
    #             self.deep_supervision
    #         ).to(self.device)
    #         # compile network for free speedup
    #         if self.compile:
    #             self.network = torch.compile(self.network)

    #         self.optimizer, self.lr_scheduler = self.configure_optimizers()
    #         # if ddp, wrap in DDP wrapper
    #         if self.is_ddp:
    #             self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
    #                 self.network)
    #             self.network = DDP(self.network, device_ids=[self.local_rank])

    #         self.loss, self.part_loss = self._build_loss()
    #         self.was_initialized = True
    #     else:
    #         raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
    #                            "That should not happen.")
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

            tr_keys = self.full_abdomen_label_case
            val_keys = self.full_abdomen_label_case
            
        else:
            # splits_file = join(self.label_data_folder, "splits_final.json")
            # dataset = Flare_Dataset(self.label_data_folder, case_identifiers=None)
            # if the split file does not exist we need to create it
            # if not isfile(splits_file):
            self.print_to_log_file("Creating new 5-fold cross-validation split...")
            splits = []
            all_keys_sorted = np.sort(self.full_abdomen_label_case)
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
                keys = np.sort(self.full_abdomen_label_case)
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
                self.is_BN
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
                self.is_BN
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
            if self.pretrained_weights_file is not None:
                load_pretrained_weights(self.network, self.pretrained_weights_file, verbose=True)
                load_pretrained_weights(self.teacher_network, self.pretrained_weights_file, verbose=True)
            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
   
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
            unpack_dataset(self.part_label_data_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(self.allowed_num_processes// 2)))
            unpack_dataset(self.unlabel_data_folder, unpack_segmentation=True, overwrite_existing=False,
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
    
    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        tr_keys, val_keys = self.do_split()
        label_dataset_tr = Flare_Dataset(self.label_data_folder,tr_keys)
        part_label_dataset_tr = Flare_Dataset(self.part_label_data_folder,self.part_abdomen_label_case)
        unlabel_dataset_tr = Flare_Dataset_two_folder(self.part_label_data_folder,self.unlabel_data_folder,self.unlabel_data_case1,self.unlabel_data_case2,load_seg=True)
        dataset_val = Flare_Dataset(self.label_data_folder,val_keys)

        dl_tr = Flare_hybrid_supervised_DataLoader(label_dataset_tr,part_label_dataset_tr, unlabel_dataset_tr,self.batch_size, self.patch_size)
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
    # def _build_loss(self):

    #     loss = DC_and_CE_loss({'batch_dice': self.batch_dice,
    #                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
    #                           ignore_label=14, dice_class=MemoryEfficientSoftDiceLoss)
        
    #     part_loss = part_DC_and_CE_loss({'batch_dice': self.batch_dice,
    #                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
    #                           ignore_label=0, dice_class=MemoryEfficientSoftDiceLoss)
       

    #     deep_supervision_scales = self._get_deep_supervision_scales(self.pool_op_kernel_sizes)

    #     # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    #     # this gives higher resolution outputs more weight in the loss
    #     weights = np.array([1 / (2 ** i)
    #                        for i in range(len(deep_supervision_scales))])
    #     weights[-1] = 0

    #     # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    #     weights = weights / weights.sum()
    #     # now wrap the loss
    #     loss = DeepSupervisionWrapper(loss, weights)
    #     part_loss = DeepSupervisionWrapper(part_loss, weights)
    #     return loss, part_loss
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

            label_output = [i[:self.sub_batch_size] for i in output]
            part_label_output = [i[self.sub_batch_size:self.sub_batch_size*2] for i in output]
            unlabel_output = [i[self.sub_batch_size*2:] for i in output]

            label_target = [i[:self.sub_batch_size] for i in target]
            part_label_target = [i[self.sub_batch_size:self.sub_batch_size*2] for i in target]

            
            part_label_predict = [i[self.sub_batch_size:self.sub_batch_size*2] for i in predict]
            unlabel_predict = [i[self.sub_batch_size*2:] for i in predict]

            part_label_pseudo_gt = [torch.argmax(torch.softmax(i, dim=1),dim=1, keepdim=True).float() for i in part_label_predict]
            unlabel_pseudo_gt = [torch.argmax(torch.softmax(i, dim=1),dim=1, keepdim=True).float() for i in unlabel_predict]
            for i in range(len(part_label_pseudo_gt)):
                part_label_pseudo_gt[i][part_label_target[i]>0] = part_label_target[i][part_label_target[i]>0]
            # unlabel_target = [t[self.sub_batch_size*2:] for t in target]
  

            # part_label_predict = [torch.argmax(torch.softmax(o[self.sub_batch_size:self.sub_batch_size*2], dim=1),dim=1, keepdim=True).float() for o in output]
            # unlabel_predict = [torch.argmax(torch.softmax(o[self.sub_batch_size*2:], dim=1),dim=1, keepdim=True).float() for o in output]
            # predict = torch.clone(predict_tmp)
            # part_label_predict = part_label_and_unlabel_predict[self.sub_batch_size:self.sub_batch_size*2].long()
            # ss_target = target[0][self.sub_batch_size:self.sub_batch_size*2].long()

            
            # for i in range(len(part_label_predict)):
            #     part_label_predict[i][part_label_target[i]>0] = part_label_target[i][part_label_target[i]>0] 
            
            
            # # part_label_loss = self.part_loss(part_label_output, part_label_target)
            label_loss = self.loss(label_output, label_target)
            part_label_loss = self.loss(part_label_output, part_label_pseudo_gt)
            unlabel_loss = self.loss(unlabel_output, unlabel_pseudo_gt)
            l = self.label_loss_weight*label_loss + \
                self.part_label_loss_weight * part_label_loss + self.unlabel_loss_weight * unlabel_loss
           
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