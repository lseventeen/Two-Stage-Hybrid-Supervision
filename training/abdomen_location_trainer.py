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
ANISO_THRESHOLD = 3


# class abdomen_location_trainer(base_trainer):
    # def __init__(self, experiment_id, fold, num_process,config):
        # super().__init__(experiment_id, fold, num_process,config)
        # self.full_abdomen_label_case = load_json(join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        #                                 'data_preprocess','analyze_result','full_abdomen_label_case.json'))       

    # def do_split(self):
    #     """
    #     The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
    #     so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
    #     Sometimes you may want to create your own split for various reasons. For this you will need to create your own
    #     splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
    #     it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
    #     and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
    #     use a random 80:20 data split.
    #     :return:
    #     """
    #     if self.fold == "all":
    #         # if fold==all then we use all images for training and validation
    #         # case_identifiers = get_case_identifiers(self.label_data_folder)

    #         tr_keys = self.full_abdomen_label_case
    #         val_keys = self.full_abdomen_label_case
            
    #     else:
    #         # splits_file = join(self.label_data_folder, "splits_final.json")
    #         # dataset = Flare_Dataset(self.label_data_folder, case_identifiers=None)
    #         # if the split file does not exist we need to create it
    #         # if not isfile(splits_file):
    #         self.print_to_log_file("Creating new 5-fold cross-validation split...")
    #         splits = []
    #         all_keys_sorted = np.sort(self.full_abdomen_label_case)
    #         kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
    #         for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
    #             train_keys = np.array(all_keys_sorted)[train_idx]
    #             test_keys = np.array(all_keys_sorted)[test_idx]
    #             splits.append({})
    #             splits[-1]['train'] = list(train_keys)
    #             splits[-1]['val'] = list(test_keys)
    #         # save_json(splits, splits_file)

    #         # else:
    #         #     self.print_to_log_file("Using splits from existing split file:", splits_file)
    #         #     splits = load_json(splits_file)
    #         #     self.print_to_log_file("The split file contains %d splits." % len(splits))

    #         self.print_to_log_file("Desired fold for training: %d" % self.fold)
    #         if self.fold < len(splits):
    #             tr_keys = splits[self.fold]['train']
    #             val_keys = splits[self.fold]['val']
    #             self.print_to_log_file("This split has %d training and %d validation cases."
    #                                    % (len(tr_keys), len(val_keys)))
    #         else:
    #             self.print_to_log_file("INFO: You requested fold %d for training but splits "
    #                                    "contain only %d folds. I am now creating a "
    #                                    "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
    #             # if we request a fold that is not in the split file, create a random 80:20 split
    #             rnd = np.random.RandomState(seed=12345 + self.fold)
    #             keys = np.sort(self.full_abdomen_label_case)
    #             idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
    #             idx_val = [i for i in range(len(keys)) if i not in idx_tr]
    #             tr_keys = [keys[i] for i in idx_tr]
    #             val_keys = [keys[i] for i in idx_val]
    #             self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
    #                                    % (len(tr_keys), len(val_keys)))
    #         if any([i in val_keys for i in tr_keys]):
    #             self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
    #                                    'splits.json or ignore if this is intentional.')
    #     return tr_keys, val_keys
   
    
    
    
    