from typing import Union, Tuple

from batchgenerators.dataloading.data_loader import DataLoader,SlimDataLoaderBase
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


class Flare_DataLoader(DataLoader):
    def __init__(self,
                 data: object,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 ):
        super().__init__(data, batch_size, 1, None, True, False, True, sampling_probabilities)
        # assert isinstance(data, nnUNetDataset), 'nnUNetDataLoaderBase only supports dictionaries as data'
        self.indices = list(data.keys())
        self.patch_size = patch_size
        self.data_shape, self.seg_shape = self.determine_shapes()
    def determine_shapes(self):
        # load one case
        data, seg, properties = self._data.load_case(self.indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape
    
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            data, seg, properties = self._data.load_case(i)
            data_all[j] = data
            seg_all[j] = seg

 
        return {'data': data_all, 'seg': seg_all, 'keys': selected_keys}


class Flare_hybrid_supervised_DataLoader(SlimDataLoaderBase):
    def __init__(self,
                 label_data: object,
                 part_label_data: object,
                 unlabel_data: object,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 num_threads_in_multithreaded=1,
                 ):
        super().__init__(label_data, batch_size, num_threads_in_multithreaded)
        # assert isinstance(data, nnUNetDataset), 'nnUNetDataLoaderBase only supports dictionaries as data'
        assert batch_size % 3 == 0 
        self.label_data = label_data
        self.part_label_data = part_label_data
        self.unlabel_data = unlabel_data
        self.label_data_indices = list(label_data.keys())
        self.part_label_data_indices = list(part_label_data.keys())
        self.unlabel_data_indices = list(unlabel_data.keys())
        self.patch_size = patch_size
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sub_batch_size = batch_size // 3
        self.sampling_probabilities = sampling_probabilities


        

    def determine_shapes(self):
        # load one case
        data, seg, properties = self.label_data.load_case(self.label_data_indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape
    
    def get_indices(self,indices,batch_size):
        # if self.infinite, this is easy
        
        return np.random.choice(indices, batch_size, replace=False, p=self.sampling_probabilities)

    def generate_train_batch(self):
        label_data_selected_keys = self.get_indices(self.label_data_indices,
                                         self.sub_batch_size)
        part_label_data_selected_keys = self.get_indices(self.part_label_data_indices,
                                         self.sub_batch_size)
        unlabel_data_selected_keys = self.get_indices(self.unlabel_data_indices,
                                         self.sub_batch_size)
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(label_data_selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            data, seg, properties = self.label_data.load_case(i)
            # np.clip(seg,0,13,out = seg)
            data_all[j] = data
            seg_all[j] = seg

        for j, i in enumerate(part_label_data_selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            data, seg, properties = self.part_label_data.load_case(i)
            # np.clip(seg,0,13,out = seg)
            data_all[j+self.sub_batch_size] = data
            seg_all[j+self.sub_batch_size] = seg

        for j, i in enumerate(unlabel_data_selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            data, seg, properties = self.unlabel_data.load_case(i)
            data_all[j+self.sub_batch_size*2] = data
            seg_all[j+self.sub_batch_size*2] = seg
         

 
        return {'data': data_all, 'seg': seg_all, 'keys': [label_data_selected_keys,part_label_data_selected_keys,unlabel_data_selected_keys]}


class Flare_semi_supervised_DataLoader(SlimDataLoaderBase):
    def __init__(self,
                 label_data: object,
                 unlabel_data: object,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 num_threads_in_multithreaded=1,
                 ):
        super().__init__(label_data, batch_size, num_threads_in_multithreaded)
        # assert isinstance(data, nnUNetDataset), 'nnUNetDataLoaderBase only supports dictionaries as data'
        assert batch_size % 2 == 0 
        self.label_data = label_data
        self.unlabel_data = unlabel_data
        self.label_data_indices = list(label_data.keys())
        self.unlabel_data_indices = list(unlabel_data.keys())
        self.patch_size = patch_size
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sub_batch_size = batch_size // 2
        self.sampling_probabilities = sampling_probabilities


        

    def determine_shapes(self):
        # load one case
        data, seg, properties = self.label_data.load_case(self.label_data_indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape
    
    def get_indices(self,indices,batch_size):
        # if self.infinite, this is easy
        
        return np.random.choice(indices, batch_size, replace=False, p=self.sampling_probabilities)

    def generate_train_batch(self):
        label_data_selected_keys = self.get_indices(self.label_data_indices,
                                         self.sub_batch_size)
       
        unlabel_data_selected_keys = self.get_indices(self.unlabel_data_indices,
                                         self.sub_batch_size)
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(label_data_selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            data, seg, properties = self.label_data.load_case(i)
            data_all[j] = data
            seg_all[j] = seg



        for j, i in enumerate(unlabel_data_selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            data, seg, properties = self.unlabel_data.load_case(i)
            data_all[j+self.sub_batch_size] = data
         

 
        return {'data': data_all, 'seg': seg_all, 'keys': [label_data_selected_keys,unlabel_data_selected_keys]}