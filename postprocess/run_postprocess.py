#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
from typing import Union, Tuple
import yaml
import numpy as np
from acvl_utils.miscellaneous.ptqdm import ptqdm
from batchgenerators.utilities.file_and_folder_operations import *
# from cropping import crop_to_nonzero
from data_preprocess.normalization_schemes import CTNormalization
from data_preprocess.imageio.nibabel_reader_writer import NibabelIOWithReorient,NibabelIO
from data_preprocess.imageio.simpleitk_reader_writer import SimpleITKIO
from data_preprocess.resampling import resample_data_or_seg_to_shape,compute_new_spacing
from data_preprocess.utils import create_lists_from_splitted_dataset_folder,get_identifiers_from_splitted_dataset_folder,get_identifiers_from_splitted_seg_folder
from postprocess.remove_small_connected_component import remove_small_cc
class Postprocessor(object):
    def __init__(self, verbose: bool = False, rw =NibabelIOWithReorient(), area_least = 1000, topk = 30,file_ending: str = ".nii.gz"):
        self.verbose = verbose
        self.rw =  rw
        self.area_least = area_least
        self.topk = topk
        self.file_ending = file_ending

 
    def run_case_save(self, output_filename_truncated: str, seg_file: str):
        seg, data_properites = self.rw.read_seg(seg_file)
        # print('dtypes', data.dtype, seg.dtype)
        # assert 14 in np.unique(seg)
        # seg = np.where(seg == 14, 1, 0) 
        spacing = data_properites['spacing']
        area_least = self.area_least / spacing[0] / spacing[1] / spacing[2]
        seg_pp = remove_small_cc(seg[0],area_least, self.topk)
        # seg = np.where(seg == 1, 14, 0) 
        self.rw.write_seg(seg_pp, output_filename_truncated + self.file_ending,
                        data_properites)


    def run(self, seg_path, output_path, 
            num_processes: int = 8,):
        identifiers = get_identifiers_from_splitted_seg_folder(seg_path,self.file_ending)
        if isdir(output_path):
            shutil.rmtree(output_path)
        maybe_mkdir_p(output_path)
 
        output_filenames_truncated = [join(output_path, i) for i in identifiers]
        seg_fnames = [join(seg_path, i + self.file_ending ) for i in identifiers]
     

        _ = ptqdm(self.run_case_save, (output_filenames_truncated, seg_fnames),
                  processes=num_processes, zipped=True, disable=self.verbose)





def run_preprocess_entry():
    area_least = 100
    topk = 30
    # seg_path = '/ai/code/nnUNet_results/Dataset165_FLARE2023_label_tumour/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/label_tumour_e1000fall_230707_103116/checkpoint_final_14'
    # output_path = f'/ai/code/nnUNet_results/Dataset165_FLARE2023_label_tumour/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/label_tumour_e1000fall_230707_103116/checkpoint_final_14_{area_least}_{topk}'
    seg_path = '/ai/code/nnUNet_results/Dataset164_FLARE2023_abdomen/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/e1000_fall_230705_234121/checkpoint_final'
    output_path  = f'/ai/code/nnUNet_results/Dataset164_FLARE2023_abdomen/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/e1000_fall_230705_234121/checkpoint_final_{area_least}_{topk}'
    seg_pp = Postprocessor(area_least = area_least, topk = topk)
    seg_pp.run(seg_path,output_path,num_processes=8)

  

  

if __name__ == '__main__':
    run_preprocess_entry()

