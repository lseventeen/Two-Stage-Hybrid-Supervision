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
import shutil
from typing import Union, Tuple
import yaml
import numpy as np
from acvl_utils.miscellaneous.ptqdm import ptqdm
from batchgenerators.utilities.file_and_folder_operations import *
# from cropping import crop_to_nonzero
from normalization_schemes import CTNormalization
from resampling import resample_data_or_seg_to_shape,compute_new_spacing
from utils import create_lists_from_splitted_dataset_folder,get_identifiers_from_splitted_dataset_folder,create_two_class_mask
from preprocessor import Preprocessor
from data_preprocess.imageio.nibabel_reader_writer import NibabelIOWithReorient,NibabelIO
from data_preprocess.imageio.simpleitk_reader_writer import SimpleITKIO
class First_stage_full_abdomen_label_preprocessor(Preprocessor):
    def __init__(self, new_shape=None, identifiers: list = None,verbose: bool = True):
        super().__init__(new_shape, verbose)
        self.identifiers = identifiers

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict):
        
        
        # crop, remember to store size before cropping!
        shape = data.shape[1:]
        original_spacing = properties['spacing']
        properties['shape'] = shape
        data = self._normalize(data)
        old_shape = data.shape[1:]
        target_spacing = None
        if self.new_shape != None:
            target_spacing = compute_new_spacing(data.shape[1:], original_spacing, self.new_shape)
            data = resample_data_or_seg_to_shape(data, self.new_shape, original_spacing, target_spacing,is_seg=False,order=3,order_z=0)
        
        if self.new_shape != None:
            seg = resample_data_or_seg_to_shape(seg, self.new_shape, original_spacing, target_spacing,is_seg=True,order=1,order_z=0)
        seg = create_two_class_mask(seg)
        assert seg.min() >= 0, 'MIN value of seg is -1'
 
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {self.new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {resample_data_or_seg_to_shape}')

        return data, seg

    def run(self, data_path, output_path, seg_path,
            num_processes: int = 4,file_ending = ".nii.gz"):
        
             
        
        if isdir(output_path):
            shutil.rmtree(output_path)
        maybe_mkdir_p(output_path)
        
        identifiers = get_identifiers_from_splitted_dataset_folder(data_path,file_ending) if self.identifiers == None else self.identifiers
        output_filenames_truncated = [join(output_path, i) for i in identifiers]

       
        # list of lists with image filenames
        image_fnames = create_lists_from_splitted_dataset_folder(data_path, file_ending,identifiers)
        # list of segmentation filenames
        seg_fnames = [join(seg_path, i + file_ending) for i in identifiers]
        
        

        _ = ptqdm(self.run_case_save, (output_filenames_truncated, image_fnames, seg_fnames),
                  processes=num_processes, zipped=True, disable=self.verbose)





def run_preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', type=int, default=8, required=False,
                        help='[OPTIONAL] Number of processes for data argmentation. Default: 8')
 
    parser.add_argument('-config', type=str, required=False, default="/ai/code/Flare2023/config/first_stage_config.yaml",
                        help='Config file.')
 
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        CFG = yaml.safe_load(file)

    # full_abdomen_label_case = load_json(join(os.path.dirname(os.path.realpath(__file__)),
    #                                     'analyze_result','full_abdomen_label_case.json')) 
    # pp = First_stage_full_abdomen_label_preprocessor(new_shape=CFG['Dataloader']['data_size'],identifiers=full_abdomen_label_case,verbose=True)
    # pp.run(CFG['Data_path']['raw_label_image_dir'],
    #                   CFG['Data_path']['preprocess_label_data_dir'], 
    #                   CFG['Data_path']['raw_label_seg_dir'],
    #                   num_processes=args.np)

    
    pp = First_stage_full_abdomen_label_preprocessor(new_shape=CFG['Dataloader']['data_size'],verbose=True)
    pp.run(CFG['Data_path']['val_image_dir'],
                      CFG['Data_path']['preprocess_val_data_dir'], 
                      CFG['Data_path']['val_seg_dir'],
                      num_processes=args.np)

if __name__ == '__main__':
    run_preprocess_entry()

