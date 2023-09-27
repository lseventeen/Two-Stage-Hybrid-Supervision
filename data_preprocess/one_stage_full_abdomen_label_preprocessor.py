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
from preprocessor import Preprocessor
from normalization_schemes import CTNormalization
from data_preprocess.imageio.nibabel_reader_writer import NibabelIOWithReorient
from resampling import resample_data_or_seg_to_shape,compute_new_spacing
from utils import create_lists_from_splitted_dataset_folder,get_identifiers_from_splitted_dataset_folder,crop_image_according_to_mask,get_bbox_from_mask
from first_stage_full_abdomen_label_preprocessor import First_stage_full_abdomen_label_preprocessor
from postprocess.remove_small_connected_component import remove_small_cc
class one_stage_full_abdomen_label_preprocessor(First_stage_full_abdomen_label_preprocessor):
   

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict):
        
        # crop, remember to store size before cropping!
        shape = data.shape[1:]
        original_spacing = properties['spacing']
        properties['shape'] = shape
          # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        np.clip(seg,0,None,out = seg)
      
        data = self._normalize(data)
        
        target_spacing = compute_new_spacing(old_shape, original_spacing, self.new_shape)

        data = resample_data_or_seg_to_shape(data, self.new_shape, original_spacing, target_spacing,is_seg=False,order=3,order_z=0)
        seg = resample_data_or_seg_to_shape(seg, self.new_shape, original_spacing, target_spacing,is_seg=True,order=1,order_z=0)
        
        assert seg.min() >= 0, 'MIN value of seg is -1'
 
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)

       

        
        if self.verbose:
             
            
            print(f'old shape: {old_shape}, new_shape: {self.new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}')

        return data, seg







def run_preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', type=int, default=1, required=False,
                        help='[OPTIONAL] Number of processes for data argmentation. Default: 8')
 
    parser.add_argument('-config', type=str, required=False, default="/ai/code/Flare2023/config/one_stage_abdomen_config.yaml",
                        help='Config file.')
 
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        CFG = yaml.safe_load(file)

    
    # full_abdomen_label_case = load_json(join(os.path.dirname(os.path.realpath(__file__)),
    #                                     'analyze_result','full_abdomen_label_case.json')) 
    
    # pp = second_stage_full_abdomen_label_preprocessor(new_shape = CFG['Dataloader']['data_size'],
    #                              identifiers = full_abdomen_label_case,
    #                             extend_size = CFG['Dataloader']['extend_size']

    #                              )
    # pp.run(CFG['Data_path']['raw_label_image_dir'],
    #                   CFG['Data_path']['preprocess_label_data_dir'], 
    #                   CFG['Data_path']['raw_label_seg_dir'],
    #                   num_processes=args.np)
    

    full_abdomen_label_case = load_json(join(os.path.dirname(os.path.realpath(__file__)),
                                        'analyze_result','full_abdomen_label_case.json')) 
    pp = one_stage_full_abdomen_label_preprocessor(new_shape = CFG['Dataloader']['data_size'],
                                 identifiers = full_abdomen_label_case,

                                 )
    pp.run(CFG['Data_path']['raw_label_image_dir'],
                      CFG['Data_path']['preprocess_label_data_dir'], 
                      CFG['Data_path']['raw_label_seg_dir'],
                      num_processes=args.np)
    
    
    pp = one_stage_full_abdomen_label_preprocessor(new_shape = CFG['Dataloader']['data_size'])
    pp.run(CFG['Data_path']['raw_unlabel_image_dir'],
                      CFG['Data_path']['preprocess_unlabel_data_dir'], 
                      CFG['Data_path']['unlabel_predict_dir'],
                      num_processes=args.np)
  

if __name__ == '__main__':
    run_preprocess_entry()

