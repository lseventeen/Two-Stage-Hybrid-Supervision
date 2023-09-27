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
from typing import Union, Tuple, List
import yaml
import numpy as np
from acvl_utils.miscellaneous.ptqdm import ptqdm
from batchgenerators.utilities.file_and_folder_operations import *
# from cropping import crop_to_nonzero
from preprocessor import Preprocessor
from normalization_schemes import CTNormalization
from data_preprocess.imageio.nibabel_reader_writer import NibabelIOWithReorient,NibabelIO
# from data_preprocess.imageio.simpleitk_reader_writer import SimpleITKIO
from resampling import resample_data_or_seg_to_shape,compute_new_spacing
from first_stage_full_abdomen_label_preprocessor import First_stage_full_abdomen_label_preprocessor
from postprocess.remove_small_connected_component import remove_small_cc
from utils import create_lists_from_splitted_dataset_folder,get_identifiers_from_splitted_dataset_folder,crop_image_according_to_mask,get_bbox_from_mask,crop_image_according_to_bbox
class Second_stage_unlabel_abdomen_preprocessor(First_stage_full_abdomen_label_preprocessor):
    def __init__(self, new_shape=None, extend_size: list = [30,30,30], identifiers: List[str] = None, area_least: int = 100, topk: int = 30, verbose: bool = True):
        super().__init__(new_shape, identifiers, verbose)
        self.area_least = area_least
        self.topk = topk
      
        self.extend_size = extend_size


    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict):
        
        # crop, remember to store size before cropping!
        shape = data.shape[1:]
        original_spacing = properties['spacing']
        properties['shape'] = shape
          # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        
        try: 
            bbox = get_bbox_from_mask(seg[0])
            crop_shape = [i[1]-i[0] for i in bbox ]
        except:
            return None,None
        target_spacing = compute_new_spacing(crop_shape, original_spacing, self.new_shape)
        extend_size = [ int(self.extend_size[i] // target_spacing[i])  for i in range(len(target_spacing))]
        data = self._normalize(data)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        crop_image, crop_seg = crop_image_according_to_mask(data[0], seg[0], extend_size)
        # crop_image,_ = crop_image_according_to_bbox(data[0],bbox,extend_size)
        
        # crop_seg,_ = crop_image_according_to_bbox(seg[0],bbox,extend_size)
        target_spacing = compute_new_spacing(crop_image.shape, original_spacing, self.new_shape)
        area_least = self.area_least / target_spacing[0] / target_spacing[1] / target_spacing[2]
        crop_seg = remove_small_cc(crop_seg,area_least, self.topk)
        seg = resample_data_or_seg_to_shape(crop_seg[None,...], self.new_shape, original_spacing, target_spacing,is_seg=True,order=1,order_z=0)
        
        assert seg.min() >= 0, 'MIN value of seg is -1'
 
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)

        data = resample_data_or_seg_to_shape(crop_image[None,...], self.new_shape, original_spacing, target_spacing,is_seg=False,order=3,order_z=0)

        
       
        
        if self.verbose:
            crop_shape = crop_image.shape
            unique = np.unique(seg)
            print(f'unique: {unique}, old shape: {old_shape}, crop shape: {crop_shape} new_shape: {self.new_shape},  crop_shape: {crop_shape}, '
                  f'old_spacing: {original_spacing}, new_spacing: {target_spacing}, extend_size: {extend_size}')

        return data, seg


    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str):
        data, seg, properties = self.run_case(image_files, seg_file)
        # print('dtypes', data.dtype, seg.dtype)
        if data is not None:
            np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
            write_pickle(properties, output_filename_truncated + '.pkl')
        else: 
            print(output_filename_truncated.split('/')[-1])

def run_preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', type=int, default=1, required=False,
                        help='[OPTIONAL] Number of processes for data argmentation. Default: 8')
 
    parser.add_argument('-config', type=str, required=False, default="/ai/code/Flare2023/config/second_stage_abdomen_config.yaml",
                        help='Config file.')
 
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        CFG = yaml.safe_load(file)

    
    
    part_label_data_pp = Second_stage_unlabel_abdomen_preprocessor(new_shape = CFG['Dataloader']['data_size'],
                                extend_size = CFG['Dataloader']['extend_size'],
                                topk=CFG['Postprocess']['topk'],
                                area_least=CFG['Postprocess']['area_least']

                                 )
    part_label_data_pp.run(CFG['Data_path']['raw_unlabel_image_dir'],
                      CFG['Data_path']['preprocess_unlabel_data_dir'], 
                      CFG['Data_path']['unlabel_predict_dir'],
                      num_processes=args.np)
    
 

  

if __name__ == '__main__':
    run_preprocess_entry()

