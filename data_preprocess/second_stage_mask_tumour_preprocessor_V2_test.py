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
from resampling import resample_data_or_seg_to_shape,compute_new_spacing
from first_stage_not_full_abdomen_label_preprocessor import First_stage_not_full_abdomen_label_preprocessor
from second_stage_no_full_abdomen_label_preprocessor import Second_stage_no_full_abdomen_label_preprocessor
from postprocess.remove_small_connected_component import remove_small_cc
from utils import create_lists_from_splitted_dataset_folder,get_identifiers_from_splitted_dataset_folder,crop_image_according_to_mask,get_bbox_from_mask,crop_image_according_to_bbox,extract_label_mask
import cc3d
import fastremap
from data_preprocess.imageio.nibabel_reader_writer import NibabelIOWithReorient,NibabelIO
class Second_stage_tumour_label_preprocessor(Second_stage_no_full_abdomen_label_preprocessor):
    def __init__(self, new_shape=None, extend_size: list = [30,30,30], area_least: int = 100, topk: int = 30, mask_extend: list = [10,10,10], verbose: bool = True):
        super().__init__(new_shape, extend_size, area_least, topk, verbose)
      
        self.mask_extend = mask_extend
    def run_case(self, image_files: List[str], seg_file: Union[str, None], predict_file):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
    
        rw = NibabelIO()
        # name = image_files[0].split('/')[-1]
        # print(name)
        # load image(s)
        data, data_properites = rw.read_images(image_files)

        # if possible, load seg
        if predict_file is not None:
            pre, _ = rw.read_seg(predict_file)

        seg, _ = rw.read_seg(seg_file)
        
        seg,_ = self.run_case_npy(data, seg, pre, data_properites)
        return seg, data_properites
    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], pre, properties: dict):
        
        # crop, remember to store size before cropping!
        shape = data.shape[1:]
        original_spacing = properties['spacing']
        properties['shape'] = shape
          # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        mask_extend = [ int(self.mask_extend[i] // original_spacing[i])  for i in range(len(original_spacing))]
        np.clip(seg,0,None,out = seg)
        seg[seg ==0] = 0
        seg [(seg > 0) & (seg < 14)] = 2
        seg[seg==14] = 1
        # seg = extract_label_mask(seg,mask_extend)
        seg = extract_label_mask(seg,mask_extend)
        seg[seg == 2] = 0 
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return seg, properties



    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str, predict_file: str):
        seg, properties = self.run_case(image_files, seg_file, predict_file)
        # print('dtypes', data.dtype, seg.dtype)
        rw = NibabelIO()
        # rw = NibabelIO()
        rw.write_seg(seg[0], output_filename_truncated +  ".nii.gz",
                    properties)
        
    def run(self, data_path, output_path, seg_path,predict_path,
                num_processes: int = 4,file_ending = ".nii.gz"):
            
        if isdir(output_path):
            shutil.rmtree(output_path)
        maybe_mkdir_p(output_path)
        
        identifiers = get_identifiers_from_splitted_dataset_folder(data_path,file_ending)
        output_filenames_truncated = [join(output_path, i) for i in identifiers]


        # list of lists with image filenames
        image_fnames = create_lists_from_splitted_dataset_folder(data_path, file_ending,identifiers)
        predict_fnames = [join(predict_path, i + file_ending) for i in identifiers]
        # list of segmentation filenames
        seg_fnames = [join(seg_path, i + file_ending) for i in identifiers]
        
        

        _ = ptqdm(self.run_case_save, (output_filenames_truncated, image_fnames, seg_fnames,predict_fnames),
            processes=num_processes, zipped=True, disable=self.verbose)
    # def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str):
    #     data, seg, properties = self.run_case(image_files, seg_file)
    #     # print('dtypes', data.dtype, seg.dtype)
    #     if data is not None:
    #         np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
    #         write_pickle(properties, output_filename_truncated + '.pkl')
    #     else: 
    #         print(output_filename_truncated.split('/')[-1])
        
def run_preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', type=int, default=1, required=False,
                        help='[OPTIONAL] Number of processes for data argmentation. Default: 8')
 
    parser.add_argument('-config', type=str, required=False, default="/ai/code/Flare2023/config/second_stage_tumour_config.yaml",
                        help='Config file.')
 
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        CFG = yaml.safe_load(file)

    
    
    pp = Second_stage_tumour_label_preprocessor(new_shape = CFG['Dataloader']['data_size'],
                                extend_size = CFG['Preprocess']['extend_size'],
                                topk=CFG['Postprocess']['topk'],
                                area_least=CFG['Postprocess']['area_least'],
                                mask_extend = CFG['Preprocess']['mask_extend']
    )
    # case = 'FLARE23_0095'
    # out = join(CFG['Data_path']['preprocess_part_label_data_dir'],case+'.nii.gz')
    # img = join(CFG['Data_path']['raw_label_image_dir'],case+'_0000.nii.gz')
    # seg = join(CFG['Data_path']['raw_label_seg_dir'],case+'.nii.gz')
    # pre = join(CFG['Data_path']['part_label_predict_dir'],case+'.nii.gz')
    
    # pp.run_case_save(out, [img], seg,pre)
                                 
    pp.run(CFG['Data_path']['raw_label_image_dir'],
                      CFG['Data_path']['preprocess_label_data_dir_test'], 
                      CFG['Data_path']['raw_label_seg_dir'],
                      CFG['Data_path']['label_abdomen_predict_dir'],
                      num_processes=args.np)
    
 

  

if __name__ == '__main__':
    run_preprocess_entry()

