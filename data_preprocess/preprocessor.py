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
from data_preprocess.utils import create_lists_from_splitted_dataset_folder,get_identifiers_from_splitted_dataset_folder
class Preprocessor(object):
    def __init__(self, new_shape = None, verbose: bool = True):
        self.new_shape = new_shape
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict):
        
        # crop, remember to store size before cropping!
        shape = data.shape[1:]
        original_spacing = properties['spacing']
        properties['shape'] = shape
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        # data, seg, bbox = crop_to_nonzero(data, seg)
        # properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        # properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]
    
        
        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data)
        

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        target_spacing = None
        if self.new_shape != None:
            target_spacing = compute_new_spacing(data.shape[1:], original_spacing, self.new_shape)
            data = resample_data_or_seg_to_shape(data, self.new_shape, original_spacing, target_spacing,is_seg=False,order=3,order_z=0)
        if seg is not None:
            if self.new_shape != None:
                seg = resample_data_or_seg_to_shape(seg, self.new_shape, original_spacing, target_spacing,is_seg=True,order=1,order_z=0)
            # np.clip(seg,0,None,out = seg)
            # assert seg.min() >= 0, 'MIN value of seg is -1'
 
            if np.max(seg) > 127:
                seg = seg.astype(np.int16)
            else:
                seg = seg.astype(np.int8)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {self.new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {resample_data_or_seg_to_shape}')

        return data, seg

    def run_case(self, image_files: List[str], seg_file: Union[str, None]):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
    
        rw = NibabelIO()

        # load image(s)
        data, data_properites = rw.read_images(image_files)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None
       
        data, seg = self.run_case_npy(data, seg, data_properites)
        return data, seg, data_properites

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str):
        data, seg, properties = self.run_case(image_files, seg_file)
        # print('dtypes', data.dtype, seg.dtype)
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')


    def _normalize(self, data: np.ndarray) -> np.ndarray:
        dataset_fingerprint_file = join((os.path.dirname(os.path.realpath(__file__))),'analyze_result','dataset_fingerprint.json')
        dataset_fingerprint = load_json(dataset_fingerprint_file)
        normalizer = CTNormalization(use_mask_for_norm=False,
                                          intensityproperties=dataset_fingerprint['foreground_intensity_properties_per_channel'][str(0)])
        data = normalizer.run(data)
        return data

    def run(self, data_path, output_path, seg_path=None,
            num_processes: int = 4,file_ending = ".nii.gz"):
      
        
        if isdir(output_path):
            shutil.rmtree(output_path)
        maybe_mkdir_p(output_path)
       
        identifiers = get_identifiers_from_splitted_dataset_folder(data_path,file_ending)
        output_filenames_truncated = [join(output_path, i) for i in identifiers]

       
        # list of lists with image filenames
        image_fnames = create_lists_from_splitted_dataset_folder(data_path, file_ending,identifiers)
        # list of segmentation filenames
       
        if seg_path is not None:
            seg_fnames = [join(seg_path, i + file_ending) for i in identifiers]
        else: 
            seg_fnames = [None for _ in identifiers]

        _ = ptqdm(self.run_case_save, (output_filenames_truncated, image_fnames, seg_fnames),
                  processes=num_processes, zipped=True, disable=self.verbose)





def run_preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', type=int, default=8, required=False,
                        help='[OPTIONAL] Number of processes for data argmentation. Default: 8')
 
    parser.add_argument('-config', type=str, required=False, default="/ai/code/Flare2023/config/one_stage_config.yaml",
                        help='Config file.')
 
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        CFG = yaml.safe_load(file)

    
    label_data_pp = Preprocessor(CFG['Dataloader']['data_size'])
    label_data_pp.run(CFG['Data_path']['raw_label_image_dir'],
                      CFG['Data_path']['preprocess_label_data_dir'], 
                      CFG['Data_path']['raw_label_seg_dir'],
                      num_processes=args.np)

    unlabel_data_pp = Preprocessor(CFG['Dataloader']['data_size'])
    unlabel_data_pp.run(CFG['Data_path']['raw_unlabel_image_dir'], 
                        CFG['Data_path']['preprocess_unlabel_data_dir'], 
                        num_processes=args.np)

  

if __name__ == '__main__':
    run_preprocess_entry()

