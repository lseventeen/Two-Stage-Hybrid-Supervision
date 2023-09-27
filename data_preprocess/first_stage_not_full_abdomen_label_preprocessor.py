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
from postprocess.remove_small_connected_component import remove_small_cc
class First_stage_not_full_abdomen_label_preprocessor(Preprocessor):
    def __init__(self, new_shape=None, area_least: int = 100, topk: int = 30, verbose: bool = True):
        super().__init__(new_shape, verbose)
        self.area_least = area_least
        self.topk = topk
        


    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None],pre: np.ndarray, properties: dict):
        
        # crop, remember to store size before cropping!
        shape = data.shape[1:]
        original_spacing = properties['spacing']
        properties['shape'] = shape
        data = self._normalize(data)

        old_shape = data.shape[1:]
        
        target_spacing = compute_new_spacing(data.shape[1:], original_spacing, self.new_shape)
        data = resample_data_or_seg_to_shape(data, self.new_shape, original_spacing, target_spacing,is_seg=False,order=3,order_z=0)
        
      
        label_unique = np.unique(seg)
        for i in label_unique:
            pre[pre == i] = 0

        pre[seg > 0] = seg[seg > 0]

        # pre[pre == 14]==0
        seg = pre
        
        area_least = self.area_least / target_spacing[0] / target_spacing[1] / target_spacing[2]
        seg = remove_small_cc(seg[0],area_least, self.topk)
        
        seg = resample_data_or_seg_to_shape(seg[None,...], self.new_shape, original_spacing, target_spacing,is_seg=True,order=1,order_z=0)
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
        
        data, seg = self.run_case_npy(data, seg, pre, data_properites)
        return data, seg, data_properites
    
    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str, predict_file: str):
        data, seg, properties = self.run_case(image_files, seg_file, predict_file)
        # print('dtypes', data.dtype, seg.dtype)
        # np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        # write_pickle(properties, output_filename_truncated + '.pkl')

        if data is not None:
            np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
            write_pickle(properties, output_filename_truncated + '.pkl')
        else: 
            print(output_filename_truncated.split('/')[-1])

    def run(self, data_path, output_path, seg_path,predict_path,
            num_processes: int = 4,file_ending = ".nii.gz"):
        
             
        
        if isdir(output_path):
            shutil.rmtree(output_path)
        maybe_mkdir_p(output_path)
        
        not_full_abdomen_label_case = load_json(join(os.path.dirname(os.path.realpath(__file__)),
                                        'analyze_result','not_full_abdomen_label_case.json')) 
        
        output_filenames_truncated = [join(output_path, i) for i in not_full_abdomen_label_case]

       
        # list of lists with image filenames
        image_fnames = create_lists_from_splitted_dataset_folder(data_path, file_ending,not_full_abdomen_label_case)
        # list of segmentation filenames
       
        predict_fnames = [join(predict_path, i + file_ending) for i in not_full_abdomen_label_case]
        seg_fnames = [join(seg_path, i + file_ending) for i in not_full_abdomen_label_case]
        
        

        _ = ptqdm(self.run_case_save, (output_filenames_truncated, image_fnames, seg_fnames,predict_fnames),
                  processes=num_processes, zipped=True, disable=self.verbose)





def run_preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', type=int, default=1, required=False,
                        help='[OPTIONAL] Number of processes for data argmentation. Default: 8')
 
    parser.add_argument('-config', type=str, required=False, default="/ai/code/Flare2023/config/first_stage_config.yaml",
                        help='Config file.')
 
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        CFG = yaml.safe_load(file)
    label_data_pp = First_stage_not_full_abdomen_label_preprocessor(CFG['Dataloader']['data_size'],
                                                                    
                                topk=CFG['Postprocess']['topk'],
                                area_least=CFG['Postprocess']['area_least'])
    label_data_pp.run(CFG['Data_path']['raw_label_image_dir'],
                      CFG['Data_path']['preprocess_label_data_dir'], 
                      CFG['Data_path']['raw_label_seg_dir'],
                      CFG['Data_path']['predict_data_dir'], 

                      num_processes=args.np)

if __name__ == '__main__':
    run_preprocess_entry()

