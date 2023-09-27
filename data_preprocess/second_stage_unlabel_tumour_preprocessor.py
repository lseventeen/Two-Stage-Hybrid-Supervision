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
from postprocess.remove_small_connected_component import remove_small_cc
from data_preprocess.imageio.nibabel_reader_writer import NibabelIOWithReorient,NibabelIO
from utils import create_lists_from_splitted_dataset_folder,get_identifiers_from_splitted_seg_folder,crop_image_according_to_mask,get_bbox_from_mask,crop_image_according_to_bbox
class Second_stage_unlabel_tumour_preprocessor(First_stage_not_full_abdomen_label_preprocessor):
    def __init__(self, new_shape=None, identifiers=None, extend_size: list = [30,30,30], area_least: int = 100, topk: int = 30, verbose: bool = True):
        super().__init__(new_shape, area_least, topk, verbose)
        self.identifiers = identifiers
        self.extend_size = extend_size


    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], pre, properties: dict):
        
        # crop, remember to store size before cropping!
        shape = data.shape[1:]
        original_spacing = properties['spacing']
        properties['shape'] = shape
          # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        
        data = self._normalize(data)

        pre= np.where(pre ==1,14,0)

        # label_unique = np.unique(seg)
        # for i in label_unique:
        #     pre[pre == i] = 0
        seg[pre > 0] = pre[pre > 0]

        try: 
            bbox = get_bbox_from_mask(seg[0])
            crop_shape = [i[1]-i[0] for i in bbox ]
        except:
            return None
        crop_shape = [i[1]-i[0] for i in bbox ]

        target_spacing = compute_new_spacing(crop_shape, original_spacing, self.new_shape)
        
        extend_size = [ int(self.extend_size[i] // target_spacing[i])  for i in range(len(target_spacing))]
        crop_image, _= crop_image_according_to_bbox(data[0], bbox, extend_size)
        # crop_image, crop_seg = crop_image_according_to_mask(data[0], seg[0], extend_size)
        target_spacing = compute_new_spacing(crop_image.shape, original_spacing, self.new_shape)
        # area_least = self.area_least / target_spacing[0] / target_spacing[1] / target_spacing[2]
        # crop_seg = remove_small_cc(crop_seg,area_least, self.topk)
        # seg = resample_data_or_seg_to_shape(crop_seg[None,...], self.new_shape, original_spacing, target_spacing,is_seg=True,order=1,order_z=0)
        # seg= np.where(seg ==14,1,0)
        # assert seg.min() >= 0, 'MIN value of seg is -1'
   
        # print()
        # if np.max(seg) > 127:
        #     seg = seg.astype(np.int16)
        # else:
        #     seg = seg.astype(np.int8)

        data = resample_data_or_seg_to_shape(crop_image[None,...], self.new_shape, original_spacing, target_spacing,is_seg=False,order=3,order_z=0)

        
       
        
        if self.verbose:
            crop_shape = crop_image.shape
            
            print(f'unique: {np.unique(seg)}, old shape: {old_shape}, crop shape: {crop_shape} new_shape: {self.new_shape}, crop_shape: {crop_shape}, '
                  f'old_spacing: {original_spacing}, new_spacing: {target_spacing}, extend_size: {extend_size}')

        return data
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
        
        data = self.run_case_npy(data, seg, pre, data_properites)
        return data, data_properites
    
    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str, predict_file: str):
        data, properties = self.run_case(image_files, seg_file, predict_file)
        # print('dtypes', data.dtype, seg.dtype)
        if data is not None:
            np.savez_compressed(output_filename_truncated + '.npz', data=data)
            write_pickle(properties, output_filename_truncated + '.pkl')
        else: 
            print(output_filename_truncated.split('/')[-1])

    def run(self, data_path, output_path, seg_path,predict_path,
            num_processes: int = 4,file_ending = ".nii.gz", overwrite = False):
        
             
        if overwrite:
            if isdir(output_path):
                shutil.rmtree(output_path)
        maybe_mkdir_p(output_path)
        
        
        
        output_filenames_truncated = [join(output_path, i) for i in self.identifiers]

       
        # list of lists with image filenames
        image_fnames = create_lists_from_splitted_dataset_folder(data_path, file_ending,self.identifiers)
        # list of segmentation filenames
       
        predict_fnames = [join(predict_path, i + file_ending) for i in self.identifiers]
        seg_fnames = [join(seg_path, i + file_ending) for i in self.identifiers]
        
        

        _ = ptqdm(self.run_case_save, (output_filenames_truncated, image_fnames, seg_fnames,predict_fnames),
                  processes=num_processes, zipped=True, disable=self.verbose)

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

    
    # identifiers = load_json(join(os.path.dirname(os.path.realpath(__file__)),
    #                                     'analyze_result','not_tumour_label_case.json')) 
    
    # pp = Second_stage_unlabel_tumour_preprocessor(new_shape = CFG['Dataloader']['data_size'],
    #                                               identifiers = identifiers,
    #                             extend_size = CFG['Preprocess']['extend_size'],
    #                             topk=CFG['Postprocess']['topk'],
    #                             area_least=CFG['Postprocess']['area_least']
    # )
    # pp.run(CFG['Data_path']['raw_label_image_dir'],
    #                   CFG['Data_path']['preprocess_unlabel_data_dir'], 
    #                   CFG['Data_path']['label_abdomen_predict_dir'],
    #                   CFG['Data_path']['not_tumour_predict_dir'],
    #                   num_processes=args.np,
    #                   overwrite = True)


    identifiers = get_identifiers_from_splitted_seg_folder(CFG['Data_path']['unlabel_abdomen_predict_dir'],".nii.gz")   
    pp = Second_stage_unlabel_tumour_preprocessor(new_shape = CFG['Dataloader']['data_size'],
                                                  identifiers = identifiers,
                                extend_size = CFG['Preprocess']['extend_size'],
                                topk=CFG['Postprocess']['topk'],
                                area_least=CFG['Postprocess']['area_least']
                                )

         
    pp.run(CFG['Data_path']['raw_unlabel_image_dir'],
                      CFG['Data_path']['preprocess_unlabel_data_dir'], 
                      CFG['Data_path']['unlabel_abdomen_predict_dir'],
                      CFG['Data_path']['unlabel_tumour_predict_dir'],
                      num_processes=args.np,
                      overwrite = True
                      )
    
 

  

if __name__ == '__main__':
    run_preprocess_entry()

