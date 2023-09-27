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

class Second_stage_tumour_label_preprocessor(Second_stage_no_full_abdomen_label_preprocessor):
    def __init__(self, new_shape=None, extend_size: list = [30,30,30], area_least: int = 100, topk: int = 30, mask_extend: list = [10,10,10], verbose: bool = True):
        super().__init__(new_shape, extend_size, area_least, topk, verbose)
      
        self.mask_extend = mask_extend

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], pre, properties: dict):
        
        # crop, remember to store size before cropping!
        shape = data.shape[1:]
        original_spacing = properties['spacing']
        properties['shape'] = shape
          # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        
        np.clip(seg,0,None,out = seg)
        try:
            bbox_pre = get_bbox_from_mask(pre[0])
            crop_shape_pre = [i[1]-i[0] for i in bbox_pre ]
        except:
            crop_shape_pre = None


        label_unique = np.unique(seg)
        for i in label_unique:
            pre[pre == i] = 0
        pre[seg > 0] = seg[seg > 0]
      




        # seg_box = seg.copy()
        # seg_box[seg_box == 14] == 0

        bbox = get_bbox_from_mask(seg[0])

        seg[seg ==0] = -1
        seg [(seg > 0) & (seg < 14)] = 0
        seg[seg>0] = 1


        crop_shape = [i[1]-i[0] for i in bbox ]

        target_spacing = compute_new_spacing(crop_shape, original_spacing, self.new_shape)
        
        extend_size = [ int(self.extend_size[i] // target_spacing[i])  for i in range(len(target_spacing))]
        # if crop_shape[1] < 100 and  crop_shape[2] < 100:
        #     extend_size = [ int(extend_size[i] + 50)  for i in range(len(extend_size))]
        data = self._normalize(data)
        
        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        
        crop_image,_ = crop_image_according_to_bbox(data[0],bbox,extend_size)
        

        crop_seg,_ = crop_image_according_to_bbox(seg[0],bbox,extend_size)
        # crop_image, crop_seg = crop_image_according_to_mask(data[0], seg[0], extend_size)
        target_spacing = compute_new_spacing(crop_image.shape, original_spacing, self.new_shape)
        area_least = self.area_least / target_spacing[0] / target_spacing[1] / target_spacing[2]
        crop_seg = remove_small_cc(crop_seg,area_least, self.topk)
        seg = resample_data_or_seg_to_shape(crop_seg[None,...], self.new_shape, original_spacing, target_spacing,is_seg=True,order=1,order_z=0)
        
        assert seg.min() >= 0, 'MIN value of seg is -1'
        # seg[seg <= 13]=0

        
        # if seg.max() != 1:
        #     return None, None

        mask_extend = [ int(self.mask_extend[i] // target_spacing[i])  for i in range(len(target_spacing))]

        seg = extract_label_mask(seg,mask_extend)

        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)




        data = resample_data_or_seg_to_shape(crop_image[None,...], self.new_shape, original_spacing, target_spacing,is_seg=False,order=3,order_z=0)

        
       
        
        if self.verbose:
            crop_shape = crop_image.shape
            
            print(f'unqiue: {np.unique(seg)}, old shape: {old_shape}, crop shape: {crop_shape} new_shape: {self.new_shape}, crop_shape_pre: {crop_shape_pre}, crop_shape: {crop_shape}, '
                  f'old_spacing: {original_spacing}, new_spacing: {target_spacing}, extend_size: {extend_size}')

        return data, seg
    def run(self, data_path, output_path, seg_path,predict_path,
                num_processes: int = 4,file_ending = ".nii.gz"):
            
                
            
        if isdir(output_path):
            shutil.rmtree(output_path)
        maybe_mkdir_p(output_path)
        
        tumour_label_case = load_json(join(os.path.dirname(os.path.realpath(__file__)),
                                        'analyze_result','tumour_label_case.json')) 
        
        output_filenames_truncated = [join(output_path, i) for i in tumour_label_case]

        
        # list of lists with image filenames
        image_fnames = create_lists_from_splitted_dataset_folder(data_path, file_ending,tumour_label_case)
        # list of segmentation filenames
        
        predict_fnames = [join(predict_path, i + file_ending) for i in tumour_label_case]
        seg_fnames = [join(seg_path, i + file_ending) for i in tumour_label_case]
        
        

        _ = ptqdm(self.run_case_save, (output_filenames_truncated, image_fnames, seg_fnames,predict_fnames),
                    processes=num_processes, zipped=True, disable=self.verbose)

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
                      CFG['Data_path']['preprocess_label_data_dir'], 
                      CFG['Data_path']['raw_label_seg_dir'],
                      CFG['Data_path']['label_abdomen_predict_dir'],
                      num_processes=args.np)
    
 

  

if __name__ == '__main__':
    run_preprocess_entry()

