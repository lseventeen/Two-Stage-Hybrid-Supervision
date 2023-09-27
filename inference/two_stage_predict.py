import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional
import yaml
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch._dynamo import OptimizedModule
from create_network import create_network
from utilities.helpers import empty_cache, dummy_context
from data_preprocess.utils import create_lists_from_splitted_dataset_folder,resize_with_GPU,crop_image_according_to_bbox,get_bbox_from_mask
from data_preprocess.preprocessor import Preprocessor
from data_preprocess.imageio.nibabel_reader_writer import NibabelIOWithReorient,NibabelIO
from data_preprocess.imageio.simpleitk_reader_writer import SimpleITKIO
from data_preprocess.resampling import resample_data_or_seg,compute_new_spacing
from postprocess.remove_small_connected_component import remove_small_cc
from eval.FLARE23_DSC_NSD_Eval import eval_dsc_nsd

import shutil

class Predictor(object):
    def __init__(self, config
                ):
        self.input_dir = config['input_dir']
        self.output_dir = config['output_dir']
        self.gt_dir = config['gt_dir']
        self.score_dir = config['score_dir']
        self.first_stage_checkpoint_dir = config['first_stage_checkpoint_dir']
        self.second_stage_abdomen_checkpoint_dir = config['second_stage_abdomen_checkpoint_dir']
        self.second_stage_tumour_checkpoint_dir = config['second_stage_tumour_checkpoint_dir']
        self.checkpoint_name = config['checkpoint_name']
        self.file_ending = config['file_ending']

        self.downsample_using_GPU = config['downsample_using_GPU']
        self.upsample_using_GPU = config['upsample_using_GPU']
        
        self.do_first_stage_RMCC = config['do_first_stage_RMCC']
        self.first_stage_area_least = config['first_stage_area_least']
        self.first_stage_topk = config['first_stage_topk']

        self.do_second_stage_tumour_RMCC = config['do_second_stage_tumour_RMCC']
        self.second_stage_tumour_area_least = config['second_stage_tumour_area_least']
        self.second_stage_tumour_topk = config['second_stage_tumour_topk']

        self.do_second_stage_abdomen_RMCC = config['do_second_stage_abdomen_RMCC']
        self.second_stage_abdomen_area_least = config['second_stage_abdomen_area_least']
        self.second_stage_abdomen_topk = config['second_stage_abdomen_topk']

        self.extend_size = config['extend_size']    
        self.cover_tumour = config['cover_tumour']
        self.overwrite = config['overwrite']
        self.verbose_preprocessing = config['verbose']
        self.verbose = config['verbose']
        self.do_eval = config['eval']
        self.threshold = config['threshold']
        self.upsample_first = config['upsample_first']
        self.device = torch.device('cuda')



    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                            
                                             checkpoint_name: str = 'checkpoint_final.pth'):
     

        checkpoint = torch.load(join(model_training_output_dir, checkpoint_name), map_location=torch.device('cpu'))
        model_config = checkpoint['init_args']['config']['Model']
        new_shape = checkpoint['init_args']['config']['Dataloader']['data_size']
        model_config['deep_supervision'] = False
        model_config['is_inference'] = True
        network = create_network(**model_config)
        if not isinstance(network, OptimizedModule):
            network.load_state_dict(checkpoint['network_weights'])
        else:
            network._orig_mod.load_state_dict(checkpoint['network_weights'])
        return network,new_shape
    

    def _manage_input_and_output_lists(self, input_folder: Union[str, List[List[str]]],
                                       output_folder: Union[None, str, List[str]],
                                       overwrite: bool=False,
                                       ):
        
        list_of_input_folder = create_lists_from_splitted_dataset_folder(input_folder, self.file_ending)
        if self.verbose:
            print(f'There are {len(list_of_input_folder)} cases in the source folder')
        caseids = [os.path.basename(i[0])[:-(len(self.file_ending) + 5)] for i in
                   list_of_input_folder]
        if self.verbose:
            print(f'There are {len(caseids)} cases that I would like to predict')
        output_filename_truncated = [join(output_folder, i) for i in caseids]
  
        # remove already predicted files form the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.file_ending) for i in output_filename_truncated]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]
            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_input_folder = [list_of_input_folder[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. 'f'That\'s {len(not_existing_indices)} cases.')
        return list_of_input_folder, output_filename_truncated
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor,network) -> torch.Tensor:

        with torch.no_grad():
           
            # messing with state dict names...
           
            assert isinstance(data, torch.Tensor)
            network = network.to(self.device, non_blocking=False)
            network.eval()
            empty_cache(self.device)
    
            with torch.no_grad():
                with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                    assert len(data.shape) == 5, 'input_image must be a 4D np.ndarray or torch.Tensor (n, c, x, y, z)'
                    if self.verbose: print(f'Input shape: {data.shape}')
                    # data = data[None]
                    empty_cache(self.device)
                    if self.verbose: print('running prediction')
                    prediction = network(data)
            empty_cache(self.device)        
           
                
        
        return prediction
    def predict_from_files(self):
        
        list_of_input_folder, output_filenames_truncated = self._manage_input_and_output_lists(self.input_dir,
                                                self.output_dir,
                                                self.overwrite)
        if len(list_of_input_folder) == 0:
            return

        preprocessor = Preprocessor(None,self.verbose_preprocessing)
        for idx in range(len(list_of_input_folder)):
            
            start_time = time.time()
            output_file_truncated = output_filenames_truncated[idx]
            raw_image, _, data_properites = preprocessor.run_case(list_of_input_folder[idx], None)
            raw_spacing = data_properites['spacing']
            raw_image_shape = raw_image.shape[1:]
            model,first_stage_shape = self.initialize_from_trained_model_folder(self.first_stage_checkpoint_dir,self.checkpoint_name)
            if self.downsample_using_GPU: 
                first_stage_image = torch.from_numpy(raw_image).contiguous().float().to(self.device, non_blocking=False)[None]
                first_stage_image = resize_with_GPU(first_stage_image,first_stage_shape)
            else:
                first_stage_image = resample_data_or_seg(raw_image, first_stage_shape)[None,...]
                first_stage_image = torch.from_numpy(first_stage_image).contiguous().float().to(self.device, non_blocking=False)
            
            if self.verbose:
                print(f'\nPredicting image of shape {raw_image_shape}:')
            
            location_output = self.predict_logits_from_preprocessed_data(first_stage_image,model)
            del model
            torch.cuda.empty_cache()
            
            first_stage_image = first_stage_image.cpu()
            location_seg = torch.softmax(location_output, 1)[0].argmax(0).cpu().numpy().astype(np.uint8)
            del location_output
            torch.cuda.empty_cache()
            
            
            if self.do_first_stage_RMCC:
                target_spacing = compute_new_spacing(raw_image_shape, raw_spacing, first_stage_shape)
                area_least = self.first_stage_area_least / target_spacing[0] / target_spacing[1] / target_spacing[2]
                location_seg = remove_small_cc(location_seg,area_least, self.first_stage_topk)

            location_bbox = get_bbox_from_mask(location_seg)
            first_stage_resize_factor = np.array(raw_image_shape) / np.array(first_stage_shape)
            raw_bbox = [[int(location_bbox[0][0] * first_stage_resize_factor[0]),
                            int(location_bbox[0][1] * first_stage_resize_factor[0])],
                        [int(location_bbox[1][0] * first_stage_resize_factor[1]),
                            int(location_bbox[1][1] * first_stage_resize_factor[1])],
                        [int(location_bbox[2][0] * first_stage_resize_factor[2]),
                            int(location_bbox[2][1] * first_stage_resize_factor[2])]]
            margin = [self.extend_size[i] / raw_spacing[i] for i in range(3)]
            crop_image, crop_second_stage_bbox = crop_image_according_to_bbox(raw_image[0], raw_bbox, margin)
            if self.verbose:
                print(crop_second_stage_bbox)

            crop_image_size = crop_image.shape
           
           
            model,second_stage_shape = self.initialize_from_trained_model_folder(self.second_stage_abdomen_checkpoint_dir,self.checkpoint_name)
            if self.downsample_using_GPU: 
                crop_image = torch.from_numpy(crop_image).contiguous().float().to(self.device, non_blocking=False)[None,None]
                crop_image = resize_with_GPU(crop_image,second_stage_shape)
            else:
                crop_image = resample_data_or_seg(crop_image, second_stage_shape)[None,...]
                crop_image = torch.from_numpy(crop_image).contiguous().float().to(self.device, non_blocking=False)
            
            second_stage_abdomen_output = self.predict_logits_from_preprocessed_data(crop_image,model)
            del model
            torch.cuda.empty_cache()
            
            
            model,_ = self.initialize_from_trained_model_folder(self.second_stage_tumour_checkpoint_dir, self.checkpoint_name)
            second_stage_tumour_output = self.predict_logits_from_preprocessed_data(crop_image,model)
            del model
            torch.cuda.empty_cache()
            
            if self.upsample_first:
                
                second_stage_abdomen_output = second_stage_abdomen_output.contiguous().float().to(self.device, non_blocking=False)
                second_stage_abdomen_output = resize_with_GPU(second_stage_abdomen_output,crop_image_size,mode = 'trilinear',align_corners=None).cpu()

                second_stage_tumour_output = second_stage_tumour_output.contiguous().float().to(self.device, non_blocking=False)
                second_stage_tumour_output = resize_with_GPU(second_stage_tumour_output,crop_image_size,mode = 'trilinear',align_corners=None).cpu()


            abdomen_seg = torch.softmax(second_stage_abdomen_output, 1)[0].argmax(0).cpu().numpy().astype(np.uint8)
            
            del second_stage_abdomen_output
            torch.cuda.empty_cache()
            
            if self.threshold == None:
                tumour_seg = torch.softmax(second_stage_tumour_output, 1)[0].argmax(0).cpu().numpy().astype(np.uint8)
                tumour_seg = np.where(tumour_seg==1,14,0)
                
            else:
                
                tumour_seg = torch.softmax(second_stage_tumour_output, 1)[0].cpu().numpy()
                tumour_seg = np.where(tumour_seg[1,...] > self.threshold, 14, 0)
            del second_stage_tumour_output
            torch.cuda.empty_cache()
            if self.do_second_stage_abdomen_RMCC:
                target_spacing = compute_new_spacing(crop_image_size, raw_spacing,second_stage_shape) if not self.upsample_first else raw_spacing
                area_least = self.second_stage_abdomen_area_least / target_spacing[0] / target_spacing[1] / target_spacing[2]
                abdomen_seg = remove_small_cc(abdomen_seg,area_least, self.second_stage_abdomen_topk)

            if self.do_second_stage_tumour_RMCC:
                target_spacing = compute_new_spacing(crop_image_size, raw_spacing,second_stage_shape) if not self.upsample_first else raw_spacing
                area_least = self.second_stage_tumour_area_least / target_spacing[0] / target_spacing[1] / target_spacing[2]
                tumour_seg = remove_small_cc(tumour_seg,area_least, self.second_stage_tumour_topk)

            if self.cover_tumour:
                tumour_seg[abdomen_seg > 0] = abdomen_seg[abdomen_seg > 0]
                seg = tumour_seg
            else:
                abdomen_seg[tumour_seg == 14] = tumour_seg[tumour_seg == 14]
                seg = abdomen_seg
            seg = abdomen_seg 
            # seg = tumour_seg
            if self.verbose:
                print(np.unique(seg))
            
            if not self.upsample_first:
                if self.upsample_using_GPU:
                    seg = torch.from_numpy(seg).contiguous().float().to(self.device, non_blocking=False)[None,None]
                    seg = resize_with_GPU(seg,crop_image_size,mode = 'nearest',align_corners=None).cpu().numpy()[0,0]
            
                else: 
                    seg = resample_data_or_seg(seg, crop_image_size,is_seg =True, order = 1)

               
            # out_mask = np.ones(raw_image_shape, np.uint8)*15
            out_mask = np.zeros(raw_image_shape, np.uint8)
            out_mask[crop_second_stage_bbox[0][0]:crop_second_stage_bbox[0][1],
                        crop_second_stage_bbox[1][0]:crop_second_stage_bbox[1][1],
                        crop_second_stage_bbox[2][0]:crop_second_stage_bbox[2][1]] = seg
            

        

            rw = NibabelIO()
            rw.write_seg(out_mask, output_file_truncated + self.file_ending,
                        data_properites)
            run_time = time.time()-start_time
            print(f"Run time: {run_time}")
            print(f"Inference done: {output_file_truncated}")
        if self.do_eval:
            eval_dsc_nsd(self.output_dir,self.gt_dir,self.score_dir)







def predict_entry_point():
    
    import argparse
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('-i', type=str, required=False, default= None,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=False, default= None,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.') 
    parser.add_argument('-config', type=str, required=False, default="/ai/code/Flare2023/config/two_stage_inference.yaml",
                        help='Config file.')
    parser.add_argument('-verbose',  action='store_true', required=False)
    
    parser.add_argument('-eval',  action='store_true', required=False)

    
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        CFG = yaml.safe_load(file)
  

    start_time = time.time()

    if args.o != None:
        CFG['output_dir'] = args.o
    
    if args.i != None:
        CFG['input_dir'] = args.i
    CFG['verbose']= args.verbose
    CFG['eval']= args.eval

    if isdir(CFG['output_dir'] ):
        shutil.rmtree(CFG['output_dir'] )
    maybe_mkdir_p(CFG['output_dir'])

    save_json(CFG,join(CFG['output_dir'],'config.json'))
    predictor = Predictor(CFG)
    predictor.predict_from_files()
    run_time = time.time()-start_time
    print(f"total run time: {run_time}")
if __name__ == '__main__':

    predict_entry_point()