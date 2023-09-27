import os
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
from utilities.helpers import empty_cache, dummy_context,create_lists_from_splitted_dataset_folder

from data_preprocess.preprocessor import Preprocessor
from data_preprocess.imageio.nibabel_reader_writer import NibabelIOWithReorient
class Predictor(object):
    def __init__(self,
                config,
                model_training_dir,
                
                file_ending = '.nii.gz',
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 checkpoint_name: str = "checkpoint_final.pth",
                 device: torch.device = torch.device('cuda'),
                ):
        
        self.config = config
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.file_ending = file_ending
        self.device = device
        self.initialize_from_trained_model_folder(model_training_dir,checkpoint_name)
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                            
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        parameters = []
        
        checkpoint = torch.load(join(model_training_output_dir, checkpoint_name),
                                    map_location=torch.device('cpu'))
        parameters.append(checkpoint['network_weights'])
        self.list_of_parameters = parameters
        model_config = checkpoint['init_args']['config']['Model']
        model_config['deep_supervision'] = False
        self.network = create_network(**model_config)
        print(self.network)
    

    def _manage_input_and_output_lists(self, input_folder: Union[str, List[List[str]]],
                                       output_folder: Union[None, str, List[str]],
                                       overwrite: bool=False,
                                       ):
        
        list_of_input_folder = create_lists_from_splitted_dataset_folder(input_folder, self.file_ending)
        print(f'There are {len(list_of_input_folder)} cases in the source folder')
        caseids = [os.path.basename(i[0])[:-(len(self.file_ending) + 5)] for i in
                   list_of_input_folder]
        print(f'There are {len(caseids)} cases that I would like to predict')
        output_filename_truncated = [join(output_folder, i) for i in caseids]
  
        # remove already predicted files form the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.file_ending) for i in output_filename_truncated]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]
            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_input_folder = [list_of_input_folder[i] for i in not_existing_indices]
            
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_input_folder, output_filename_truncated

    def predict_from_files(self,
                           input_folder: str,
                           output_folder: str,
                           overwrite: bool = True,
                           num_threads_torch: int = 8
                          ):
      
        list_of_input_folder, output_filenames_truncated = self._manage_input_and_output_lists(input_folder,
                                                output_folder,
                                                overwrite)
        if len(list_of_input_folder) == 0:
            return

        preprocessor = Preprocessor(verbose=self.verbose_preprocessing)
        for idx in range(len(list_of_input_folder)):
            data, _, data_properites = preprocessor.run_case(list_of_input_folder[idx], None,
                                                               self.config['Dataloader']['data_size'])

            data = torch.from_numpy(data).contiguous().float()
            output_file_truncated = output_filenames_truncated[idx]

            print(f'\nPredicting image of shape {data.shape}:')
         
            prediction = self.predict_logits_from_preprocessed_data(data).cpu()
            old_threads = torch.get_num_threads()
            torch.set_num_threads(num_threads_torch)
            
            segmentation = torch.softmax(prediction.float(), 0)[0].argmax(0).cpu().float().numpy()
            del prediction
            
            torch.set_num_threads(old_threads)
        

            rw = NibabelIOWithReorient()
            rw.write_seg(segmentation, output_file_truncated + self.file_ending,
                        data_properites)
            
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
           
            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(self.list_of_parameters[0])
            else:
                self.network._orig_mod.load_state_dict(self.list_of_parameters[0])
            assert isinstance(data, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()
            empty_cache(self.device)
    
            with torch.no_grad():
                with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                    assert len(data.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'
                    if self.verbose: print(f'Input shape: {data.shape}')
                
        
                    if self.verbose: print('preallocating arrays')
                    data = data.to(self.device, non_blocking=False)[None]
                    empty_cache(self.device)
                    if self.verbose: print('running prediction')
                    prediction = self.network(data)
            empty_cache(self.device)        
           
                
           

            print('Prediction done, transferring to CPU if needed')
            prediction = prediction.to('cpu')
        
        return prediction






def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-config', type=str, required=False, default="/ai/code/Flare2023/config/config.yaml",
                        help='Config file.')
    parser.add_argument('-i', type=str, required=False, default= "/ai/data/flare2023/validation",
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=False, default= None,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.') 
    parser.add_argument('-m', type=str, required=False, default= None,
                        help='training checkpoint file file folder') 
    parser.add_argument('-cn', type=str, required=False, default=None,
                        help='training checkpoint name')
  
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    
    args = parser.parse_args()
  
    with open(args.config, 'r') as file:
        CFG = yaml.safe_load(file)

    if args.m == None:
        args.m = join(CFG["Data_path"]["checkpoint_dir"],args.cn)
    if args.o == None:
        args.o = join(args.m,"result")
    
    
    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive agressive haha

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    predictor = Predictor( CFG,args.m,'.nii.gz',True,True,'checkpoint_best.pth')

    predictor.predict_from_files(args.i,args.o,True,8)

if __name__ == '__main__':
    predict_entry_point()