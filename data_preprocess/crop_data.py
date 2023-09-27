from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import argparse


def crop_data(input_path,output_path,json_file,file_ending):
    case_id = load_json(json_file)   
    if isdir(output_path):
        shutil.rmtree(output_path)
    maybe_mkdir_p(output_path)
    data_size = len(case_id)
    print(f"DATA SIZE: {data_size}")

    for i in case_id:
        shutil.copy(join(input_path, i+file_ending), join(output_path, i+file_ending))
        print(f'{i} Done')

if __name__ == "__main__":
    
    crop_data(input_path = '/ai/data/flare2023/imagesTr2200', 
              output_path = '/ai/data/flare2023/not_tumour_label_case' ,
              json_file = '/ai/code/Flare2023/data_preprocess/analyze_result/not_tumour_label_case.json',
              file_ending = '_0000.nii.gz')


       