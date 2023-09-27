import os
import shutil
from batchgenerators.utilities.file_and_folder_operations import *


def main():
    file_dir = "/ai/data/flare2023/val-gt"
    image_source_dir = "/ai/data/flare2023/validation"
    # label_source_dir = "/ai/data/surgtoolloc23/STLv2/train/image"
    
    image_target_dir = "/ai/data/flare2023/val50"
    # label_target_dir = "/ai/data/surgtoolloc23/test_set/label"

    # 获取文件夹a中的文件名列表x
    file_list = os.listdir(file_dir)
    file_list = [i.split('.')[0] for i in file_list]

    os.makedirs(image_target_dir,exist_ok=True)
    # os.makedirs(label_target_dir,exist_ok=True)


    for file_name in file_list:
        image_source_path = os.path.join(image_source_dir, file_name+'_0000.nii.gz')
        if isfile(image_source_path):
       
            shutil.copy2(image_source_path, image_target_dir)
            # label_source_path = os.path.join(label_source_dir, file_name+'.txt')
            # shutil.copy2(label_source_path, label_target_dir)


            
            print(f"Copied {file_name} to {image_source_path}")

if __name__ == "__main__":
    main()






