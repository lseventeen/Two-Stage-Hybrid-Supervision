from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from utils import create_lists_from_splitted_dataset_folder,get_identifiers_from_splitted_dataset_folder


def crop_part_label_abdomen_case():
    not_full_abdomen_label_case = load_json('/ai/code/Flare2023/data_preprocess/analyze_result/not_full_abdomen_label_case.json')   
    input_dir = '/ai/data/flare2023/imagesTr2200'
    output_dir = '/ai/data/flare2023/not_full_tumour_label_case'


    if isdir(output_dir):
        shutil.rmtree(output_dir)
    maybe_mkdir_p(output_dir)

    data_size = len(not_full_abdomen_label_case)
    print(f"DATA SIZE: {data_size}")
    for i in not_full_abdomen_label_case:
        # train_names.append(i)
        shutil.copy(join(input_dir, i+'_0000.nii.gz'), join(output_dir, i+'_0000.nii.gz'))
        print(f'{i} Done')
    


if __name__ == "__main__":
    crop_part_label_abdomen_case()

       