import os
from typing import List

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from .utils import get_case_identifiers



# def get_case_identifiers(folder: str) -> List[str]:
#     """
#     finds all npz files in the given folder and reconstructs the training case names from them
#     """
#     case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz")]
#     return case_identifiers


class Flare_Dataset(object):
    def __init__(self, folder: str, case_identifiers: List[str] = None, load_seg: bool = True):
        """
        This does not actually load the dataset. It merely creates a dictionary where the keys are training case names and
        the values are dictionaries containing the relevant information for that case.
        dataset[training_case] -> info
        Info has the following key:value pairs:
        - dataset[case_identifier]['properties']['data_file'] -> the full path to the npz file associated with the training case
        - dataset[case_identifier]['properties']['properties_file'] -> the pkl file containing the case properties

        In addition, if the total number of cases is < num_images_properties_loading_threshold we load all the pickle files
        (containing auxiliary information). This is done for small datasets so that we don't spend too much CPU time on
        reading pkl files on the fly during training. However, for large datasets storing all the aux info (which also
        contains locations of foreground voxels in the images) can cause too much RAM utilization. In that
        case is it better to load on the fly.

        If properties are loaded into the RAM, the info dicts each will have an additional entry:
        - dataset[case_identifier]['properties'] -> pkl file content

        IMPORTANT! THIS CLASS ITSELF IS READ-ONLY. YOU CANNOT ADD KEY:VALUE PAIRS WITH nnUNetDataset[key] = value
        USE THIS INSTEAD:
        nnUNetDataset.dataset[key] = value
        (not sure why you'd want to do that though. So don't do it)
        """
        super().__init__()
        self.load_seg = load_seg
        # print('loading dataset')
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder, "%s.npz" % c)
            self.dataset[c]['properties_file'] = join(folder, "%s.pkl" % c)

    def __getitem__(self, key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def load_case(self, key):
        entry = self[key]
        data = np.load(entry['data_file'][:-4] + ".npy", 'r')
        seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r') if self.load_seg else None
        return data, seg, entry['properties']


    

class Flare_Dataset_two_folder(Flare_Dataset):
    def __init__(self, folder1: str, folder2: str, case_identifiers1: List[str] = None, case_identifiers2: List[str] = None,load_seg: bool = True):
        """
        This does not actually load the dataset. It merely creates a dictionary where the keys are training case names and
        the values are dictionaries containing the relevant information for that case.
        dataset[training_case] -> info
        Info has the following key:value pairs:
        - dataset[case_identifier]['properties']['data_file'] -> the full path to the npz file associated with the training case
        - dataset[case_identifier]['properties']['properties_file'] -> the pkl file containing the case properties

        In addition, if the total number of cases is < num_images_properties_loading_threshold we load all the pickle files
        (containing auxiliary information). This is done for small datasets so that we don't spend too much CPU time on
        reading pkl files on the fly during training. However, for large datasets storing all the aux info (which also
        contains locations of foreground voxels in the images) can cause too much RAM utilization. In that
        case is it better to load on the fly.

        If properties are loaded into the RAM, the info dicts each will have an additional entry:
        - dataset[case_identifier]['properties'] -> pkl file content

        IMPORTANT! THIS CLASS ITSELF IS READ-ONLY. YOU CANNOT ADD KEY:VALUE PAIRS WITH nnUNetDataset[key] = value
        USE THIS INSTEAD:
        nnUNetDataset.dataset[key] = value
        (not sure why you'd want to do that though. So don't do it)
        """
        self.load_seg = load_seg
        case_identifiers1.sort()
        case_identifiers2.sort()

        self.dataset = {}
        for c in case_identifiers1:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder1, "%s.npz" % c)
            self.dataset[c]['properties_file'] = join(folder1, "%s.pkl" % c)
        for c in case_identifiers2:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder2, "%s.npz" % c)
            self.dataset[c]['properties_file'] = join(folder2, "%s.pkl" % c)




    


