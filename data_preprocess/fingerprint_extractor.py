import os
from typing import List, Type, Union
from collections import Counter
import numpy as np
from acvl_utils.miscellaneous.ptqdm import ptqdm
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from imageio.nibabel_reader_writer import NibabelIOWithReorient
from imageio.simpleitk_reader_writer import SimpleITKIO
from cropping import crop_to_nonzero
from utils import get_identifiers_from_splitted_dataset_folder, create_lists_from_splitted_dataset_folder
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

class DatasetFingerprintExtractor(object):
    def __init__(self, num_processes: int = 8, verbose: bool = False):
        """
        extracts the dataset fingerprint used for experiment planning. The dataset fingerprint will be saved as a
        json file in the input_folder

        Philosophy here is to do only what we really need. Don't store stuff that we can easily read from somewhere
        else. Don't compute stuff we don't need (except for intensity_statistics_per_channel)
        """
        
        self.verbose = verbose

        self.input_folder = "/ai/data/flare2023"
        self.num_processes = num_processes

        # We don't want to use all foreground voxels because that can accumulate a lot of data (out of memory). It is
        # also not critically important to get all pixels as long as there are enough. Let's use 10e7 voxels in total
        # (for the entire dataset)
        self.num_foreground_voxels_for_intensitystats = 10e7

    @staticmethod
    def collect_foreground_intensities(segmentation: np.ndarray, images: np.ndarray, seed: int = 1234,
                                       num_samples: int = 10000):
        """
        images=image with multiple channels = shape (c, x, y(, z))
        """
        assert len(images.shape) == 4
        assert len(segmentation.shape) == 4

        assert not np.any(np.isnan(segmentation)), "Segmentation contains NaN values. grrrr.... :-("
        assert not np.any(np.isnan(images)), "Images contains NaN values. grrrr.... :-("

        rs = np.random.RandomState(seed)

        intensities_per_channel = []
        # we don't use the intensity_statistics_per_channel at all, it's just something that might be nice to have
        intensity_statistics_per_channel = []

        # segmentation is 4d: 1,x,y,z. We need to remove the empty dimension for the following code to work
        foreground_mask = segmentation[0] > 0

        for i in range(len(images)):
            foreground_pixels = images[i][foreground_mask]
            num_fg = len(foreground_pixels)
            # sample with replacement so that we don't get issues with cases that have less than num_samples
            # foreground_pixels. We could also just sample less in those cases but that would than cause these
            # training cases to be underrepresented
            intensities_per_channel.append(
                rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else [])
            intensity_statistics_per_channel.append({
                'mean': np.mean(foreground_pixels) if num_fg > 0 else np.nan,
                'median': np.median(foreground_pixels) if num_fg > 0 else np.nan,
                'min': np.min(foreground_pixels) if num_fg > 0 else np.nan,
                'max': np.max(foreground_pixels) if num_fg > 0 else np.nan,
                'percentile_99_5': np.percentile(foreground_pixels, 99.5) if num_fg > 0 else np.nan,
                'percentile_00_5': np.percentile(foreground_pixels, 0.5) if num_fg > 0 else np.nan,

            })

        return intensities_per_channel, intensity_statistics_per_channel

    @staticmethod
    def analyze_case(image_files: List[str], segmentation_file: str, reader_writer_class,
                     num_samples: int = 10000):
        rw = reader_writer_class()
        data_id = segmentation_file.split("/")[-1].split(".")[0]
        data, properties_images = rw.read_images(image_files)
        seg, properties_seg = rw.read_seg(segmentation_file)
        np.clip(seg,0,None,out = seg)
        label_unique, counts = np.unique(seg, return_counts=True)
        # we no longer crop and save the cropped images before this is run. Instead we run the cropping on the fly.
        # Downside is that we need to do this twice (once here and once during preprocessing). Upside is that we don't
        # need to save the cropped data anymore. Given that cropping is not too expensive it makes sense to do it this
        # way. This is only possible because we are now using our new input/output interface.
        # data_cropped, seg_cropped, bbox = crop_to_nonzero(images, seg)

        foreground_intensities_per_channel, foreground_intensity_stats_per_channel = \
            DatasetFingerprintExtractor.collect_foreground_intensities(seg, data,
                                                                       num_samples=num_samples)
        # label_unique, counts = np.unique(seg, return_counts=True)
        # label_unique = np.unique(segmentation).tolist()
        # value_counts = Counter(segmentation)
        spacing = properties_images['spacing']

        shape = data.shape[1:]
        # shape_after_crop = data.shape[1:]
        # relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)
        return foreground_intensities_per_channel, foreground_intensity_stats_per_channel, \
               {data_id:[label_unique.tolist(),counts.tolist(),spacing,shape,]}  
        # {label_unique, shape_before_crop, spacing}

    def run(self, overwrite_existing: bool = False) -> dict:
        # we do not save the properties file in self.input_folder because that folder might be read-only. We can only
        # reliably write in nnUNet_preprocessed and nnUNet_results, so nnUNet_preprocessed it is
        preprocessed_output_folder =  os.path.join(os.path.dirname(os.path.realpath(__file__)),"analyze_result")
        maybe_mkdir_p(preprocessed_output_folder)
        properties_file = join(preprocessed_output_folder, 'dataset_fingerprint.json')
        full_label_case_file = join(preprocessed_output_folder, 'full_label_case.json')
        full_abdomen_label_case_file = join(preprocessed_output_folder, 'full_abdomen_label_case.json')
        part_abdomen_label_case_file = join(preprocessed_output_folder, 'part_abdomen_label_case.json')
        tumour_label_case_file = join(preprocessed_output_folder, 'tumour_label_case.json')
        not_full_abdomen_label_case_file = join(preprocessed_output_folder, 'not_full_abdomen_label_case.json')
        not_tumour_label_case_file = join(preprocessed_output_folder, 'not_tumour_label_case.json')
        not_abdomen_label_case_file = join(preprocessed_output_folder, 'not_abdomen_label_case.json')
        if not isfile(properties_file) or overwrite_existing:
            file_ending = ".nii.gz"
            
            training_identifiers = get_identifiers_from_splitted_dataset_folder(join(self.input_folder, 'imagesTr2200'),
                                                                                file_ending)
            training_images_per_case = create_lists_from_splitted_dataset_folder(join(self.input_folder, 'imagesTr2200'),
                                                                                 file_ending)
            training_labels_per_case = [join(self.input_folder, 'labelsTr2200', i + file_ending) for i in
                                        training_identifiers]

            # determine how many foreground voxels we need to sample per training case
            num_foreground_samples_per_case = int(self.num_foreground_voxels_for_intensitystats //
                                                  len(training_identifiers))

            results = ptqdm(DatasetFingerprintExtractor.analyze_case,
                            (training_images_per_case, training_labels_per_case),
                            processes=self.num_processes, zipped=True, reader_writer_class=NibabelIOWithReorient,
                            num_samples=num_foreground_samples_per_case, disable=self.verbose)

            

            label = [r[-1] for r in results]
            foreground_intensities_per_channel = [np.concatenate([r[0][i] for r in results]) for i in
                                                  range(len(results[0][0]))]
            full_label_case = []
            full_abdomen_label_case = []
            part_abdomen_label_case = []
            tumour_label_case = []
            not_full_abdomen_label_case = []
            not_tumour_label_case = []
            not_abdomen_label_case = []

            for r in results:
                if all(i in list(r[-1].values())[0][0] for i in range(1, 14)):
                    full_abdomen_label_case.append(list(r[-1].keys())[0])
                if len(list(r[-1].values())[0][0]) == 15:
                    full_label_case.append(list(r[-1].keys())[0])

                if 14 in list(r[-1].values())[0][0]:
                    tumour_label_case.append(list(r[-1].keys())[0])
                if not 14 in list(r[-1].values())[0][0]:
                    not_tumour_label_case.append(list(r[-1].keys())[0])
                if any(i in list(r[-1].values())[0][0] for i in range(1, 14)) and len(list(r[-1].values())[0][0]) < 15 and 14 in list(r[-1].values())[0][0]:
                    part_abdomen_label_case.append(list(r[-1].keys())[0])
                elif any(i in list(r[-1].values())[0][0] for i in range(1, 14)) and len(list(r[-1].values())[0][0]) < 14 and not 14 in list(r[-1].values())[0][0]:
                    part_abdomen_label_case.append(list(r[-1].keys())[0])

                if len(list(r[-1].values())[0][0]) < 15 and 14 in list(r[-1].values())[0][0]:
                    not_full_abdomen_label_case.append(list(r[-1].keys())[0])
                elif len(list(r[-1].values())[0][0]) < 14 and not 14 in list(r[-1].values())[0][0]:
                    not_full_abdomen_label_case.append(list(r[-1].keys())[0])
                    
                if not any(i in list(r[-1].values())[0][0] for i in range(1, 14)):
                    not_abdomen_label_case.append(list(r[-1].keys())[0])
                
                # for i in range(1,13):
                #     if i in list(r[-1].values())[0][0]:
                #         tumour_label_case.append(list(r[-1].keys())[0])
                #         break

            num_channels = 1
            intensity_statistics_per_channel = {}
            for i in range(num_channels):
                intensity_statistics_per_channel[i] = {
                    'mean': float(np.mean(foreground_intensities_per_channel[i])),
                    'median': float(np.median(foreground_intensities_per_channel[i])),
                    'std': float(np.std(foreground_intensities_per_channel[i])),
                    'min': float(np.min(foreground_intensities_per_channel[i])),
                    'max': float(np.max(foreground_intensities_per_channel[i])),
                    'percentile_99_5': float(np.percentile(foreground_intensities_per_channel[i], 99.5)),
                    'percentile_00_5': float(np.percentile(foreground_intensities_per_channel[i], 0.5)),
                }

            fingerprint = {
                    "label":label,
                    'foreground_intensity_properties_per_channel': intensity_statistics_per_channel,
 
                }

            try:
                save_json(fingerprint, properties_file)
                save_json(full_label_case, full_label_case_file)
                save_json(full_abdomen_label_case, full_abdomen_label_case_file)
                save_json(part_abdomen_label_case, part_abdomen_label_case_file)
                save_json(tumour_label_case, tumour_label_case_file)
                save_json(not_full_abdomen_label_case, not_full_abdomen_label_case_file)
                save_json(not_tumour_label_case,not_tumour_label_case_file)
                save_json(not_abdomen_label_case,not_abdomen_label_case_file)
            except Exception as e:
                if isfile(properties_file):
                    os.remove(properties_file)
                raise e
        else:
            fingerprint = load_json(properties_file)
        return fingerprint


if __name__ == '__main__':
    dfe = DatasetFingerprintExtractor(10)
    dfe.run(overwrite_existing=True)
