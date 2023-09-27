

import numpy as np
from acvl_utils.miscellaneous.ptqdm import ptqdm
from batchgenerators.utilities.file_and_folder_operations import join
import torch
import torch.nn.functional as F

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import re
import cc3d
from scipy.ndimage import binary_fill_holes, binary_closing
def create_two_class_mask(mask):

    mask = np.clip(mask, 0, 1)
    mask = binary_fill_holes(mask, origin=1,)
    # nonzero_mask = binary_closing(nonzero_mask,origin=1,iterations=1)
    return mask

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return np.array([[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]])


def crop_image_according_to_mask(npy_image, npy_mask, margin=None):
    if margin is None:
        margin = [20, 20, 20]

    bbox = get_bbox_from_mask(npy_mask)

    extend_bbox = np.concatenate(
        [np.max([[0, 0, 0], bbox[:, 0] - margin], axis=0)[:, np.newaxis],
         np.min([npy_image.shape, bbox[:, 1] + margin], axis=0)[:, np.newaxis]], axis=1)


    crop_mask = crop_to_bbox(npy_mask,extend_bbox)
    crop_image = crop_to_bbox(npy_image,extend_bbox)
       

    return crop_image, crop_mask


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(
        bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def crop_image_according_to_bbox(npy_image, bbox, margin=None):
    if margin is None:
        margin = [20, 20, 20]

    image_shape = npy_image.shape
    extend_bbox = [[max(0, int(bbox[0][0]-margin[0])),
                   min(image_shape[0], int(bbox[0][1]+margin[0]))],
                   [max(0, int(bbox[1][0]-margin[1])),
                   min(image_shape[1], int(bbox[1][1]+margin[1]))],
                   [max(0, int(bbox[2][0]-margin[2])),
                   min(image_shape[2], int(bbox[2][1]+margin[2]))]]


    crop_image = crop_to_bbox(npy_image, extend_bbox)
   
  
    return crop_image, extend_bbox






def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)
    # all files must be .nii.gz and have 4 digit channel index
    crop = len(file_ending) + 5
    files = [i[:-crop] for i in files]
    # only unique image ids
    files = np.unique(files)
    return files


def get_identifiers_from_splitted_seg_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)
    # all files must be .nii.gz and have 4 digit channel index
    crop = len(file_ending)
    files = [i[:-crop] for i in files]
    # only unique image ids
    files = np.unique(files)
    return files

def create_lists_from_splitted_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None) -> List[List[str]]:
    """
    does not rely on dataset.json
    """
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    list_of_lists = []
    for f in identifiers:
        p = re.compile(re.escape(f) + r"_\d\d\d\d" + re.escape(file_ending))
        list_of_lists.append([join(folder, i) for i in files if p.fullmatch(i)])
    return list_of_lists



def input_normal_and_downsample(x, input_size, clip_window = None, mean=-34.07, std=333.52, is_clip=False):
    """Input layer, including re-sample, clip and normalization image."""

    x = F.interpolate(x, size=input_size, mode='trilinear',
                      align_corners=False)
    # print(f"min,max: {x.min()}, {x.max()}")
    if is_clip:
        # lower_bound = torch.quantile(
        #     x.float(), clip_window[0], interpolation='nearest')
        # upper_bound = torch.quantile(
        #     x.float(), clip_window[1], interpolation='nearest')
        # print(f"low, max: {lower_bound}, {upper_bound}")
        x = torch.clamp(x, min=clip_window[0], max=clip_window[1])
    mean = torch.mean(x)
    std = torch.std(x)
    x = (x - mean) / (1e-5 + std)
    return x


def input_downsample(x, input_size, clip_window = None, mean=-34.07, std=333.52, is_clip=False):
    """Input layer, including re-sample, clip and normalization image."""

    x = F.interpolate(x, size=input_size, mode='trilinear',
                      align_corners=False)
    # print(f"min,max: {x.min()}, {x.max()}")
    if is_clip:
        # lower_bound = torch.quantile(
        #     x.float(), clip_window[0], interpolation='nearest')
        # upper_bound = torch.quantile(
        #     x.float(), clip_window[1], interpolation='nearest')
        # print(f"low, max: {lower_bound}, {upper_bound}")
        x = torch.clamp(x, min=clip_window[0], max=clip_window[1])
    mean = torch.mean(x)
    std = torch.std(x)
    x = (x - mean) / (1e-5 + std)
    return x


def resize_with_GPU(x, output_size,mode='trilinear',align_corners=False):

    x = F.interpolate(x, size=output_size,
                      mode=mode, align_corners=align_corners)
    return x


# def extract_label_mask(seg,mask_extend):
#     seg_copy = seg[0].copy()
#     out_mask = np.zeros_like(seg_copy).astype(np.int8)-1

#     # try:
#     seg_cc,N = cc3d.connected_components(seg_copy, connectivity=26,return_N=True)
#     print(f'CC_num: {N}')
#     for _, cc in cc3d.each(seg_cc, binary=True, in_place=True):
        
#         bbox = get_bbox_from_mask(cc.astype(int))
#         extend_bbox = np.concatenate(
#             [np.max([[0, 0, 0], bbox[:, 0] - mask_extend], axis=0)[:, np.newaxis],
#             np.min([seg_copy.shape, bbox[:, 1] + mask_extend], axis=0)[:, np.newaxis]], axis=1)
        
#         assert len(out_mask.shape) == 3, "only supports 3d images"

#         out_mask[extend_bbox[0][0]:extend_bbox[0][1],
#                     extend_bbox[1][0]:extend_bbox[1][1],
#                     extend_bbox[2][0]:extend_bbox[2][1]] = seg_copy[extend_bbox[0][0]:extend_bbox[0][1],
#                     extend_bbox[1][0]:extend_bbox[1][1],
#                     extend_bbox[2][0]:extend_bbox[2][1]] 
#     return out_mask[None]


def extract_label_mask(seg,mask_extend):
    seg_copy = seg[0].copy()
    seg_copy[seg_copy==-1]=0
    out_mask = seg[0].copy()

    # try:
    seg_cc,N = cc3d.connected_components(seg_copy, connectivity=26,return_N=True)
    print(f'CC_num: {N}')
    for _, cc in cc3d.each(seg_cc, binary=True, in_place=True):
        
        bbox = get_bbox_from_mask(cc.astype(int))
        extend_bbox = np.concatenate(
            [np.max([[0, 0, 0], bbox[:, 0] - mask_extend], axis=0)[:, np.newaxis],
            np.min([seg_copy.shape, bbox[:, 1] + mask_extend], axis=0)[:, np.newaxis]], axis=1)
        
        assert len(out_mask.shape) == 3, "only supports 3d images"
        
        

        out_mask[extend_bbox[0][0]:extend_bbox[0][1],
                    extend_bbox[1][0]:extend_bbox[1][1],
                    extend_bbox[2][0]:extend_bbox[2][1]] = seg_copy[extend_bbox[0][0]:extend_bbox[0][1],
                    extend_bbox[1][0]:extend_bbox[1][1],
                    extend_bbox[2][0]:extend_bbox[2][1]] 
    return out_mask[None]