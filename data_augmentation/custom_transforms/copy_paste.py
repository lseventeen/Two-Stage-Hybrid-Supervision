from batchgenerators.transforms.abstract_transforms import AbstractTransform
import cc3d
from scipy.ndimage import binary_fill_holes, binary_closing
import numpy as np
from data_preprocess.utils import get_bbox_from_mask
import random
class Copy_paste(AbstractTransform):
    """ Crops data and seg (if available) in the center

    Args:
        output_size (int or tuple of int): Output patch size

    """

    def __init__(self, data_key="data", label_key="seg",copy_num=10, copy_size = 20, ):
        self.data_key = data_key
        self.label_key = label_key
        self.copy_size = copy_size
        self.copy_num = copy_num

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        B,C,H,W,Z = data.shape
        data_result = np.zeros((B,C,H,W,Z),dtype=np.float32)
        seg_result = np.zeros((B,C,H,W,Z),dtype=np.float32)
        
        
        for i in range(B):
            d = data[i]
            s = seg[i]
            mask = seg[i].copy()

            mask[:,:self.copy_size//2,:,:] = 0
            mask[:,H-self.copy_size//2:,:,:] = 0

            mask[:,:,:self.copy_size//2,:] = 0
            mask[:,:,W-self.copy_size//2:,:] = 0

            mask[:,:,:,:self.copy_size//2] = 0
            mask[:,:,:,Z-self.copy_size//2:] = 0

            mask_voxel_coords = np.where(mask > 0)
            if len(mask_voxel_coords[0]) != 0:
                for _ in range(self.copy_num):
                    h = random.randint(self.copy_size//2,H-self.copy_size)
                    w = random.randint(self.copy_size//2,W-self.copy_size)
                    z = random.randint(self.copy_size//2,Z-self.copy_size)

                    center_id = random.randrange(0,len(mask_voxel_coords[0]))
                    center_x = mask_voxel_coords[1][center_id]
                    center_y = mask_voxel_coords[2][center_id]
                    center_z = mask_voxel_coords[3][center_id]

                    d[:,h-self.copy_size//2:h+self.copy_size//2,
                        w-self.copy_size//2:w+self.copy_size//2,
                        z-self.copy_size//2:z+self.copy_size//2] = d[:,center_x-self.copy_size//2:center_x+self.copy_size//2,
                                                                    center_y-self.copy_size//2:center_y+self.copy_size//2,
                                                                    center_z-self.copy_size//2:center_z+self.copy_size//2]
                    s[:,h-self.copy_size//2:h+self.copy_size//2,
                        w-self.copy_size//2:w+self.copy_size//2,
                        z-self.copy_size//2:z+self.copy_size//2] = s[:,center_x-self.copy_size//2:center_x+self.copy_size//2,
                                                                    center_y-self.copy_size//2:center_y+self.copy_size//2,
                                                                    center_z-self.copy_size//2:center_z+self.copy_size//2]
            data_result[i]=d
            seg_result[i]=s
            

        data_dict[self.data_key] = data_result
       
        data_dict[self.label_key] = seg_result

        return data_dict