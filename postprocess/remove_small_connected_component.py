

import cc3d
import fastremap
import numpy as np
from typing import List, Optional


def keep_topk_cc(mask: np.ndarray, area_least: int, topk: int, out_label: int, out_mask: Optional[np.ndarray] = None) -> np.ndarray:
    labeled_mask = mask.copy()
    labeled_mask = cc3d.connected_components(labeled_mask, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labeled_mask, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)

    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    if out_mask is None:
        out_mask = np.zeros_like(mask)
    for i in range(min(topk, len(candidates))):
        if candidates[i][1] > area_least:
            coords = np.where(labeled_mask == int(candidates[i][0]))
            out_mask[coords] = out_label
        else:
            break

    return out_mask


def remove_small_cc(mask: np.ndarray, area_least: int, topk: int, out_mask: Optional[np.ndarray] = None) -> np.ndarray:
    labeled_mask = mask.copy()
    # try:
    labeled_mask,N = cc3d.connected_components(labeled_mask, connectivity=26,return_N=True)
    # except:
    #     return mask
    areas = {}
    for label, extracted in cc3d.each(labeled_mask, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)

    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    if out_mask is None:
        out_mask = np.zeros_like(mask)
    for i in range(min(topk, len(candidates))):
        if candidates[i][1] > area_least:
            coords = np.where(labeled_mask == int(candidates[i][0]))
            out_mask[coords] = mask[coords[0][0], coords[1][0], coords[2][0]]
        else:
            break

    return out_mask