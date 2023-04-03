from ultralytics import YOLO
from glob import glob
import numpy as np
from skimage.morphology import skeletonize
from cv2 import Canny


def calculate_pixel_size(results, catsize:float=2.3):

    cat = results.masks.masks[np.where(results.boxes.cls.numpy() == 25)[0]][0].numpy()

    A = np.array(np.where(Canny(cat.astype(np.uint8), 1, 1)>0)).T
    B = np.array(np.where(skeletonize(cat))).T
    res = (A[:, np.newaxis] - B)
    out = (res**2).sum(axis=2).min(axis=0).mean()
    return catsize/out

def calculate_region_width(source:np.ndarray, pixel_size:float=0.24):
    A = np.array(np.where(Canny(source.astype(np.uint8), 1, 1)>0)).T
    B = np.array(np.where(skeletonize(source))).T
    res = (A[:, np.newaxis] - B)
    out_list = (res**2).sum(axis=2).min(axis=0)*pixel_size
    return list(out_list)