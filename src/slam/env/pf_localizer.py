import numpy as np
from pathlib import Path
from .gridmap import GridMap
from .robot import *
from numba import njit, prange
import cv2
from scipy.ndimage import distance_transform_edt



class PFLocalizer:
    def __init__(self,env:Environment,n_part:int):
        self.env=env
        self.distmap=distance_transform_edt(env.map.grid)

        np.arra

        assert 0
    


    def render(self, lidar_scan=None):
        h, w = self.map.grid.shape
        img_true = np.ones((h, w, 3), dtype=np.uint8) * 255 
        img_true[self.map.grid]=np.zeros(3,dtype=np.uint8)
        img_guess=img_true.copy()
        self.env.render_robot_to_img(img_true,lidar_scan)


        return img_true
        