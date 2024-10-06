import os
import numpy as np

def get_camera_matrix():
    fx = float(os.environ.get('CAMERA_FX', 1.97547873e+03))
    fy = float(os.environ.get('CAMERA_FY', 2.05341424e+03))
    cx = float(os.environ.get('CAMERA_CX', 2.05341424e+03))
    cy = float(os.environ.get('CAMERA_CY', 2.05341424e+03))

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])