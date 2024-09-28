# config.py
import cv2

# Configuration parameters for Shi-Tomasi corner detection
SHI_TOMASI_PARAMS = {
    'maxCorners': 300,
    'qualityLevel': 0.02,
    'minDistance': 10
}

# Configuration parameters for Lucas-Kanade optical flow
LK_PARAMS = {
    'winSize': (21, 21),
    'maxLevel': 3,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
}

# Camera intrinsic parameters
CAMERA_INTRINSICS = {
    'fx': 1.97547873e+03,
    'fy': 2.05341424e+03,
    'cx': 2.05341424e+03,
    'cy': 2.05341424e+03
}
