# extractors.py

import cv2
import numpy as np
from config import SHI_TOMASI_PARAMS

class ShiTomasiExtractor:
    """Extracts Shi-Tomasi corners from video frames."""
    def __init__(self, maxCorners=300, qualityLevel=0.02, minDistance=10):
        self.params = {
            'maxCorners': maxCorners,
            'qualityLevel': qualityLevel,
            'minDistance': minDistance
        }

    def extract_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, mask=None, **self.params)
        return np.float32(corners).reshape(-1, 2) if corners is not None else np.array([])
