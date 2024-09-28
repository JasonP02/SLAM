# trackers.py

import cv2
import numpy as np
from config import LK_PARAMS

class FeatureTracker:
    """Tracks features across video frames using Lucas-Kanade optical flow."""
    def __init__(self, lk_params=None):
        self.lk_params = lk_params if lk_params is not None else LK_PARAMS
        self.prev_gray = None
        self.prev_features = None

    def initialize(self, frame, features):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_features = features.reshape(-1, 1, 2).astype(np.float32)

    def track(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None or self.prev_features is None:
            return np.array([]), np.array([])

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_features,
            None,
            **self.lk_params
        )

        if p1 is not None and st is not None:
            good_new = p1[st.flatten() == 1].reshape(-1, 2)
            good_prev = self.prev_features[st.flatten() == 1].reshape(-1, 2)
        else:
            good_new = np.array([])
            good_prev = np.array([])

        self.prev_gray = gray.copy()
        self.prev_features = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

        return good_prev, good_new
