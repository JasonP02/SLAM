import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray_frame, 1000, 0.01, 10)
        
        if features is None:
            return [], None

        keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=10) for f in features]
        keypoints, descriptors = self.orb.compute(frame, keypoints)
        
        return keypoints, descriptors

    def match(self, desc1, desc2):
        matches = self.matcher.match(desc1, desc2)
        return sorted(matches, key=lambda x: x.distance)[:50]