# video_processor.py

import cv2
import numpy as np
from config import CAMERA_INTRINSICS
from visualizers import Visualizer
from bundle_adjuster import BundleAdjuster

class VideoProcessor:
    def __init__(self, filepath):
        self.visualizer = Visualizer()
        self.features = {"Keypoints": [], "Descriptors": []}
        self.global_map = np.empty((0, 3))
        self.points_3d_global = None
        self.orb = cv2.ORB_create()
        self.frame_count = 0
        self.K = np.array([
            [CAMERA_INTRINSICS['fx'], 0, CAMERA_INTRINSICS['cx']],
            [0, CAMERA_INTRINSICS['fy'], CAMERA_INTRINSICS['cy']],
            [0, 0, 1]
        ])
        self.bundle_adjuster = BundleAdjuster(self.K)

    def process_video(self, filepath):
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {filepath}")
        
        try:
            while True:
                ret, self.frame = cap.read()
                if not ret:
                    print("End of video or cannot read the frame.")
                    break

                self.update_global_map()
                self.frame_count += 1

                if self.visualizer.display_frame(self.frame) == ord('q'):
                    print("Processing stopped by user.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"Total frames processed: {self.frame_count}")
        
        return np.vstack(self.global_map) if self.global_map else np.array([])

    def match_features(self, desc1, desc2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:50]

    def get_kp_and_desc(self):
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray_frame, 1000, 0.01, 10)
        
        if features is None:
            return [], None

        keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=10) for f in features]
        keypoints, descriptors = self.orb.compute(self.frame, keypoints)
        
        self.features["Keypoints"].append(keypoints)
        self.features["Descriptors"].append(descriptors)
        return keypoints, descriptors

    def update_global_map(self):
        if self.frame_count == 0:
            self.get_kp_and_desc()
            self.bundle_adjuster.initialize()
        else:
            kp_curr, desc_curr = self.get_kp_and_desc()
            kp_prev, desc_prev = self.features["Keypoints"][-2], self.features["Descriptors"][-2]
            matches = self.match_features(desc_prev, desc_curr)

            if len(matches) < 8:
                print("Not enough matches found. Skipping frame.")
                return

            pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
            pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])

            F, inliers = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.FM_RANSAC, 1.0, 0.99)
            
            if F is None or inliers is None:
                print("Failed to compute Fundamental matrix. Skipping frame.")
                return

            inliers = inliers.ravel() == 1
            pts1_inliers = pts_prev[inliers]
            pts2_inliers = pts_curr[inliers]

            # Pass in features to our bundle adjuster class
            self.bundle_adjuster.update(pts1_inliers, pts2_inliers)
            
            if self.frame_count % 5 == 0:
                self.bundle_adjuster.optimize(frame_count=self.frame_count,
                                              features=np.array(pts1_inliers),
                                              global_map=self.global_map)

            # Update our global points at the end
            self.points_3d_global = self.bundle_adjuster.get_global_points()
            self.global_map = np.vstack([self.global_map, self.points_3d_global])  # Concatenate new points