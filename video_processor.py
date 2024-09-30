# video_processor.py

import cv2
import numpy as np
from config import CAMERA_INTRINSICS
from visualizers import Visualizer
from scipy.optimize import least_squares

class VideoProcessor:
    """Processes the video to extract, track features, and build a 3D map."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.visualizer = Visualizer()

        self.features = {"Keypoints": [], "Descriptors": []}

        self.global_map = []
        self.points_3d_global = None

        self.pts_inliers1 = None
        self.pts_inlier2 = None

        self.orb = cv2.ORB_create()

        self.frame = None
        self.frame_count = 0

        self.R_total = np.eye(3)
        self.R = None
        self.t_total = np.zeros((3, 1))
        self.t = None
        self.K = np.array([
            [CAMERA_INTRINSICS['fx'], 0, CAMERA_INTRINSICS['cx']],
            [0, CAMERA_INTRINSICS['fy'], CAMERA_INTRINSICS['cy']],
            [0, 0, 1]
        ])


    def process_video(self):
        cap = cv2.VideoCapture(self.filepath)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.filepath}")
        
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
        return matches[:50]  # Return top 50 matches

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

            # This function appends the current shi-tomasi features, and orb descriptors to our features dict.
            self.get_kp_and_desc() # Do this for frame i (0 index)

            self.frame = self.visualizer.draw_features(self.frame, self.features["Keypoints"][-1])

            self.P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        else:
            # Get our features for the current and previous frame
            kp_curr, desc_curr = self.get_kp_and_desc()
            kp_prev, desc_prev = self.features["Keypoints"][-2], self.features["Descriptors"][-2]
            i = self.frame_count
            # Perform matching for bundle ajustment (later)
            matches = self.match_features(desc_prev, desc_curr)

            if len(matches) < 8:
                print("Not enough matches found. Skipping frame.")
                return

            # Convert these to useful datatypes
            pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
            pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])

            # Inliers is used to find the most relevant kps
            F, inliers = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.FM_RANSAC, 1.0, 0.99)
            
            if F is None or inliers is None:
                print("Failed to compute Fundamental matrix. Skipping frame.")
                return

            # Filtering... 
            inliers = inliers.ravel() == 1
            self.pts1_inliers = pts_prev[inliers]
            self.pts2_inliers = pts_curr[inliers]

            # Essential matrix to get R, t
            E, _ = cv2.findEssentialMat(self.pts1_inliers, self.pts2_inliers, self.K)
            _, self.R, self.t, _ = cv2.recoverPose(E, self.pts1_inliers, self.pts2_inliers, self.K)

            # Projection matrix which maps 2D -> 3D
            P1 = self.K @ np.hstack((self.R_total, self.t_total))
            P2 = self.K @ np.hstack((self.R, self.t))

            # Combine R (3x3) and t (3x1) to a 4D matrix
            points_4d_hom = cv2.triangulatePoints(P1, P2, self.pts1_inliers.T, self.pts2_inliers.T)
            
            # Use the SE(3) representation to get our new 3D estimation for frame i
            points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1, 3)

            # Updating R, t
            self.R_total = self.R @ self.R_total
            self.t_total += self.R_total @ self.t

            # Not sure...
            self.points_3d_global = (self.R_total @ points_3d.T).T + self.t_total.T
            
            # self.bundle_adjustment()

            # Perhaps we should update this rarely (e.g. several new features)
            self.global_map.append(self.points_3d_global)


    def bundle_adjustment(self):
        # Need to redo this...
        pass