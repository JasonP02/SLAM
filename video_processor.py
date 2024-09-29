# video_processor.py

import cv2
import numpy as np
from config import CAMERA_INTRINSICS
from extractors import ShiTomasiExtractor
from trackers import FeatureTracker
from visualizers import Visualizer
from scipy.optimize import least_squares

class VideoProcessor:
    """Processes the video to extract, track features, and build a 3D map."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.extractor = ShiTomasiExtractor(**{
            'maxCorners': 300,
            'qualityLevel': 0.02,
            'minDistance': 10
        })
        self.tracker = FeatureTracker()
        self.visualizer = Visualizer()
        self.global_map = []
        self.points_3d_global = None
        self.std = 0
        self.mean = 0
        self.pts_inliers1 = None
        self.pts_inlier2 = None
        self.frame = None
        self.frame_count = 0
        self.R_total = np.eye(3)
        self.R = None
        self.t_total = np.zeros((3, 1))
        self.t = None
        self.P1 = None
        self.K = np.array([
            [CAMERA_INTRINSICS['fx'], 0, CAMERA_INTRINSICS['cx']],
            [0, CAMERA_INTRINSICS['fy'], CAMERA_INTRINSICS['cy']],
            [0, 0, 1]
        ])
        self.pts1_inliers = None
        self.pts2_inliers = None



    def process_video(self):
        cap = cv2.VideoCapture(self.filepath)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.filepath}")
        try:
            while True:
                ret, self.frame = cap.read()
                
                self.update_global_map() # The good stuff is here
                
                if not ret:
                    print("End of video or cannot read the frame.")
                    break

                # Display the processed frame
                if self.visualizer.display_frame(self.frame) == ord('q'):
                    print("Processing stopped by user.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"Total frames processed: {self.frame_count}")
        return np.vstack(self.global_map) if self.global_map else np.array([])



    def update_global_map(self):
        if self.frame_count == 0:
            # Initial feature extraction    
            features = self.extractor.extract_features(self.frame)
            if features.size > 0:
                self.tracker.initialize(self.frame, features)
            print(f"Frame {self.frame_count}: {features.shape[0]} features detected.")
            self.frame = self.visualizer.draw_features(self.frame, features)
            self.P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        else:
            # Track features
            flow_prev, flow_curr = self.tracker.track(self.frame)
            if flow_prev.size > 0 and flow_curr.size > 0:
                print(f"Frame {self.frame_count}: {flow_curr.shape[0]} features tracked.")
                self.frame = self.visualizer.draw_features(self.frame, flow_curr)
                self.frame = self.visualizer.draw_flow(self.frame, flow_prev, flow_curr)

                if flow_prev.shape[0] >= 8:
                    # Compute Fundamental matrix to get our bounds
                    F, inliers = cv2.findFundamentalMat(
                        flow_prev, flow_curr, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.9
                    )
                    if F is not None and inliers is not None:
                        inliers = inliers.ravel() == 1
                        self.pts1_inliers = flow_prev[inliers]
                        self.pts2_inliers = flow_curr[inliers]

                        if self.pts1_inliers.shape[0] >= 8:
                            # Compute Essential matrix and recover pose in order to get the rotation matrix
                            E, _ = cv2.findEssentialMat(self.pts1_inliers, self.pts2_inliers, self.K)
                            _, self.R, self.t, _ = cv2.recoverPose(E, self.pts1_inliers, self.pts2_inliers, self.K)

                            # Update projection matrices 
                            # The projection matrix is found by relating the camera parameters to the estimated rotation matrix and position vector
                            self.P1 = self.K @ np.hstack((self.R_total, self.t_total))
                            P2 = self.K @ np.hstack((self.R, self.t))

                            # Triangulate points
                            points_4d_hom = cv2.triangulatePoints(self.P1, P2, self.pts1_inliers.T, self.pts2_inliers.T)
                            points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1, 3)

                            # Update global pose
                            self.R_total = self.R @ self.R_total
                            self.t_total += self.R_total @ self.t

                            # Transform points to global frame
                            self.points_3d_global = (self.R_total @ points_3d.T).T + self.t_total.T
                             
                            self.bundle_adjustment() # Perform bundle adjustment on our points before saving

                            self.global_map.append(self.points_3d_global)
            else:
                # If tracking fails, re-initialize features
                print(f"Frame {self.frame_count}: Tracking failed. Re-initializing features.")
                features = self.extractor.extract_features(self.frame)
                if features.size > 0:
                    self.tracker.initialize(self.frame, features)
                self.frame = self.visualizer.draw_features(self.frame, features)
                self.P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.frame_count += 1


    def bundle_adjustment(self):

        # Reproject 3D points to 2D using current estimates for the first view (previous frame)
        rvec1, _ = cv2.Rodrigues(self.R_total)
        tvec1 = self.t_total.flatten()
        points_reprojected1, J1 = cv2.projectPoints(
            objectPoints=self.points_3d_global,
            rvec=rvec1,
            tvec=tvec1,
            cameraMatrix=self.K,
            distCoeffs=None
        )

        # Reproject 3D points to 2D using current estimates for the second view (current frame)
        rvec2, _ = cv2.Rodrigues(self.R_total @ self.R)  # Combined rotation for the second frame
        tvec2 = (self.t_total + self.R_total @ self.t).flatten()  # Translation for the second frame
        points_reprojected2, J2 = cv2.projectPoints(
            objectPoints=self.points_3d_global,
            rvec=rvec2,
            tvec=tvec2,
            cameraMatrix=self.K,
            distCoeffs=None
        )

        # Compute the residual error (observed - projected) for both views
        r1 = (self.pts1_inliers - points_reprojected1.reshape(-1, 2)).flatten()  # Residual for the first view
        r2 = (self.pts2_inliers - points_reprojected2.reshape(-1, 2)).flatten()  # Residual for the second view

        # Combine residuals into one vector
        residuals = np.hstack([r1, r2])

        # Stack Jacobians for both views
        J_combined = np.vstack([J1, J2])

        # Compute the Jacobian transpose times the residuals (J_combined.T @ residuals)
        JT_residuals = J_combined.T @ residuals

        # Compute the approximate Hessian (J_combined.T @ J_combined)
        JTJ_combined = J_combined.T @ J_combined

        # Apply Levenberg-Marquardt damping parameter
        lam = 1e-3
        damped_matrix = JTJ_combined + lam * np.eye(JTJ_combined.shape[0])

        # Solve for parameter updates (Δx)
        delta_x = np.linalg.solve(damped_matrix, -JT_residuals)

        # Update camera parameters and 3D points using delta_x
        # Interpret and apply delta_x to update R, t, and points_3d

