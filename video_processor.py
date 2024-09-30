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

        self.params = []

        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))
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
            _, R, t, _ = cv2.recoverPose(E, self.pts1_inliers, self.pts2_inliers, self.K)

            # Projection matrix which maps 2D -> 3D
            P1 = self.K @ np.hstack((self.R_total, self.t_total))
            P2 = self.K @ np.hstack((self.R, self.t))

            # Combine R (3x3) and t (3x1) to a 4D matrix
            points_4d_hom = cv2.triangulatePoints(P1, P2, self.pts1_inliers.T, self.pts2_inliers.T)
            
            # Use the SE(3) representation to get our new 3D estimation for frame i
            points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1, 3)

            # Updating R, t
            self.R_total = R @ self.R_total
            self.t_total += self.R_total @ t

            # Convert R to Rodrigues vector
            R_vec, _ = cv2.Rodrigues(R)

            # Ensure params is a 1D array
            self.params.append(np.concatenate([R_vec.flatten(), t.flatten()]))

            # Not sure...
            self.points_3d_global = (self.R_total @ points_3d.T).T + self.t_total.T
            
            if self.frame_count % 5 == 0:
                '''
                Every 10 frames we perform optimization over all R and t, giving us a more reliable map
                '''
                optimized_R, optimized_t = self.bundle_adjustment(len(matches), R, t, points_3d, )
                self.R_total = optimized_R
                self.t_total = optimized_t

            # Perhaps we should update this rarely (e.g. several new features)
            self.global_map.append(self.points_3d_global)


    def project_points(points_3d, R, t, K): 
        '''
        This gives us our point in the 2d space for error calulation
        '''
    
        points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        points_2d_hom = K @ (R @ points_3d_hom.T + t)
        points_2d = points_2d_hom[:2] / points_2d_hom[2]  # normalize by the third component
        return points_2d.T  # return 2D points (u, v)

    def reprojection_error_full(self, num_points, points_2d):
        '''
        Error calculation based on the estimated rotation matrix, and ground 
        truth position vector (our features)
        '''
        num_cameras = self.num_frames
        R_vecs = self.params[:num_cameras*3].reshape((num_cameras, 3))
        t_vecs = self.params[num_cameras*3:num_cameras*6].reshape((num_cameras, 3, 1))
        points_3d = self.params[num_cameras*6:].reshape((num_points, 3))

        error = []
        for i in range(num_cameras):
            R, _ = cv2.Rodrigues(R_vecs[i])
            t = t_vecs[i]
            points_proj = self.project_points(points_3d, R, t, self.K)
            error.append(points_2d[i] - points_proj)

        return np.concatenate(error).ravel()

    def optimize_bundle_adjustment(self, points_3d, points_2d, reprojection_error):
        '''
        Scipy optimizer. Might have to revise this, but its a nonlinear least squares optimizer
        '''
        result = least_squares(reprojection_error, self.params, args=(points_3d, points_2d, self.K))

        # unpack optimized R and t
        R_opt, t_opt = result.x[:3], result.x[3:6]
        R_opt, _ = cv2.Rodrigues(R_opt)
        t_opt = t_opt.reshape(3, 1)
        
        return R_opt, t_opt

    def bundle_adjustment(self, num_features, R, t, points_3d):
        # Bundle adjustment is the following optimization:
        # x_3d = f(R,K,x_2d)
        # obj: min( x_true - x_proj )^2
        # thus, we optimize for R,K, x_proj
        points_2d = self.project_points(R,t,self.K)
        error = self.reprojection_error_full(num_features, points_2d)
        R_opt, t_opt = self.optimize_bundle_adjustment(points_3d, points_2d, error)
        
        return R_opt, t_opt