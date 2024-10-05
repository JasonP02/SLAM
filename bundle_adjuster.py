# bundle_adjuster.py

import cv2
import numpy as np
from scipy.optimize import least_squares
from utils import Map, MapPoint, Keyframe

class BundleAdjuster:
    def __init__(self, K):
        self.K = K
        self.map = Map()
        self.points_3D_global = None
        self.frame_count = 0

    def initialize(self):
        initial_keyframe = Keyframe(
            id=self.frame_count,
            R=np.eye(3),
            t=np.zeros((3, 1)),
            keypoints=None,
            descriptors=None
        )
        self.map.add_keyframe(initial_keyframe)
    
    def update(self, pts1, pts2, E):
        _, R_new, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
        
        new_keyframe = Keyframe(
            id=self.frame_count,
            R=R_new,
            t=t,
            keypoints=pts2,
            descriptors=None
        )
        
        prev_keyframe = self.map.get_latest_keyframe()
        
        P1 = self.K @ np.hstack((prev_keyframe.R, prev_keyframe.t))
        P2 = self.K @ np.hstack((R_new, t))
        
        points_4D_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3D = cv2.convertPointsFromHomogeneous(points_4D_hom.T).reshape(-1, 3)
        
        for point_3d in points_3D:
            map_point = MapPoint(position=point_3d)
            self.map.add_map_point(map_point)
        
        if self.frame_count > 0:
            new_keyframe.R = prev_keyframe.R @ R_new
            new_keyframe.t = prev_keyframe.t + (prev_keyframe.R @ t)
        
        self.map.add_keyframe(new_keyframe)
        self.frame_count += 1
        
        self.points_3D_global = self.calculate_global_points(points_3D, new_keyframe)

    def calculate_global_points(self, points_3D, keyframe):
        return (keyframe.R @ points_3D.T).T + keyframe.t.T

    def optimize(self):
        if len(self.map.keyframes) < 2:
            return

        params = self.get_optimization_parameters()
        result = least_squares(
            self.reprojection_error,
            params,
            verbose=2
        )
        
        self.update_parameters_from_optimization(result.x)

    def get_optimization_parameters(self):
        params = []
        for kf in self.map.keyframes:
            R_vec, _ = cv2.Rodrigues(kf.R)
            params.extend(R_vec.flatten())
            params.extend(kf.t.flatten())
        return np.array(params)

    def update_parameters_from_optimization(self, optimized_params):
        n_keyframes = len(self.map.keyframes)
        for i in range(n_keyframes):
            start_idx = i * 6
            R_vec = optimized_params[start_idx:start_idx+3]
            t = optimized_params[start_idx+3:start_idx+6].reshape(3, 1)
            
            R, _ = cv2.Rodrigues(R_vec)
            self.map.keyframes[i].R = R
            self.map.keyframes[i].t = t

    def reprojection_error(self, params):
        # Implement reprojection error calculation
        # This is a placeholder implementation
        return np.zeros(1)

    def get_global_points(self):
        return self.points_3D_global