# bundle_adjuster.py

import cv2
import numpy as np
from scipy.optimize import least_squares
from utils import Map, MapPoint, Keyframe
class BundleAdjuster:
    def __init__(self, K):
        self.K = K
        self.local_window_size = 25  # Number of recent keyframes to consider
        self.min_observations = 3  # Minimum number of observations for a map point
        self.max_reprojection_error = 5.0  # Maximum allowed reprojection error (in pixels)
        
    def optimize(self, map):
        if len(map.keyframes) < 2:
            return

        keyframes_to_optimize, map_points_to_optimize = self.select_keyframes_and_map_points(map)

        if len(keyframes_to_optimize) < 2 or len(map_points_to_optimize) < 10:
            print(f"Not enough keyframes ({len(keyframes_to_optimize)}) or map points ({len(map_points_to_optimize)}) for optimization")
            return

        params = self.get_optimization_parameters(keyframes_to_optimize, map_points_to_optimize)
        result = least_squares(
            self.reprojection_error,
            params,
            args=(keyframes_to_optimize, map_points_to_optimize),
            verbose=2,
            method='trf',
            ftol=1e-4,
            xtol=1e-4,
            gtol=1e-4,
            max_nfev=100
        )
        
        self.update_parameters_from_optimization(result.x, keyframes_to_optimize, map_points_to_optimize)

    def select_keyframes_and_map_points(self, map):
        keyframes_to_optimize = map.keyframes[-self.local_window_size:]
        print(f"Keyframes to optimize: {len(keyframes_to_optimize)}")
        
        map_points_to_optimize = set()
        for kf in keyframes_to_optimize:
            map_points_to_optimize.update(kf.map_points)
        print(f"Initial map points to optimize: {len(map_points_to_optimize)}")
        
        filtered_map_points = []
        for mp in map_points_to_optimize:
            observations = len(mp.observations)
            reprojection_error = self.compute_reprojection_error(mp, keyframes_to_optimize)
            print(f"Map point - Observations: {observations}, Reprojection error: {reprojection_error}")
            if observations >= self.min_observations and reprojection_error < self.max_reprojection_error:
                filtered_map_points.append(mp)
        
        print(f"Filtered map points to optimize: {len(filtered_map_points)}")
        
        return keyframes_to_optimize, filtered_map_points
    def compute_reprojection_error(self, map_point, keyframes):
        if not map_point.observations:
            return float('inf')
        
        total_error = 0
        valid_observations = 0
        for kf_id, feature_idx in map_point.observations:
            keyframe = next((kf for kf in keyframes if kf.id == kf_id), None)
            if keyframe is None:
                print(f"Warning: Keyframe {kf_id} not found in optimization window")
                continue
            
            if feature_idx >= len(keyframe.keypoints):
                print(f"Warning: Feature index {feature_idx} out of range for keyframe {kf_id} (total keypoints: {len(keyframe.keypoints)})")
                continue
            
            projected_point, _ = cv2.projectPoints(map_point.position[np.newaxis], 
                                                   keyframe.R, keyframe.t, self.K, None)
            observed_point = np.array(keyframe.keypoints[feature_idx].pt)
            error = np.linalg.norm(projected_point.flatten() - observed_point)
            total_error += error
            valid_observations += 1
        
        if valid_observations == 0:
            print(f"Warning: No valid observations for map point")
            return float('inf')
        
        return total_error / valid_observations
    def get_optimization_parameters(self, keyframes, map_points):
        params = []
        for kf in keyframes:
            R_vec, _ = cv2.Rodrigues(kf.R)
            params.extend(R_vec.flatten())
            params.extend(kf.t.flatten())
        
        for mp in map_points:
            params.extend(mp.position.flatten())
        
        return np.array(params)

    def update_parameters_from_optimization(self, optimized_params, keyframes, map_points):
        n_keyframes = len(keyframes)
        keyframe_params = optimized_params[:n_keyframes * 6].reshape(-1, 6)
        map_point_params = optimized_params[n_keyframes * 6:].reshape(-1, 3)
        
        for i, keyframe in enumerate(keyframes):
            keyframe.R, _ = cv2.Rodrigues(keyframe_params[i, :3])
            keyframe.t = keyframe_params[i, 3:].reshape(3, 1)
        
        for i, map_point in enumerate(map_points):
            map_point.position = map_point_params[i]

    def reprojection_error(self, params, keyframes, map_points):
        n_keyframes = len(keyframes)
        keyframe_params = params[:n_keyframes * 6].reshape(-1, 6)
        map_point_params = params[n_keyframes * 6:].reshape(-1, 3)
        
        errors = []
        
        for i, keyframe in enumerate(keyframes):
            R, _ = cv2.Rodrigues(keyframe_params[i, :3])
            t = keyframe_params[i, 3:].reshape(3, 1)
            
            for j, map_point in enumerate(map_points):
                point_3d = map_point_params[j]
                projected_point, _ = cv2.projectPoints(point_3d[np.newaxis], R, t, self.K, None)
                projected_point = projected_point.flatten()
                
                observation = next((obs for obs in map_point.observations if obs[0] == keyframe.id), None)
                
                if observation is not None:
                    feature_idx = observation[1]
                    observed_point = np.array(keyframe.keypoints[feature_idx].pt)
                    error = projected_point - observed_point
                    errors.extend(error)
        
        return np.array(errors)