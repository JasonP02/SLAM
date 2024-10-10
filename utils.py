# utils.py

import numpy as np

class MapPoint:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float64)  # 3D coordinates as NumPy array
        self.observations = []    # List of (keyframe_id, feature_index)

class Keyframe:
    def __init__(self, id, R, t, keypoints, descriptors):
        self.id = id
        self.R = np.array(R, dtype=np.float64)
        self.t = np.array(t, dtype=np.float64).reshape(3, 1)
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.map_points = []  # Associated MapPoints
        # Note: Incomplete comment removed

class Map:
    def __init__(self):
        self.keyframes = []
        self.map_points = []
    
    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)
    
    def add_map_point(self, map_point):
        self.map_points.append(map_point)
    
    def get_global_points(self):
        return np.array([map_point.position for map_point in self.map_points])

def filter_outliers(points_3d, threshold=3):
    if points_3d.size == 0:
        return points_3d
    std_dev = np.std(points_3d, axis=0)
    std_dev[std_dev == 0] = 1  # Prevent division by zero
    z_scores = np.abs((points_3d - np.mean(points_3d, axis=0)) / std_dev)
    return points_3d[(z_scores < threshold).all(axis=1)]

