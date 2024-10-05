# utils.py
import numpy as np



class MapPoint:
    def __init__(self, position):
        self.position = position  # 3D coordinates
        self.observations = []    # List of (keyframe_id, feature_index)

class Keyframe:
    def __init__(self, id, pose, keypoints, descriptors):
        self.id = id
        self.pose = pose  # Rotation and translation
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.map_points = []  # Associated MapPoints

class Map:
    def __init__(self):
        self.keyframes = []
        self.map_points = []

    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)

    def add_map_point(self, map_point):
        self.map_points.append(map_point)

def filter_outliers(points_3d, threshold=3):
    if points_3d.size == 0:
        return points_3d

    std_dev = points_3d.std(axis=0)
    std_dev[std_dev == 0] = 1  # Prevent division by zero

    z_scores = np.abs((points_3d - points_3d.mean(axis=0)) / std_dev)
    filtered_points = points_3d[(z_scores < threshold).all(axis=1)]
    
    return filtered_points