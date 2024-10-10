import numpy as np
import cv2
from feature_extractor import FeatureExtractor
from bundle_adjuster import BundleAdjuster
from utils import Map, Keyframe, MapPoint
from epipolar_geometry import EpipolarGeometry

class SLAM:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.map = Map()
        self.feature_extractor = FeatureExtractor()
        self.bundle_adjuster = BundleAdjuster(camera_matrix)
        self.frame_count = 0
        self.keyframe_interval = 100  # Perform bundle adjustment every n frames
        self.epipolar_geometry = EpipolarGeometry(camera_matrix)

    def process_frame(self, frame):
        if self.frame_count == 0:
            self._initialize(frame)
        else:
            self._update(frame)

        self.frame_count += 1

        if self.frame_count % self.keyframe_interval == 0:
            self._perform_bundle_adjustment()

        return self.map.get_global_points()

    def _initialize(self, frame):
        keypoints, descriptors = self.feature_extractor.extract(frame)
        initial_keyframe = self._create_keyframe(0, np.eye(3), np.zeros((3, 1)), keypoints, descriptors)
        self.map.add_keyframe(initial_keyframe)
        self._initialize_map_points(initial_keyframe)

    def _update(self, frame):
        prev_keyframe = self.map.keyframes[-1]
        keypoints, descriptors = self.feature_extractor.extract(frame)
        matches = self.feature_extractor.match(prev_keyframe.descriptors, descriptors)

        if not self._has_enough_matches(matches):
            return

        R, t = self._estimate_pose(prev_keyframe, keypoints, matches)
        if R is None or t is None:
            return

        new_keyframe = self._create_keyframe(self.frame_count, R, t, keypoints, descriptors)
        self._update_map_points(prev_keyframe, new_keyframe, matches)
        self.map.add_keyframe(new_keyframe)
        self._triangulate_new_points(prev_keyframe, new_keyframe, matches)

    def _has_enough_matches(self, matches, min_matches=8):
        if len(matches) < min_matches:
            print("Not enough matches found. Skipping frame.")
            return False
        return True

    def _estimate_pose(self, prev_keyframe, curr_keypoints, matches):
        return self.epipolar_geometry.estimate_pose(np.float32([prev_keyframe.keypoints[m.queryIdx].pt for m in matches]), np.float32([curr_keypoints[m.trainIdx].pt for m in matches]))

    def _create_keyframe(self, id, R, t, keypoints, descriptors):
        return Keyframe(
            id=id,
            R=R,
            t=t,
            keypoints=keypoints,
            descriptors=descriptors
        )

    def _initialize_map_points(self, keyframe):
        for i, kp in enumerate(keyframe.keypoints):
            map_point = MapPoint(position=np.array([0, 0, 0]))
            map_point.observations.append((keyframe.id, i))
            self.map.add_map_point(map_point)
            keyframe.map_points.append(map_point)

    def _update_map_points(self, prev_keyframe, new_keyframe, matches):
        # Revise ? 
        for m in matches:
            prev_feature_idx = m.queryIdx
            curr_feature_idx = m.trainIdx
            
            if prev_feature_idx < len(prev_keyframe.map_points):
                map_point = prev_keyframe.map_points[prev_feature_idx]
                map_point.observations.append((new_keyframe.id, curr_feature_idx))
                new_keyframe.map_points.append(map_point)
            else:
                new_map_point = MapPoint(position=np.array([0, 0, 0]))
                new_map_point.observations.append((prev_keyframe.id, prev_feature_idx))
                new_map_point.observations.append((new_keyframe.id, curr_feature_idx))
                self.map.add_map_point(new_map_point)
                prev_keyframe.map_points.append(new_map_point)
                new_keyframe.map_points.append(new_map_point)

    def _triangulate_new_points(self, prev_keyframe, new_keyframe, matches):
        points_3D = self.epipolar_geometry.triangulate_points(prev_keyframe, new_keyframe, np.float32([prev_keyframe.keypoints[m.queryIdx].pt for m in matches]), np.float32([new_keyframe.keypoints[m.trainIdx].pt for m in matches]))
        
        for i, point_3d in enumerate(points_3D):
            if i < len(prev_keyframe.map_points) and prev_keyframe.map_points[i] is not None:
                map_point = prev_keyframe.map_points[i]
                map_point.position = point_3d
            else:
                map_point = MapPoint(position=point_3d)
                self.map.add_map_point(map_point)
                
                map_point.observations.append((prev_keyframe.id, matches[i].queryIdx))
                
                if i >= len(prev_keyframe.map_points) or prev_keyframe.map_points[i] is None:
                    prev_keyframe.map_points.append(map_point)
            
            map_point.observations.append((new_keyframe.id, matches[i].trainIdx))
            new_keyframe.map_points.append(map_point)

    def _perform_bundle_adjustment(self):
        self.bundle_adjuster.optimize(self.map)