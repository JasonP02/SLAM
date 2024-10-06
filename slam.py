import numpy as np
from feature_extractor import FeatureExtractor
from bundle_adjuster import BundleAdjuster
from utils import Map, Keyframe

class SLAM:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.map = Map()
        self.feature_extractor = FeatureExtractor()
        self.bundle_adjuster = BundleAdjuster(camera_matrix)
        self.frame_count = 0

    def process_frame(self, frame):
        if self.frame_count == 0:
            self.initialize(frame)
        else:
            self.update(frame)

        self.frame_count += 1

        if self.frame_count % 5 == 0:
            self.bundle_adjuster.optimize(self.map)

        return self.map.get_global_points()

    def initialize(self, frame):
        keypoints, descriptors = self.feature_extractor.extract(frame)
        initial_keyframe = Keyframe(
            id=self.frame_count,
            R=np.eye(3),
            t=np.zeros((3, 1)),
            keypoints=keypoints,
            descriptors=descriptors
        )
        self.map.add_keyframe(initial_keyframe)

    def update(self, frame):
        prev_keyframe = self.map.get_latest_keyframe()
        keypoints, descriptors = self.feature_extractor.extract(frame)
        matches = self.feature_extractor.match(prev_keyframe.descriptors, descriptors)

        if len(matches) < 8:
            return

        pts_prev = np.float32([prev_keyframe.keypoints[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts_prev, pts_curr, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            return

        _, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, self.camera_matrix)

        new_keyframe = Keyframe(
            id=self.frame_count,
            R=prev_keyframe.R @ R,
            t=prev_keyframe.t + (prev_keyframe.R @ t),
            keypoints=keypoints,
            descriptors=descriptors
        )

        self.map.add_keyframe(new_keyframe)
        self.bundle_adjuster.triangulate_new_points(self.map, pts_prev, pts_curr)