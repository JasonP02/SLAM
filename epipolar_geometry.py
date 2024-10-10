import cv2
import numpy as np
from utils import MapPoint

class EpipolarGeometry:
    def __init__(self, camera_matrix):
        self.K = camera_matrix

    def triangulate_points(self, kf1, kf2, pts1, pts2):
        P1 = self.K @ np.hstack((kf1.R, kf1.t))
        P2 = self.K @ np.hstack((kf2.R, kf2.t))
        
        points_4D_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3D = cv2.convertPointsFromHomogeneous(points_4D_hom.T).reshape(-1, 3)
        
        return points_3D

    def estimate_pose(self, pts1, pts2):
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            print("Failed to compute Essential matrix.")
            return None, None

        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        return R, t