# bundle_adjuster.py

import cv2
import numpy as np
from scipy.optimize import least_squares

class BundleAdjuster:
    def __init__(self, K):
        self.K = K
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))
        self.R_and_t = []
        self.points_3d_global = None

    def initialize(self):
        self.P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    def update(self, pts1, pts2):
        E, _ = cv2.findEssentialMat(pts1, pts2, self.K)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)

        P1 = self.K @ np.hstack((self.R_total, self.t_total))
        P2 = self.K @ np.hstack((R, t))

        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1, 3)

        R_new, _ = cv2.Rodrigues(R)
        self.R_total = R_new @ self.R_total
        self.t_total += self.R_total @ t

        R_vec, _ = cv2.Rodrigues(R_new)
        self.R_and_t.append(np.concatenate([R_vec.flatten(), t.flatten()]))

        self.points_3d_global = (self.R_total @ points_3d.T).T + self.t_total.T

    def optimize(self):
        optimized_R, optimized_t = self.bundle_adjustment()
        self.R_total = optimized_R
        self.t_total = optimized_t

    def project_points(self, R, t):
        points_3d_hom = np.hstack((self.points_3d_global, np.ones((self.points_3d_global.shape[0], 1))))
        points_2d_hom = self.K @ (R @ points_3d_hom.T + t)
        points_2d = points_2d_hom[:2] / points_2d_hom[2]
        return points_2d.T

    def reprojection_error(self, params, points_2d):
        R = cv2.Rodrigues(params[:3])[0]
        t = params[3:6].reshape(3, 1)
        points_proj = self.project_points(R, t)
        return (points_2d - points_proj).ravel()

    def bundle_adjustment(self):
        initial_params = np.concatenate([cv2.Rodrigues(self.R_total)[0].ravel(), self.t_total.ravel()])
        result = least_squares(self.reprojection_error, initial_params, args=(self.project_points(self.R_total, self.t_total),))
        
        R_opt, _ = cv2.Rodrigues(result.x[:3])
        t_opt = result.x[3:6].reshape(3, 1)
        
        return R_opt, t_opt

    def get_global_points(self):
        return self.points_3d_global