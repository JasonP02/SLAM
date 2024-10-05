# Bundle_Adjuster.py

import cv2
import numpy as np
from scipy.optimize import least_squares

class BundleAdjuster:
    def __init__(self, K):
        self.K = K
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))
        self.R_and_t = []
        self.points_3D_global = None
        self.features = None
        self.global_map = None
        self.frame_count = None 

    def initialize(self):
        self.P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    def update(self, pts1, pts2): 
        '''
        for each frame we perform this update
        '''
        E, _ = cv2.findEssentialMat(pts1, pts2, self.K)
        _, R_new, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
        print(f'R {self.R_total.shape} and t {self.t_total.shape}')
        P1 = self.K @ np.hstack((self.R_total, self.t_total))
        P2 = self.K @ np.hstack((R_new, t))

        points_4D_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3D = cv2.convertPointsFromHomogeneous(points_4D_hom.T).reshape(-1, 3)

        self.R_total = self.R_total @ R_new 
        self.t_total += self.R_total @ t  # Apply new translation in global frame


        R_vec, _ = cv2.Rodrigues(R_new)
        self.R_and_t.append(np.concatenate([R_vec.flatten(), t.flatten()]))

        self.points_3D_global = (self.R_total @ points_3D.T).T + self.t_total.T

    def optimize(self, frame_count, features, global_map):
        '''
        optimize for every nth frame (tbd)
        !! note that the update of R,t_total is probably incorrect
        '''
        self.global_map = global_map
        self.frame_count = frame_count
        self.features = features
        optimized_R, optimized_t = self.bundle_adjustment()
        print(optimized_R)
        print(optimized_t)
        self.R_total = optimized_R
        # After updating R_total
        U, _, Vt = np.linalg.svd(self.R_total)
        self.R_total = U @ Vt

        self.t_total = optimized_t

    def project_points(self, points_3D, R, t):
        '''
        project 3D points to 2D using given R and t
        '''
        points_3D_hom = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))
        RT = np.hstack((R, t))  # combine rotation and translation into one matrix
        points_2D_hom = self.K @ RT @ points_3D_hom.T  # apply projection
        points_2D = points_2D_hom[:2] / points_2D_hom[2]
        return points_2D.T


    def reprojection_error(self, params, points_3D, points_2D):
        '''
        Error calculation for optimizer across all frames
        '''
        total_error = []
        RT_np = params.reshape((self.frame_count, 6))  # split parameters into R and t vectors

        for i in range(self.frame_count):
            # Validate the structure of points_2D
            if not isinstance(points_2D, (list, np.ndarray)):
                raise TypeError("points_2D should be a list or numpy array indexed by frame.")

            # Extract R and t for this frame from params
            R_vec = RT_np[i, 0:3]
            t = RT_np[i, 3:6].reshape(3, 1)  # reshape t to (3, 1)

            # Convert R_vec to rotation matrix
            R, _ = cv2.Rodrigues(R_vec)

            # Project 3D points to 2D for this frame
            points_proj = self.project_points(points_3D, R, t)
            
            # Calculate error for this frame
            frame_error = (points_2D[i] - points_proj).ravel()
            total_error.append(frame_error)

        # Concatenate errors from all frames
        return np.concatenate(total_error)
    
    def bundle_adjustment(self):
        '''
        Scipy nonlinear least squares optimizer; increases accuracy of R and t
        '''
        if not self.R_and_t:
            raise ValueError("No rotation and translation data available for optimization.")

        # Initial parameters: concatenate [R1, t1, R2, t2, ..., Rn, tn] as a 1D array
        initial_params = np.array(self.R_and_t).ravel()  # Shape: (6 * frame_count,)

        # Call least_squares with reprojection_error as the objective function
        result = least_squares(
            self.reprojection_error,        # The objective function
            initial_params,                 # Initial guesses for parameters
            args=(self.global_map, self.features),  # Additional arguments
            verbose=2                        # Optional: for more detailed output
        )

        # Reshape the optimized parameters back into [R1, t1, R2, t2, ..., Rn, tn]
        optimized_params = result.x.reshape((self.frame_count, 6))  # Shape: (frame_count, 6)

        # Split into rotation vectors and translation vectors
        R_opt_vecs = optimized_params[:, :3]  # Shape: (frame_count, 3)
        t_opt_vecs = optimized_params[:, 3:]  # Shape: (frame_count, 3)

        # Convert rotation vectors back to rotation matrices
        R_opt = [cv2.Rodrigues(R_vec)[0] for R_vec in R_opt_vecs]

        t_opt = [np.array(t_vec).reshape(3, 1) for t_vec in t_opt_vecs]
        # Initialize total rotation and translation
        R_total = np.eye(3)  # Identity matrix to accumulate rotations
        t_total = np.zeros((3, 1))  # Zero vector for translations

        # Accumulate rotations and translations
        for i in range(self.frame_count):
            # Accumulate rotation by multiplying the current rotation with the previous total
            R_total = R_total @ R_opt[i]
            
            # Transform the current translation vectort_total += R_total @ t_opt[i]t_total += R_total @ t_opt[i] by the current total rotation

            print(f"t_opt[{i}] shape: {t_opt[i].shape}")
            t_total += R_total @ t_opt[i]


        # Update the total R and t
        print(f' t: {t_total.shape}')
        self.R_total = R_total
        self.t_total = t_total

        return R_total, t_total

    def get_global_points(self):
        '''
        good practice to make this its own method
        '''
        return self.points_3D_global
