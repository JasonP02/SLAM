# visualizers.py

import cv2
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """Handles visualization of features, optical flow, and 3D point cloud."""
    @staticmethod
    def draw_features(frame, features, color=(0, 255, 0), radius=3):
        for x, y in features:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
        return frame

    @staticmethod
    def draw_flow(frame, prev_features, current_features, color=(0, 0, 255)):
        for (x_prev, y_prev), (x_curr, y_curr) in zip(prev_features, current_features):
            cv2.arrowedLine(
                frame,
                (int(x_prev), int(y_prev)),
                (int(x_curr), int(y_curr)),
                color=color,
                thickness=1,
                tipLength=0.03
            )
        return frame

    @staticmethod
    def display_frame(frame, window_name='Processed Frame'):
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF

    @staticmethod
    def visualize_3d_map(points_3d):
        if points_3d.size == 0:
            print("No 3D points to display.")
            return

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        ax.scatter(X, Y, Z, c='b', marker='o', s=20, alpha=0.6)

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('3D Point Cloud Visualization')

        # Set equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()
