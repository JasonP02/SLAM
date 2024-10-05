import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    @staticmethod
    def visualize_3d_map(points_3d):
        if points_3d.size == 0:
            print("No 3D points to display.")
            return

        # Filter outliers
        points_3d = filter_outliers(points_3d)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c='b', marker='o', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()