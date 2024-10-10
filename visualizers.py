import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Button, RadioButtons
from matplotlib.colors import LinearSegmentedColormap

class Visualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.scatter = None
        self.points_3d = None
        self.camera_poses = None
        self.color_mode = 'depth'
        self.show_cameras = False

    def show_3d_map(self, points_3d, camera_poses=None):
        if points_3d.size == 0:
            print("No 3D points to display.")
            return

        self.points_3d = points_3d
        self.camera_poses = camera_poses
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self._update_plot()

        # Add reset button
        reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        reset_button = Button(reset_ax, 'Reset View')
        reset_button.on_clicked(self.reset_view)

        # Add radio buttons for color mode
        color_mode_ax = plt.axes([0.025, 0.5, 0.15, 0.15])
        color_mode_radio = RadioButtons(color_mode_ax, ('Depth', 'Timestamp'))
        color_mode_radio.on_clicked(self._update_color_mode)

        # Add toggle button for camera poses
        camera_toggle_ax = plt.axes([0.025, 0.4, 0.15, 0.04])
        camera_toggle_button = Button(camera_toggle_ax, 'Toggle Cameras')
        camera_toggle_button.on_clicked(self._toggle_cameras)

        # Connect mouse events
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.press = None
        plt.show()

    def _update_plot(self):
        self.ax.clear()
        
        if self.color_mode == 'depth':
            colors = self.points_3d[:, 2]
            cmap = 'viridis'
            cbar_label = 'Depth'
        else:  # timestamp
            colors = np.arange(len(self.points_3d))
            cmap = 'plasma'
            cbar_label = 'Timestamp'

        self.scatter = self.ax.scatter(
            self.points_3d[:, 0], self.points_3d[:, 1], self.points_3d[:, 2],
            c=colors, cmap=cmap, s=5, alpha=0.5
        )
        
        if self.show_cameras and self.camera_poses is not None:
            self._plot_camera_poses()

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        cbar = self.fig.colorbar(self.scatter, ax=self.ax, label=cbar_label)
        cbar.set_alpha(1)
        # Remove the draw_all() call as it's not needed

        self.fig.canvas.draw_idle()

    def _plot_camera_poses(self):
        for pose in self.camera_poses:
            R, t = pose
            cam_pos = -R.T @ t
            self.ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='red', s=50, marker='o')
            
            # Plot camera orientation
            for i in range(3):
                axis = R.T[:, i]
                self.ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                               axis[0], axis[1], axis[2],
                               length=0.5, color=['r', 'g', 'b'][i])

    def _update_color_mode(self, label):
        self.color_mode = label.lower()
        self._update_plot()

    def _toggle_cameras(self, event):
        self.show_cameras = not self.show_cameras
        self._update_plot()

    def reset_view(self, event):
        self.ax.set_xlim3d(np.min(self.points_3d[:, 0]), np.max(self.points_3d[:, 0]))
        self.ax.set_ylim3d(np.min(self.points_3d[:, 1]), np.max(self.points_3d[:, 1]))
        self.ax.set_zlim3d(np.min(self.points_3d[:, 2]), np.max(self.points_3d[:, 2]))
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None: return
        scale = 0.9 if event.button == 'up' else 1.1
        x_min, x_max = ax.get_xlim3d()
        y_min, y_max = ax.get_ylim3d()
        z_min, z_max = ax.get_zlim3d()
        ax.set_xlim3d(x_min * scale, x_max * scale)
        ax.set_ylim3d(y_min * scale, y_max * scale)
        ax.set_zlim3d(z_min * scale, z_max * scale)
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        self.press = event.xdata, event.ydata

    def on_release(self, event):
        self.press = None
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.ax: return
        dx = event.xdata - self.press[0]
        dy = event.ydata - self.press[1]
        self.ax.azim += dx
        self.ax.elev += dy
        self.press = event.xdata, event.ydata
        self.fig.canvas.draw_idle()