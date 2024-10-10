from video_processor import VideoProcessor
from slam import SLAM
from visualizers import Visualizer
from config import get_camera_matrix
import cv2
import numpy as np

def main():
    video_path = "videos/robo-vid.MP4"
    camera_matrix = get_camera_matrix()

    video_processor = VideoProcessor(video_path)
    slam = SLAM(camera_matrix)
    visualizer = Visualizer()

    while True:
        frame = video_processor.get_next_frame()
        if frame is None:
            break

        slam.process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_processor.release()
    cv2.destroyAllWindows()

    # Show the final 3D map
    final_global_points = slam.map.get_global_points()
    camera_poses = [(kf.R, kf.t) for kf in slam.map.keyframes]
    visualizer.show_3d_map(final_global_points, camera_poses)

if __name__ == "__main__":
    main()