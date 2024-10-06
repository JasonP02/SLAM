from video_processor import VideoProcessor
from slam import SLAM
from visualizers import Visualizer
from config import get_camera_matrix

def main():
    video_path = "path/to/your/video.mp4"
    camera_matrix = get_camera_matrix()

    video_processor = VideoProcessor(video_path)
    slam = SLAM(camera_matrix)
    visualizer = Visualizer()

    while True:
        frame = video_processor.get_next_frame()
        if frame is None:
            break

        global_points = slam.process_frame(frame)
        visualizer.update(frame, global_points)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_processor.release()
    cv2.destroyAllWindows()

    visualizer.show_3d_map(slam.map.get_global_points())

if __name__ == "__main__":
    main()