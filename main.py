import cv2
from slam import SLAM
from visualizers import Visualizer
from config import get_camera_matrix

def main():
    video_path = "path/to/your/video.mp4"
    camera_matrix = get_camera_matrix()

    slam = SLAM(camera_matrix)
    visualizer = Visualizer()

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        global_points = slam.process_frame(frame)
        visualizer.update(frame, global_points)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    visualizer.show_3d_map(slam.map.get_global_points())

if __name__ == "__main__":
    main()