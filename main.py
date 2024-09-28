# main.py

from video_processor import VideoProcessor
from visualizers import Visualizer
import os

def display_3d_map(points_3d):
    """Visualize the accumulated 3D points."""
    Visualizer.visualize_3d_map(points_3d)

def main():
    # Specify the correct path to your video file
    # Use raw strings (prefix with 'r') or double backslashes '\\' in Windows paths
    video_path = r"C:\Users\jason\Dropbox\Python\SLAM\videos\robo-vid.MP4"
    # Alternatively, use forward slashes '/'
    # video_path = "C:/Users/jason/Dropbox/Python/SLAM/videos/robo-vid.MP4"

    # Verify that the video file exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file does not exist at the specified path: {video_path}")
        return

    # Create an instance of VideoProcessor
    processor = VideoProcessor(video_path)

    try:
        # Process the video and retrieve the global 3D map
        global_map = processor.process_video()
    except IOError as e:
        print(e)
        return
    except KeyboardInterrupt:
        print("Processing interrupted by user.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Display the 3D map
    if global_map.size > 0:
        display_3d_map(global_map)
    else:
        print("No 3D points were generated.")

if __name__ == "__main__":
    main()
