# main.py

from video_processor import VideoProcessor
from visualizers import Visualizer
import os


def main():
    video_path = r"C:\Users\jason\Dropbox\Python\SLAM\videos\robo-vid.MP4"
    
    if not os.path.isfile(video_path):
        print(f"Error: Video file does not exist at the specified path: {video_path}")
        return
    
    processor = VideoProcessor(video_path)

    try:
        # Process the video and retrieve the global 3D map
        global_map = processor.process_video(video_path)
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
        Visualizer.visualize_3d_map(global_map)
    else:
        print("No 3D points were generated.")

if __name__ == "__main__":
    main()
