# video_processor.py

import cv2

class VideoProcessor:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.frame_count = 0

    def get_next_frame(self):
        ret, frame = self.video.read()
        if ret:
            self.frame_count += 1
            return frame
        return None

    def release(self):
        self.video.release()

    def get_frame_count(self):
        return self.frame_count