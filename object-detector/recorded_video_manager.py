import sys; sys.path.append("../common")
from text_to_speech import text_to_speech
import cv2
import os
from time import sleep
import numpy as np

class RecordedVideoManager:

    def __init__(self, mada_config_dict):

        in_video_dir = mada_config_dict.get("in_video_dir", "../videos")

        in_rgb_video = mada_config_dict.get("in_rgb_video", "rgb_4.mp4")
        in_rgb_video_path = os.path.join(in_video_dir, f"{in_rgb_video}")

        if not os.path.isfile(in_rgb_video_path):
            print(f"Cannot find file {in_rgb_video_path}")
            audio_error_message = "Cannot find color video file"
            text_to_speech(audio_error_message)
            return

        self.rgb_cap = cv2.VideoCapture(in_rgb_video_path)

        fps = self.rgb_cap.get(cv2.CAP_PROP_FPS)
        print("fps=", fps)
        self.image_width = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        in_depth_video = in_rgb_video.replace("rgb", "depth")
        in_depth_video = in_depth_video.replace(".mp4", ".mkv")
        in_depth_video_path = os.path.join(in_video_dir, f"{in_depth_video}")
        if not os.path.isfile(in_depth_video_path):
            print(f"Cannot find file {in_depth_video_path}")
            audio_error_message = "Cannot find depth video file"
            text_to_speech(audio_error_message)
            self.depth_cap = None
        else:
            self.depth_cap = cv2.VideoCapture(in_depth_video_path,
                                     apiPreference=cv2.CAP_FFMPEG,
                                     params=[
                                         # BGR conversion turned OFF, decoded frame will be returned in its original format.
                                         # Multiplanar formats are not supported by the backend. Only GRAY8/GRAY16LE pixel formats have been tested.
                                         cv2.CAP_PROP_CONVERT_RGB,
                                         0,  # false
                                     ],
                                     )

        print(f"[INFO] Start reading files '{in_rgb_video_path}' and '{in_depth_video_path}'...")

        self.n_frame = 0
        self.init_frame = mada_config_dict.get("init_frame", 0)
        self.sleep_time = mada_config_dict.get("sleep_time", 0)

    def get_color_and_depth_images(self):

        self.n_frame += 1
        print(f"\n[FRAME] {self.n_frame}")
        rgb_ret_ok, color_image = self.rgb_cap.read()

        if self.depth_cap is None:
            depth_image = None
        else:
            depth_ret_ok, depth_image = self.depth_cap.read()
        # depth_image = np.array(depth_image, dtype=np.uint16)
        # print("depth", depth_image.shape, depth_image.dtype)  # (480, 640, 3) uint8 en mp4v/.mp4 y H264/.avi

        if self.n_frame < self.init_frame:
            skip_frame = True
        else:
            sleep(self.sleep_time)
            skip_frame = False

        return skip_frame, color_image, depth_image

    def stop(self):

        self.rgb_cap.release()

        if self.depth_cap is not None:
           self.depth_cap.release()

