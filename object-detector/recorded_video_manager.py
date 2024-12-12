import sys; sys.path.append("../common")
from text_to_speech import text_to_speech
import cv2
import os
from time import sleep, time

class RecordedVideoManager:

    def __init__(self, mada_config_dict):

        recorded_video = mada_config_dict["recorded_video"]
        in_video_dir = recorded_video["in_video_dir"]

        in_rgb_video = recorded_video["in_rgb_video"]
        in_rgb_video_path = os.path.join(in_video_dir, f"{in_rgb_video}")

        if not os.path.isfile(in_rgb_video_path):
            print(f"Cannot find file {in_rgb_video_path}")
            audio_error_message = "Cannot find color video file"
            text_to_speech(audio_error_message)
            return

        self.rgb_cap = cv2.VideoCapture(in_rgb_video_path)

        self.fps = self.rgb_cap.get(cv2.CAP_PROP_FPS)
        print("fps (frames per second) =", self.fps)
        self.spf = 1/self.fps  # second per frame

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

        speed_file = in_rgb_video.replace("rgb", "speed")
        speed_file = speed_file.replace(".mp4", ".txt")
        speed_file_path = os.path.join(in_video_dir, speed_file)
        self.frame_speed = None
        if os.path.isfile(speed_file_path):
            with open(speed_file_path) as f:
                speed_frame_lines = f.readlines()
            self.frame_speed = {int(speed_frame_line.split()[1]):int(speed_frame_line.split()[0])
                                for speed_frame_line in speed_frame_lines}
            # print(self.frame_speed)

        self.n_frame = 0
        self.n_frame_init_time = 0
        self.init_frame = recorded_video.get("init_frame", 0)
        self.sleep_time = recorded_video.get("sleep_time", 0)

    def get_color_and_depth_images(self, log=True):

        self.n_frame += 1
        self.n_frame_init_time = time()
        if log:
            print(f"\n[FRAME] {self.n_frame}")
        if self.frame_speed is not None:
            self.speed = self.frame_speed.get(self.n_frame, None)
        else:
            self.speed = None

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

    def sleep_frame_resting_time(self, log=False):
        n_frame_end_time = time()
        n_frame_time = n_frame_end_time - self.n_frame_init_time
        n_frame_remaining_time = self.spf - n_frame_time
        if log:
            print(f"n_frame_time is {int(n_frame_time * 1000)}, n_frame_remaining_time is {int(n_frame_remaining_time * 1000)} ms")
        if n_frame_remaining_time > 0:
            sleep(n_frame_remaining_time)

        return

    def stop(self):

        self.rgb_cap.release()

        if self.depth_cap is not None:
           self.depth_cap.release()

