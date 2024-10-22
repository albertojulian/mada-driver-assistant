import pyrealsense2 as rs
import numpy as np
import cv2
import os
import glob
import yaml
from realsense_camera import RealSenseCamera


class VideoRecorder:

    def __init__(self, mada_config_dict):

        self.image_width = mada_config_dict.get("image_width", 640)  # 848
        self.image_height = mada_config_dict.get("image_height", 480)
        self.fps = mada_config_dict.get("fps", 15)  # 30

        out_video_dir = "videos"

        out_rgb_video_base = "rgb"
        rgb_videos = glob.glob(os.path.join(out_video_dir, f"{out_rgb_video_base}*.mp4"))
        next_id = len(rgb_videos)
        out_rgb_video_path = os.path.join(out_video_dir, f"{out_rgb_video_base}_{next_id}.mp4")

        out_depth_video_base = "depth"
        out_depth_video_path = os.path.join(out_video_dir, f"{out_depth_video_base}_{next_id}.mkv")

        self.rgb_vid_writer = cv2.VideoWriter(
            out_rgb_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.image_width, self.image_height)
        )

        self.depth_vid_writer = cv2.VideoWriter(
            out_depth_video_path,
            apiPreference=cv2.CAP_FFMPEG,
            fourcc=cv2.VideoWriter_fourcc(*'FFV1'),
            fps=self.fps,
            frameSize=(self.image_width, self.image_height),
            params=[
                cv2.VIDEOWRITER_PROP_DEPTH,
                cv2.CV_16U,
                cv2.VIDEOWRITER_PROP_IS_COLOR,
                0,  # false
            ],
        )

    def record_color_and_depth_images(self, color_image, depth_image):

        self.rgb_vid_writer.write(color_image)

        # depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)  # hace falta??
        self.depth_vid_writer.write(depth_image)

    def stop(self):

        self.rgb_vid_writer.release()
        self.depth_vid_writer.release()


def record_color_and_depth_videos_loop(mada_config_dict):

    camera = RealSenseCamera(mada_config_dict)

    video_recorder = VideoRecorder(mada_config_dict)

    try:
        while True:

            color_image, depth_image = camera.get_color_and_depth_images()

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)

            video_recorder.record_color_and_depth_images(color_image, depth_image)

            key = cv2.waitKey(1)
            if key == 27:  # ESCAPE
                break

    finally:
        camera.stop()
        video_recorder.stop()

        cv2.destroyAllWindows()

    # depth_image = depth_image[depth_image < 10000]
    # plt.hist(depth_image, bins=100)
    # plt.show()


if __name__ == "__main__":

    mada_file = "object_detector.yaml"
    with open(mada_file) as file:
        mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

    record_color_and_depth_videos_loop(mada_config_dict)