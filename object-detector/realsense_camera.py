import pyrealsense2 as rs
import numpy as np

class RealSenseCamera:
    def __init__(self, mada_config_dict):

        camera_conf = mada_config_dict["camera"]
        self.image_width = camera_conf["image_width"]
        self.image_height = camera_conf["image_height"]
        self.fps = camera_conf["fps"]

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.image_width, self.image_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.image_width, self.image_height, rs.format.bgr8, self.fps)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        rs_device = pipeline_profile.get_device()
        device_product_line = str(rs_device.get_info(rs.camera_info.product_line))
        print(device_product_line)  # D400

        print(f"[INFO] {device_product_line} start streaming...")
        self.pipeline.start(config)

        # alignment between color and depth
        self.aligned_stream = rs.align(rs.stream.color)

    def get_color_and_depth_images(self):

        frames = self.pipeline.wait_for_frames()
        frames = self.aligned_stream.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        skip_frame = False  # for compatibility with RecordedVideoManager
        return skip_frame, color_image, depth_image

    def stop(self):

        # Stop streaming
        self.pipeline.stop()
