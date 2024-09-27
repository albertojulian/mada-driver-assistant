import pyrealsense2 as rs
import numpy as np
import cv2
import os
import glob

pipe = rs.pipeline()

config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipe)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
print(device_product_line)  # D400

align_to = rs.stream.color
aligned_stream = rs.align(align_to=align_to)  # alignment between color and depth
# point_cloud = rs.pointcloud()

width = 640
height = 480
fps = 15  # 6??, 25, 30
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

pipe.start(config)

out_video_dir = "videos"

out_rgb_video_base = "rgb"
rgb_videos = glob.glob(os.path.join(out_video_dir, f"{out_rgb_video_base}*.mp4"))
next_id = len(rgb_videos)
out_rgb_video_path = os.path.join(out_video_dir, f"{out_rgb_video_base}_{next_id}.mp4")

out_depth_video_base = "depth"
out_depth_video_path = os.path.join(out_video_dir, f"{out_depth_video_base}_{next_id}.mkv")


rgb_vid_writer = cv2.VideoWriter(
    out_rgb_video_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

depth_vid_writer = cv2.VideoWriter(
    out_depth_video_path,
    apiPreference=cv2.CAP_FFMPEG,
    fourcc=cv2.VideoWriter_fourcc(*'FFV1'),
    fps=fps,
    frameSize=(width, height),
    params=[
            cv2.VIDEOWRITER_PROP_DEPTH,
            cv2.CV_16U,
            cv2.VIDEOWRITER_PROP_IS_COLOR,
            0,  # false
    ],
)

try:
  while True:
    frames = pipe.wait_for_frames()
    frames = aligned_stream.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())  # (480, 640) uint16
    # print("depth", depth_image.shape, depth_image.dtype)
    color_image = np.asanyarray(color_frame.get_data())
    # print("color", color_image.shape, color_image.dtype)

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)

    rgb_vid_writer.write(color_image)

    # depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)  # hace falta??
    depth_vid_writer.write(depth_image)

    key = cv2.waitKey(1)
    if key == 27:  # ESCAPE
      break

finally:
    pipe.stop()

    rgb_vid_writer.release()
    depth_vid_writer.release()

    cv2.destroyAllWindows()

# depth_image = depth_image[depth_image < 10000]
# plt.hist(depth_image, bins=100)
# plt.show()
