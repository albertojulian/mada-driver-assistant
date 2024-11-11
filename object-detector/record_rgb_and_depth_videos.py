import cv2
import os
import glob
import yaml
from realsense_camera import RealSenseCamera
import asyncio
import websockets
import sys; sys.path.append("../common")
from text_to_speech import text_to_speech

class VideoRecorder:

    def __init__(self, mada_config_dict):

        camera_conf = mada_config_dict["camera"]
        self.image_width = camera_conf["image_width"]
        self.image_height = camera_conf["image_height"]
        self.fps = camera_conf["fps"]

        recorded_video = mada_config_dict["recorded_video"]
        out_video_dir = recorded_video["in_video_dir"]

        out_rgb_video_base = "rgb"
        rgb_videos = glob.glob(os.path.join(out_video_dir, f"{out_rgb_video_base}*.mp4"))
        self.video_id = len(rgb_videos)
        out_rgb_video = f"{out_rgb_video_base}_{self.video_id}.mp4"
        out_rgb_video_path = os.path.join(out_video_dir, out_rgb_video)

        out_depth_video_base = "depth"
        out_depth_video_path = os.path.join(out_video_dir, f"{out_depth_video_base}_{self.video_id}.mkv")

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

        self.n_frame = 0

    def record_color_and_depth_images(self, color_image, depth_image):

        self.n_frame += 1

        self.rgb_vid_writer.write(color_image)

        # depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)  # hace falta??
        self.depth_vid_writer.write(depth_image)

    def stop(self):

        self.rgb_vid_writer.release()
        self.depth_vid_writer.release()


async def record_color_and_depth_videos_loop():

    mada_file = "../mada.yaml"
    with open(mada_file) as file:
        mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

    communications = mada_config_dict["communications"]
    ws_ip = communications["ws_ip"]
    ws_port = communications["ws_port"]
    uri = f"{ws_ip}:{ws_port}"  # driver agent events handler address: IP and port

    websocket = None

    camera = RealSenseCamera(mada_config_dict)

    video_recorder = VideoRecorder(mada_config_dict)

    try:
        while True:
            try:
                if websocket is None or websocket.closed:
                    websocket = await websockets.connect(uri)

                skip_frame, color_image, depth_image = camera.get_color_and_depth_images()

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', color_image)

                video_recorder.record_color_and_depth_images(color_image, depth_image)

                space_event_message = f"frameEvent {video_recorder.video_id}_{video_recorder.n_frame}"
                await websocket.send(space_event_message)

                key = cv2.waitKey(1)
                if key == 27:  # ESCAPE
                    break

            except websockets.ConnectionClosedError as e:
                print(f"Connection closed: {e}. Retrying in 1 second...")
                websocket = None
                await asyncio.sleep(1)  # Wait before trying to reconnect

            except Exception as e:
                video_recorder_unknown_error = communications.get("video_recorder_unknown_error", "Video Recorder or connection error")
                text_to_speech(video_recorder_unknown_error)
                break

    finally:
        camera.stop()
        video_recorder.stop()

        cv2.destroyAllWindows()

    # depth_image = depth_image[depth_image < 10000]
    # plt.hist(depth_image, bins=100)
    # plt.show()


if __name__ == "__main__":

    print("\n<<<<<<<<<<<< Starting Video Recorder >>>>>>>>>>>>\n")

    asyncio.get_event_loop().run_until_complete(record_color_and_depth_videos_loop())