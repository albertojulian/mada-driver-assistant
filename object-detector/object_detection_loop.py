# detection: 23 classes, including COCO (person, car, bus, bicycle, truck, traffic light), speed limit, give way, etc
# track
# If LIVE == True, read from camera Intel Realsense
# If LIVE == False, read color and depth videos from "grabar color y depth alineados uint16"
# TTS

import cv2
import websockets
import asyncio
import yaml
from realsense_camera import RealSenseCamera
from recorded_video_manager import RecordedVideoManager
from object_detector import ObjectDetector
import sys; sys.path.append("../common")
from text_to_speech import text_to_speech

async def detect_and_track_objects():

    mada_file = "object_detector.yaml"
    with open(mada_file) as file:
        mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

    LIVE = mada_config_dict.get("live", False)

    ws_ip = mada_config_dict.get("ws_ip", "ws://192.168.43.233)")
    ws_port = mada_config_dict.get("ws_port", 8765)
    uri = f"{ws_ip}:{ws_port}"  # WebSocket server address: IP and port

    websocket = None

    if LIVE is True:
        # Get images from camera
        video_device = RealSenseCamera(mada_config_dict)

    else:
        # LIVE is False => Get images from two disc files: .mp4 for RGB, .mkv for DEPTH
        video_device = RecordedVideoManager(mada_config_dict)

    image_width = video_device.image_width
    image_height = video_device.image_height

    object_detector = ObjectDetector(mada_config_dict, image_width, image_height)

    try:

        while True:
            try:
                if websocket is None or websocket.closed:
                    websocket = await websockets.connect(uri)

                color_image, depth_image = video_device.get_color_and_depth_images()
                if color_image is None:
                    continue

                space_event_message = object_detector.detection_and_tracking(color_image, depth_image)

                if space_event_message != "":
                    await websocket.send(space_event_message)

                # Show images
                if object_detector.show_distance:
                    cv2.namedWindow('MADA RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('MADA RealSense', color_image)

                key = cv2.waitKey(1)
                if key == 27:  # ESCAPE
                    break

            except websockets.ConnectionClosedError as e:
                print(f"Connection closed: {e}. Retrying in 3 seconds...")
                websocket = None
                await asyncio.sleep(3)  # Wait before trying to reconnect

            except Exception as e:
                connection_error = mada_config_dict.get("object_detector_connection_error", "Object Detector connection error")
                text_to_speech(connection_error)
                break

    finally:
        cv2.destroyAllWindows()

        video_device.stop()

        await websocket.close()


if __name__ == "__main__":

    print("\n<<<<<<<<<<<< Starting Object Detector >>>>>>>>>>>>\n")

    asyncio.get_event_loop().run_until_complete(detect_and_track_objects())
