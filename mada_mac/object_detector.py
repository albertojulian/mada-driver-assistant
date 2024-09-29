# detection: normal COCO
# track
# If LIVE == True, read from camera Intel Realsense
# If LIVE == False, read color and depth videos from "grabar color y depth alineados uint16"
# TTS


from ultralytics import YOLO  # pip install ultralytics
import pyrealsense2 as rs
import numpy as np
import os
import cv2
import torch
import websockets
import asyncio
import yaml
from text_to_speech import text_to_speech


async def space_events():
    """
    Generates space events, defined by:
    - an object, defined by:
        - class_name/id: identifier of an object class : car, person, traffic signal, traffic light
        - track_id: identifier of an object, or instance of a class name, that is detected in several frames.
    - bounding box: rectangle that contains the object in a given frame
    - distance: from the camera to the object
    :return:
    """

    ws_ip = mada_config_dict.get("ws_ip", "ws://192.168.43.233)")
    ws_port = mada_config_dict.get("ws_port", 8765)
    uri = f"{ws_ip}:{ws_port}"  # WebSocket server address: IP and port

    websocket = None

    if LIVE is True:  # Get images from camera

        image_width = mada_config_dict.get("image_width", 640)  # 848
        image_height = mada_config_dict.get("image_height", 480)
        fps = mada_config_dict.get("fps", 15)  # 30

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, fps)

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        rs_device = pipeline_profile.get_device()
        device_product_line = str(rs_device.get_info(rs.camera_info.product_line))
        print(device_product_line)  # D400

        print(f"[INFO] {device_product_line} start streaming...")
        pipeline.start(config)

        aligned_stream = rs.align(rs.stream.color)  # alignment between color and depth

    else:
        # LIVE is False => Get images from two disc files: .mp4 for RGB, .mkv for DEPTH
        in_video_dir = mada_config_dict.get("in_video_dir", "../videos")

        in_rgb_video = mada_config_dict.get("in_rgb_video", "rgb_4.mp4")
        in_rgb_video_path = os.path.join(in_video_dir, f"{in_rgb_video}")

        rgb_cap = cv2.VideoCapture(in_rgb_video_path)
        fps = rgb_cap.get(cv2.CAP_PROP_FPS)
        print("fps=", fps)
        image_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        in_depth_video = mada_config_dict.get("in_depth_video", "depth_4.mkv")
        in_depth_video_path = os.path.join(in_video_dir, f"{in_depth_video}")
        depth_cap = cv2.VideoCapture(in_depth_video_path,
                                     apiPreference=cv2.CAP_FFMPEG,
                                     params=[
                                         # BGR conversion turned OFF, decoded frame will be returned in its original format.
                                         # Multiplanar formats are not supported by the backend. Only GRAY8/GRAY16LE pixel formats have been tested.
                                         cv2.CAP_PROP_CONVERT_RGB,
                                         0,  # false
                                     ],
                                     )

        print(f"[INFO] Start reading files '{in_rgb_video_path}' and '{in_depth_video_path}'...")

    coco_traffic_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 9: 'traffic light', }
    coco_traffic_class_ids = coco_traffic_classes.keys()

    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
        device = torch.device("cpu")
    else:
        print("MPS is available")
        device = torch.device("mps")

    # device = torch.device("cpu")

    yolo_model = mada_config_dict.get("yolo_model", "yolov8m.pt")
    model = YOLO(yolo_model).to(device)

    n_frame = 0

    init_second = 0  # 90
    init_frame = init_second * fps
    # init_frame = 547

    try:

        while True:
            try:
                if websocket is None or websocket.closed:
                    websocket = await websockets.connect(uri)

                if LIVE is True:
                    frames = pipeline.wait_for_frames()
                    frames = aligned_stream.process(frames)

                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()

                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())

                else:
                    print(f"\n[FRAME] {n_frame}")
                    n_frame += 1
                    rgb_ret, color_image = rgb_cap.read()
                    depth_ret, depth_image = depth_cap.read()
                    # depth_image = np.array(depth_image, dtype=np.uint16)
                    # print("depth", depth_image.shape, depth_image.dtype)  # (480, 640, 3) uint8 en mp4v/.mp4 y H264/.avi

                    if n_frame < init_frame:
                        continue

                # Perform the actual detection by running the model with the image as input
                results = model.track(color_image, device=device, persist=True, show=SHOW_TRACK)
                result = results[0]
                class_names = result.names  # {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', ...
                if isinstance(result.boxes.id, list) or result.boxes.id is not None:
                    track_ids = np.array(result.boxes.id.cpu(), dtype="int")
                    classes = np.array(result.boxes.cls.cpu(), dtype="int")
                    scores = np.array(result.boxes.conf.cpu(), dtype="float")
                    boxes = np.array(result.boxes.xyxy.cpu(), dtype="int")

                    for track_id, class_id, score, box in zip(track_ids, classes, scores, boxes):

                        score = float("{:.2f}".format(score))
                        print(f"[DEBUG] track_id: {track_id}, class: {class_id} {class_names[class_id]}, score: {score}")

                        if score > 0.8 and class_id in coco_traffic_class_ids:
                            class_name = class_names[class_id]

                            object_distance = cv2_rect_text(color_image, depth_image, box, class_name)

                            params = dict()
                            params["IMAGE_WIDTH"] = image_width
                            params["IMAGE_HEIGHT"] = image_height
                            params["TRACK_ID"] = track_id
                            params["CLASS_ID"] = class_id
                            params["CLASS_NAME"] = class_name
                            params["BOUNDING_BOX"] = list(box)  # box example: array([314, 166, 445, 302]) => array generates error when decoding
                            params["OBJECT_DISTANCE"] = object_distance

                            message = "setSpaceEvent " + str(params)
                            await websocket.send(message)

                # Show images
                if SHOW_DISTANCE:
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

        if LIVE is True:
            # Stop streaming
            pipeline.stop()
        else:
            rgb_cap.release()
            depth_cap.release()

        await websocket.close()


def cv2_rect_text(color_image, depth_image, box, class_name):
    left = int(box[0])
    top = int(box[1])
    right = int(box[2])
    bottom = int(box[3])

    # draw box
    cv2.rectangle(color_image, (left, top), (right, bottom), (255, 0, 0), 2, 1)

    zs = depth_image[top:bottom, left:right]
    z = np.median(zs)
    object_distance = float("{:.1f}".format(z / factor))

    if object_distance > max_distance:
        object_distance_txt = f"{class_name} at >{max_distance} m"
    else:
        object_distance_txt = f"{class_name} at {object_distance} m"

    print("[INFO] ", object_distance_txt)

    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_text_corner = (left, top + 20)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2
    cv2.putText(color_image, object_distance_txt,
                bottom_left_text_corner,
                font,
                font_scale,
                font_color,
                line_type)

    return object_distance


if __name__ == "__main__":
    mada_file = "mada.yaml"
    with open(mada_file) as file:
        mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

    LIVE = mada_config_dict.get("live", False)

    # if SHOW_TRACK True, image is displayed in model.track
    SHOW_TRACK = mada_config_dict.get("show_track", False)
    # if SHOW_DISTANCE True, image is displayed in cv2.imshow (SHOW_DISTANCE is the opposite of SHOW_TRACK)
    SHOW_DISTANCE = not SHOW_TRACK  # either show track id OR distance

    max_distance = mada_config_dict.get("max_distance_from_camera", 6)
    factor = 1000  # from mm to m; used for the depth image

    print("\n<<<<<<<<<<<< Starting Object Detector >>>>>>>>>>>>\n")

    asyncio.get_event_loop().run_until_complete(space_events())
