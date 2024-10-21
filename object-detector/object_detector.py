# detection: 23 classes, including COCO (person, car, bus, bicycle, truck, traffic light), speed limit, give way, etc
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
from time import sleep

import sys
sys.path.append("../common")
from utils import is_int
from text_to_speech import text_to_speech
from paddleocr import PaddleOCR    # pip install paddlepaddle paddleocr

async def detect_and_track_objects():
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
        if not os.path.isfile(in_rgb_video_path):
            print(f"Cannot find file {in_rgb_video_path}")
            audio_error_message = "Cannot find color video file"
            text_to_speech(audio_error_message)
            return

        rgb_cap = cv2.VideoCapture(in_rgb_video_path)
        fps = rgb_cap.get(cv2.CAP_PROP_FPS)
        print("fps=", fps)
        image_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        in_depth_video = mada_config_dict.get("in_depth_video", "depth_4.mkv")
        in_depth_video_path = os.path.join(in_video_dir, f"{in_depth_video}")
        if not os.path.isfile(in_depth_video_path):
            print(f"Cannot find file {in_depth_video_path}")
            audio_error_message = "Cannot find depth video file"
            text_to_speech(audio_error_message)
            return

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

    mada_class_names = mada_config_dict.get("mada_class_names", None)
    # mada_class_name2id = {mada_class_names[id]: id for id in range(len(mada_class_names))}
    score_thresh = mada_config_dict.get("score_thresh", 0.8)

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
    init_frame = mada_config_dict.get("init_frame", 0)

    paddle_ocr = PaddleOCR(use_angle_cls=False, ocr_version='PP-OCRv4', lang='en', show_log=False)
    # paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

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

                    sleep(sleep_time)

                # Perform the actual detection by running the model with the image as input
                results = model.track(color_image, device=device, persist=True, show=SHOW_TRACK)
                result = results[0]
                if isinstance(result.boxes.id, list) or result.boxes.id is not None:
                    track_ids = np.array(result.boxes.id.cpu(), dtype="int")
                    classes = np.array(result.boxes.cls.cpu(), dtype="int")
                    scores = np.array(result.boxes.conf.cpu(), dtype="float")
                    boxes = np.array(result.boxes.xyxy.cpu(), dtype="int")

                    for track_id, class_id, score, box in zip(track_ids, classes, scores, boxes):

                        score = float("{:.2f}".format(score))
                        print(f"[DEBUG] track_id: {track_id}, class: {class_id} {mada_class_names[class_id]}, score: {score}")

                        if score > score_thresh:
                            params = dict()

                            pre_text = ""
                            post_text = ""
                            class_name = mada_class_names[class_id]
                            if class_name == "speed limit":
                                # ocr
                                bbox_image = get_bbox_image(color_image, box)
                                bbox_image = resize_to_min_dimension(bbox_image, target_min_size=60)
                                result = paddle_ocr.ocr(bbox_image, cls=False)
                                # result = paddle_ocr.ocr(bbox_image, cls=True)
                                if result[0] is not None:
                                    speed_limit = result[0][0][1][0]
                                    prob = result[0][0][1][1]
                                    prob = int(100 * round(prob, 2))

                                    print(f"Speed limit is {speed_limit} with prob {prob}%")
                                    if is_int(speed_limit):   # discard text values
                                        speed_limit = int(speed_limit)
                                        if speed_limit in speed_limits:
                                            params["SPEED_LIMIT"] = speed_limit
                                            post_text = f" {speed_limit}"

                            elif class_name == "traffic light":
                                # color
                                bbox_image = get_bbox_image(color_image, box)
                                traffic_light_color = classify_traffic_light(bbox_image)
                                # print(f"traffic_light_color is {traffic_light_color}")
                                params["TRAFFIC_LIGHT_COLOR"] = traffic_light_color
                                pre_text = f"{traffic_light_color} "

                            object_distance = cv2_rect_text(color_image, depth_image, box,
                                                            class_name, pre_text, post_text)

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


def get_bbox_image(image, bbox):

    left, top, right, bottom = map(int, bbox)
    bbox_image = image[top:bottom, left:right]

    return bbox_image


def resize_to_min_dimension(image, target_min_size=60):
    # Obtener dimensiones actuales
    height, width = image.shape[:2]

    if height >= target_min_size and width >= target_min_size:    # no hay que escalar
        return image

    # Calcular el factor de escalado
    if height < width:
        scale_factor = target_min_size / height
    else:
        scale_factor = target_min_size / width

    # Calcular las nuevas dimensiones manteniendo la proporción
    new_width = int(round(width * scale_factor, 0))
    new_height = int(round(height * scale_factor, 0))

    # Redimensionar la imagen
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image


def classify_traffic_light(bbox_image):

    # Convertir a espacio de color HSV para mejor segmentación
    hsv = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2HSV)

    # Definir rangos de color para rojo, amarillo y verde
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([15, 70, 50])
    yellow_upper = np.array([35, 255, 255])

    green_lower = np.array([40, 70, 50])
    green_upper = np.array([90, 255, 255])

    # Crear máscaras para cada color
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    # Aplicar operaciones morfológicas para eliminar ruido
    kernel = np.ones((5,5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    # Contar píxeles en cada máscara
    red_count = cv2.countNonZero(mask_red)
    yellow_count = cv2.countNonZero(mask_yellow)
    green_count = cv2.countNonZero(mask_green)

    # Determinar el estado basado en el conteo de píxeles
    counts = {'red': red_count, 'yellow': yellow_count, 'green': green_count}
    state = max(counts, key=counts.get)

    total_pixels = bbox_image.shape[0] * bbox_image.shape[1]
    # traffic_light_threshold_perc = 0.05  # below this percentaje the light is considered off
    traffic_light_threshold = int(traffic_light_threshold_perc * total_pixels)
    print(f"thresh es {traffic_light_threshold}\n")

    if counts[state] < traffic_light_threshold:
        return "off"
    return state


def classify_traffic_light2(bbox_image):

    # Convertir la imagen de BGR a HSV para trabajar mejor con colores
    hsv = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2HSV)

    # Definir los rangos de color para cada luz de semáforo en el espacio HSV
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    
    green_lower = np.array([50, 100, 100])
    green_upper = np.array([70, 255, 255])

    # Crear máscaras para los colores rojo, amarillo y verde
    mask_red = cv2.inRange(hsv, red_lower, red_upper)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    # Aplicar un pequeño recorte lateral para evitar los píxeles claros en los bordes
    height, width = mask_red.shape
    margin = int(width * 0.1)  # Recortar el 10% de cada lado
    mask_red = mask_red[:, margin:width - margin]
    mask_yellow = mask_yellow[:, margin:width - margin]
    mask_green = mask_green[:, margin:width - margin]

    # Dividir el bounding box en tercios (superior, central, inferior)
    top_section_red = mask_red[:height // 3, :]
    middle_section_red = mask_red[height // 3:2 * height // 3, :]
    bottom_section_red = mask_red[2 * height // 3:, :]

    top_section_yellow = mask_yellow[:height // 3, :]
    middle_section_yellow = mask_yellow[height // 3:2 * height // 3, :]
    bottom_section_yellow = mask_yellow[2 * height // 3:, :]

    top_section_green = mask_green[:height // 3, :]
    middle_section_green = mask_green[height // 3:2 * height // 3, :]
    bottom_section_green = mask_green[2 * height // 3:, :]

    # Contar los píxeles blancos en cada sección para cada color
    top_red_pixels = cv2.countNonZero(top_section_red)
    middle_red_pixels = cv2.countNonZero(middle_section_red)
    bottom_red_pixels = cv2.countNonZero(bottom_section_red)

    top_yellow_pixels = cv2.countNonZero(top_section_yellow)
    middle_yellow_pixels = cv2.countNonZero(middle_section_yellow)
    bottom_yellow_pixels = cv2.countNonZero(bottom_section_yellow)

    top_green_pixels = cv2.countNonZero(top_section_green)
    middle_green_pixels = cv2.countNonZero(middle_section_green)
    bottom_green_pixels = cv2.countNonZero(bottom_section_green)

    # Evaluar en qué sección hay más píxeles para cada color
    if top_red_pixels > middle_red_pixels and top_red_pixels > bottom_red_pixels:
        return "red"
    elif middle_yellow_pixels > top_yellow_pixels and middle_yellow_pixels > bottom_yellow_pixels:
        return "yellow"
    elif bottom_green_pixels > top_green_pixels and bottom_green_pixels > middle_green_pixels:
        return "green"
    else:
        return "off"


def cv2_rect_text(color_image, depth_image, bbox, class_name, pre_text, post_text):

    left, top, right, bottom = map(int, bbox)

    # draw box
    cv2.rectangle(color_image, (left, top), (right, bottom), (255, 0, 0), 2, 1)

    zs = depth_image[top:bottom, left:right]
    z = np.median(zs)
    object_distance = float("{:.1f}".format(z / factor))

    rect_text = f"{pre_text}{class_name}{post_text}"
    if object_distance > max_distance:
        object_distance_txt = f"{rect_text} at >{max_distance} m"
    else:
        object_distance_txt = f"{rect_text} at {object_distance} m"

    print("[INFO] ", object_distance_txt)

    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_text_corner = (left, top - 10)
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
    mada_file = "object_detector.yaml"
    with open(mada_file) as file:
        mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

    LIVE = mada_config_dict.get("live", False)

    # if SHOW_TRACK True, image is displayed in model.track
    SHOW_TRACK = mada_config_dict.get("show_track", False)

    # else if SHOW_DISTANCE True, image is displayed in cv2.imshow (SHOW_DISTANCE is the opposite of SHOW_TRACK)
    SHOW_DISTANCE = not SHOW_TRACK  # either show track id OR distance

    max_distance = mada_config_dict.get("max_distance_from_camera", 6)
    factor = 1000  # from mm to m; used for the depth image

    sleep_time = mada_config_dict.get("sleep_time", 0)

    speed_limits = mada_config_dict.get("speed_limits", None)

    traffic_light_threshold_perc = mada_config_dict.get("traffic_light_threshold_perc", 0.05)

    print("\n<<<<<<<<<<<< Starting Object Detector >>>>>>>>>>>>\n")

    asyncio.get_event_loop().run_until_complete(detect_and_track_objects())
