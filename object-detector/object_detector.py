import torch
from ultralytics import YOLO  # pip install ultralytics
import os
import cv2
import numpy as np
import sys; sys.path.append("../common")
from utils import is_int
from paddleocr import PaddleOCR    # pip install paddlepaddle paddleocr

class ObjectDetector:

    def __init__(self, mada_config_dict, image_width, image_height):

        self.image_width = image_width
        self.image_height = image_height

        object_detection_model = mada_config_dict["object_detection_model"]
        self.mada_class_names = object_detection_model["mada_class_names"]
        # mada_class_name2id = {mada_class_names[id]: id for id in range(len(mada_class_names))}
        self.score_thresh = object_detection_model["score_thresh"]

        # if show_track True, image is displayed in model.track
        self.show_track = object_detection_model.get("show_track", False)
        # else if show_distance True, image is displayed in cv2.imshow (SHOW_DISTANCE is the opposite of SHOW_TRACK)
        self.show_distance = not self.show_track   # either show track id OR distance

        camera_conf = mada_config_dict["camera"]
        self.max_distance = camera_conf.get("max_distance_from_camera", 6)

        # from mm to m; used for the depth image
        self.factor = 1000

        # Check that MPS is available
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                      "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                      "and/or you do not have an MPS-enabled device on this machine.")
            self.compute_device = torch.device("cpu")
        else:
            print("MPS is available")
            self.compute_device = torch.device("mps")

        # self.compute_device = torch.device("cpu")

        yolo_model = object_detection_model["yolo_model"]
        self.model = YOLO(yolo_model).to(self.compute_device)

        self.paddle_ocr = PaddleOCR(use_angle_cls=False, ocr_version='PP-OCRv4', lang='en', show_log=False)
        # paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

        speed_conf = mada_config_dict["speed"]

        self.speed_limits = speed_conf["speed_limits"]

        traffic_light_conf = mada_config_dict["traffic_light"]
        self.traffic_light_threshold_ratio = traffic_light_conf["traffic_light_threshold_ratio"]

        hue_ranges = traffic_light_conf["hue_ranges"]
        self.red_H_lower1 = hue_ranges["red_H_lower1"]
        self.red_H_upper1 = hue_ranges["red_H_upper1"]
        self.red_H_lower2 = hue_ranges["red_H_lower2"]
        self.red_H_upper2 = hue_ranges["red_H_upper2"]
        self.yellow_H_lower = hue_ranges["yellow_H_lower"]
        self.yellow_H_upper = hue_ranges["yellow_H_upper"]
        self.green_H_lower = hue_ranges["green_H_lower"]
        self.green_H_upper = hue_ranges["green_H_upper"]

    def detection_and_tracking(self, color_image, depth_image, log=True):
        """
        Generates space events from the detection and tracking of objects, defined by:
        - an object, defined by:
            - class_name/id: identifier of an object class : car, person, traffic signal, traffic light
            - track_id: identifier of an object, or instance of a class name, that is detected in several frames.
        - bounding box: rectangle that contains the object in a given frame
        - distance: from the camera to the object
        """

        space_event_messages = []

        # Perform the actual detection and tracking by running track on the model with the image as input
        results = self.model.track(color_image, device=self.compute_device, persist=True, show=self.show_track)
        result = results[0]

        if isinstance(result.boxes.id, list) or result.boxes.id is not None:
            track_ids = np.array(result.boxes.id.cpu(), dtype="int")
            classes = np.array(result.boxes.cls.cpu(), dtype="int")
            scores = np.array(result.boxes.conf.cpu(), dtype="float")
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")

            for track_id, class_id, score, bbox in zip(track_ids, classes, scores, bboxes):

                score = float("{:.2f}".format(score))
                if log:
                    print(f"[TRACK] track_id: {track_id}, class: {class_id} {self.mada_class_names[class_id]}, score: {score}")

                if score > self.score_thresh:

                    class_name = self.mada_class_names[class_id]
                    pre_text = ""  # for traffic light and speed limit
                    # post_text = ""  # for speed limit
                    params = dict()
                    params["IMAGE_WIDTH"] = self.image_width
                    params["IMAGE_HEIGHT"] = self.image_height
                    params["TRACK_ID"] = track_id
                    params["CLASS_ID"] = class_id
                    params["CLASS_NAME"] = class_name
                    params["BOUNDING_BOX"] = list(bbox)  # box example: array([314, 166, 445, 302]) => array generates error when decoding

                    if class_name == "traffic light":
                        # color
                        bbox_image = self.get_bbox_image(color_image, bbox)
                        traffic_light_color = self.classify_traffic_light(bbox_image)
                        if log:
                            print(f"\ntraffic_light_color is {traffic_light_color}\n")
                        params["TRAFFIC_LIGHT_COLOR"] = traffic_light_color
                        pre_text = f"{traffic_light_color} "
                    elif class_name == "speed limit":
                        # ocr
                        bbox_image = self.get_bbox_image(color_image, bbox)
                        bbox_image = self.resize_to_min_dimension(bbox_image, target_min_size=60)
                        result = self.paddle_ocr.ocr(bbox_image, cls=False)
                        # result = paddle_ocr.ocr(bbox_image, cls=True)
                        if result[0] is not None:
                            speed_limit = result[0][0][1][0]
                            prob = result[0][0][1][1]
                            prob = int(100 * round(prob, 2))
                            if log:
                                print(f"Speed limit is {speed_limit} with prob {prob}%")
                            if is_int(speed_limit):  # discard text values
                                speed_limit = int(speed_limit)
                                if speed_limit in self.speed_limits:
                                    params["SPEED_LIMIT"] = speed_limit
                                    # post_text = f" {speed_limit}"
                                    pre_text = f"{speed_limit} km/h "

                    if depth_image is None:
                        object_distance = 0
                    else:
                        object_distance = self.get_object_distance(depth_image, bbox)

                    self.cv2_rect_text(color_image, object_distance, bbox, class_name, pre_text)

                    params["OBJECT_DISTANCE"] = object_distance
                    space_event_message = "setSpaceEvent " + str(params)

                    space_event_messages.append(space_event_message)

        return space_event_messages

    def get_bbox_image(self, image, bbox):

        left, top, right, bottom = map(int, bbox)
        bbox_image = image[top:bottom, left:right]

        return bbox_image


    def classify_traffic_light(self, bbox_image):

        def get_most_colored_part(mask, height_part, traffic_light_threshold):
            top_count = cv2.countNonZero(mask[:height_part, :])
            middle_count = cv2.countNonZero(mask[height_part:2 * height_part, :])
            bottom_count = cv2.countNonZero(mask[2 * height_part:, :])

            # Determinar el estado basado en el conteo de píxeles
            counts = {'top': top_count, 'middle': middle_count, 'bottom': bottom_count}
            most_colored_part = max(counts, key=counts.get)

            if counts[most_colored_part] > traffic_light_threshold:
                part_count = counts[most_colored_part]
            else:
                most_colored_part = "dark"
                part_count = 0
            # print("counts ", counts, most_colored_part)

            return most_colored_part, part_count

        # Convertir a espacio de color HSV para mejor segmentación
        hsv = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2HSV)

        # Definir rangos de color para rojo, amarillo y verde
        # El estándar de Hue es circular: 0 a 360 grados; en OpenCV se normaliza de 0 a 180 (8 bits)
        # En OpenCV el color rojo está en valores de H de 0-10 y 170-180 (que es el 0) aprox
        red_lower1 = np.array([self.red_H_lower1, 70, 50])
        red_upper1 = np.array([self.red_H_upper1, 255, 255])
        red_lower2 = np.array([self.red_H_lower2, 70, 50])
        red_upper2 = np.array([self.red_H_upper2, 255, 255])

        yellow_lower = np.array([self.yellow_H_lower, 70, 50])
        yellow_upper = np.array([self.yellow_H_upper, 255, 255])

        green_lower = np.array([self.green_H_lower, 70, 50])
        green_upper = np.array([self.green_H_upper, 255, 255])

        # Crear máscaras para cada color
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)

        height, width, _ = bbox_image.shape
        height_margin = int(height * 0.1)
        width_margin = int(width * 0.1)  # Recortar el 10% de cada lado
        mask_red = mask_red[height_margin:height - height_margin, width_margin:width - width_margin]
        mask_yellow = mask_yellow[height_margin:height - height_margin, width_margin:width - width_margin]
        mask_green = mask_green[height_margin:height - height_margin, width_margin:width - width_margin]

        height_cut = height - 2 * height_margin
        height_part = height_cut // 3
        width_cut = width - 2 * width_margin
        part_pixels = height_part * width_cut
        traffic_light_threshold = int(self.traffic_light_threshold_ratio * part_pixels)
        # print(f"thresh es {traffic_light_threshold}\n")

        most_red, red_part_count = get_most_colored_part(mask_red, height_part, traffic_light_threshold)
        most_yellow, yellow_part_count = get_most_colored_part(mask_yellow, height_part, traffic_light_threshold)
        most_green, green_part_count = get_most_colored_part(mask_green, height_part, traffic_light_threshold)

        if most_red == "top" and red_part_count > yellow_part_count and red_part_count > green_part_count:
            state = "red"
        elif most_yellow in ["middle", "bottom"] and yellow_part_count > red_part_count and yellow_part_count > green_part_count:
            state = "yellow"
        elif most_green == "bottom" and green_part_count > red_part_count and green_part_count > yellow_part_count:
            state = "green"
        else:
            state = "off"

        return state


    def resize_to_min_dimension(self, image, target_min_size=60):
        """
        PaddleOCR works better with images greater than 60 pixels in width and height
        :param image:
        :param target_min_size:
        :return:
        """
        # Obtener dimensiones actuales
        height, width = image.shape[:2]

        if height >= target_min_size and width >= target_min_size:  # no hay que escalar
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

    def get_object_distance(self, depth_image, bbox):

        left, top, right, bottom = map(int, bbox)

        zs = depth_image[top:bottom, left:right]
        z = np.median(zs)
        object_distance = float("{:.1f}".format(z / self.factor))

        return object_distance

    def cv2_rect_text(self, color_image, object_distance, bbox, class_name, pre_text, log=True):

        left, top, right, bottom = map(int, bbox)

        # draw box
        cv2.rectangle(color_image, (left, top), (right, bottom), (255, 0, 0), 2, 1)

        rect_text = f"{pre_text}{class_name}"
        if object_distance > self.max_distance:
            name_and_object_distance_txt = f"{rect_text} at >{self.max_distance} m"
        else:
            name_and_object_distance_txt = f"{rect_text} at {object_distance} m"

        if log:
            print("[BBOX TITLE] ", name_and_object_distance_txt)

        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_text_corner = (left, top - 10)
        font_scale = 1
        font_color = (255, 255, 255)
        line_type = 2
        cv2.putText(color_image, name_and_object_distance_txt,
                    bottom_left_text_corner,
                    font,
                    font_scale,
                    font_color,
                    line_type)

