import torch
from ultralytics import YOLO  # pip install ultralytics
import cv2
from paddleocr import PaddleOCR    # pip install paddlepaddle paddleocr
import numpy as np
import sys; sys.path.append("../common")
from utils import is_int

class ObjectDetector:

    def __init__(self, mada_config_dict, image_width, image_height):

        self.image_width = image_width
        self.image_height = image_height

        self.mada_class_names = mada_config_dict.get("mada_class_names", None)
        # mada_class_name2id = {mada_class_names[id]: id for id in range(len(mada_class_names))}
        self.score_thresh = mada_config_dict.get("score_thresh", 0.8)

        # if show_track True, image is displayed in model.track
        self.show_track = mada_config_dict.get("show_track", False)
        # else if show_distance True, image is displayed in cv2.imshow (SHOW_DISTANCE is the opposite of SHOW_TRACK)
        self.show_distance = not self.show_track   # either show track id OR distance

        self.max_distance = mada_config_dict.get("max_distance_from_camera", 6)

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

        yolo_model = mada_config_dict.get("yolo_model", "yolov8m.pt")
        self.model = YOLO(yolo_model).to(self.compute_device)

        self.paddle_ocr = PaddleOCR(use_angle_cls=False, ocr_version='PP-OCRv4', lang='en', show_log=False)
        # paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

        self.speed_limits = mada_config_dict.get("speed_limits", None)
        self.traffic_light_threshold_ratio = mada_config_dict.get("traffic_light_threshold_ratio", 0.05)

    def detection_and_tracking(self, color_image, depth_image):
        """
        Generates space events from the detection and tracking of objects, defined by:
        - an object, defined by:
            - class_name/id: identifier of an object class : car, person, traffic signal, traffic light
            - track_id: identifier of an object, or instance of a class name, that is detected in several frames.
        - bounding box: rectangle that contains the object in a given frame
        - distance: from the camera to the object
        """

        space_event_message = ""

        # Perform the actual detection and tracking by running track on the model with the image as input
        results = self.model.track(color_image, device=self.compute_device, persist=True, show=self.show_track)
        result = results[0]

        if isinstance(result.boxes.id, list) or result.boxes.id is not None:
            track_ids = np.array(result.boxes.id.cpu(), dtype="int")
            classes = np.array(result.boxes.cls.cpu(), dtype="int")
            scores = np.array(result.boxes.conf.cpu(), dtype="float")
            boxes = np.array(result.boxes.xyxy.cpu(), dtype="int")

            for track_id, class_id, score, box in zip(track_ids, classes, scores, boxes):

                score = float("{:.2f}".format(score))
                print(f"[DEBUG] track_id: {track_id}, class: {class_id} {self.mada_class_names[class_id]}, score: {score}")

                if score > self.score_thresh:

                    class_name = self.mada_class_names[class_id]
                    pre_text = ""  # for traffic light
                    post_text = ""  # for speed limit

                    params = dict()
                    params["IMAGE_WIDTH"] = self.image_width
                    params["IMAGE_HEIGHT"] = self.image_height
                    params["TRACK_ID"] = track_id
                    params["CLASS_ID"] = class_id
                    params["CLASS_NAME"] = class_name
                    params["BOUNDING_BOX"] = list(
                        box)  # box example: array([314, 166, 445, 302]) => array generates error when decoding

                    if class_name == "traffic light":
                        # color
                        bbox_image = self.get_bbox_image(color_image, box)
                        traffic_light_color = self.classify_traffic_light(bbox_image)
                        # print(f"traffic_light_color is {traffic_light_color}")
                        params["TRAFFIC_LIGHT_COLOR"] = traffic_light_color
                        pre_text = f"{traffic_light_color} "
                    elif class_name == "speed limit":
                        # ocr
                        bbox_image = self.get_bbox_image(color_image, box)
                        bbox_image = self.resize_to_min_dimension(bbox_image, target_min_size=60)
                        result = self.paddle_ocr.ocr(bbox_image, cls=False)
                        # result = paddle_ocr.ocr(bbox_image, cls=True)
                        if result[0] is not None:
                            speed_limit = result[0][0][1][0]
                            prob = result[0][0][1][1]
                            prob = int(100 * round(prob, 2))

                            print(f"Speed limit is {speed_limit} with prob {prob}%")
                            if is_int(speed_limit):  # discard text values
                                speed_limit = int(speed_limit)
                                if speed_limit in self.speed_limits:
                                    params["SPEED_LIMIT"] = speed_limit
                                    post_text = f" {speed_limit}"

                    object_distance = self.get_object_distance(depth_image, box)

                    self.cv2_rect_text(color_image, object_distance, box, class_name, pre_text, post_text)

                    params["OBJECT_DISTANCE"] = object_distance
                    space_event_message = "setSpaceEvent " + str(params)

        return space_event_message

    def get_bbox_image(self, image, bbox):

        left, top, right, bottom = map(int, bbox)
        bbox_image = image[top:bottom, left:right]

        return bbox_image

    def classify_traffic_light(self, bbox_image):

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
        kernel = np.ones((5, 5), np.uint8)
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
        traffic_light_threshold = int(self.traffic_light_threshold_ratio * total_pixels)
        print(f"thresh es {traffic_light_threshold}\n")

        if counts[state] < traffic_light_threshold:
            return "off"
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

    def cv2_rect_text(self, color_image, object_distance, bbox, class_name, pre_text, post_text):

        left, top, right, bottom = map(int, bbox)

        # draw box
        cv2.rectangle(color_image, (left, top), (right, bottom), (255, 0, 0), 2, 1)

        rect_text = f"{pre_text}{class_name}{post_text}"
        if object_distance > self.max_distance:
            object_distance_txt = f"{rect_text} at >{self.max_distance} m"
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

    def classify_traffic_light2(self, bbox_image):

        # TODO: revisar si interesa comprobar top, middle, bottom y añadir a classify_traffic_light

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
