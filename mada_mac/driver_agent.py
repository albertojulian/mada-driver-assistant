import threading
from memory import Memory
from planner import Planner
from text_to_speech import text_to_speech
from ast import literal_eval
import yaml
from typing import Literal
from time import time

# Executing a function inside a string: 'check_safety_distance_from_vehicle_v0(vehicle_type="car", position="in front")'
# - Use eval() to execute the function if it's a simple expression,
# - or exec() if it's a statement or requires more flexibility.

mada_file = "mada.yaml"
with open(mada_file) as file:
    mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

# time between actions: avoids redundant actions about the same object
TIME_BETWEEN_ACTIONS = mada_config_dict.get("time_between_actions", 2)


class DriverAgent:

    def __init__(self):
        self.memory = Memory()
        self.planner = Planner()

    def assess_automatic_action(self, params_str, log=False):
        """

        :param params_str
        :param log
        :return:
        """

        params = literal_eval(params_str)
        # params = eval(params_str)

        object, _ = self.memory.add_space_event(params, log=log)

        class_name = params["CLASS_NAME"]
        class_id = params["CLASS_ID"]

        object_distance = params["OBJECT_DISTANCE"]  # distance from the camera (ego car) to the object

        time_since_last_action = object.time_since_last_action()

        if class_name in ["car", "bus"]:

            if log:
                print(f"time_since_last_action {time_since_last_action}")

            # if last action was more than x seconds ago, perform action
            if time_since_last_action > TIME_BETWEEN_ACTIONS:
                space_event = object.space_events[-1]
                object_position = space_event.object_position  # position of last space event
                if object_position == "in front":
                    too_close, output_message = check_safety_distance_from_vehicle(vehicle=object, space_event=space_event, report_always=False)
                    if too_close:
                        self.memory.add_action_event(params, output_message)

        elif class_name in ["person", "traffic light"]:
            # if last action was more than x seconds ago, perform action
            if time_since_last_action > TIME_BETWEEN_ACTIONS:
                object_position = object.space_events[-1].object_position  # position of last space event
                max_distance = mada_config_dict.get("max_distance_from_camera", 6)
                if object_distance > max_distance:
                    output_message = f"{class_name} {object_position} is more than {max_distance} meters away."
                else:
                    output_message = f"{class_name} {object_position} at {object_distance} meters."

                if log:
                    print(f"[ACTION] class_id: {class_id}, output message: {output_message}")

                self.memory.add_action_event(params, output_message)

                if self.memory.listen_mode is False:
                    audio_thread = threading.Thread(target=text_to_speech, args=(output_message,))
                    # Start audio in a separate thread
                    audio_thread.start()
        # elif class_name in [speed_limit, ...]
        # TODO: a speed limit should be reported only once, but a traffic light more times

    def assess_request_action(self, text_input_message, log=True):

        text_input_message_event = self.memory.add_text_input_message(text_input_message, log=log)

        start_time = time()
        parsing_ok, response = self.planner.generate_response(text_input_message)
        processing_time = round(time() - start_time, 1)
        text_input_message_event.processing_time = processing_time
        print(f"LLM Response took {processing_time} seconds")

        if parsing_ok:
            eval(response)
        else:
            text_to_speech(response)


_driver_agent_instance = None


def get_driver_agent(log=False):
    global _driver_agent_instance
    if _driver_agent_instance is None:
        _driver_agent_instance = DriverAgent()
        print("\n<<<<<<<<<<<< Starting Driver Agent >>>>>>>>>>>>")
    else:
        if log:
            print("Driver Agent already exists")

    return _driver_agent_instance


def check_safety_distance_from_vehicle_v0(vehicle_type: Literal["car", "bus"],
                                          position: Literal["on the left", "in front", "on the right"],
                                          log: bool = False):
    """

    :param vehicle_type: car, bus
    :param position: on the left, in front, on the right
    :param log
    :return:
    """

    vehicle, space_event = get_vehicle_instance([vehicle_type], position)
    if vehicle is not None and space_event is not None:

        vehicle_distance = space_event.object_distance

        max_distance = mada_config_dict.get("max_distance_from_camera", 6)
        if vehicle_distance > max_distance:
            output_message = f"{vehicle_type} {position} is more than {max_distance} meters away, which is the camera limit."
        else:
            current_speed, current_speed_str = get_current_speed(tts=False)
            output_message = f"There is a {vehicle_type} {position} at {vehicle_distance} meters"

            if current_speed is None:
                output_message += f", but your speed is unknown and thus safety distance cannot be calculated."
            else:
                safety_distance = get_safety_distance(current_speed)
                if vehicle_distance < safety_distance:
                    output_message = f"You should reduce the speed. {output_message}, but safety distance is bigger: {safety_distance} meters."
                else:
                    output_message += f"You can keep the speed. {output_message}, and safety distance is smaller: {safety_distance} meters."
    else:
        output_message = f"There is no {vehicle_type} {position}"

    if log:
        print(f"[ACTION] Output message: {output_message}")

    text_to_speech(output_message)


def check_safety_distance_from_vehicle(vehicle=None, space_event=None, report_always: bool = True):
    position = "in front"
    too_close = False
    if vehicle is None:  # executed from stt => llm => function calling
        vehicle_types = ["car", "bus"]

        vehicle, space_event = get_vehicle_instance(vehicle_types, position)
        if vehicle is None or space_event is None:
            vehicle_types_str = " nor ".join(vehicle_types)
            output_message = f"There is no {vehicle_types_str} {position}"
            text_to_speech(output_message)
            return

    # Vehicle is not None and space_event is not None
    vehicle_type = vehicle.class_name

    vehicle_distance = space_event.object_distance

    max_distance = mada_config_dict.get("max_distance_from_camera", 6)
    if vehicle_distance > max_distance:
        output_message = f"{vehicle_type} {position} is more than {max_distance} meters away, which is the camera limit"
    else:
        current_speed, current_speed_str = get_current_speed(tts=False)
        output_message = f"{vehicle_type} {position} at {vehicle_distance} meters"

        if current_speed is None:
            output_message += f", but your speed is unknown and thus safety distance cannot be calculated."
        else:
            safety_distance = get_safety_distance(current_speed)

            if vehicle_distance < safety_distance and current_speed > 0:
                output_message = f"Reduce the speed. {output_message}, but safety distance is {safety_distance} meters."
                too_close = True
            else:
                output_message += f"You can keep the speed. {output_message}, and safety distance is {safety_distance} meters."

    if report_always:  # execute from stt => llm => function calling
        print(f"[ACTION] Output message: {output_message}")
        text_to_speech(output_message)
    elif too_close is True:
        print(f"[ACTION] Output message: {output_message}")
        audio_thread = threading.Thread(target=text_to_speech, args=(output_message,))
        # Start audio in a separate thread
        audio_thread.start()

    return too_close, output_message


def get_vehicle_instance(vehicle_types, position):
    # TODO: debe considerarse una ventana entre dos momentos:
    # - cuando se detecta el mensaje de entrada (el llm tarda varios segundos en procesar)
    # - unos segundos antes
    """
    returns most recent vehicle at a given position
    :param vehicle_types:
    :param position:
    :return:
    """
    driver_agent = get_driver_agent()
    memory = driver_agent.memory
    # objects_list is browsed in reverse order: from most recent
    for object in memory.objects_list[::-1]:
        if object.class_name in vehicle_types:
            # space event list is browsed in reverse order: from most recent
            for space_event in object.space_events[::-1]:
                if space_event.object_position == position:
                    return object, space_event
    return None, None


def get_current_speed(tts=True):
    driver_agent = get_driver_agent()
    memory = driver_agent.memory

    if len(memory.speed_events) == 0:
        current_speed = None
        current_speed_str = "unknown"
    else:
        current_speed = memory.speed_events[-1].speed
        current_speed_str = f"{current_speed} km/h"

    if tts:
        output_message = f"Your speed is {current_speed_str}."
        text_to_speech(output_message)

    return current_speed, current_speed_str


def get_safety_distance(current_speed, safety_time=2, min_distance=2):
    """

    :param current_speed: km/h
    :param safety_time: in seconds; should depend on weather conditions: greater with rain
    :param min_distance: m; if current_speed == 0 or very low, distance should be 2m at least (there is 1m from camera to car plate)
    :return safety_distance: m, for the current speed
    """
    safety_distance = round(current_speed * 1000/3600 * safety_time, 1)

    safety_distance = max(safety_distance, min_distance)

    return safety_distance


def main1():
    current_speed_l = [0, 5, 20, 50, 120]

    for current_speed in current_speed_l:
        safety_distance = get_safety_distance(current_speed)
        print(f"At {current_speed} km/h, safety distance {safety_distance} m")


if __name__ == "__main__":

    main1()
