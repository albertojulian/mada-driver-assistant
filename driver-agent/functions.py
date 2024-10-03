import sys
sys.path.append('../common')
from text_to_speech import text_to_speech
import yaml
from typing import Literal
import threading
from memory import get_memory

mada_file = "driver_agent.yaml"
with open(mada_file) as file:
    mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)


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

    memory = get_memory()
    # objects_list is browsed in reverse order: from most recent
    for object in memory.objects_list[::-1]:
        if object.class_name in vehicle_types:
            # space event list is browsed in reverse order: from most recent
            for space_event in object.space_events[::-1]:
                if space_event.object_position == position:
                    return object, space_event
    return None, None


def get_current_speed(tts=True):
    memory = get_memory()

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
    current_speed_l = [0, 5, 13, 15, 20, 30, 120]

    for current_speed in current_speed_l:
        safety_distance = get_safety_distance(current_speed)
        print(f"At {current_speed} km/h, safety distance {safety_distance} m")


if __name__ == "__main__":

    main1()
