import yaml
from typing import Literal
from memory import get_memory, SpaceEvent, MadaObject

import sys; sys.path.append("../common")

from text_to_speech import text_to_speech, text_to_speech_async
from utils import get_most_frequent_value, is_float


mada_file = "../mada.yaml"

with open(mada_file) as file:
    mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)


def check_safety_distance_from_vehicle(vehicle: MadaObject, space_event: SpaceEvent,
                                           report_always: bool = True) -> (bool, str):
    """
    Checks if we are at a safe distance from the vehicle in front.
    :param vehicle:
    :param space_event:
    :param report_always:
    :return:
    """

    vehicle_types = ["car", "bus"]
    position = "in front"

    if vehicle is None or space_event is None:
        vehicle_types_str = " nor ".join(vehicle_types)
        output_message = f"There is no {vehicle_types_str} {position}"
        text_to_speech(output_message)
        return False, None


    vehicle_type = vehicle.class_name
    position = "in front"
    too_close = False

    vehicle_distance = space_event.object_distance

    camera_conf = mada_config_dict["camera"]
    max_distance = camera_conf.get("max_distance_from_camera", 6)

    speed_conf = mada_config_dict["speed"]
    speed_threshold = speed_conf["speed_threshold"]  # in km/h; minimum to check safety distance

    if vehicle_distance > max_distance:
        output_message = f"{vehicle_type} {position} is more than {max_distance} meters away, which is the camera limit"
    else:
        current_speed, current_speed_str = get_current_speed(tts=False)
        output_message = f"{vehicle_type} {position} at {vehicle_distance} meters. "

        if current_speed is None:
            output_message += f", but your speed is unknown and thus safety distance cannot be calculated."
        else:
            safety_distance = get_safety_distance(current_speed)

            if vehicle_distance < safety_distance and current_speed > speed_threshold:
                output_message = f"Reduce the speed. {output_message}, but safety distance is {safety_distance} meters."
                too_close = True
            else:
                output_message = f"You can keep the speed. {output_message}, and safety distance is {safety_distance} meters."

    if report_always:  # executed from stt => llm => function calling
        pass
        # print(f"[ACTION] Output message: {output_message}")

    elif too_close is True:
        space_event2tts(output_message)

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
    for object in memory.mada_objects_list[::-1]:
        if object.class_name in vehicle_types:
            # space event list is browsed in reverse order: from most recent
            for space_event in object.space_events[::-1]:
                if space_event.object_position == position:
                    return object, space_event
    return None, None



def get_current_speed(tts: bool = True) -> (float, str):
    """
    Gets the current speed in km/h.
    :param tts:
    :return:
    """
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


def get_safety_distance(current_speed: float, reaction_time: float = 2, min_distance: float = 2) -> float:
    """
    Calculates the safety distance (in meters) with respect to a vehicle in front, given the current speed and the reaction time.
    :param current_speed: km/h
    :param reaction_time: in seconds; default 2 (typical rule of thumb); also depends on weather condition (rain) or darkness
    :param min_distance: m; if current_speed == 0 or very low, distance should be 2m at least (there is 1m from camera to car plate)
    :return safety_distance: m, for the current speed
    """
    safety_distance = round(current_speed * 1000/3600 * reaction_time, 1)

    safety_distance = max(safety_distance, min_distance)

    return safety_distance


def get_last_speed_limit_sign(tts: bool = True) -> (float, str):
    """
    Returns the speed limit (in km/h) from the last speed limit sign detected.
    :return:
    """

    memory = get_memory()

    # objects_list is browsed in reverse order: from most recent
    for object in memory.mada_objects_list[::-1]:
        if object.class_name == "speed limit":
            if len(object.speed_limits) > 0:
                speed_limit = get_most_frequent_value(object.speed_limits)
                speed_limit_str = f"{speed_limit} km/h"

                if tts:
                    output_message = f"Speed limit is {speed_limit_str}"
                    text_to_speech(output_message)

                return speed_limit, speed_limit_str

    if tts:
        output_message = f"No speed limit sign has been well detected"
        text_to_speech_async(output_message)

    return None, None


def space_event2tts(output_message):
    from driver_agent import get_driver_agent

    driver_agent = get_driver_agent()
    disable_audio = driver_agent.listen_mode
    # audio is disabled when driver agent is in listen mode to avoid mixing driver speech with audio automatic notifications
    text_to_speech(output_message, disable_audio=disable_audio)


def main1():
    # current_speed_l = [0, 5, 13, 15, 20, 30, 40, 50, 60, 120]

    current_speed_l = [10, 50, 58]

    reaction_time = 2
    for current_speed in current_speed_l:
        safety_distance = get_safety_distance(current_speed, reaction_time=reaction_time)
        print(f"At {current_speed} km/h, with a reaction time of {reaction_time} s, safety distance {safety_distance} m")

    reaction_time = 3
    for current_speed in current_speed_l:
        safety_distance = get_safety_distance(current_speed, reaction_time=reaction_time)
        print(f"At {current_speed} km/h, with a reaction time of {reaction_time} s, safety distance {safety_distance} m")


def main2():
    get_last_speed_limit_sign()


def main3():
    check_safety_distance_from_vehicle(vehicle=None, space_event=None, report_always=True)


if __name__ == "__main__":

    main1()
    # main2()
    # main3()
