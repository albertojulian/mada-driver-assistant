# from langchain.agents import tool  # @tool, old version
import json

from functions import *

from langchain_core.tools import tool  # @tool, new version

@tool
def check_safety_distance_from_vehicle_lg(inputs: str) -> (bool, str):
    """
    Checks if we are at a safe distance from the vehicle in front.
    """
    vehicle_types = ["car", "bus"]
    report_always = True
    position = "in front"

    vehicle, space_event = get_vehicle_instance(vehicle_types, position)
    too_close, output_message = check_safety_distance_from_vehicle(vehicle, space_event, report_always)
    return too_close, output_message


@tool
def get_current_speed_lg() -> (float, str):
    """
    Gets the current speed in km/h.
    """
    current_speed, current_speed_str = get_current_speed(tts=False)

    return current_speed, "km/h"


@tool
def get_safety_distance_lg(current_speed: float = None, reaction_time: float = 2, min_distance: float = 2) -> (float, str):
    """
        Calculates the safety distance (meters) given the current speed (km/h) and the reaction time (seconds; optional; default is 2).
    """
    if current_speed is None:
        return 0

    safety_distance = get_safety_distance(current_speed, reaction_time, min_distance)
    return safety_distance, "meters"


@tool
def get_last_speed_limit_sign_lg() -> (float, str):
    """
    Returns the speed limit (km/h) from the last speed limit sign detected.
    """
    speed_limit, _ = get_last_speed_limit_sign(tts=False)

    return speed_limit, "km/h"


