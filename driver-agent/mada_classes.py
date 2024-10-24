from memory import MadaObject, get_memory
import yaml
from functions import check_safety_distance_from_vehicle, get_current_speed
import sys
sys.path.append("../common")
from text_to_speech import text_to_speech_async


mada_file = "driver_agent.yaml"
with open(mada_file) as file:
    mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

# time between actions: avoids redundant actions about the same object
TIME_BETWEEN_ACTIONS = mada_config_dict.get("time_between_actions", 2)


class Bicycle(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        time_since_last_action = self.time_since_last_action()

        if log:
            print(f"time_since_last_action {time_since_last_action}")

        # if last action was more than x seconds ago, perform action
        if time_since_last_action > TIME_BETWEEN_ACTIONS:
            if object_position == "in front":
                too_close, output_message = check_safety_distance_from_vehicle(vehicle=self, space_event=space_event,
                                                                               report_always=False)
                if too_close:
                    self.add_action_event(params, output_message)


class Bus(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        time_since_last_action = self.time_since_last_action()

        if log:
            print(f"time_since_last_action {time_since_last_action}")

        # if last action was more than x seconds ago, perform action
        if time_since_last_action > TIME_BETWEEN_ACTIONS:
            if object_position == "in front":
                too_close, output_message = check_safety_distance_from_vehicle(vehicle=self, space_event=space_event,
                                                                               report_always=False)
                if too_close:
                    self.add_action_event(params, output_message)


class Car(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        time_since_last_action = self.time_since_last_action()

        if log:
            print(f"time_since_last_action {time_since_last_action}")

        # if last action was more than x seconds ago, perform action
        if time_since_last_action > TIME_BETWEEN_ACTIONS:
            if object_position == "in front":
                too_close, output_message = check_safety_distance_from_vehicle(vehicle=self, space_event=space_event,
                                                                               report_always=False)
                if too_close:
                    self.add_action_event(params, output_message)


class Construction(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"{self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class CyclesCrossing(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"Warning, {self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class DeadEndStreet(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"{self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class GiveWay(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"Reduce speed, {self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class GoLeft(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"{self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class GoRight(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"{self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class Motorcycle(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        time_since_last_action = self.time_since_last_action()

        if log:
            print(f"time_since_last_action {time_since_last_action}")

        # if last action was more than x seconds ago, perform action
        if time_since_last_action > TIME_BETWEEN_ACTIONS:
            if object_position == "in front":
                too_close, output_message = check_safety_distance_from_vehicle(vehicle=self, space_event=space_event,
                                                                               report_always=False)
                if too_close:
                    self.add_action_event(params, output_message)


class NoEntry(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"{self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class NoLeftTurn(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"{self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class NoOvertaking(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"{self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class NoPriority(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"{self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class NoRightTurn(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"{self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class PedestrianCrossing(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"Warning, {self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class Person(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        time_since_last_action = self.time_since_last_action()
        object_distance = space_event.object_distance

        # if last action was more than x seconds ago, perform action
        if time_since_last_action > TIME_BETWEEN_ACTIONS:
            max_distance = mada_config_dict.get("max_distance_from_camera", 6)
            if object_distance > max_distance:
                output_message = f"{self.class_name} {object_position} is more than {max_distance} meters away."
            else:
                output_message = f"{self.class_name} {object_position} at {object_distance} meters."

            if log:
                print(f"[ACTION] class_id: {self.class_id}, output message: {output_message}")

            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class Roundabout(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"{self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class SchoolCrossing(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"Warning, {self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class SpeedLimit(MadaObject):

    def __init__(self, params, init_time):
        self.speed_limit = []

        super().__init__(params, init_time)

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:

            speed_limit = params.get("SPEED_LIMIT", None)

            if speed_limit is not None:
                current_speed, current_speed_str = get_current_speed(tts=False)

                if current_speed is None:
                    output_message = f"There is a {speed_limit} km/h {self.class_name} {object_position}"
                    self.add_action_event(params, output_message)
                    space_event2tts(output_message)
                elif current_speed > speed_limit:
                    output_message = (f"Reduce speed, your current speed is {current_speed_str} "
                                      f"but there is a {speed_limit} km/h {self.class_name} {object_position}")
                    self.add_action_event(params, output_message)
                    space_event2tts(output_message)


class Stop(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        # report only once
        if len(self.action_events) == 0:
            output_message = f"{self.class_name} sign {object_position}"
            self.add_action_event(params, output_message)

            space_event2tts(output_message)


class TrafficLight(MadaObject):

    def __init__(self, params, init_time):
        self.traffic_light_colors = []

        super().__init__(params, init_time)

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position
        # n_frame = params["N_FRAME"]  # TODO borrar
        traffic_light_color = params["TRAFFIC_LIGHT_COLOR"]
        # print(f"DA n_frame {n_frame}, traffic_light_color {traffic_light_color}")  # TODO borrar

        if traffic_light_color == "off":  # avoid storing "off" light state
            return

        if traffic_light_color == "red":
            # on red start or red from not red
            if len(self.traffic_light_colors) == 0 or self.traffic_light_colors[-1] != "red":
                output_message = f"Stop, {traffic_light_color} {self.class_name} {object_position}"
                self.add_action_event(params, output_message)
                space_event2tts(output_message)

        elif traffic_light_color == "green":
            # on green start or green from not green
            if len(self.traffic_light_colors) == 0 or self.traffic_light_colors[-1] != "green":
                output_message = f"You can go on, {traffic_light_color} {self.class_name} {object_position}"
                self.add_action_event(params, output_message)
                space_event2tts(output_message)

        else:    # yellow
            # on yellow start or yellow from red (pedestrians may cross)
            if len(self.traffic_light_colors) == 0 or self.traffic_light_colors[-1] == "red":
                output_message = f"Warning, pedestrians may cross, {traffic_light_color} {self.class_name} {object_position}"
                self.add_action_event(params, output_message)
                space_event2tts(output_message)

            # on yellow from green (reduce speed)
            elif self.traffic_light_colors[-1] == "green":
                output_message = f"Reduce speed, {traffic_light_color} {self.class_name} {object_position}"
                self.add_action_event(params, output_message)
                space_event2tts(output_message)

        self.traffic_light_colors.append(traffic_light_color)


class Truck(MadaObject):

    def manage_space_event(self, params, log=False):
        # call MadaObject manage_space_event method
        space_event = super().manage_space_event(params, log)
        object_position = space_event.object_position

        time_since_last_action = self.time_since_last_action()

        if log:
            print(f"time_since_last_action {time_since_last_action}")

        # if last action was more than x seconds ago, perform action
        if time_since_last_action > TIME_BETWEEN_ACTIONS:
            if object_position == "in front":
                too_close, output_message = check_safety_distance_from_vehicle(vehicle=self, space_event=space_event,
                                                                               report_always=False)
                if too_close:
                    self.add_action_event(params, output_message)


def space_event2tts(output_message):
    memory = get_memory()
    if memory.listen_mode is False:

        text_to_speech_async(output_message)