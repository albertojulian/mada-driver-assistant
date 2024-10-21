from time import time
import sys
sys.path.append("../common")
from utils import is_float
from text_to_speech import text_to_speech_async


class Memory:

    def __init__(self):
        self.mada_objects_dict = {}
        self.mada_objects_list = []
        self.speed_events = []
        self.accel_events = []
        self.gyro_events = []
        self.text_input_messages = []
        self.init_time = time()  # time reference for all events
        self.listen_mode = False
        self.check_safe_distance = False

        print("\n<<<<<<<<<<<< Starting Memory >>>>>>>>>>>>")

    def manage_space_event(self, params, log=False):
        mada_object, is_new_object = self.get_mada_object(params, log=log)
        if mada_object is None:
            return None

        mada_object.manage_space_event(params, log=log)

        # return mada_object, is_new_object

    def get_mada_object(self, params, log=False):
        # TODO: is is_new_object used??
        track_id = params["TRACK_ID"]
        mada_object = self.mada_objects_dict.get(track_id, None)

        if mada_object is None:  # no object with the track_id => create new object
            mada_object = self.add_mada_object(params, log=log)
            is_new_object = True
        else:
            is_new_object = False

        return mada_object, is_new_object

    def add_mada_object(self, params, log=False):
        track_id = params["TRACK_ID"]

        # mada_object = MadaObject(params, self.init_time)
        mada_class_name = params["CLASS_NAME"]
        mada_object = self.create_mada_object(mada_class_name, params)

        if mada_object is None:
            return None

        self.mada_objects_dict[track_id] = mada_object
        self.mada_objects_list.append(mada_object)
        if log:
            print(f"Creating object with id {track_id}")

        return mada_object

    def create_mada_object(self, mada_class_name, params):
        python_mada_class_name = camel_case(mada_class_name)

        # class in same module or script
        class_ref = globals().get(python_mada_class_name, None)
        if class_ref is None:
            # class in another module
            import mada_classes
            class_ref = getattr(mada_classes, python_mada_class_name, None)

        if class_ref is None:
            output = f"There is no class {python_mada_class_name} defined"
            print(output)
            text_to_speech_async(output)

            return None

        mada_object = class_ref(params, self.init_time)

        return mada_object

    def add_speed_event(self, speed, log=False):
        speed_event = SpeedEvent(speed, self.init_time)
        self.speed_events.append(speed_event)
        if log:
            print(f"Creating speed = {speed} event")

    def add_accel_event(self, accel_coords, log=False):
        if len(accel_coords) != 3:
            return

        floats = [is_float(coord) for coord in accel_coords]
        if sum(floats) != 3:
            return

        accel_event = AccelEvent(accel_coords, self.init_time)
        self.accel_events.append(accel_event)
        if log:
            print(f"Creating accel event")

    def add_gyro_event(self, gyro_coords, log=False):
        if len(gyro_coords) != 3:
            return

        floats = [is_float(coord) for coord in gyro_coords]
        if sum(floats) != 3:
            return

        gyro_event = GyroEvent(gyro_coords, self.init_time)
        self.gyro_events.append(gyro_event)
        if log:
            print(f"Creating gyro event")

        umbral = 0.1
        message = ""
        z = gyro_coords[2]
        if z > umbral:
            message = "Turning right"
        elif z < -umbral:
            message = "Turning left"

        text_to_speech_async(message)

    def add_text_input_message(self, text_input_message, log=False):
        text_input_message_event = SpeechToTextEvent(text_input_message, self.init_time)
        self.text_input_messages.append(text_input_message_event)
        if log:
            print(f"Creating input message event: {text_input_message}")

        return text_input_message_event

    """
    def add_action_event(self, params, output_message, log=False):
        object, is_new_object = self.get_object(params, log=log)
        object.add_action_event(params, output_message, log=log)
    """

    def print_content(self, extended_log=False):

        print(f"\nMemory contains:")
        print(f"- {len(self.mada_objects_list)} objects")
        print(f"- {len(self.speed_events)} speed events")
        print(f"- {len(self.accel_events)} accelerometer events")
        print(f"- {len(self.gyro_events)} gyroscope events")
        print(f"- {len(self.text_input_messages)} text input messages\n")

        if extended_log:
            for mada_object in self.mada_objects_list:
                lifetime = round(mada_object.get_lifetime(), 1)
                print(f'Object with id {mada_object.track_id} has been alive for {lifetime} seconds')
                # TODO: implement and call mada_object.print_content()

            for speed_event in self.speed_events:
                event_time = round(speed_event.creation_time, 1)
                print(f'Speed of {speed_event.speed} km/h at time {event_time}')

            for text_input_message in self.text_input_messages:
                print(f"Received input message: {text_input_message}")


# from class name "traffic light" to class ref "TrafficLight" (camel case)
def camel_case(mada_class_name):
    camel_mada_class_name = "".join([word.capitalize() for word in mada_class_name.split()])
    return camel_mada_class_name


_memory_instance = None


def get_memory(log=False):
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = Memory()
    else:
        if log:
            print("Memory already exists")

    return _memory_instance


class MadaObject:

    """
    MadaObject can be vehicles (car, bus), traffic signs, traffic lights, people?
    """
    def __init__(self, params, init_time):
        self.image_width = params["IMAGE_WIDTH"]
        self.image_height = params["IMAGE_HEIGHT"]
        self.track_id = params["TRACK_ID"]

        self.class_name = params["CLASS_NAME"]
        self.class_id = params["CLASS_ID"]

        self.init_time = init_time
        self.space_events = []
        self.action_events = []

    # overriden by subclasses
    def manage_space_event(self, params, log=False):
        space_event = self.add_space_event(params, log)
        return space_event

    def add_space_event(self, params, log=False):
        space_event = SpaceEvent(params, self.init_time)
        self.space_events.append(space_event)
        if log:
            print(f"Creating space event")

        return space_event

    def get_lifetime(self):

        if len(self.space_events) == 0:  # no space events yet
            return 0

        space_events_init_time = self.space_events[0].creation_time  # time of first space event
        space_events_end_time = self.space_events[-1].creation_time  # time of last space event
        return space_events_end_time - space_events_init_time

    def add_action_event(self, params, output_message, log=False):
        action_event = ActionEvent(output_message, self.init_time)
        self.action_events.append(action_event)
        if log:
            print(f"Creating action event")

    def time_since_last_action(self):
        if len(self.action_events) == 0:  # no action events yet
            return 1000

        current_time = time() - self.init_time

        return current_time - self.action_events[-1].creation_time

    def print_content(self):
        # TODO: implement
        pass


class Event:
    def __init__(self, init_time):
        self.creation_time = time() - init_time


class SpaceEvent(Event):
    def __init__(self, params, init_time):
        super().__init__(init_time)

        self.image_width = params["IMAGE_WIDTH"]
        self.bbox = params["BOUNDING_BOX"]
        bbox_xc = (self.bbox[0] + self.bbox[2]) / 2  # x center of bounding box
        left_limit = self.image_width / 3
        right_limit = self.image_width * 2 / 3

        if bbox_xc < left_limit:
            self.object_position = "on the left"
        elif bbox_xc > right_limit:
            self.object_position = "on the right"
        else:
            self.object_position = "in front"

        self.object_distance = params["OBJECT_DISTANCE"]


class SpeedEvent(Event):
    def __init__(self, speed, init_time):
        super().__init__(init_time)
        self.speed = speed


class AccelEvent(Event):
    def __init__(self, accel_coords, init_time):
        super().__init__(init_time)
        self.x, self.y, self.z = [float(coord) for coord in accel_coords]


class GyroEvent(Event):
    def __init__(self, gyro_coords, init_time):
        super().__init__(init_time)
        self.x, self.y, self.z = [float(coord) for coord in gyro_coords]


class ActionEvent(Event):
    def __init__(self, output_message, init_time):
        super().__init__(init_time)
        self.output_message = output_message


class SpeechToTextEvent(Event):
    def __init__(self, text_input_message, init_time):
        super().__init__(init_time)
        self.text_input_message = text_input_message
        self.processing_time = 0


if __name__ == "__main__":

    mada_class_names = ['bicycle', 'bus', 'car', 'construction', 'cycles crossing', 'dead end street',
                       'give way', 'go left', 'go right', 'motorcycle', 'no entry', 'no left turn', 'no overtaking',
                       'no priority', 'no right turn', 'pedestrian crossing', 'person', 'roundabout',
                       'school crossing', 'speed limit', 'stop', 'traffic light', 'truck']

    mada_class_name = "traffic light"
    python_mada_class_name = camel_case(mada_class_name)

    params = {}
    params["IMAGE_WIDTH"] = 640
    params["IMAGE_HEIGHT"] = 480
    params["TRACK_ID"] = 13
    params["CLASS_NAME"] = mada_class_name
    params["CLASS_ID"] = 13

    init_time = 10

    # class in same module or script
    class_ref = globals().get(python_mada_class_name, None)
    if class_ref is None:
        # class in another module
        import mada_classes
        class_ref = getattr(mada_classes, python_mada_class_name, None)

    if class_ref is None:
        print(f"There is no class {python_mada_class_name} defined")
        exit()

    mada_object = class_ref(params, init_time)

    print(mada_object.image_width)