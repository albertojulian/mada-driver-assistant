from driver_agent import get_driver_agent, check_safety_distance_from_vehicle_v0
from time import time, sleep


def memory_writing_test(log=True, add_delay=False):
    driver_agent = get_driver_agent(log=log)
    memory = driver_agent.memory

    image_width = 680
    image_height = 480

    params1 = dict()
    params1["IMAGE_WIDTH"] = image_width
    params1["IMAGE_HEIGHT"] = image_height
    params1["TRACK_ID"] = 5
    params1["CLASS_ID"] = 3
    params1["CLASS_NAME"] = "car"
    params1["BOUNDING_BOX"] = [300, 0, 500, 100]
    params1["OBJECT_DISTANCE"] = 30

    _, _ = memory.add_space_event(params1, log=log)

    if add_delay:
        sleep_time = 1.5
        sleep(sleep_time)

    speed = 50
    memory.add_speed_event(speed, log=log)

    if add_delay:
        sleep_time = 3.8
        sleep(sleep_time)

    _, _ = memory.add_space_event(params1, log=log)

    speed = 40
    memory.add_speed_event(speed, log=log)

    params2 = dict()
    params2["IMAGE_WIDTH"] = image_width
    params2["IMAGE_HEIGHT"] = image_height
    params2["TRACK_ID"] = 7
    params2["CLASS_ID"] = 4
    params2["CLASS_NAME"] = "bus"
    params2["BOUNDING_BOX"] = [0, 0, 100, 100]
    params2["OBJECT_DISTANCE"] = 20

    _, _ = memory.add_space_event(params2, log=log)

    if add_delay:
        sleep_time = 2.3
        sleep(sleep_time)

    speed = 30
    memory.add_speed_event(speed, log=log)


def memory_reading_test(extended_log=False):

    driver_agent = get_driver_agent()
    memory = driver_agent.memory

    print(f"\nMemory has:")
    print(f"- {len(memory.objects_list)} objects")
    print(f"- {len(memory.speed_events)} speed events")
    print(f"- {len(memory.accel_events)} accelerometer events")
    print(f"- {len(memory.gyro_events)} gyroscope events")
    print(f"- {len(memory.text_input_messages)} text input messages\n")

    if extended_log:
        for object in memory.objects_list:
            lifetime = round(object.get_lifetime(), 1)
            print(f'Object with id {object.track_id} has been alive for {lifetime} seconds')

        for speed_event in memory.speed_events:
            event_time = round(speed_event.creation_time, 1)
            print(f'Speed of {speed_event.speed} km/h at time {event_time}')

        for text_input_message in memory.text_input_messages:
            print(f"Received input message: {text_input_message}")


def main1():

    memory_writing_test(add_delay=True)
    memory_reading_test(extended_log=True)


def main2():
    memory_writing_test()
    log = True

    vehicle_type = "bus"
    position = "on the left"
    check_safety_distance_from_vehicle_v0(vehicle_type, position, log=log)

    vehicle_type = "car"
    position = "in front"
    check_safety_distance_from_vehicle_v0(vehicle_type, position, log=log)

    vehicle_type = "car"
    position = "on the right"
    check_safety_distance_from_vehicle_v0(vehicle_type, position, log=log)


if __name__ == "__main__":

    # main1()
    main2()
