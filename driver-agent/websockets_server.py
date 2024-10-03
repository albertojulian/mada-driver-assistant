# La app móvil informa la velocidad cada x segundos
# ORDEN:
# - Activar GPS y Conexion compartida (AndroidAJR) en el móvil
# - conectar WiFi Mac a AndroidAJR
# - ejecutar: wss (python websockets_server.py)
# - ejecutar app móvil SpeedVoiceWebSocket
# - ejecutar: od (sudo python "object_detector.py")

# Cuando el Mac usa AndroidAJR del móvil, la IP es 192.168.43.233
# IP del móvil: 192.168.0.11
# instalo websockets por PyCharm (equivale a pip install websockets)
import asyncio
import websockets
import subprocess
from driver_agent import get_driver_agent
import yaml
from time import time
import sys
sys.path.append('../common')
from text_to_speech import text_to_speech


def wifi_info_mac():
    # Ejecutar el comando `airport` para obtener el SSID
    result = subprocess.run(
        ['/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport', '-I'],
        capture_output=True, text=True)
    output = result.stdout
    for line in output.split('\n'):
        if ' SSID' in line:
            ssid = line.split(":")[1].strip()
            return ssid


def check_mac_wifi_connection():
    mada_file = "driver_agent.yaml"
    with open(mada_file) as file:
        mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

    mac_ssid = wifi_info_mac()
    android_wifi = mada_config_dict.get("android_wifi", "AndroidAJR")
    status_ok = False
    if mac_ssid == android_wifi:
        mac_connected_and_phone_app = mada_config_dict.get("mac_connected_and_phone_app", "Connection with phone OK")
        message = mac_connected_and_phone_app
        status_ok = True
    else:
        mac_not_connected = mada_config_dict.get("mac_not_connected_and_phone_app",
                                                               "Connection with phone not OK")
        message = mac_not_connected

    text_to_speech(message)

    return status_ok


async def handle_events(websocket, path):
    # Agregar la conexión WebSocket a la lista
    connections.append(websocket)
    print("Client connection")
    try:

        async for message in websocket:
            if log:
                print(f"Message received: {message}")

            if message == phone_connected:
                text_to_speech(phone_connected_start_object_detector)

            elif message.split()[0] == "setSpaceEvent":
                # print("Space event received")
                params_str = " ".join(message.split()[1:])
                driver_agent.evaluate_automatic_action(params_str)

            elif message.split()[0] == "setInputMessage":
                text_input_message = " ".join(message.split()[1:])
                if text_input_message == "listen":
                    driver_agent.memory.listen_mode = True
                    text_to_speech(listening_ack, print_message=False)

                # elif input_message == "stop":
                #     driver_agent.memory.listen_mode = False  # enables proactive actions

                else:
                    if driver_agent.memory.listen_mode is True:
                        driver_agent.evaluate_action_from_request(text_input_message)
                        driver_agent.memory.listen_mode = False  # enables proactive actions

            elif message.split()[0] == "setSpeed":
                speed = int(message.split()[1])
                driver_agent.memory.add_speed_event(speed, log=log)

            elif message.split()[0] == "setAccel":
                accel_coords = message.split()[1].split(";")
                driver_agent.memory.add_accel_event(accel_coords, log=log)

            elif message.split()[0] == "setGyro":
                gyro_coords = message.split()[1].split(";")
                driver_agent.memory.add_gyro_event(gyro_coords, log=log)

            current_time = time()
            duration = current_time - init_time
            interval = round(duration % 5, 1)  # print memory content every 5 seconds
            if interval == 0.0:
                # memory_reading_test()
                driver_agent.memory.print_content()

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Client webSocket connection closed")
    finally:
        # Eliminar la conexión WebSocket de la lista cuando se cierra
        connections.remove(websocket)


def main1():
    ssid = wifi_info_mac()
    print(f"Conectado a la red WiFi: {ssid}")


if __name__ == "__main__":

    # main1()
    mada_file = "driver_agent.yaml"
    with open(mada_file) as file:
        mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

    # Store all WebSocket connections in a list
    connections = []

    log = mada_config_dict.get("log", True)
    driver_agent = get_driver_agent(log=log)
    init_time = driver_agent.memory.init_time

    phone_connected = mada_config_dict.get("phone_connected", "Android phone is connected")
    phone_connected_start_object_detector = mada_config_dict.get("phone_connected_start_object_detector",
                                                                 "Start object detector")
    listening_ack = mada_config_dict.get("listening_ack", "I am listening")

    status_ok = check_mac_wifi_connection()

    if status_ok:
        print("\n<<<<<<<<<<<< Starting WebSockets Server >>>>>>>>>>>>\n")

        ws_port = mada_config_dict.get("ws_port", 8765)
        start_server = websockets.serve(handle_events, "0.0.0.0", ws_port)  # port to listen websockets
        # start_server = websockets.serve(handle_client, "localhost", 8000)

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()





