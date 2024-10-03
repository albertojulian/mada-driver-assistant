import asyncio
import websockets
import subprocess
import yaml
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
                print("Space event received")

            elif message.split()[0] == "setInputMessage":
                text_input_message = " ".join(message.split()[1:])
                print(f"Driver said {text_input_message}")

            elif message.split()[0] == "setSpeed":
                print("Speed event received")

            elif message.split()[0] == "setAccel":
                print("Accel event received")

            elif message.split()[0] == "setGyro":
                print("Gyro event received")

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Client webSocket connection closed")
    finally:
        # Eliminar la conexión WebSocket de la lista cuando se cierra
        connections.remove(websocket)

if __name__ == "__main__":

    mada_file = "driver_agent.yaml"
    with open(mada_file) as file:
        mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

    # Store all WebSocket connections in a list
    connections = []

    log = mada_config_dict.get("log", True)
    phone_connected = mada_config_dict.get("phone_connected", "Android phone is connected")
    phone_connected_start_object_detector = mada_config_dict.get("phone_connected_start_object_detector", "Start object detector")
    listening_ack = mada_config_dict.get("listening_ack", "I am listening")

    status_ok = check_mac_wifi_connection()

    if status_ok:
        print("\n<<<<<<<<<<<< Starting WebSockets Server >>>>>>>>>>>>\n")

        ws_port = mada_config_dict.get("ws_port", 8765)
        start_server = websockets.serve(handle_events, "0.0.0.0", ws_port)  # port to listen websockets
        # start_server = websockets.serve(handle_client, "localhost", 8000)

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

