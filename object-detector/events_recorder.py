import asyncio
import websockets
import subprocess
import threading
import yaml
from time import time
import os
import sys; sys.path.append("../common")
from text_to_speech import text_to_speech
from utils import check_mac_wifi_connection


async def handle_events(websocket, path):
    # Agregar la conexión WebSocket a la lista
    connections.append(websocket)
    print("Client connection")

    global video_id_frame

    try:

        async for message in websocket:
            if log:
                print(f"Message received: {message}")

            if message == phone_connected:
                text_to_speech(phone_connected_start_video_recorder)

            elif message.split()[0] == "frameEvent":
                video_id_frame = message.split()[1]

            elif message.split()[0] == "setSpeed":
                print("Speed event received")
                speed = int(message.split()[1])
                if video_id_frame is not None:
                    print(f"speed {speed} at video_id & frame {video_id_frame}")
                    video_id, n_frame = video_id_frame.split("_")
                    speed_file = f"speed_{video_id}.txt"
                    speed_file_path = os.path.join(out_video_dir, speed_file)
                    speed_line = f"{speed} {n_frame}\n"
                    with open(speed_file_path, "a") as file:
                        file.write(speed_line)

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Client webSocket connection closed")
    finally:
        # Eliminar la conexión WebSocket de la lista cuando se cierra
        connections.remove(websocket)

if __name__ == "__main__":

    mada_file = "../mada.yaml"
    with open(mada_file) as file:
        mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

    # Store all WebSocket connections in a list
    connections = []
    video_id_frame = None
    recorded_video = mada_config_dict["recorded_video"]
    out_video_dir = recorded_video["in_video_dir"]


    log = mada_config_dict.get("log", True)
    communications = mada_config_dict["communications"]
    phone_connected = communications.get("phone_connected", "Android phone is connected")
    phone_connected_start_video_recorder = communications.get("phone_connected_start_video_recorder",
                                                                 "Start Video Recorder")
    listening_ack = communications.get("listening_ack", "I am listening")

    status_ok = check_mac_wifi_connection()

    if status_ok:
        print("\n<<<<<<<<<<<< Starting WebSockets Server >>>>>>>>>>>>\n")

        ws_port = communications["ws_port"]
        start_server = websockets.serve(handle_events, "0.0.0.0", ws_port)  # port to listen websockets
        # start_server = websockets.serve(handle_client, "localhost", 8000)

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
