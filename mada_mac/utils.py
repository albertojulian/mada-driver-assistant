from gtts import gTTS, tts as gtts
from time import time

# pip install mycroft-mimic3-tts[all]  # Removing [all] will install support for English only.
import subprocess
import yaml

import os
import cv2

# pip install PyPDF2
# from PyPDF2 import PdfReader

# pip install pdfminer.six
from pdfminer.high_level import extract_text

mada_file = "mada.yaml"
with open(mada_file) as file:
    mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)


# Original voices are slow => increase with audio_speed=1.75 as the -r parameter in afplay
def say_message2(message, audio_speed=1.75, print_message=True, audio_file="tts_out.mp3"):

    if print_message:
        print("\n<<<<<<<<<<< Start printing audio message: ")
        print(message)
        print(">>>>>>>>>>>>>> End printing audio message\n")

    # Create the gTTS object with the message
    # tts = gTTS(message, lang='es')
    tts = gTTS(message, lang='en')

    # Save the audio file with the message
    try:
        tts.save(audio_file)

        # play the file with the message with Apple's afplay command
        # Original voices are slow => increase with the -r parameter in afplay
        # os.system(f"afplay {audio_file} -r {audio_speed}")
        subprocess.run(f"afplay {audio_file} -r {audio_speed}", shell=True)

        return True

    except gtts.gTTSError:
        print("#######################################")
        print("gTTS Connection error; check if the Mac is connected to internet (AndroidAJR)")
        print("#######################################")

        connection_error_audio_file = "gtts_connection_error.mp3"
        if os.path.exists(connection_error_audio_file):
            subprocess.run(f"afplay {connection_error_audio_file} -r {audio_speed}", shell=True)
        else:
            print(f"Cannot find file {connection_error_audio_file}")


def say_message1(message, audio_speed=1.5, audio_file="tts_out.wav"):  # mimic3
    # mimic3 --voice <voice> "<text>" > output.wav

    subprocess.run(f"mimic3 '{message}' > {audio_file}", shell=True)
    subprocess.run(f"afplay {audio_file} -r {audio_speed}", shell=True)


"""
def read_pdf_v1(pdf_file):
    drive_txt = ""
    reader = PdfReader(pdf_file)
    for page_id in range(len(reader.pages)):
        page = reader.pages[page_id]
        drive_txt += page.extract_text()
    return drive_txt
"""


def pdf2txt(file_pdf):
    file_txt = file_pdf[:-4] + ".txt"
    drive_str = extract_text(file_pdf)
    with open(file_txt, "w") as file:
        file.write(drive_str)


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


def is_float(str):
    try:
        float(str)
        return 1
    except ValueError:
        return 0



def main1():
    video_name = "rgb_4.mp4"  # 1280x720; 266 frames
    in_video_path = os.path.join("./videos", video_name)

    cap = cv2.VideoCapture(in_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(fps, "fps")


def main2():
    file = "../Small Language Models AJR/0 traffic docs/dh-chapter3.pdf"

    # drive1_txt = read_pdf_v1(file)
    # print(drive1_txt)

    pdf2txt(file)


def main3():
    ssid = wifi_info_mac()
    print(f"Conectado a la red WiFi: {ssid}")



if __name__ == "__main__":

    # main1()
    # main2()
    main3()
