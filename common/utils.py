from gtts import gTTS, tts as gtts
from time import time

# pip install mycroft-mimic3-tts[all]  # Removing [all] will install support for English only.
import subprocess
import yaml

import os

# pip install PyPDF2
# from PyPDF2 import PdfReader

# pip install pdfminer.six
# from pdfminer.high_level import extract_text


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

def pdf2txt(file_pdf):
    file_txt = file_pdf[:-4] + ".txt"
    drive_str = extract_text(file_pdf)
    with open(file_txt, "w") as file:
        file.write(drive_str)
"""


def is_float(str):
    try:
        float(str)
        return 1
    except ValueError:
        return 0

def is_int(str):
    try:
        int(str)
        return 1
    except ValueError:
        return 0

def main1():
    file = "../Small Language Models AJR/0 traffic docs/dh-chapter3.pdf"

    # drive1_txt = read_pdf_v1(file)
    # print(drive1_txt)

    pdf2txt(file)

def main2():
    start_time = time()
    message = "Your"  # speed is 70 kilometers per hour, but limit is 90. You can increase speed
    for _ in range(10):
        say_message1(message)

    end_time = time()
    process_time = end_time - start_time
    print(round(process_time, 2))

    # 10 repeticiones de una palabra tardan 20 s de mimic3 (local) y 12 de gtts (internet por móvil)

if __name__ == "__main__":

    # main1()
    main2()
