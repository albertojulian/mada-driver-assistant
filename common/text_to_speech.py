from gtts import gTTS, tts as gtts
import subprocess
import os
from time import time


def text_to_speech(message, audio_speed=1.75, print_message=True, audio_file="tts_out.mp3"):

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

        audiofile_to_speech(audio_file, audio_speed)

        return True

    except gtts.gTTSError:
        print("#######################################")
        print("gTTS Connection error; check if the Mac is connected to internet (AndroidAJR)")
        print("#######################################")

        connection_error_audio_file = "gtts_connection_error.mp3"
        audiofile_to_speech(connection_error_audio_file, audio_speed)


def audiofile_to_speech(audio_file, audio_speed=1.75):

    # play the file with the message with Apple's afplay command
    # Original voices are slow => increase with the -r parameter in afplay
    # os.system(f"afplay {audio_file} -r {audio_speed}")

    if os.path.exists(audio_file):
        subprocess.run(f"afplay {audio_file} -r {audio_speed}", shell=True)
    else:
        print(f"Cannot find file {audio_file}")


def create_gtts_connection_error_audio():

    message = "gtts connection error. Check if the Mac is connected to internet"
    gtts_ok = text_to_speech(message)
    if gtts_ok:
        import shutil
        shutil.copy("tts_out.mp3", "gtts_connection_error.mp3")


def main1():
    start_time = time()
    message = "Your"  # speed is 70 kilometers per hour, but limit is 90. You can increase speed
    for _ in range(10):
        text_to_speech(message)

    end_time = time()
    process_time = end_time - start_time
    print(round(process_time, 2))

    # 10 repeticiones de una palabra tardan 21.95s de mimic3 (local) y 12 de gtts (internet por móvil)


if __name__ == "__main__":

    # main1()
    create_gtts_connection_error_audio()
