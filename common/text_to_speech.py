from gtts import gTTS, tts as gtts
import subprocess
import os
from time import time, sleep

# import sounddevice as sd
import numpy as np
import wave
import io

# pip install mycroft-mimic3-tts[all]  # Removing [all] will install support for English only.

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
        print("gTTS Connection error; switching TTS")
        print("#######################################")

        # it seems mimic3 needs .wav rather than .mp3
        audio_wav_file = text_to_audiofile(message)
        audiofile_to_speech(audio_wav_file, audio_speed)


def text_to_audiofile(message, audio_speed=1.5, audio_wav_file="tts_out.wav"):  # mimic3
    # mimic3 --voice <voice> "<text>" > output.wav

    subprocess.run(f"mimic3 '{message}' > {audio_wav_file}", shell=True)

    return audio_wav_file


def audiofile_to_speech(audio_file, audio_speed=1.75):

    # play the file with the message with Apple's afplay command
    # Original voices are slow => increase with the -r parameter in afplay
    # os.system(f"afplay {audio_file} -r {audio_speed}")

    if os.path.exists(audio_file):
        subprocess.run(f"afplay {audio_file} -r {audio_speed}", shell=True)
    else:
        print(f"Cannot find file {audio_file}")


def check_tts():

    while True:
        message = "testing TTS"
        text_to_speech(message)
        sleep(3)

def create_gtts_connection_error_audio():

    message = "gtts connection error. Check if the Mac is connected to internet"
    gtts_ok = text_to_speech(message)
    if gtts_ok:
        import shutil
        shutil.copy("tts_out.mp3", "gtts_connection_error.mp3")

def text_to_speech2(message, audio_speed=1.75, print_message=True, lang='en'):

    if print_message:
        print("\n<<<<<<<<<<< Start printing audio message: ")
        print(message)
        print(">>>>>>>>>>>>>> End printing audio message\n")

    audio_data = gtts_tts(message, lang)

    if audio_data is None:  # Si gTTS falla o no hay conexión
        print("Usando Mimic 3 en local")
        audio_data = mimic3_tts(message)

    if audio_data:
        play_audio(audio_data)
    else:
        print("No se pudo generar el audio con ninguna opción")


def gtts_tts(message, lang="en"):

    try:
        # Create the gTTS object with the message
        # tts = gTTS(message, lang='es')
        tts = gTTS(message, lang=lang)

        # Almacena el audio en memoria (sin escribir a disco)
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)  # Reinicia el puntero

        audio_data = convert_mp3_to_wav(audio_data)

        return audio_data

    except Exception as e:
        print(f"Error con gTTS: {e}")
        return None


def mimic3_tts(message):
    try:
        process = subprocess.Popen(['mimic3', message], stdout=subprocess.PIPE)
        audio_data = process.stdout.read()
        return io.BytesIO(audio_data)
    except Exception as e:
        print(f"Error con Mimic3: {e}")
        return None

def play_audio(audio_data):
    with wave.open(audio_data, 'rb') as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        audio_frames = wf.readframes(wf.getnframes())
        audio_array = np.frombuffer(audio_frames, dtype=np.int16)

        if num_channels > 1:
            audio_array = audio_array.reshape(-1, num_channels)

        sd.play(audio_array, samplerate=sample_rate)
        sd.wait()

def convert_mp3_to_wav(mp3_data):
    process = subprocess.Popen(['ffmpeg', '-i', 'pipe:0', '-f', 'wav', 'pipe:1'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    wav_data, _ = process.communicate(input=mp3_data.read())
    return io.BytesIO(wav_data)


def main1():
    start_time = time()
    message = "Your"  # speed is 70 kilometers per hour, but limit is 90. You can increase speed
    for _ in range(10):
        text_to_speech2(message)

    end_time = time()
    process_time = end_time - start_time
    print(round(process_time, 2))

    # 10 repeticiones de una palabra tardan 21.95s de mimic3 (local) y 12 de gtts (internet por móvil)


if __name__ == "__main__":

    # create_gtts_connection_error_audio()
    # check_tts()
    main1()