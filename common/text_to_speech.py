from gtts import gTTS, tts as gtts
import subprocess
import os
import platform
from time import time, sleep
import sounddevice as sd
import io
import threading
import soundfile as sf
from pyrubberband import pyrb

# pip install mycroft-mimic3-tts[all]  # Removing [all] will install support for English only.

def text_to_speech_async(output_message):
    audio_thread = threading.Thread(target=text_to_speech, args=(output_message,))
    # Start audio in a separate thread
    audio_thread.start()


def text_to_speech(message, audio_speed=1.75, print_message=True, disable_audio=False, lang='en'):

    # sd._terminate()  # Reset PortAudio
    # sd._initialize()  # Reinitialize


    if print_message:
        print("\n<<<<<<<<<<< Printing audio message >>>>>>>>>>>>>>")
        print(message)

    # audio is disabled when driver agent is in listen mode to avoid mixing driver speech with audio automatic notifications
    if disable_audio:
        return

    if platform.system() == "Darwin":
        text_to_speech_mac(message, audio_speed)
        # text_to_speech_not_mac(message, audio_speed)  # a bit less quality, but crashes less
    else:
        text_to_speech_not_mac(message, audio_speed)

def text_to_speech_mac(message, audio_speed=1.75, audio_file="tts_out.mp3"):

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


def text_to_audiofile(message, audio_wav_file="tts_out.wav"):  # mimic3
    # mimic3 --voice <voice> "<text>" > output.wav

    subprocess.run(f"mimic3 '{message}' > {audio_wav_file}", shell=True)

    return audio_wav_file


def audiofile_to_speech(audio_file, audio_speed=1.75):

    # play the file with the message with Apple's afplay command
    # Original voices are slow => increase with the -r parameter in afplay
    # os.system(f"afplay {audio_file} -r {audio_speed}")

    if os.path.exists(audio_file):
        # subprocess.run(f"afplay {audio_file} -r {audio_speed} -d", shell=True)
        subprocess.run(
            f"ffplay -nodisp -loglevel quiet -i {audio_file} -af 'atempo={audio_speed}' -autoexit",
            shell=True)
    else:
        print(f"Cannot find file {audio_file}")


def text_to_speech_not_mac(message, audio_speed=1.75, lang='en'):

    audio_data = gtts_tts(message, lang)

    if audio_data is None:  # Si gTTS falla o no hay conexión
        print("Usando Mimic 3 en local")
        audio_data = mimic3_tts(message)

    if audio_data:
        audio_data = change_audio_speed(audio_data, audio_speed)
        try:
            play_audio(audio_data)
        except Exception as e:
            print(f"Audio playback failed: {e}")

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

        # audio_data = convert_mp3_to_wav(audio_data)

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

def change_audio_speed(audio_data, speed=1.75, librosa_stretch=True):
    # Cargar el audio desde el flujo de bytes
    audio_data.seek(0)
    data, sample_rate = sf.read(audio_data)  # 24000

    # Ajustar la velocidad sin cambiar el tono
    data_stretched = pyrb.time_stretch(data, sample_rate, rate=speed, rbargs=None)

    # Almacenar el audio modificado en un flujo de bytes
    modified_audio_data = io.BytesIO()
    sf.write(modified_audio_data, data_stretched, sample_rate, format="wav")
    modified_audio_data.seek(0)  # Reinicia el puntero
    return modified_audio_data


def play_audio(audio_data):
    with sf.SoundFile(audio_data) as sf_file:
        audio_array = sf_file.read(dtype="int16")
        sample_rate = sf_file.samplerate  # 24000

        try:
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()  # Wait until playback is finished
        except Exception as e:
            print(f"Audio playback failed: {e}")


def check_tts():

    while True:
        message = "testing TTS"
        text_to_speech(message)
        sleep(3)


def main1():
    start_time = time()
    message = "Your"  # speed is 70 kilometers per hour, but limit is 90. You can increase speed
    for _ in range(10):
        text_to_speech(message)

    end_time = time()
    process_time = end_time - start_time
    print(round(process_time, 2))

    # 10 repeticiones de una palabra tardan 21.95s de mimic3 (local) y 12 de gtts (internet por móvil)


def main2():
    message = "Speed limit is 50 km/h"
    text_to_speech_not_mac(message)
    text_to_speech_mac(message)


def main3():
    import sounddevice as sd
    print(sd.query_devices())

    """
      0 LS32A70, Core Audio (0 in, 2 out)
    > 1 MacBook Pro (micrófono), Core Audio (1 in, 0 out)
    < 2 MacBook Pro (altavoces), Core Audio (0 in, 2 out)
      3 Microsoft Teams Audio, Core Audio (1 in, 1 out)
      4 ZoomAudioDevice, Core Audio (2 in, 2 out)
    """

if __name__ == "__main__":

    # check_tts()
    # main1()
    main2()
    # main3()