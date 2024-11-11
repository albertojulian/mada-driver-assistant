from collections import Counter
import subprocess
import yaml
import os
from time import time
from text_to_speech import text_to_speech
# pip install PyPDF2
# from PyPDF2 import PdfReader

# pip install pdfminer.six
# from pdfminer.high_level import extract_text

# Cuando el Mac usa AndroidAJR del móvil, la IP es 192.168.43.233
# IP del móvil: 192.168.0.11

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
    mada_file = "../mada.yaml"
    with open(mada_file) as file:
        mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

    mac_ssid = wifi_info_mac()
    communications = mada_config_dict["communications"]
    android_wifi = communications.get("android_wifi", "AndroidAJR")
    status_ok = False
    if mac_ssid == android_wifi:
        mac_connected_and_phone_app = communications.get("mac_connected_and_phone_app", "Connection with phone OK")
        message = mac_connected_and_phone_app
        status_ok = True
    else:
        if mac_ssid is None:
            message = "Mac is not connected to any wi-fi"
        else:
            message = f"Mac is connected to {mac_ssid} rather than {android_wifi}"

    text_to_speech(message)

    return status_ok


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


def get_most_frequent_value(items_list):
    if isinstance(items_list, list) and len(items_list) > 0:
        items_count = Counter(items_list)
        most_frequent_value = items_count.most_common(1)[0][0]
    else:
        most_frequent_value = None

    return most_frequent_value


def main1():
    check_mac_wifi_connection()


"""
def main2():
    file = "../Small Language Models AJR/0 traffic docs/dh-chapter3.pdf"

    # drive1_txt = read_pdf_v1(file)
    # print(drive1_txt)

    pdf2txt(file)

"""


if __name__ == "__main__":

    main1()
    # main2()
