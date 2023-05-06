import os
import sys
import cv2
from base64 import b64encode

import requests

BASE_URL = "http://localhost:7860"


def setup_test_env():
    ext_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if ext_root not in sys.path:
        sys.path.append(ext_root)


def readImage(path):
    img = cv2.imread(path)
    retval, buffer = cv2.imencode('.jpg', img)
    b64img = b64encode(buffer).decode("utf-8")
    return b64img


def get_model():
    r = requests.get(BASE_URL+"/controlnet/model_list")
    result = r.json()
    if "model_list" in result:
        result = result["model_list"]
        for item in result:
            print("Using model: ", item)
            return item
    return "None"


def get_modules():
    return requests.get(f"{BASE_URL}/controlnet/module_list").json()


def detect(json):
    return requests.post(BASE_URL+"/controlnet/detect", json=json)
