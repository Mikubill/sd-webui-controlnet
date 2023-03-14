import os, sys, cv2
from base64 import b64encode

import requests


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
    r = requests.get("http://localhost:7860/controlnet/model_list")
    result = r.json()
    if "model_list" in result:
        result = result["model_list"]
        for item in result:
            print("Using model: ", item)
            return item
    return "None"
