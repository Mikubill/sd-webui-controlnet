import os
import sys
import cv2
from base64 import b64encode
from pathlib import Path

import requests

BASE_URL = "http://localhost:7860"


def setup_test_env():
    os.environ['IGNORE_CMD_ARGS_ERRORS'] = 'True'

    file_path = Path(__file__).resolve()
    ext_root = file_path.parent.parent
    a1111_root = ext_root.parent.parent

    for p in (ext_root, a1111_root):
        if p not in sys.path:
            sys.path.append(str(p))

    # Initialize shared opts.
    import webui


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
