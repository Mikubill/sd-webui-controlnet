import os
import numpy as np
from PIL import Image, ImageDraw

from modules import devices
from modules.paths import models_path

from annotator.mmpkg.mmpose.apis import inference_top_down_pose_model, init_pose_model
from annotator.mmpkg.mmpose.datasets import DatasetInfo


try:
    import face_recognition
    has_face_det = True
except (ImportError, ModuleNotFoundError):
    has_face_det = False


model_path = "https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_300w_256x256-eea53406_20211019.pth"

modeldir = os.path.join(models_path, "face_model")

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "config.py")

face_model = None

def process_face_det_results(face_det_results):
    person_results = []
    for bbox in face_det_results:
        person = {}
        person['bbox'] = [bbox[3], bbox[0], bbox[1], bbox[2]]
        person_results.append(person)

    return person_results


def unload_face_model():
    global face_model
    if face_model is not None:
        face_model.cpu()


def apply_face_model(image):

    global face_model
    if face_model is None:
        face_modelpath = os.path.join(modeldir, "hrnetv2_w18_300w.pth")
        
        if not os.path.exists(face_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(model_path, model_dir=modeldir, file_name="hrnetv2_w18_300w.pth")

        face_model = init_pose_model(
            config_file, face_modelpath, device=devices.get_device_for("controlnet"))

    dataset = face_model.cfg.data['test']['type']
    dataset_info = face_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)

    image = image[:, :, ::-1].copy()

    face_det_results = face_recognition.face_locations(image)
    face_results = process_face_det_results(face_det_results)

    pose_results, returned_outputs = inference_top_down_pose_model(
        face_model,
        image,
        face_results,
        bbox_thr=None,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=False,
        outputs=None)

    def draw_landmarks(image, landmarks, color="white", radius=2.5):
        draw = ImageDraw.Draw(image)
        for dot in landmarks:
            x, y, _ = dot
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

    con_img = Image.new('RGB', (image.shape[1], image.shape[0]), color=(0, 0, 0))
    if len(pose_results) == 0:
        return np.array(con_img)
    for face in pose_results:
        draw_landmarks(con_img, face['keypoints'])
    con_img = np.array(con_img)

    return con_img
