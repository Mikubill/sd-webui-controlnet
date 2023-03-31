from PIL import Image

from .mediapipe_face_common import generate_annotation


def apply_mediapipe_face(image, max_faces: int = 1):
    return generate_annotation(image, max_faces)
