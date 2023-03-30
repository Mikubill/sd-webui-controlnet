from PIL import Image

from .laion_face_common import generate_annotation


def apply_mediapipe_laion_face(image: Image.Image, max_faces: int = 1):
    return generate_annotation(image.convert("RGB"), max_faces)
