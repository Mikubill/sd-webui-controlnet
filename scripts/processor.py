import cv2
import numpy as np
from annotator.util import resize_image, HWC3

from typing import Callable, Tuple

model_canny = None


def canny(img, res=512, thr_a=100, thr_b=200, **kwargs):
    l, h = thr_a, thr_b
    img = resize_image(HWC3(img), res)
    global model_canny
    if model_canny is None:
        from annotator.canny import apply_canny
        model_canny = apply_canny
    result = model_canny(img, l, h)
    return result, True

def scribble_thr(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    result = np.zeros_like(img, dtype=np.uint8)
    result[np.min(img, axis=2) < 127] = 255
    return result, True


def scribble_xdog(img, res=512, thr_a=32, **kwargs):
    img = resize_image(HWC3(img), res)
    g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.5)
    g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 5.0)
    dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
    result = np.zeros_like(img, dtype=np.uint8)
    result[2 * (255 - dog) > thr_a] = 255
    return result, True


def tile_resample(img, res=512, thr_a=1.0, **kwargs):
    img = HWC3(img)
    if thr_a < 1.1:
        return img, True
    H, W, C = img.shape
    H = int(float(H) / float(thr_a))
    W = int(float(W) / float(thr_a))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img, True


def threshold(img, res=512, thr_a=127, **kwargs):
    img = resize_image(HWC3(img), res)
    result = np.zeros_like(img, dtype=np.uint8)
    result[np.min(img, axis=2) > thr_a] = 255
    return result, True


def inpaint(img, res=512, **kwargs):
    return img, True


def invert(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    img = 255 - img
    return img, True


model_hed = None


def hed(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_hed
    if model_hed is None:
        from annotator.hed import apply_hed
        model_hed = apply_hed
    result = model_hed(img)
    return result, True

def hed_safe(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_hed
    if model_hed is None:
        from annotator.hed import apply_hed
        model_hed = apply_hed
    result = model_hed(img, is_safe=True)
    return result, True

def unload_hed():
    global model_hed
    if model_hed is not None:
        from annotator.hed import unload_hed_model
        unload_hed_model()

def scribble_hed(img, res=512, **kwargs):
    result, _ = hed(img, res)
    import cv2
    from annotator.util import nms
    result = nms(result, 127, 3.0)
    result = cv2.GaussianBlur(result, (0, 0), 3.0)
    result[result > 4] = 255
    result[result < 255] = 0
    return result, True


model_mediapipe_face = None


def mediapipe_face(img, res=512, thr_a: int = 10, thr_b: float = 0.5, **kwargs):
    max_faces = int(thr_a)
    min_confidence = thr_b
    img = resize_image(HWC3(img), res)
    global model_mediapipe_face
    if model_mediapipe_face is None:
        from annotator.mediapipe_face import apply_mediapipe_face
        model_mediapipe_face = apply_mediapipe_face
    result = model_mediapipe_face(img, max_faces=max_faces, min_confidence=min_confidence)
    return result, True


model_mlsd = None


def mlsd(img, res=512, thr_a=0.1, thr_b=0.1, **kwargs):
    thr_v, thr_d = thr_a, thr_b
    img = resize_image(HWC3(img), res)
    global model_mlsd
    if model_mlsd is None:
        from annotator.mlsd import apply_mlsd
        model_mlsd = apply_mlsd
    result = model_mlsd(img, thr_v, thr_d)
    return result, True

def unload_mlsd():
    global model_mlsd
    if model_mlsd is not None:
        from annotator.mlsd import unload_mlsd_model
        unload_mlsd_model()


model_midas = None


def midas(img, res=512, a=np.pi * 2.0, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_midas
    if model_midas is None:
        from annotator.midas import apply_midas
        model_midas = apply_midas
    results, _ = model_midas(img, a)
    return results, True

def midas_normal(img, res=512, a=np.pi * 2.0, thr_a=0.4, **kwargs): # bg_th -> thr_a
    bg_th = thr_a
    img = resize_image(HWC3(img), res)
    global model_midas
    if model_midas is None:
        from annotator.midas import apply_midas
        model_midas = apply_midas
    _, results  = model_midas(img, a, bg_th)
    return results, True

def unload_midas():
    global model_midas
    if model_midas is not None:
        from annotator.midas import unload_midas_model
        unload_midas_model()

model_leres = None

def leres(img, res=512, a=np.pi * 2.0, thr_a=0, thr_b=0, boost=False, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_leres
    if model_leres is None:
        from annotator.leres import apply_leres
        model_leres = apply_leres
    results = model_leres(img, thr_a, thr_b, boost=boost)
    return results, True

def unload_leres():
    global model_leres
    if model_leres is not None:
        from annotator.leres import unload_leres_model
        unload_leres_model()

class OpenposeModel(object):
    def __init__(self) -> None:
        self.model_openpose = None
        
    def run_model(
            self, 
            img: np.ndarray, 
            include_body: bool, 
            include_hand: bool, 
            include_face: bool,
            json_pose_callback: Callable[[str], None] = None,
            res: int = 512, 
            **kwargs  # Ignore rest of kwargs
        ) -> Tuple[np.ndarray, bool]:
        """Run the openpose model. Returns a tuple of
        - result image
        - is_image flag

        The JSON format pose string is passed to `json_pose_callback`.
        """
        if json_pose_callback is None:
            json_pose_callback = lambda x: None

        img = resize_image(HWC3(img), res)
        if self.model_openpose is None:
            from annotator.openpose import OpenposeDetector
            self.model_openpose = OpenposeDetector()
                
        return self.model_openpose(
            img, 
            include_body=include_body, 
            include_hand=include_hand, 
            include_face=include_face,
            json_pose_callback=json_pose_callback
        ), True
    
    def unload(self):
        if self.model_openpose is not None:
            self.model_openpose.unload_model()

g_openpose_model = OpenposeModel()


model_uniformer = None


def uniformer(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_uniformer
    if model_uniformer is None:
        from annotator.uniformer import apply_uniformer
        model_uniformer = apply_uniformer
    result = model_uniformer(img)
    return result, True

def unload_uniformer():
    global model_uniformer
    if model_uniformer is not None:
        from annotator.uniformer import unload_uniformer_model
        unload_uniformer_model()
        
        
model_pidinet = None


def pidinet(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_pidinet
    if model_pidinet is None:
        from annotator.pidinet import apply_pidinet
        model_pidinet = apply_pidinet
    result = model_pidinet(img)
    return result, True

def pidinet_ts(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_pidinet
    if model_pidinet is None:
        from annotator.pidinet import apply_pidinet
        model_pidinet = apply_pidinet
    result = model_pidinet(img, apply_fliter=True)
    return result, True

def pidinet_safe(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_pidinet
    if model_pidinet is None:
        from annotator.pidinet import apply_pidinet
        model_pidinet = apply_pidinet
    result = model_pidinet(img, is_safe=True)
    return result, True

def scribble_pidinet(img, res=512, **kwargs):
    result, _ = pidinet(img, res)
    import cv2
    from annotator.util import nms
    result = nms(result, 127, 3.0)
    result = cv2.GaussianBlur(result, (0, 0), 3.0)
    result[result > 4] = 255
    result[result < 255] = 0
    return result, True

def unload_pidinet():
    global model_pidinet
    if model_pidinet is not None:
        from annotator.pidinet import unload_pid_model
        unload_pid_model()
        

clip_encoder = None


def clip(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global clip_encoder
    if clip_encoder is None:
        from annotator.clip import apply_clip
        clip_encoder = apply_clip
    result = clip_encoder(img).squeeze(0)
    return result, False


def clip_vision_visualization(x):
    return np.ndarray((x.shape[0] * 4, x.shape[1]), dtype="uint8", buffer=x.detach().cpu().numpy().tobytes())


def unload_clip():
    global clip_encoder
    if clip_encoder is not None:
        from annotator.clip import unload_clip_model
        unload_clip_model()
        

model_color = None


def color(img, res=512, **kwargs):
    global model_color
    if model_color is None:
        from annotator.color import apply_color
        model_color = apply_color
    result = model_color(img, res=res)
    return result, True


model_binary = None


def binary(img, res=512, thr_a=0, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_binary
    if model_binary is None:
        from annotator.binary import apply_binary
        model_binary = apply_binary
    result = model_binary(img, thr_a)
    return result, True


def lineart_standard(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    x = img.astype(np.float32)
    g = cv2.GaussianBlur(x, (0, 0), 6.0)
    intensity = np.min(g - x, axis=2).clip(0, 255)
    intensity /= max(16, np.median(intensity[intensity > 8]))
    intensity *= 127
    return intensity.clip(0, 255).astype(np.uint8), True

model_lineart = None


def lineart(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_lineart
    if model_lineart is None:
        from annotator.lineart import LineartDetector
        model_lineart = LineartDetector(LineartDetector.model_default)

    # applied auto inversion
    result = 255-model_lineart(img)
    return result, True

def unload_lineart():
    global model_lineart
    if model_lineart is not None:
        model_lineart.unload_model()


model_lineart_coarse = None


def lineart_coarse(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_lineart_coarse
    if model_lineart_coarse is None:
        from annotator.lineart import LineartDetector
        model_lineart_coarse = LineartDetector(LineartDetector.model_coarse)

    # applied auto inversion
    result = 255-model_lineart_coarse(img)
    return result, True

def unload_lineart_coarse():
    global model_lineart_coarse
    if model_lineart_coarse is not None:
        model_lineart_coarse.unload_model()


model_lineart_anime = None


def lineart_anime(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_lineart_anime
    if model_lineart_anime is None:
        from annotator.lineart_anime import LineartAnimeDetector
        model_lineart_anime = LineartAnimeDetector()

    # applied auto inversion
    result = 255-model_lineart_anime(img)
    return result, True

def unload_lineart_anime():
    global model_lineart_anime
    if model_lineart_anime is not None:
        model_lineart_anime.unload_model()


model_manga_line = None


def lineart_anime_denoise(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_manga_line
    if model_manga_line is None:
        from annotator.manga_line import MangaLineExtration
        model_manga_line = MangaLineExtration()

    # applied auto inversion
    result = model_manga_line(img)
    return result, True


def unload_lineart_anime_denoise():
   global model_manga_line
   if model_manga_line is not None:
       model_manga_line.unload_model()



model_zoe_depth = None


def zoe_depth(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_zoe_depth
    if model_zoe_depth is None:
        from annotator.zoe import ZoeDetector
        model_zoe_depth = ZoeDetector()
    result = model_zoe_depth(img)
    return result, True

def unload_zoe_depth():
    global model_zoe_depth
    if model_zoe_depth is not None:
        model_zoe_depth.unload_model()


model_normal_bae = None


def normal_bae(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_normal_bae
    if model_normal_bae is None:
        from annotator.normalbae import NormalBaeDetector
        model_normal_bae = NormalBaeDetector()
    result = model_normal_bae(img)
    return result, True

def unload_normal_bae():
    global model_normal_bae
    if model_normal_bae is not None:
        model_normal_bae.unload_model()


model_oneformer_coco = None


def oneformer_coco(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_oneformer_coco
    if model_oneformer_coco is None:
        from annotator.oneformer import OneformerDetector
        model_oneformer_coco = OneformerDetector(OneformerDetector.configs["coco"])
    result = model_oneformer_coco(img)
    return result, True

def unload_oneformer_coco():
    global model_oneformer_coco
    if model_oneformer_coco is not None:
        model_oneformer_coco.unload_model()


model_oneformer_ade20k = None


def oneformer_ade20k(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_oneformer_ade20k
    if model_oneformer_ade20k is None:
        from annotator.oneformer import OneformerDetector
        model_oneformer_ade20k = OneformerDetector(OneformerDetector.configs["ade20k"])
    result = model_oneformer_ade20k(img)
    return result, True

def unload_oneformer_ade20k():
    global model_oneformer_ade20k
    if model_oneformer_ade20k is not None:
        model_oneformer_ade20k.unload_model()


model_shuffle = None


def shuffle(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_shuffle
    if model_shuffle is None:
        from annotator.shuffle import ContentShuffleDetector
        model_shuffle = ContentShuffleDetector()
    result = model_shuffle(img)
    return result, True