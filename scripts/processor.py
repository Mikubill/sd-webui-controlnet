import cv2
import numpy as np

from annotator.util import HWC3
from typing import Callable, Tuple


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad


model_canny = None


def canny(img, res=512, thr_a=100, thr_b=200, **kwargs):
    l, h = thr_a, thr_b
    img, remove_pad = resize_image_with_pad(img, res)
    global model_canny
    if model_canny is None:
        from annotator.canny import apply_canny
        model_canny = apply_canny
    result = model_canny(img, l, h)
    return remove_pad(result), True


def scribble_thr(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    result = np.zeros_like(img, dtype=np.uint8)
    result[np.min(img, axis=2) < 127] = 255
    return remove_pad(result), True


def scribble_xdog(img, res=512, thr_a=32, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.5)
    g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 5.0)
    dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
    result = np.zeros_like(img, dtype=np.uint8)
    result[2 * (255 - dog) > thr_a] = 255
    return remove_pad(result), True


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
    img, remove_pad = resize_image_with_pad(img, res)
    result = np.zeros_like(img, dtype=np.uint8)
    result[np.min(img, axis=2) > thr_a] = 255
    return remove_pad(result), True


def identity(img, **kwargs):
    return img, True


def invert(img, res=512, **kwargs):
    return 255 - HWC3(img), True


model_hed = None


def hed(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_hed
    if model_hed is None:
        from annotator.hed import apply_hed
        model_hed = apply_hed
    result = model_hed(img)
    return remove_pad(result), True


def hed_safe(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_hed
    if model_hed is None:
        from annotator.hed import apply_hed
        model_hed = apply_hed
    result = model_hed(img, is_safe=True)
    return remove_pad(result), True


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
    img, remove_pad = resize_image_with_pad(img, res)
    global model_mediapipe_face
    if model_mediapipe_face is None:
        from annotator.mediapipe_face import apply_mediapipe_face
        model_mediapipe_face = apply_mediapipe_face
    result = model_mediapipe_face(img, max_faces=max_faces, min_confidence=min_confidence)
    return remove_pad(result), True


model_mlsd = None


def mlsd(img, res=512, thr_a=0.1, thr_b=0.1, **kwargs):
    thr_v, thr_d = thr_a, thr_b
    img, remove_pad = resize_image_with_pad(img, res)
    global model_mlsd
    if model_mlsd is None:
        from annotator.mlsd import apply_mlsd
        model_mlsd = apply_mlsd
    result = model_mlsd(img, thr_v, thr_d)
    return remove_pad(result), True


def unload_mlsd():
    global model_mlsd
    if model_mlsd is not None:
        from annotator.mlsd import unload_mlsd_model
        unload_mlsd_model()


model_midas = None


def midas(img, res=512, a=np.pi * 2.0, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_midas
    if model_midas is None:
        from annotator.midas import apply_midas
        model_midas = apply_midas
    result, _ = model_midas(img, a)
    return remove_pad(result), True


def midas_normal(img, res=512, a=np.pi * 2.0, thr_a=0.4, **kwargs):  # bg_th -> thr_a
    bg_th = thr_a
    img, remove_pad = resize_image_with_pad(img, res)
    global model_midas
    if model_midas is None:
        from annotator.midas import apply_midas
        model_midas = apply_midas
    _, result = model_midas(img, a, bg_th)
    return remove_pad(result), True


def unload_midas():
    global model_midas
    if model_midas is not None:
        from annotator.midas import unload_midas_model
        unload_midas_model()


model_leres = None


def leres(img, res=512, a=np.pi * 2.0, thr_a=0, thr_b=0, boost=False, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_leres
    if model_leres is None:
        from annotator.leres import apply_leres
        model_leres = apply_leres
    result = model_leres(img, thr_a, thr_b, boost=boost)
    return remove_pad(result), True


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
            use_dw_pose: bool = False,
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

        img, remove_pad = resize_image_with_pad(img, res)

        if self.model_openpose is None:
            from annotator.openpose import OpenposeDetector
            self.model_openpose = OpenposeDetector()

        return remove_pad(self.model_openpose(
            img,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
            use_dw_pose=use_dw_pose,
            json_pose_callback=json_pose_callback
        )), True

    def unload(self):
        if self.model_openpose is not None:
            self.model_openpose.unload_model()


g_openpose_model = OpenposeModel()

model_uniformer = None


def uniformer(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_uniformer
    if model_uniformer is None:
        from annotator.uniformer import apply_uniformer
        model_uniformer = apply_uniformer
    result = model_uniformer(img)
    return remove_pad(result), True


def unload_uniformer():
    global model_uniformer
    if model_uniformer is not None:
        from annotator.uniformer import unload_uniformer_model
        unload_uniformer_model()


model_pidinet = None


def pidinet(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_pidinet
    if model_pidinet is None:
        from annotator.pidinet import apply_pidinet
        model_pidinet = apply_pidinet
    result = model_pidinet(img)
    return remove_pad(result), True


def pidinet_ts(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_pidinet
    if model_pidinet is None:
        from annotator.pidinet import apply_pidinet
        model_pidinet = apply_pidinet
    result = model_pidinet(img, apply_fliter=True)
    return remove_pad(result), True


def pidinet_safe(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_pidinet
    if model_pidinet is None:
        from annotator.pidinet import apply_pidinet
        model_pidinet = apply_pidinet
    result = model_pidinet(img, is_safe=True)
    return remove_pad(result), True


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
    img = HWC3(img)
    global clip_encoder
    if clip_encoder is None:
        from annotator.clip import apply_clip
        clip_encoder = apply_clip
    result = clip_encoder(img)
    return result, False


def clip_vision_visualization(x):
    x = x.detach().cpu().numpy()[0]
    x = np.ascontiguousarray(x).copy()
    return np.ndarray((x.shape[0] * 4, x.shape[1]), dtype="uint8", buffer=x.tobytes())


def unload_clip():
    global clip_encoder
    if clip_encoder is not None:
        from annotator.clip import unload_clip_model
        unload_clip_model()


model_color = None


def color(img, res=512, **kwargs):
    img = HWC3(img)
    global model_color
    if model_color is None:
        from annotator.color import apply_color
        model_color = apply_color
    result = model_color(img, res=res)
    return result, True


def lineart_standard(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    x = img.astype(np.float32)
    g = cv2.GaussianBlur(x, (0, 0), 6.0)
    intensity = np.min(g - x, axis=2).clip(0, 255)
    intensity /= max(16, np.median(intensity[intensity > 8]))
    intensity *= 127
    result = intensity.clip(0, 255).astype(np.uint8)
    return remove_pad(result), True


model_lineart = None


def lineart(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_lineart
    if model_lineart is None:
        from annotator.lineart import LineartDetector
        model_lineart = LineartDetector(LineartDetector.model_default)

    # applied auto inversion
    result = 255 - model_lineart(img)
    return remove_pad(result), True


def unload_lineart():
    global model_lineart
    if model_lineart is not None:
        model_lineart.unload_model()


model_lineart_coarse = None


def lineart_coarse(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_lineart_coarse
    if model_lineart_coarse is None:
        from annotator.lineart import LineartDetector
        model_lineart_coarse = LineartDetector(LineartDetector.model_coarse)

    # applied auto inversion
    result = 255 - model_lineart_coarse(img)
    return remove_pad(result), True


def unload_lineart_coarse():
    global model_lineart_coarse
    if model_lineart_coarse is not None:
        model_lineart_coarse.unload_model()


model_lineart_anime = None


def lineart_anime(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_lineart_anime
    if model_lineart_anime is None:
        from annotator.lineart_anime import LineartAnimeDetector
        model_lineart_anime = LineartAnimeDetector()

    # applied auto inversion
    result = 255 - model_lineart_anime(img)
    return remove_pad(result), True


def unload_lineart_anime():
    global model_lineart_anime
    if model_lineart_anime is not None:
        model_lineart_anime.unload_model()


model_manga_line = None


def lineart_anime_denoise(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_manga_line
    if model_manga_line is None:
        from annotator.manga_line import MangaLineExtration
        model_manga_line = MangaLineExtration()

    # applied auto inversion
    result = model_manga_line(img)
    return remove_pad(result), True


def unload_lineart_anime_denoise():
    global model_manga_line
    if model_manga_line is not None:
        model_manga_line.unload_model()


model_lama = None


def lama_inpaint(img, res=512, **kwargs):
    H, W, C = img.shape
    raw_color = img[:, :, 0:3].copy()
    raw_mask = img[:, :, 3:4].copy()

    res = 256  # Always use 256 since lama is trained on 256

    img_res, remove_pad = resize_image_with_pad(img, res, skip_hwc3=True)

    global model_lama
    if model_lama is None:
        from annotator.lama import LamaInpainting
        model_lama = LamaInpainting()

    # applied auto inversion
    prd_color = model_lama(img_res)
    prd_color = remove_pad(prd_color)
    prd_color = cv2.resize(prd_color, (W, H))

    alpha = raw_mask.astype(np.float32) / 255.0
    fin_color = prd_color.astype(np.float32) * alpha + raw_color.astype(np.float32) * (1 - alpha)
    fin_color = fin_color.clip(0, 255).astype(np.uint8)

    result = np.concatenate([fin_color, raw_mask], axis=2)

    return result, True


def unload_lama_inpaint():
    global model_lama
    if model_lama is not None:
        model_lama.unload_model()


model_zoe_depth = None


def zoe_depth(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_zoe_depth
    if model_zoe_depth is None:
        from annotator.zoe import ZoeDetector
        model_zoe_depth = ZoeDetector()
    result = model_zoe_depth(img)
    return remove_pad(result), True


def unload_zoe_depth():
    global model_zoe_depth
    if model_zoe_depth is not None:
        model_zoe_depth.unload_model()


model_normal_bae = None


def normal_bae(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_normal_bae
    if model_normal_bae is None:
        from annotator.normalbae import NormalBaeDetector
        model_normal_bae = NormalBaeDetector()
    result = model_normal_bae(img)
    return remove_pad(result), True


def unload_normal_bae():
    global model_normal_bae
    if model_normal_bae is not None:
        model_normal_bae.unload_model()


model_oneformer_coco = None


def oneformer_coco(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_oneformer_coco
    if model_oneformer_coco is None:
        from annotator.oneformer import OneformerDetector
        model_oneformer_coco = OneformerDetector(OneformerDetector.configs["coco"])
    result = model_oneformer_coco(img)
    return remove_pad(result), True


def unload_oneformer_coco():
    global model_oneformer_coco
    if model_oneformer_coco is not None:
        model_oneformer_coco.unload_model()


model_oneformer_ade20k = None


def oneformer_ade20k(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_oneformer_ade20k
    if model_oneformer_ade20k is None:
        from annotator.oneformer import OneformerDetector
        model_oneformer_ade20k = OneformerDetector(OneformerDetector.configs["ade20k"])
    result = model_oneformer_ade20k(img)
    return remove_pad(result), True


def unload_oneformer_ade20k():
    global model_oneformer_ade20k
    if model_oneformer_ade20k is not None:
        model_oneformer_ade20k.unload_model()


model_shuffle = None


def shuffle(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    img = remove_pad(img)
    global model_shuffle
    if model_shuffle is None:
        from annotator.shuffle import ContentShuffleDetector
        model_shuffle = ContentShuffleDetector()
    result = model_shuffle(img)
    return result, True


model_free_preprocessors = [
    "reference_only",
    "reference_adain",
    "reference_adain+attn"
]

flag_preprocessor_resolution = "Preprocessor Resolution"
preprocessor_sliders_config = {
    "none": [],
    "inpaint": [],
    "inpaint_only": [],
    "canny": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        },
        {
            "name": "Canny Low Threshold",
            "value": 100,
            "min": 1,
            "max": 255
        },
        {
            "name": "Canny High Threshold",
            "value": 200,
            "min": 1,
            "max": 255
        },
    ],
    "mlsd": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "MLSD Value Threshold",
            "min": 0.01,
            "max": 2.0,
            "value": 0.1,
            "step": 0.01
        },
        {
            "name": "MLSD Distance Threshold",
            "min": 0.01,
            "max": 20.0,
            "value": 0.1,
            "step": 0.01
        }
    ],
    "hed": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "scribble_hed": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "hed_safe": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "openpose": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "openpose_full": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "dw_openpose_full": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "segmentation": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "depth": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "depth_leres": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "Remove Near %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        },
        {
            "name": "Remove Background %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        }
    ],
    "depth_leres++": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "Remove Near %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        },
        {
            "name": "Remove Background %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        }
    ],
    "normal_map": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "Normal Background Threshold",
            "min": 0.0,
            "max": 1.0,
            "value": 0.4,
            "step": 0.01
        }
    ],
    "threshold": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        },
        {
            "name": "Binarization Threshold",
            "min": 0,
            "max": 255,
            "value": 127
        }
    ],

    "scribble_xdog": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        },
        {
            "name": "XDoG Threshold",
            "min": 1,
            "max": 64,
            "value": 32,
        }
    ],
    "tile_resample": [
        None,
        {
            "name": "Down Sampling Rate",
            "value": 1.0,
            "min": 1.0,
            "max": 8.0,
            "step": 0.01
        }
    ],
    "tile_colorfix": [
        None,
        {
            "name": "Variation",
            "value": 8.0,
            "min": 3.0,
            "max": 32.0,
            "step": 1.0
        }
    ],
    "tile_colorfix+sharp": [
        None,
        {
            "name": "Variation",
            "value": 8.0,
            "min": 3.0,
            "max": 32.0,
            "step": 1.0
        },
        {
            "name": "Sharpness",
            "value": 1.0,
            "min": 0.0,
            "max": 2.0,
            "step": 0.01
        }
    ],
    "reference_only": [
        None,
        {
            "name": r'Style Fidelity (only for "Balanced" mode)',
            "value": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        }
    ],
    "reference_adain": [
        None,
        {
            "name": r'Style Fidelity (only for "Balanced" mode)',
            "value": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        }
    ],
    "reference_adain+attn": [
        None,
        {
            "name": r'Style Fidelity (only for "Balanced" mode)',
            "value": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        }
    ],
    "inpaint_only+lama": [],
    "color": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048,
        }
    ],
    "mediapipe_face": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048,
        },
        {
            "name": "Max Faces",
            "value": 1,
            "min": 1,
            "max": 10,
            "step": 1
        },
        {
            "name": "Min Face Confidence",
            "value": 0.5,
            "min": 0.01,
            "max": 1.0,
            "step": 0.01
        }
    ],
}

preprocessor_filters = {
    "All": "none",
    "Canny": "canny",
    "Depth": "depth_midas",
    "Normal": "normal_bae",
    "OpenPose": "openpose_full",
    "MLSD": "mlsd",
    "Lineart": "lineart_standard (from white bg & black line)",
    "SoftEdge": "softedge_pidinet",
    "Scribble": "scribble_pidinet",
    "Seg": "seg_ofade20k",
    "Shuffle": "shuffle",
    "Tile": "tile_resample",
    "Inpaint": "inpaint_only",
    "IP2P": "none",
    "Reference": "reference_only",
    "T2IA": "none",
}
