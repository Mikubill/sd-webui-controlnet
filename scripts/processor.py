
import numpy as np
from annotator.util import resize_image, HWC3


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

def simple_scribble(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    result = np.zeros_like(img, dtype=np.uint8)
    result[np.min(img, axis=2) < 127] = 255
    return result, True


model_hed = None


def hed(img, res=512, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_hed
    if model_hed is None:
        from annotator.hed import apply_hed
        model_hed = apply_hed
    result = model_hed(img)
    return result, True

def unload_hed():
    global model_hed
    if model_hed is not None:
        from annotator.hed import unload_hed_model
        unload_hed_model()

def fake_scribble(img, res=512, **kwargs):
    result, _ = hed(img, res)
    import cv2
    from annotator.hed import nms
    result = nms(result, 127, 3.0)
    result = cv2.GaussianBlur(result, (0, 0), 3.0)
    result[result > 10] = 255
    result[result < 255] = 0
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

def leres(img, res=512, a=np.pi * 2.0, thr_a=0, thr_b=0, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_leres
    if model_leres is None:
        from annotator.leres import apply_leres
        model_leres = apply_leres
    results = model_leres(img, thr_a, thr_b)
    return results, True

def unload_leres():
    global model_leres
    if model_leres is not None:
        from annotator.leres import unload_leres_model
        unload_leres_model()

model_openpose = None


def openpose(img, res=512, has_hand=False, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_openpose
    if model_openpose is None:
        from annotator.openpose import apply_openpose
        model_openpose = apply_openpose
    result, _ = model_openpose(img, has_hand)
    return result, True

def openpose_hand(img, res=512, has_hand=True, **kwargs):
    img = resize_image(HWC3(img), res)
    global model_openpose
    if model_openpose is None:
        from annotator.openpose import apply_openpose
        model_openpose = apply_openpose
    result, _ = model_openpose(img, has_hand)
    return result, True

def unload_openpose():
    global model_openpose
    if model_openpose is not None:
        from annotator.openpose import unload_openpose_model
        unload_openpose_model()


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