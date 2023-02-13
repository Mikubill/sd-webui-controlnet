
from annotator.util import resize_image, HWC3

model_canny = None


def canny(img, res=512, l=100, h=200):
    img = resize_image(HWC3(img), res)
    global model_canny
    if model_canny is None:
        from annotator.canny import apply_canny
        model_canny = apply_canny
    result = model_canny(img, l, h)
    return result


model_hed = None


def hed(img, res=512):
    img = resize_image(HWC3(img), res)
    global model_hed
    if model_hed is None:
        from annotator.hed import apply_hed
        model_hed = apply_hed
    result = model_hed(img)
    return result


model_mlsd = None


def mlsd(img, res, thr_v, thr_d):
    img = resize_image(HWC3(img), res)
    global model_mlsd
    if model_mlsd is None:
        from annotator.mlsd import apply_mlsd
        model_mlsd = apply_mlsd
    result = model_mlsd(img, thr_v, thr_d)
    return result


model_midas = None


def midas(img, res, a):
    img = resize_image(HWC3(img), res)
    global model_midas
    if model_midas is None:
        from annotator.midas import apply_midas
        model_midas = apply_midas
    results = model_midas(img, a)
    return results


model_openpose = None


def openpose(img, res, has_hand):
    img = resize_image(HWC3(img), res)
    global model_openpose
    if model_openpose is None:
        from annotator.openpose import apply_openpose
        model_openpose = apply_openpose
    result, _ = model_openpose(img, has_hand)
    return result


# model_uniformer = None


# def uniformer(img, res):
#     img = resize_image(HWC3(img), res)
#     global model_uniformer
#     if model_uniformer is None:
#         from annotator.uniformer import apply_uniformer
#         model_uniformer = apply_uniformer
#     result = model_uniformer(img)
#     return result