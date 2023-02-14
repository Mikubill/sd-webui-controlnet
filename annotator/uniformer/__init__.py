from annotator.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from annotator.uniformer.mmseg.core.evaluation import get_palette
from modules.shared import extensions
import os

modeldir = os.path.join(extensions.extensions_dir, "sd-webui-controlnet", "annotator", "uniformer")
checkpoint_file = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth"
config_file = os.path.join(extensions.extensions_dir, "sd-webui-controlnet", "annotator", "uniformer", "exp", "upernet_global_small", "config.py")
model = None

def apply_uniformer(img):
    global model
    if model is None:
        modelpath = os.path.join(modeldir, "body_pose_model.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(checkpoint_file, model_dir=modeldir)
            
        model = init_segmentor(config_file, checkpoint_file).cuda()
    
    result = inference_segmentor(model, img)
    res_img = show_result_pyplot(model, img, result, get_palette('ade'), opacity=1)
    return res_img
