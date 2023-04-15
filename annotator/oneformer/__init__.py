import os
from annotator.util import annotator_ckpts_path
from .api import make_detectron2_model, semantic_run


class OneformerCOCODetector:
    def __init__(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/150_16_swin_l_oneformer_coco_100ep.pth"
        modelpath = os.path.join(annotator_ckpts_path, "150_16_swin_l_oneformer_coco_100ep.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        config = os.path.join(os.path.dirname(__file__), 'configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml')
        self.model, self.meta = make_detectron2_model(config, modelpath)

    def __call__(self, img):
        return semantic_run(img, self.model, self.meta)


class OneformerADE20kDetector:
    def __init__(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/250_16_swin_l_oneformer_ade20k_160k.pth"
        modelpath = os.path.join(annotator_ckpts_path, "250_16_swin_l_oneformer_ade20k_160k.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        config = os.path.join(os.path.dirname(__file__), 'configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml')
        self.model, self.meta = make_detectron2_model(config, modelpath)

    def __call__(self, img):
        return semantic_run(img, self.model, self.meta)

