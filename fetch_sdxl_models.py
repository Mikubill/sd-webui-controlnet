diffuser_models_fp16 = [
    # Small models
    "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small/resolve/main/diffusion_pytorch_model.fp16.safetensors",
    "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-small/resolve/main/diffusion_pytorch_model.fp16.safetensors",
    # Mid models
    "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-mid/resolve/main/diffusion_pytorch_model.fp16.safetensors",
    "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-mid/resolve/main/diffusion_pytorch_model.fp16.safetensors"
    # Full size models
    "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors",
    "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors",
]

sai_control_loras = [
    # Small models
    "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors",
    "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-depth-rank128.safetensors",
    "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-recolor-rank128.safetensors",
    "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-sketch-rank128-metadata.safetensors",
    # Mid models
    "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors",
    "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors",
    "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-recolor-rank256.safetensors",
    "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors",
]

community_models = [
    # Openpose
    "https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors"
    # Seg
    "https://huggingface.co/SargeZT/sdxl-controlnet-seg/resolve/main/diffusion_pytorch_model.bin",
    # depth-faid-vidit
    "https://huggingface.co/SargeZT/controlnet-sd-xl-1.0-depth-faid-vidit/resolve/main/diffusion_pytorch_model.bin",
    # depth-zeed
    "https://huggingface.co/SargeZT/controlnet-sd-xl-1.0-depth-zeed/resolve/main/diffusion_pytorch_model.bin",
    # softedge
    "https://huggingface.co/SargeZT/controlnet-sd-xl-1.0-softedge-dexined/resolve/main/controlnet-sd-xl-1.0-softedge-dexined.safetensors",
]

all_models = diffuser_models_fp16 + sai_control_loras + community_models

import sys
import os
from urllib.parse import urlparse
from torch.hub import download_url_to_file
from typing import List


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def fetch_models(model_urls: List[str], model_dir: str):
    for model_url in model_urls:
        filename = model_url.split("/")[-1].replace(".bin", ".pth")
        load_file_from_url(model_url, model_dir, file_name=filename)


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    default_model_path = os.path.join(os.path.dirname(current_file_path), "models")

    model_path = sys.argv[1] if len(sys.argv) > 1 else default_model_path
    user_input = (
        input(f"Downloading to model path: {model_path} (Y/N): ").strip().upper()
    )

    if user_input == "Y":
        print("Proceeding...")
        fetch_models(all_models, model_path)
    else:
        print("Operation cancelled.")
