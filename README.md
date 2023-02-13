## sd-webui-controlnet
(WIP) WebUI extension for ControlNet

This extension is for AUTOMATIC1111's [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), allows the Web UI to add [ControlNet](https://github.com/lllyasviel/ControlNet) to the original Stable Diffusion model to generate images. The addition is on-the-fly, the merging is not required.

ControlNet is a neural network structure to control diffusion models by adding extra conditions. 

Thanks & Inspired: kohya-ss/sd-webui-additional-networks

## Limits

Dragging a file on the Web UI will freeze the entire page. It is better to use the upload file option instead.

## Install

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter URL of this repo to "URL for extension's git repository".
4. Press "Install" button.
5. Reload/Restart Web UI.

## Usage

1. Put the ControlNet models (`.pt`, `.pth`, `.ckpt` or `.safetensors`) inside the sd-webui-controlnet/models folder.
2. Press "Refresh models" to update the models list.
3. Upload your image and select preprocessor, done.

Currently it supports both full models and trimmed models. Use `extract_controlnet.py` to extract controlnet from original `.pth` file.

## Examples

| Input | Processed | Image |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro_input.png?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro_canny.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro_gen.png?raw=true"> |
