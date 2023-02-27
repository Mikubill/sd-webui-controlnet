## sd-webui-controlnet
(WIP) WebUI extension for ControlNet

This extension is for AUTOMATIC1111's [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), allows the Web UI to add [ControlNet](https://github.com/lllyasviel/ControlNet) to the original Stable Diffusion model to generate images. The addition is on-the-fly, the merging is not required.

ControlNet is a neural network structure to control diffusion models by adding extra conditions. 

Thanks & Inspired: kohya-ss/sd-webui-additional-networks

### Limits

* Dragging large file on the Web UI may freeze the entire page. It is better to use the upload file option instead.
* Just like WebUI's [hijack](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/3715ece0adce7bf7c5e9c5ab3710b2fdc3848f39/modules/sd_hijack_unet.py#L27), we used some interpolate to accept arbitrary size configure (see `scripts/cldm.py`)

### Install

Upgrade gradio if any ui issues occured: `pip install gradio==3.16.2`

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter URL of this repo to "URL for extension's git repository".
4. Press "Install" button.
5. Reload/Restart Web UI.

### Usage

1. Put the ControlNet models (`.pt`, `.pth`, `.ckpt` or `.safetensors`) inside the `models/ControlNet` folder.
2. Open "txt2img" or "img2img" tab, write your prompts.
3. Press "Refresh models" and select the model you want to use. (If nothing appears, try reload/restart the webui)
4. Upload your image and select preprocessor, done.

Currently it supports both full models and trimmed models. Use `extract_controlnet.py` to extract controlnet from original `.pth` file.

Pretrained Models: https://huggingface.co/lllyasviel/ControlNet/tree/main/models

### Extraction

Two methods can be used to reduce the model's filesize:

1. Directly extract controlnet from original .pth file using `extract_controlnet.py`.

2. Transfer control from original checkpoint by making difference using `extract_controlnet_diff.py`.

All type of models can be correctly recognized and loaded. The results of different extraction methods are discussed in https://github.com/lllyasviel/ControlNet/discussions/12 and https://github.com/Mikubill/sd-webui-controlnet/issues/73. 

Pre-extracted model: https://huggingface.co/webui/ControlNet-modules-safetensors

Pre-extracted difference model: https://huggingface.co/kohya-ss/ControlNet-diff-modules

### T2I-Adapter Support (Experimental)

Currently support both sketch Adapter and image Adapter. Note that the impl is experimental, result may differ from original repo. See `Adapter Examples` for reference.

To use these models:
1. Download files from https://huggingface.co/TencentARC/T2I-Adapter
2. Setup correct config in settings panel - `sketch_adapter_v14.yaml` for sketch model and `image_adapter_v14.yaml` for keypose and segmentation model.
3. It's better to use a slightly lower strength (t) when generating images with sketch model, such as 0.6-0.8. (ref: [ldm/models/diffusion/plms.py](https://github.com/TencentARC/T2I-Adapter/blob/5f41a0e38fc6eac90d04bc4cede85a2bc4570653/ldm/models/diffusion/plms.py#L158))

### Tips 

* Don't forget to add some negative prompt, default negative prompt in ControlNet repo is "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality".
* Regarding canvas height/width: they are designed for canvas generation. If you want to upload images directly, you can safely ignore them.

### Examples

| Source | Input | Output |
|:-------------------------:|:-------------------------:|:-------------------------:|
| (no preprocessor) |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/bal-source.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/bal-gen.png?raw=true"> |
| (no preprocessor) |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/dog_rel.jpg?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/dog_rel.png?raw=true"> |
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro_input.png?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro_canny.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro-out.png?raw=true"> |
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/evt_source.jpg?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/evt_hed.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/evt_gen.png?raw=true"> |
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/an-source.jpg?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/an-pose.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/an-gen.png?raw=true"> |
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/sk-b-src.png?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/sk-b-dep.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/sk-b-out.png?raw=true"> |
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/nm-src.png?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/nm-gen.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/nm-out.png?raw=true"> |

### Adapter Examples

| Input | Output |
|:-------------------------:|:-------------------------:|
|  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/dog_sk-2.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/dog_out-2.png?raw=true"> |
|  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/cat_sk-2.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/cat_out-2.png?raw=true"> |
|  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/kp_a-2.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/kp_o-2.png?raw=true"> |
|  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/kp_o2-2.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/kp_a2-2.png?raw=true"> |

### Minimum Requirements

* (Windows) (NVIDIA: Ampere) 4gb - with `--xformers` enabled, and `Low VRAM` mode ticked in the UI, goes up to 768x832

### CFG Based ControlNet (Experimental)

The original ControlNet applies control to both conditional (cond) and unconditional (uncond) parts. Enabling this option will make the control only apply to the cond part. Some experiments indicate that this approach improves image quality.

To enable this option, tick `Enable CFG-Based guidance for ControlNet` in the settings.

Note that you need to use a low cfg scale/guidance scale (such as 3-5) and proper weight tuning to get good result.

### Guess Mode (Non-Prompt Mode, Experimental)

Guess Mode is CFG Based ControlNet + Exponential decay in weighting. 

See issue https://github.com/Mikubill/sd-webui-controlnet/issues/236 for more details.

Original introduction from controlnet:

The "guess mode" (or called non-prompt mode) will completely unleash all the power of the very powerful ControlNet encoder.

In this mode, you can just remove all prompts, and then the ControlNet encoder will recognize the content of the input control map, like depth map, edge map, scribbles, etc.

This mode is very suitable for comparing different methods to control stable diffusion because the non-prompted generating task is significantly more difficult than prompted task. In this mode, different methods' performance will be very salient.

For this mode, we recommend to **use 50 steps and guidance scale between 3 and 5.**

### Multi-ControlNet / Joint Conditioning (Experimental)

This option allows multiple ControlNet inputs for a single generation. To enable this option, change `Multi ControlNet: Max models amount (requires restart)` in the settings. Note that you will need to restart the WebUI for changes to take effect.

* Guess Mode will apply to all ControlNet if any of them are enabled.

| Source A | Source B | Output |
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/220448620-cd3ede92-8d3f-43d5-b771-32dd8417618f.png"> |  <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/220448619-beed9bdb-f6bb-41c2-a7df-aa3ef1f653c5.png"> | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/220448613-c99a9e04-0450-40fd-bc73-a9122cefaa2c.png"> |

### Weight and Guidance Strength/Start/End

Weight is the weight of the controlnet "influence". It's analogous to prompt attention/emphasis. E.g. (myprompt: 1.2). Technically, it's the factor by which to multiply the ControlNet outputs before merging them with original SD Unet.

Guidance Start/End is the percentage of total steps the controlnet applies (guidance strength = guidance end). It's analogous to prompt editing/shifting. E.g. \[myprompt::0.8\] (It applies from the beginning until 80% of total steps)

### API/Script Access

This extension can accept txt2img or img2img tasks via API or external extension call. Note that you may need to enable `Allow other scripts to control this extension` in settings for external calls.

To use the API: start WebUI with argument `--api` and go to `http://webui-address/docs` for documents or checkout [examples](https://github.com/Mikubill/sd-webui-controlnet/blob/main/example/api_txt2img.ipynb).

To use external extension call: Pass your config to p(pipeline). For more details see [scripts/controlnet.py](https://github.com/Mikubill/sd-webui-controlnet/blob/8ca06ba8eb2989cbd434063c9a7c0e7a3bdfabea/scripts/controlnet.py#L459-L462)

### MacOS Support

Tested with pytorch nightly: https://github.com/Mikubill/sd-webui-controlnet/pull/143#issuecomment-1435058285

To use this extension with mps and normal pytorch, currently you may need to start WebUI with `--no-half`.
