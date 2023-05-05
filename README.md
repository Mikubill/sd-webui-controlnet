## sd-webui-controlnet

(WIP) WebUI extension for ControlNet and other injection-based SD controls.

![image](https://user-images.githubusercontent.com/19834515/235606305-229b3d1e-5bfc-467f-9d55-0976eab71652.png)

This extension is for AUTOMATIC1111's [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), allows the Web UI to add [ControlNet](https://github.com/lllyasviel/ControlNet) to the original Stable Diffusion model to generate images. The addition is on-the-fly, the merging is not required.

ControlNet is a neural network structure to control diffusion models by adding extra conditions. 

Thanks & Inspired by: kohya-ss/sd-webui-additional-networks

### Install

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter `https://github.com/Mikubill/sd-webui-controlnet.git` to "URL for extension's git repository".
4. Press "Install" button.
5. Wait 5 seconds, and you will see the message "Installed into stable-diffusion-webui\extensions\sd-webui-controlnet. Use Installed tab to restart".
6. Go to "Installed" tab, click "Check for updates", and then click "Apply and restart UI". (The next time you can also use this method to update ControlNet.)
7. Completely restart A1111 webui including your terminal. (If you do not know what is a "terminal", you can reboot your computer: turn your computer off and turn it on again.)
8. Download models (see below).
9. After you put models in the correct folder, you may need to refresh to see the models. The refresh button is right to your "Model" dropdown.

### Download Models

Right now all the 14 models of ControlNet 1.1 are in the beta test. [Here is the discussion and bug report](https://github.com/Mikubill/sd-webui-controlnet/issues/736).

Download the models from ControlNet 1.1: https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main

You need to download model files ending with ".pth" .

**Put models in your "stable-diffusion-webui\extensions\sd-webui-controlnet\models". Now we have already included all "yaml" files. You only need to download "pth" files.** 

Note: If you download models elsewhere, please make sure that yaml file names and model files names are same. Please manually rename all yaml files if you download from other sources. Otherwise, models may have unexpected behaviors. You can ignore this if you download models from official sources. (Some models like "shuffle" needs the YAML file so that we know the outputs of ControlNet should pass a global average pooling before inject to SD U-Nets.)

**Do not right click the filenames in HuggingFace website to download. Some users right clicked those HuggingFace HTML websites and saved those HTML pages as PTH/YAML files. They are not downloading correct PTH/YAML files. Instead, please click the small download arrow “↓” icon in HuggingFace to download.**

### New Features in A1111 ControlNet Extension 1.1

**Perfect Support for All ControlNet 1.0/1.1 and T2I Adapter Models.**

Now we have perfect support all available models and preprocessors, including perfect support for T2I style adapter and ControlNet 1.1 Shuffle. (Make sure that your YAML file names and model file names are same, see also YAML files in "stable-diffusion-webui\extensions\sd-webui-controlnet\models".)

**Perfect Support for A1111 High-Res. Fix**

Now if you turn on High-Res Fix in A1111, each controlnet will output two different control images: a small one and a large one. The small one is for your basic generating, and the big one is for your High-Res Fix generating. The two control images are computed by a smart algorithm called "super high-quality control image resampling". This is turned on by default, and you do not need to change any setting.

**Perfect Support for A1111 I2I and Mask**

Now ControlNet is extensively tested with A1111's different types of masks, including "Inpaint masked"/"Inpaint not masked", and "Whole picture"/"Only masked", and "Only masked padding"&"Mask blur". The resizing perfectly matches A1111's "Just resize"/"Crop and resize"/"Resize and fill". This means you can use ControlNet in nearly everywhere in your A1111 UI without difficulty!

**Pixel Perfect Mode**

Now if you turn on pixel-perfect mode, you do not need to set preprocessor (annotator) resolutions manually. The ControlNet will automatically compute the best annotator resolution for you so that each pixel perfectly matches Stable Diffusion.

**User-Friendly GUI and Preprocessor Preview**

We reorganized some previously confusing UI like "canvas width/height for new canvas" and it is in the 📝 button now. Now the preview GUI is controlled by the "allow preview" option and the trigger button 💥. The preview image size is better than before, and you do not need to scroll up and down - your a1111 GUI will not be messed up anymore!

**Support for Upscaling Scripts**

Now ControlNet 1.1 can support almost all Upscaling/Tile methods. ControlNet 1.1 support the script "Ultimate SD upscale" and almost all other tile-based extensions. Please do not confuse ["Ultimate SD upscale"](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111) with "SD upscale" - they are different scripts. Note that the most recommended upscaling method is ["Tiled VAE/Diffusion"](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) but we test as many methods/extensions as possible. Note that "SD upscale" is supported since 1.1.117, and if you use it, you need to leave all ControlNet images as blank (We do not recommend "SD upscale" since it is somewhat buggy and cannot be maintained).

**Control Mode (previously called Guess Mode)**

We have fixed many bugs in previous 1.0’s Guess Mode and now it is called Control Mode

![image](https://user-images.githubusercontent.com/19834515/233897484-e84232f5-be3f-418b-bdaa-d9d7393f3988.png)

Now you can control which aspect is more important (your prompt or your ControlNet)

| Input (depth+canny+hed) | Control Mode: "Balanced" | Control Mode: "My prompt is more important" | Control Mode: "ControlNet is more important"
| --- | --- | --- | --- |
| ![image](https://user-images.githubusercontent.com/19834515/233898012-3b7f970e-a599-42b2-a56a-65899b816085.png) | ![image](https://user-images.githubusercontent.com/19834515/233897765-8ecf4ee7-a605-46fc-b124-57474c3d7575.png) | ![image](https://user-images.githubusercontent.com/19834515/233897802-6bf46513-9203-482b-8997-e889d578a381.png) | ![image](https://user-images.githubusercontent.com/19834515/233897826-54033ff3-3251-4a6a-bf4a-62f877cb6434.png) |

"Balanced" = put ControlNet on both sides of cfg scale, same as turn off "Guess Mode" in ControlNet 1.0

"My prompt is more important" = put ControlNet on both sides of cfg scale and use progressively reduced SD U-Net injections (layer_weight*=0.825**I, where 0<=I <13, and the 13 means ControlNet injected SD 13 times). In this way, you can make sure that your prompts are perfectly displayed in your generated images.

"ControlNet is more important" = put ControlNet only on the Conditional Side (the cond in A1111's batch-cond-uncond). This means the ControlNet will be X times stronger if your cfg-scale is X. For example, if your cfg-scale is 7, then ControlNet is 7 times stronger. Note that here the X times stronger is different from "Control Weights" since your weights are not modified. This "stronger" effect usually has less artifacts and give ControlNet more space to guess what is missing from your prompts (and in 1.0, it is called "Guess Mode"). 

### See Also

Documents of ControlNet 1.1: https://github.com/lllyasviel/ControlNet-v1-1-nightly

### Update from ControlNet 1.0 to 1.1

If you are a previous user of ControlNet 1.0, you may:

* If you are not sure, you can back up and remove the folder "stable-diffusion-webui\extensions\sd-webui-controlnet", and then start from the step 1 in the above Install section. 

* Or you can start from the step 6 in the above Install section.


### Default Setting

This is my setting. If you run into any problem, you can use this setting as a sanity check

![image](https://user-images.githubusercontent.com/19834515/235620638-17937171-8ac1-45bc-a3cb-3aebf605b4ef.png)

### Previous Models

Big Models: https://huggingface.co/lllyasviel/ControlNet/tree/main/models

Small Models: https://huggingface.co/webui/ControlNet-modules-safetensors

You can still use all previous models in the previous ControlNet 1.0. Now, the previous "depth" is now called "depth_midas", the previous "normal" is called "normal_midas", the previous "hed" is called "softedge_hed". And starting from 1.1, all line maps, edge maps, lineart maps, boundary maps will have black background and white lines.

### Use Previous Version 1.0

The previous version (sd-webui-controlnet 1.0) is archived in 

https://github.com/lllyasviel/webui-controlnet-v1-archived

Using this version is not a temporary stop of updates. You will stop all updates forever.

Please consider this version if you work with professional studios that requires 100% reproducing of all previous results pixel by pixel.

In the new controlnet 1.1, your inputs are always correct as long as you follow the one and only one rule: set preprocessor as invert if your image has black lines and white background. If you prefer the previous 1.0 way to manually find out correct combinations by testing all correct/wrong combinations of preprocessors+invert+rgb2bgr, you may opt-out the updating and use the above old version 1.0.


### Usage

1. Open "txt2img" or "img2img" tab, write your prompts.
2. Press "Refresh models" and select the model you want to use. (If nothing appears, try reload/restart the webui)
3. Upload your image and select preprocessor, done.

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

### T2I-Adapter Support

(From TencentARC/T2I-Adapter)

T2I-Adapter is a small network that can provide additional guidance for pre-trained text-to-image models. 

To use T2I-Adapter models:
1. Download files from https://huggingface.co/TencentARC/T2I-Adapter
2. Copy corresponding config file and rename it to the same name as the model - see list below.
3. It's better to use a slightly lower strength (t) when generating images with sketch model, such as 0.6-0.8. (ref: [ldm/models/diffusion/plms.py](https://github.com/TencentARC/T2I-Adapter/blob/5f41a0e38fc6eac90d04bc4cede85a2bc4570653/ldm/models/diffusion/plms.py#L158))

| Adapter | Config |
|:-------------------------:|:-------------------------:|
| t2iadapter_canny_sd14v1.pth | sketch_adapter_v14.yaml |
| t2iadapter_sketch_sd14v1.pth | sketch_adapter_v14.yaml |
| t2iadapter_seg_sd14v1.pth | image_adapter_v14.yaml |
| t2iadapter_keypose_sd14v1.pth | image_adapter_v14.yaml |
| t2iadapter_openpose_sd14v1.pth | image_adapter_v14.yaml |
| t2iadapter_color_sd14v1.pth | t2iadapter_color_sd14v1.yaml |
| t2iadapter_style_sd14v1.pth | t2iadapter_style_sd14v1.yaml |

Note: 

* This implement is experimental, result may differ from original repo.
* Some adapters may have mapping deviations (see issue https://github.com/lllyasviel/ControlNet/issues/255)

### Adapter Examples

| Source | Input | Output |
|:-------------------------:|:-------------------------:|:-------------------------:|
| (no preprocessor) | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/dog_sk-2.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/dog_out-2.png?raw=true"> |
| (no preprocessor) | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/cat_sk-2.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/cat_out-2.png?raw=true"> |
| (no preprocessor) | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222967315-dc50406d-2930-47c5-8027-f76b95969f2b.png"> | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222967311-724d9531-4b93-4770-8409-cd9480434112.png"> |
| (no preprocessor) | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222966824-8f6c36f1-525b-40c2-ae9e-d3f5d148b5c9.png"> | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222966821-110541a4-5014-4cee-90f8-758edf540eae.png"> |
| <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222947416-ec9e52a4-a1d0-48d8-bb81-736bf636145e.jpeg"> | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222947435-1164e7d8-d857-42f9-ab10-2d4a4b25f33a.png"> | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222947557-5520d5f8-88b4-474d-a576-5c9cd3acac3a.png"> |
| <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222947416-ec9e52a4-a1d0-48d8-bb81-736bf636145e.jpeg"> | (clip, non-image) | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222965711-7b884c9e-7095-45cb-a91c-e50d296ba3a2.png"> |

Examples by catboxanon, no tweaking or cherrypicking. (Color Guidance)

| Image | Disabled | Enabled |
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width="256" alt="" src="https://user-images.githubusercontent.com/122327233/222869104-0830feab-a0a1-448e-8bcd-add54b219cba.png"> |  <img width="256" alt="" src="https://user-images.githubusercontent.com/122327233/222869047-d0111979-0ef7-4152-8523-8a45c47217c0.png"> | <img width="256" alt="" src="https://user-images.githubusercontent.com/122327233/222869079-7e5a62e0-fffe-4a19-8875-cba4c68b9428.png"> |
| <img width="256" alt="" src="https://user-images.githubusercontent.com/122327233/222869253-44f94dfa-5448-48b2-85be-73db867bdbbb.png"> |  <img width="256" alt="" src="https://user-images.githubusercontent.com/122327233/222869261-92e598d0-2950-4874-8b6c-c159bda38365.png"> | <img width="256" alt="" src="https://user-images.githubusercontent.com/122327233/222869272-a4883524-7804-4013-addd-4d1ac56c5d0d.png"> |

### Minimum Requirements

* (Windows) (NVIDIA: Ampere) 4gb - with `--xformers` enabled, and `Low VRAM` mode ticked in the UI, goes up to 768x832

### Multi-ControlNet / Joint Conditioning (Experimental)

This option allows multiple ControlNet inputs for a single generation. To enable this option, change `Multi ControlNet: Max models amount (requires restart)` in the settings. Note that you will need to restart the WebUI for changes to take effect.

* Guess Mode will apply to all ControlNet if any of them are enabled.

| Source A | Source B | Output |
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/220448620-cd3ede92-8d3f-43d5-b771-32dd8417618f.png"> |  <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/220448619-beed9bdb-f6bb-41c2-a7df-aa3ef1f653c5.png"> | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/220448613-c99a9e04-0450-40fd-bc73-a9122cefaa2c.png"> |

### Weight and Guidance Strength/Start/End

Weight is the weight of the controlnet "influence". It's analogous to prompt attention/emphasis. E.g. (myprompt: 1.2). Technically, it's the factor by which to multiply the ControlNet outputs before merging them with original SD Unet.

Guidance Start/End is the percentage of total steps the controlnet applies (guidance strength = guidance end). It's analogous to prompt editing/shifting. E.g. \[myprompt::0.8\] (It applies from the beginning until 80% of total steps)

### Batch Mode

Put any unit into batch mode to activate batch mode for all units. Specify a batch directory for each unit, or use the new textbox in the img2img batch tab as a fallback. Although the textbox is located in the img2img batch tab, you can use it to generate images in the txt2img tab as well.

Note that this feature is only available in the gradio user interface. Call the APIs as many times as you want for custom batch scheduling.

### API/Script Access

This extension can accept txt2img or img2img tasks via API or external extension call. Note that you may need to enable `Allow other scripts to control this extension` in settings for external calls.

To use the API: start WebUI with argument `--api` and go to `http://webui-address/docs` for documents or checkout [examples](https://github.com/Mikubill/sd-webui-controlnet/blob/main/example/api_txt2img.ipynb).

To use external call: Checkout [Wiki](https://github.com/Mikubill/sd-webui-controlnet/wiki/API)

### Command Line Arguments

This extension adds these command line arguments to the webui:

```
    --controlnet-dir <path to directory with controlnet models>                                ADD a controlnet models directory
    --controlnet-annotator-models-path <path to directory with annotator model directories>    SET the directory for annotator models
    --no-half-controlnet                                                                       load controlnet models in full precision
```

### MacOS Support

Tested with pytorch nightly: https://github.com/Mikubill/sd-webui-controlnet/pull/143#issuecomment-1435058285

To use this extension with mps and normal pytorch, currently you may need to start WebUI with `--no-half`.

### Example: Visual-ChatGPT (by API)

Quick start:

```base
# Run WebUI in API mode
python launch.py --api --xformers

# Install/Upgrade transformers
pip install -U transformers

# Install deps
pip install langchain==0.0.101 openai 

# Run exmaple
python example/chatgpt.py
```

### Limits

* Dragging large file on the Web UI may freeze the entire page. It is better to use the upload file option instead.
* Just like WebUI's [hijack](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/3715ece0adce7bf7c5e9c5ab3710b2fdc3848f39/modules/sd_hijack_unet.py#L27), we used some interpolate to accept arbitrary size configure (see `scripts/cldm.py`)

