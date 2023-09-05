# ControlNet for Stable Diffusion WebUI

The WebUI extension for ControlNet and other injection-based SD controls.

![image](https://github.com/Mikubill/sd-webui-controlnet/assets/19834515/00787fd1-1bc5-4b90-9a23-9683f8458b85)

This extension is for AUTOMATIC1111's [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), allows the Web UI to add [ControlNet](https://github.com/lllyasviel/ControlNet) to the original Stable Diffusion model to generate images. The addition is on-the-fly, the merging is not required.

# Installation

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter `https://github.com/Mikubill/sd-webui-controlnet.git` to "URL for extension's git repository".
4. Press "Install" button.
5. Wait for 5 seconds, and you will see the message "Installed into stable-diffusion-webui\extensions\sd-webui-controlnet. Use Installed tab to restart".
6. Go to "Installed" tab, click "Check for updates", and then click "Apply and restart UI". (The next time you can also use these buttons to update ControlNet.)
7. Completely restart A1111 webui including your terminal. (If you do not know what is a "terminal", you can reboot your computer to achieve the same effect.)
8. Download models (see below).
9. After you put models in the correct folder, you may need to refresh to see the models. The refresh button is right to your "Model" dropdown.

# Download Models

Right now all the 14 models of ControlNet 1.1 are in the beta test.

Download the models from ControlNet 1.1: https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main

You need to download model files ending with ".pth" .

Put models in your "stable-diffusion-webui\extensions\sd-webui-controlnet\models". You only need to download "pth" files.

Do not right-click the filenames in HuggingFace website to download. Some users right-clicked those HuggingFace HTML websites and saved those HTML pages as PTH/YAML files. They are not downloading correct files. Instead, please click the small download arrow “↓” icon in HuggingFace to download.

# Download Models for SDXL

See instructions [here](https://github.com/Mikubill/sd-webui-controlnet/discussions/2039).

# Features in ControlNet 1.1

### Perfect Support for All ControlNet 1.0/1.1 and T2I Adapter Models.

Now we have perfect support all available models and preprocessors, including perfect support for T2I style adapter and ControlNet 1.1 Shuffle. (Make sure that your YAML file names and model file names are same, see also YAML files in "stable-diffusion-webui\extensions\sd-webui-controlnet\models".)

### Perfect Support for A1111 High-Res. Fix

Now if you turn on High-Res Fix in A1111, each controlnet will output two different control images: a small one and a large one. The small one is for your basic generating, and the big one is for your High-Res Fix generating. The two control images are computed by a smart algorithm called "super high-quality control image resampling". This is turned on by default, and you do not need to change any setting.

### Perfect Support for All A1111 Img2Img or Inpaint Settings and All Mask Types

Now ControlNet is extensively tested with A1111's different types of masks, including "Inpaint masked"/"Inpaint not masked", and "Whole picture"/"Only masked", and "Only masked padding"&"Mask blur". The resizing perfectly matches A1111's "Just resize"/"Crop and resize"/"Resize and fill". This means you can use ControlNet in nearly everywhere in your A1111 UI without difficulty!

### The New "Pixel-Perfect" Mode

Now if you turn on pixel-perfect mode, you do not need to set preprocessor (annotator) resolutions manually. The ControlNet will automatically compute the best annotator resolution for you so that each pixel perfectly matches Stable Diffusion.

### User-Friendly GUI and Preprocessor Preview

We reorganized some previously confusing UI like "canvas width/height for new canvas" and it is in the 📝 button now. Now the preview GUI is controlled by the "allow preview" option and the trigger button 💥. The preview image size is better than before, and you do not need to scroll up and down - your a1111 GUI will not be messed up anymore!

### Support for Almost All Upscaling Scripts

Now ControlNet 1.1 can support almost all Upscaling/Tile methods. ControlNet 1.1 support the script "Ultimate SD upscale" and almost all other tile-based extensions. Please do not confuse ["Ultimate SD upscale"](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111) with "SD upscale" - they are different scripts. Note that the most recommended upscaling method is ["Tiled VAE/Diffusion"](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) but we test as many methods/extensions as possible. Note that "SD upscale" is supported since 1.1.117, and if you use it, you need to leave all ControlNet images as blank (We do not recommend "SD upscale" since it is somewhat buggy and cannot be maintained - use the "Ultimate SD upscale" instead).

### More Control Modes (previously called Guess Mode)

We have fixed many bugs in previous 1.0’s Guess Mode and now it is called Control Mode

![image](https://user-images.githubusercontent.com/19834515/236641759-6c44ddf6-c7ad-4bda-92be-e90a52911d75.png)

Now you can control which aspect is more important (your prompt or your ControlNet)：

* "Balanced": ControlNet on both sides of CFG scale, same as turning off "Guess Mode" in ControlNet 1.0

* "My prompt is more important": ControlNet on both sides of CFG scale, with progressively reduced SD U-Net injections (layer_weight*=0.825**I, where 0<=I <13, and the 13 means ControlNet injected SD 13 times). In this way, you can make sure that your prompts are perfectly displayed in your generated images.

* "ControlNet is more important": ControlNet only on the Conditional Side of CFG scale (the cond in A1111's batch-cond-uncond). This means the ControlNet will be X times stronger if your cfg-scale is X. For example, if your cfg-scale is 7, then ControlNet is 7 times stronger. Note that here the X times stronger is different from "Control Weights" since your weights are not modified. This "stronger" effect usually has less artifact and give ControlNet more room to guess what is missing from your prompts (and in the previous 1.0, it is called "Guess Mode"). 

<table width="100%">
<tr>
<td width="25%" style="text-align: center">Input (depth+canny+hed)</td>
<td width="25%" style="text-align: center">"Balanced"</td>
<td width="25%" style="text-align: center">"My prompt is more important"</td>
<td width="25%" style="text-align: center">"ControlNet is more important"</td>
</tr>
<tr>
<td width="25%" style="text-align: center"><img src="samples/cm1.png"></td>
<td width="25%" style="text-align: center"><img src="samples/cm2.png"></td>
<td width="25%" style="text-align: center"><img src="samples/cm3.png"></td>
<td width="25%" style="text-align: center"><img src="samples/cm4.png"></td>
</tr>
</table>

### Reference-Only Control

Now we have a `reference-only` preprocessor that does not require any control models. It can guide the diffusion directly using images as references.

(Prompt "a dog running on grassland, best quality, ...")

![image](samples/ref.png)

This method is similar to inpaint-based reference but it does not make your image disordered. 

Many professional A1111 users know a trick to diffuse image with references by inpaint. For example, if you have a 512x512 image of a dog, and want to generate another 512x512 image with the same dog, some users will connect the 512x512 dog image and a 512x512 blank image into a 1024x512 image, send to inpaint, and mask out the blank 512x512 part to diffuse a dog with similar appearance. However, that method is usually not very satisfying since images are connected and many distortions will appear.

This `reference-only` ControlNet can directly link the attention layers of your SD to any independent images, so that your SD will read arbitary images for reference. You need at least ControlNet 1.1.153 to use it.

To use, just select `reference-only` as preprocessor and put an image. Your SD will just use the image as reference.

*Note that this method is as "non-opinioned" as possible. It only contains very basic connection codes, without any personal preferences, to connect the attention layers with your reference images. However, even if we tried best to not include any opinioned codes, we still need to write some subjective implementations to deal with weighting, cfg-scale, etc - tech report is on the way.*

More examples [here](https://github.com/Mikubill/sd-webui-controlnet/discussions/1236).

# Technical Documents

See also the documents of ControlNet 1.1: 

https://github.com/lllyasviel/ControlNet-v1-1-nightly#model-specification

# Default Setting

This is my setting. If you run into any problem, you can use this setting as a sanity check

![image](https://user-images.githubusercontent.com/19834515/235620638-17937171-8ac1-45bc-a3cb-3aebf605b4ef.png)

# Use Previous Models

### Use ControlNet 1.0 Models

https://huggingface.co/lllyasviel/ControlNet/tree/main/models

You can still use all previous models in the previous ControlNet 1.0. Now, the previous "depth" is now called "depth_midas", the previous "normal" is called "normal_midas", the previous "hed" is called "softedge_hed". And starting from 1.1, all line maps, edge maps, lineart maps, boundary maps will have black background and white lines.

### Use T2I-Adapter Models

(From TencentARC/T2I-Adapter)

To use T2I-Adapter models:

1. Download files from https://huggingface.co/TencentARC/T2I-Adapter/tree/main/models
2. Put them in "stable-diffusion-webui\extensions\sd-webui-controlnet\models".
3. Make sure that the file names of pth files and yaml files are consistent.

*Note that "CoAdapter" is not implemented yet.*

# Gallery

The below results are from ControlNet 1.0.

| Source | Input | Output |
|:-------------------------:|:-------------------------:|:-------------------------:|
| (no preprocessor) |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/bal-source.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/bal-gen.png?raw=true"> |
| (no preprocessor) |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/dog_rel.jpg?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/dog_rel.png?raw=true"> |
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro_input.png?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro_canny.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro-out.png?raw=true"> |
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/evt_source.jpg?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/evt_hed.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/evt_gen.png?raw=true"> |
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/an-source.jpg?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/an-pose.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/an-gen.png?raw=true"> |
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/sk-b-src.png?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/sk-b-dep.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/sk-b-out.png?raw=true"> |

The below examples are from T2I-Adapter.

From `t2iadapter_color_sd14v1.pth` :

| Source | Input | Output |
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222947416-ec9e52a4-a1d0-48d8-bb81-736bf636145e.jpeg"> | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222947435-1164e7d8-d857-42f9-ab10-2d4a4b25f33a.png"> | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222947557-5520d5f8-88b4-474d-a576-5c9cd3acac3a.png"> |

From `t2iadapter_style_sd14v1.pth` :

| Source | Input | Output |
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222947416-ec9e52a4-a1d0-48d8-bb81-736bf636145e.jpeg"> | (clip, non-image) | <img width="256" alt="" src="https://user-images.githubusercontent.com/31246794/222965711-7b884c9e-7095-45cb-a91c-e50d296ba3a2.png"> |

# Minimum Requirements

* (Windows) (NVIDIA: Ampere) 4gb - with `--xformers` enabled, and `Low VRAM` mode ticked in the UI, goes up to 768x832

# Multi-ControlNet

This option allows multiple ControlNet inputs for a single generation. To enable this option, change `Multi ControlNet: Max models amount (requires restart)` in the settings. Note that you will need to restart the WebUI for changes to take effect.

<table width="100%">
<tr>
<td width="25%" style="text-align: center">Source A</td>
<td width="25%" style="text-align: center">Source B</td>
<td width="25%" style="text-align: center">Output</td>
</tr>
<tr>
<td width="25%" style="text-align: center"><img src="https://user-images.githubusercontent.com/31246794/220448620-cd3ede92-8d3f-43d5-b771-32dd8417618f.png"></td>
<td width="25%" style="text-align: center"><img src="https://user-images.githubusercontent.com/31246794/220448619-beed9bdb-f6bb-41c2-a7df-aa3ef1f653c5.png"></td>
<td width="25%" style="text-align: center"><img src="https://user-images.githubusercontent.com/31246794/220448613-c99a9e04-0450-40fd-bc73-a9122cefaa2c.png"></td>
</tr>
</table>

# Control Weight/Start/End

Weight is the weight of the controlnet "influence". It's analogous to prompt attention/emphasis. E.g. (myprompt: 1.2). Technically, it's the factor by which to multiply the ControlNet outputs before merging them with original SD Unet.

Guidance Start/End is the percentage of total steps the controlnet applies (guidance strength = guidance end). It's analogous to prompt editing/shifting. E.g. \[myprompt::0.8\] (It applies from the beginning until 80% of total steps)

# Batch Mode

Put any unit into batch mode to activate batch mode for all units. Specify a batch directory for each unit, or use the new textbox in the img2img batch tab as a fallback. Although the textbox is located in the img2img batch tab, you can use it to generate images in the txt2img tab as well.

Note that this feature is only available in the gradio user interface. Call the APIs as many times as you want for custom batch scheduling.

# API and Script Access

This extension can accept txt2img or img2img tasks via API or external extension call. Note that you may need to enable `Allow other scripts to control this extension` in settings for external calls.

To use the API: start WebUI with argument `--api` and go to `http://webui-address/docs` for documents or checkout [examples](https://github.com/Mikubill/sd-webui-controlnet/blob/main/example/api_txt2img.ipynb).

To use external call: Checkout [Wiki](https://github.com/Mikubill/sd-webui-controlnet/wiki/API)

# Command Line Arguments

This extension adds these command line arguments to the webui:

```
    --controlnet-dir <path to directory with controlnet models>                                ADD a controlnet models directory
    --controlnet-annotator-models-path <path to directory with annotator model directories>    SET the directory for annotator models
    --no-half-controlnet                                                                       load controlnet models in full precision
    --controlnet-preprocessor-cache-size                                                       Cache size for controlnet preprocessor results
    --controlnet-loglevel                                                                      Log level for the controlnet extension
```

# MacOS Support

Tested with pytorch nightly: https://github.com/Mikubill/sd-webui-controlnet/pull/143#issuecomment-1435058285

To use this extension with mps and normal pytorch, currently you may need to start WebUI with `--no-half`.

# Archive of Deprecated Versions

The previous version (sd-webui-controlnet 1.0) is archived in 

https://github.com/lllyasviel/webui-controlnet-v1-archived

Using this version is not a temporary stop of updates. You will stop all updates forever.

Please consider this version if you work with professional studios that requires 100% reproducing of all previous results pixel by pixel.

# Thanks

This implementation is inspired by kohya-ss/sd-webui-additional-networks
