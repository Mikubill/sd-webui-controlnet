

import torch
import torch.nn as nn

from omegaconf import OmegaConf
from copy import deepcopy
from modules import devices, lowvram, shared, scripts
from ldm.modules.diffusionmodules.util import timestep_embedding
from ldm.modules.diffusionmodules.openaimodel import UNetModel


class TorchHijackForUnet:
    """
    This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;
    this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64
    """

    def __getattr__(self, item):
        if item == 'cat':
            return self.cat

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    def cat(self, tensors, *args, **kwargs):
        if len(tensors) == 2:
            a, b = tensors
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        return torch.cat(tensors, *args, **kwargs)


th = TorchHijackForUnet()


def align(hint, size):
    b, c, h1, w1 = hint.shape
    h, w = size
    if h != h1 or w != w1:
         hint = th.nn.functional.interpolate(hint, size=size, mode="nearest")
    return hint


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


class PlugableAdapter(nn.Module):
    def __init__(self, state_dict, config_path, weight=1.0, lowvram=False, base_model=None) -> None:
        super().__init__()
        config = OmegaConf.load(config_path)
        
        self.control_model = Adapter(**config.model.params)           
        self.control_model.load_state_dict(state_dict)
        self.lowvram = lowvram            
        self.weight = weight
        self.control = None
        self.hint_cond = None
        
        if not self.lowvram:
            self.control_model.to(devices.get_device_for("controlnet"))

    def hook(self, model, parent_model):
        outer = self
        
        def guidance_schedule_handler(x):
            if (x.sampling_step / x.total_sampling_steps) > self.stop_guidance_percent:
                # stop guidance
                self.guidance_stopped = True

        def forward(self, x, timesteps=None, context=None, **kwargs):            
            features_adapter = kwargs["features"]
            assert timesteps is not None, ValueError(f"insufficient timestep: {timesteps}")
            hs = []
            with th.no_grad():
                t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
                emb = self.time_embed(t_emb)
                h = x.type(self.dtype)
                for i, module in enumerate(self.input_blocks):
                    h = module(h, emb, context)
                    # same as openaimodel.py:744
                    if ((i+1)%3 == 0) and len(features_adapter) and not outer.guidance_stopped:
                        h = h + features_adapter.pop(0) * outer.weight
                    hs.append(h)
                h = self.middle_block(h, emb, context)

            for i, module in enumerate(self.output_blocks):
                h = th.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context)

            h = h.type(x.dtype)
            return self.out(h)

        def forward2(*args, **kwargs):
            # webui will handle other compoments 
            try:
                if shared.cmd_opts.lowvram:
                    lowvram.send_everything_to_cpu()
                    
                if self.lowvram:
                    self.control_model.to(devices.get_device_for("controlnet"))
                    
                if not hasattr(outer, "features"):
                    if self.control_model.conv_in.in_channels == 64:
                        outer.hint_cond = outer.hint_cond[0].unsqueeze(0).unsqueeze(0)
                    else:
                        outer.hint_cond = outer.hint_cond.unsqueeze(0)
                
                outer.features = self.control_model(outer.hint_cond)
                return forward(features=deepcopy(outer.features), *args, **kwargs)
            finally:
                if self.lowvram:
                    self.control_model.cpu()
        
        model._original_forward = model.forward
        model.forward = forward2.__get__(model, UNetModel)
        scripts.script_callbacks.on_cfg_denoiser(guidance_schedule_handler)
    
    def notify(self, cond_like, weight, stop_guidance_percent):
        if hasattr(self, "features"):
            del self.features
            
        self.stop_guidance_percent = stop_guidance_percent
        self.guidance_stopped = False
        
        self.hint_cond = cond_like
        self.weight = weight

    def restore(self, model):
        scripts.script_callbacks.remove_current_script_callbacks()
        if not hasattr(model, "_original_forward"):
            # no such handle, ignore
            return
        
        model.forward = model._original_forward
        del model._original_forward


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize//2
        if in_c != out_c or sk==False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk==False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_sk')
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None: # edit
            h = self.in_conv(x)
            # x = self.in_conv(x)
        # else:
        #     x = x

        h = self.block1(h)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize//2
        if in_c != out_c or sk==False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk==False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None: # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class Adapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True):
        super(Adapter, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i!=0) and (j==0):
                    self.body.append(ResnetBlock(channels[i-1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i*self.nums_rb +j
                x = self.body[idx](x)
            features.append(x)

        return features