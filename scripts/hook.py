
import torch
import torch.nn as nn
from modules import devices, lowvram, shared, scripts

cond_cast_unet = getattr(devices, 'cond_cast_unet', lambda x: x)

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


class ControlParams:
    def __init__(
        self, 
        control_model, 
        hint_cond, 
        guess_mode, 
        weight, 
        guidance_stopped,
        start_guidance_percent,
        stop_guidance_percent, 
        advanced_weighting, 
        is_adapter,
        is_extra_cond
    ):
        self.control_model = control_model
        self.hint_cond = hint_cond
        self.guess_mode = guess_mode
        self.weight = weight
        self.guidance_stopped = guidance_stopped
        self.start_guidance_percent = start_guidance_percent
        self.stop_guidance_percent = stop_guidance_percent
        self.advanced_weighting = advanced_weighting
        self.is_adapter = is_adapter
        self.is_extra_cond = is_extra_cond


class UnetHook(nn.Module):
    def __init__(self, lowvram=False) -> None:
        super().__init__()
        self.lowvram = lowvram
        self.batch_cond_available = True
        self.only_mid_control = shared.opts.data.get("control_net_only_mid_control", False)
        
    def hook(self, model):
        outer = self
        
        def guidance_schedule_handler(x):
            for param in self.control_params:
                current_sampling_percent = (x.sampling_step / x.total_sampling_steps)
                param.guidance_stopped = current_sampling_percent < param.start_guidance_percent or current_sampling_percent > param.stop_guidance_percent
   
        def cfg_based_adder(base, x, require_autocast, is_adapter=False):
            if isinstance(x, float):
                return base + x
            
            if require_autocast:
                zeros = torch.zeros_like(base)
                zeros[:, :x.shape[1], ...] = x
                x = zeros
                
            # assume the input format is [cond, uncond] and they have same shape
            # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/0cc0ee1bcb4c24a8c9715f66cede06601bfc00c8/modules/sd_samplers_kdiffusion.py#L114
            if base.shape[0] % 2 == 0 and (self.guess_mode or shared.opts.data.get("control_net_cfg_based_guidance", False)):
                if self.is_vanilla_samplers:  
                    uncond, cond = base.chunk(2)
                    if x.shape[0] % 2 == 0:
                        _, x_cond = x.chunk(2)
                        return torch.cat([uncond, cond + x_cond], dim=0)
                    if is_adapter:
                        return torch.cat([uncond, cond + x], dim=0)
                else:
                    cond, uncond = base.chunk(2)
                    if x.shape[0] % 2 == 0:
                        x_cond, _ = x.chunk(2)
                        return torch.cat([cond + x_cond, uncond], dim=0)
                    if is_adapter:
                        return torch.cat([cond + x, uncond], dim=0)
            
            # resize to sample resolution
            base_h, base_w = base.shape[-2:]
            xh, xw = x.shape[-2:]
            if base_h != xh or base_w != xw:
                x = th.nn.functional.interpolate(x, size=(base_h, base_w), mode="nearest")
            
            return base + x

        def forward(self, x, timesteps=None, context=None, **kwargs):
            total_control = [0.0] * 13
            total_adapter = [0.0] * 4
            total_extra_cond = torch.zeros([0, context.shape[-1]]).to(devices.get_device_for("controlnet"))
            only_mid_control = outer.only_mid_control
            require_inpaint_hijack = False
            
            # handle external cond first
            for param in outer.control_params:
                if param.guidance_stopped or not param.is_extra_cond:
                    continue
                if outer.lowvram:
                    param.control_model.to(devices.get_device_for("controlnet"))
                control = param.control_model(x=x, hint=param.hint_cond, timesteps=timesteps, context=context)
                total_extra_cond = torch.cat([total_extra_cond, control.clone().squeeze(0) * param.weight])
                
            # check if it's non-batch-cond mode (lowvram, edit model etc)
            if context.shape[0] % 2 != 0 and outer.batch_cond_available:
                outer.batch_cond_available = False
                if len(total_extra_cond) > 0 or outer.guess_mode or shared.opts.data.get("control_net_cfg_based_guidance", False):
                    print("Warning: StyleAdapter and cfg/guess mode may not works due to non-batch-cond inference")
                
            # concat styleadapter to cond, pad uncond to same length
            if len(total_extra_cond) > 0 and outer.batch_cond_available:
                total_extra_cond = torch.repeat_interleave(total_extra_cond.unsqueeze(0), context.shape[0] // 2, dim=0)
                if outer.is_vanilla_samplers:  
                    uncond, cond = context.chunk(2)
                    cond = torch.cat([cond, total_extra_cond], dim=1)
                    uncond = torch.cat([uncond, uncond[:, -total_extra_cond.shape[1]:, :]], dim=1)
                    context = torch.cat([uncond, cond], dim=0)
                else:
                    cond, uncond = context.chunk(2)
                    cond = torch.cat([cond, total_extra_cond], dim=1)
                    uncond = torch.cat([uncond, uncond[:, -total_extra_cond.shape[1]:, :]], dim=1)
                    context = torch.cat([cond, uncond], dim=0)
                
            # handle unet injection stuff
            for param in outer.control_params:
                if param.guidance_stopped or param.is_extra_cond:
                    continue
                if outer.lowvram:
                    param.control_model.to(devices.get_device_for("controlnet"))
                    
                # hires stuffs
                # note that this method may not works if hr_scale < 1.1
                if abs(x.shape[-1] - param.hint_cond.shape[-1] // 8) > 8:
                    only_mid_control = shared.opts.data.get("control_net_only_midctrl_hires", True)
                    # If you want to completely disable control net, uncomment this.
                    # return self._original_forward(x, timesteps=timesteps, context=context, **kwargs)
                    
                # inpaint model workaround
                x_in = x
                control_model = param.control_model.control_model
                if not param.is_adapter and x.shape[1] != control_model.input_blocks[0][0].in_channels and x.shape[1] == 9: 
                    # inpaint_model: 4 data + 4 downscaled image + 1 mask
                    x_in = x[:, :4, ...]
                    require_inpaint_hijack = True
                    
                assert param.hint_cond is not None, f"Controlnet is enabled but no input image is given"  
                control = param.control_model(x=x_in, hint=param.hint_cond, timesteps=timesteps, context=context)
                control_scales = ([param.weight] * 13)
                
                if outer.lowvram:
                    param.control_model.to("cpu")
                if param.guess_mode:
                    if param.is_adapter:
                        # see https://github.com/Mikubill/sd-webui-controlnet/issues/269
                        control_scales = param.weight * [0.25, 0.62, 0.825, 1.0]
                    else:    
                        control_scales = [param.weight * (0.825 ** float(12 - i)) for i in range(13)]
                if param.advanced_weighting is not None:
                    control_scales = param.advanced_weighting
                    
                control = [c * scale for c, scale in zip(control, control_scales)]
                for idx, item in enumerate(control):
                    target = total_adapter if param.is_adapter else total_control
                    target[idx] += item
                        
            control = total_control
            assert timesteps is not None, ValueError(f"insufficient timestep: {timesteps}")
            hs = []
            with th.no_grad():
                t_emb = cond_cast_unet(timestep_embedding(timesteps, self.model_channels, repeat_only=False))
                emb = self.time_embed(t_emb)
                h = x.type(self.dtype)
                for i, module in enumerate(self.input_blocks):
                    h = module(h, emb, context)
                    
                    # t2i-adatper, same as openaimodel.py:744
                    if ((i+1)%3 == 0) and len(total_adapter):
                        h = cfg_based_adder(h, total_adapter.pop(0), require_inpaint_hijack, is_adapter=True)
                        
                    hs.append(h)
                h = self.middle_block(h, emb, context)

            control_in = control.pop()
            h = cfg_based_adder(h, control_in, require_inpaint_hijack)

            for i, module in enumerate(self.output_blocks):
                if only_mid_control:
                    hs_input = hs.pop()
                    h = th.cat([h, hs_input], dim=1)
                else:
                    hs_input, control_input = hs.pop(), control.pop()
                    h = th.cat([h, cfg_based_adder(hs_input, control_input, require_inpaint_hijack)], dim=1)
                h = module(h, emb, context)

            h = h.type(x.dtype)
            return self.out(h)

        def forward2(*args, **kwargs):
            # webui will handle other compoments 
            try:
                if shared.cmd_opts.lowvram:
                    lowvram.send_everything_to_cpu()
                                            
                return forward(*args, **kwargs)
            finally:
                if self.lowvram:
                    [param.control_model.to("cpu") for param in self.control_params]
                        
        model._original_forward = model.forward
        model.forward = forward2.__get__(model, UNetModel)
        scripts.script_callbacks.on_cfg_denoiser(guidance_schedule_handler)
    
    def notify(self, params, is_vanilla_samplers): # lint: list[ControlParams]
        self.is_vanilla_samplers = is_vanilla_samplers
        self.control_params = params
        self.guess_mode = any([param.guess_mode for param in params])

    def restore(self, model):
        scripts.script_callbacks.remove_current_script_callbacks()
        if hasattr(self, "control_params"):
            del self.control_params
        
        if not hasattr(model, "_original_forward"):
            # no such handle, ignore
            return
        
        model.forward = model._original_forward
        del model._original_forward
