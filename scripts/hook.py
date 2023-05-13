import torch
import torch.nn as nn

from enum import Enum
from modules import devices, lowvram, shared, scripts

cond_cast_unet = getattr(devices, 'cond_cast_unet', lambda x: x)

from ldm.modules.diffusionmodules.util import timestep_embedding
from ldm.modules.diffusionmodules.openaimodel import UNetModel


class ControlModelType(Enum):
    """
    The type of Control Models (supported or not).
    """

    ControlNet = "ControlNet, Lvmin Zhang"
    T2I_Adapter = "T2I_Adapter, Chong Mou"
    T2I_StyleAdapter = "T2I_StyleAdapter, Chong Mou"
    T2I_CoAdapter = "T2I_CoAdapter, Chong Mou"
    MasaCtrl = "MasaCtrl, Mingdeng Cao"
    GLIGEN = "GLIGEN, Yuheng Li"
    AttentionInjection = "AttentionInjection, Lvmin Zhang"  # A simple attention injection written by Lvmin
    StableSR = "StableSR, Jianyi Wang"
    PromptDiffusion = "PromptDiffusion, Zhendong Wang"
    ControlLoRA = "ControlLoRA, Wu Hecong"


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
        weight,
        guidance_stopped,
        start_guidance_percent,
        stop_guidance_percent, 
        advanced_weighting, 
        control_model_type,
        hr_hint_cond,
        global_average_pooling,
        batch_size,
        instance_counter,
        is_vanilla_samplers,
        cfg_scale,
        soft_injection,
        cfg_injection
    ):
        self.control_model = control_model
        self._hint_cond = hint_cond
        self.weight = weight
        self.guidance_stopped = guidance_stopped
        self.start_guidance_percent = start_guidance_percent
        self.stop_guidance_percent = stop_guidance_percent
        self.advanced_weighting = advanced_weighting
        self.control_model_type = control_model_type
        self.global_average_pooling = global_average_pooling
        self.hr_hint_cond = hr_hint_cond
        self.used_hint_cond = None
        self.used_hint_cond_latent = None
        self.batch_size = batch_size
        self.instance_counter = instance_counter
        self.is_vanilla_samplers = is_vanilla_samplers
        self.cfg_scale = cfg_scale
        self.soft_injection = soft_injection
        self.cfg_injection = cfg_injection

    def generate_uc_mask(self, length, dtype, device):
        if self.is_vanilla_samplers and self.cfg_scale == 1:
            return torch.tensor([1 for _ in range(length)], dtype=dtype, device=device)

        y = []

        for i in range(length):
            p = (self.instance_counter + i) % (self.batch_size * 2)
            if self.is_vanilla_samplers:
                y += [0] if p < self.batch_size else [1]
            else:
                y += [1] if p < self.batch_size else [0]

        self.instance_counter += length

        return torch.tensor(y, dtype=dtype, device=device)

    @property
    def hint_cond(self):
        return self._hint_cond

    # fix for all the extensions that modify hint_cond,
    # by forcing used_hint_cond to update on the next timestep
    # hr_hint_cond can stay the same, since most extensions dont modify the hires pass
    # but if they do, it will cause problems
    @hint_cond.setter
    def hint_cond(self, new_hint_cond):
        self._hint_cond = new_hint_cond
        self.used_hint_cond = None
        self.used_hint_cond_latent = None


class UnetHook(nn.Module):
    def __init__(self, lowvram=False) -> None:
        super().__init__()
        self.lowvram = lowvram

    def guidance_schedule_handler(self, x):
        for param in self.control_params:
            current_sampling_percent = (x.sampling_step / x.total_sampling_steps)
            param.guidance_stopped = current_sampling_percent < param.start_guidance_percent or current_sampling_percent > param.stop_guidance_percent

    def hook(self, model):
        outer = self

        def cfg_based_adder(base, x, require_autocast):
            if isinstance(x, float):
                return base + x
            
            if require_autocast:
                zeros = torch.zeros_like(base)
                zeros[:, :x.shape[1], ...] = x
                x = zeros

            # resize to sample resolution
            base_h, base_w = base.shape[-2:]
            xh, xw = x.shape[-2:]
            if base_h != xh or base_w != xw:
                x = th.nn.functional.interpolate(x, size=(base_h, base_w), mode="nearest")
            
            return base + x

        def forward(self, x, timesteps=None, context=None, **kwargs):
            total_controlnet_embedding = [0.0] * 13
            total_t2i_adapter_embedding = [0.0] * 4
            total_extra_prompt_tokens = None
            require_inpaint_hijack = False

            # High-res fix
            is_in_high_res_fix = False
            for param in outer.control_params:
                # select which hint_cond to use
                param.used_hint_cond = param.hint_cond

                # Attention Injection do not need high-res fix
                if param.control_model_type in [ControlModelType.AttentionInjection]:
                    continue

                # has high-res fix
                if param.hr_hint_cond is not None and x.ndim == 4 and param.hint_cond.ndim == 3 and param.hr_hint_cond.ndim == 3:
                    _, h_lr, w_lr = param.hint_cond.shape
                    _, h_hr, w_hr = param.hr_hint_cond.shape
                    _, _, h, w = x.shape
                    h, w = h * 8, w * 8
                    if abs(h - h_lr) < abs(h - h_hr):
                        # we are in low-res path
                        param.used_hint_cond = param.hint_cond
                    else:
                        # we are in high-res path
                        param.used_hint_cond = param.hr_hint_cond
                        is_in_high_res_fix = True

            # Convert control image to latent
            for param in outer.control_params:
                if param.used_hint_cond_latent is not None:
                    continue
                if param.control_model_type not in [ControlModelType.AttentionInjection]:
                    continue
                param.used_hint_cond_latent = 'No Implementation'

            # handle prompt token control
            for param in outer.control_params:
                if param.guidance_stopped:
                    continue

                if param.control_model_type not in [ControlModelType.T2I_StyleAdapter]:
                    continue

                param.control_model.to(devices.get_device_for("controlnet"))
                query_size = int(x.shape[0])
                control = param.control_model(x=x, hint=param.used_hint_cond, timesteps=timesteps, context=context)
                uc_mask = param.generate_uc_mask(query_size, dtype=x.dtype, device=x.device)[:, None, None]
                control = torch.cat([control.clone() for _ in range(query_size)], dim=0)
                control *= param.weight
                control *= uc_mask
                if total_extra_prompt_tokens is None:
                    total_extra_prompt_tokens = control.clone()
                else:
                    total_extra_prompt_tokens = torch.cat([total_extra_prompt_tokens, control.clone()], dim=1)
                
            if total_extra_prompt_tokens is not None:
                context = torch.cat([context, total_extra_prompt_tokens], dim=1)
                
            # handle ControlNet / T2I_Adapter
            for param in outer.control_params:
                if param.guidance_stopped:
                    continue

                if param.control_model_type not in [ControlModelType.ControlNet, ControlModelType.T2I_Adapter]:
                    continue

                param.control_model.to(devices.get_device_for("controlnet"))
                # inpaint model workaround
                x_in = x
                control_model = param.control_model.control_model

                if param.control_model_type == ControlModelType.ControlNet:
                    if x.shape[1] != control_model.input_blocks[0][0].in_channels and x.shape[1] == 9:
                        # inpaint_model: 4 data + 4 downscaled image + 1 mask
                        x_in = x[:, :4, ...]
                        require_inpaint_hijack = True

                assert param.used_hint_cond is not None, f"Controlnet is enabled but no input image is given"
                control = param.control_model(x=x_in, hint=param.used_hint_cond, timesteps=timesteps, context=context)
                control_scales = ([param.weight] * 13)
                
                if outer.lowvram:
                    param.control_model.to("cpu")

                if param.cfg_injection or param.global_average_pooling:
                    query_size = int(x.shape[0])
                    if param.control_model_type == ControlModelType.T2I_Adapter:
                        control = [torch.cat([c.clone() for _ in range(query_size)], dim=0) for c in control]
                    uc_mask = param.generate_uc_mask(query_size, dtype=x.dtype, device=x.device)[:, None, None, None]
                    control = [c * uc_mask for c in control]

                if param.soft_injection or is_in_high_res_fix:
                    # important! use the soft weights with high-res fix can significantly reduce artifacts.
                    if param.control_model_type == ControlModelType.T2I_Adapter:
                        control_scales = [param.weight * x for x in (0.25, 0.62, 0.825, 1.0)]
                    elif param.control_model_type == ControlModelType.ControlNet:
                        control_scales = [param.weight * (0.825 ** float(12 - i)) for i in range(13)]

                if param.advanced_weighting is not None:
                    control_scales = param.advanced_weighting
                    
                control = [c * scale for c, scale in zip(control, control_scales)]
                if param.global_average_pooling:
                    control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
                    
                for idx, item in enumerate(control):
                    target = None
                    if param.control_model_type == ControlModelType.ControlNet:
                        target = total_controlnet_embedding
                    if param.control_model_type == ControlModelType.T2I_Adapter:
                        target = total_t2i_adapter_embedding
                    if target is not None:
                        target[idx] = item + target[idx]
                        
            assert timesteps is not None, ValueError(f"insufficient timestep: {timesteps}")
            hs = []
            with th.no_grad():
                t_emb = cond_cast_unet(timestep_embedding(timesteps, self.model_channels, repeat_only=False))
                emb = self.time_embed(t_emb)
                h = x.type(self.dtype)
                for i, module in enumerate(self.input_blocks):
                    h = module(h, emb, context)
                    
                    # t2i-adatper, same as openaimodel.py:744
                    if (i+1) % 3 == 0:
                        h = cfg_based_adder(h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack)

                    hs.append(h)
                h = self.middle_block(h, emb, context)

            h = cfg_based_adder(h, total_controlnet_embedding.pop(), require_inpaint_hijack)

            for i, module in enumerate(self.output_blocks):
                h = th.cat([h, cfg_based_adder(hs.pop(), total_controlnet_embedding.pop(), require_inpaint_hijack)], dim=1)
                h = module(h, emb, context)

            h = h.type(x.dtype)
            return self.out(h)

        def forward_webui(*args, **kwargs):
            # webui will handle other compoments 
            try:
                if shared.cmd_opts.lowvram:
                    lowvram.send_everything_to_cpu()
                                            
                return forward(*args, **kwargs)
            finally:
                if self.lowvram:
                    for param in self.control_params:
                        if param.control_model is not None:
                            param.control_model.to("cpu")

        model._original_forward = model.forward
        model.forward = forward_webui.__get__(model, UNetModel)
        scripts.script_callbacks.on_cfg_denoiser(self.guidance_schedule_handler)
    
    def notify(self, params, is_vanilla_samplers):
        self.is_vanilla_samplers = is_vanilla_samplers
        self.control_params = params

    def restore(self, model):
        scripts.script_callbacks.remove_callbacks_for_function(self.guidance_schedule_handler)
        if hasattr(self, "control_params"):
            del self.control_params
        
        if not hasattr(model, "_original_forward"):
            # no such handle, ignore
            return
        
        model.forward = model._original_forward
        del model._original_forward
