import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union
import math
from typing import Callable, Dict, List, Optional, Tuple, Union
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction


def always_round(x):
    intx = int(x)
    is_even = intx%2 == 0
    if is_even:
        if x < intx + 0.5:
            return intx
        return intx + 1
    else:
        return round(x)


def _img_importance_flatten(img: torch.tensor, w: int, h: int) -> torch.tensor:
    return F.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        # scale_factor=1 / ratio,
        size=(w, h),
        mode="bilinear",
        align_corners=True,
    ).squeeze()


def _image_context_seperator(
    img: Image.Image, color_context: dict, _tokenizer
) -> List[Tuple[List[int], torch.Tensor]]:

    ret_lists = []

    if img is not None:
        w, h = img.size
        # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        # img = img.resize((w, h), resample=Image.LANCZOS)

        for color, v in color_context.items():
            f = v.split(",")[-1]
            v = ",".join(v.split(",")[:-1])
            f = float(f)
            v_input = _tokenizer(
                v,
                max_length=_tokenizer.model_max_length,
                truncation=True,
            )
            v_as_tokens = v_input["input_ids"][1:-1]
            if isinstance(color, str):
                r, g, b = color[1:3], color[3:5], color[5:7]
                color = (int(r, 16), int(g, 16), int(b, 16))

            img_where_color = (np.array(img) == color).all(axis=-1)

            if not img_where_color.sum() > 0:
                print(f"Warning : not a single color {color} not found in image")

            img_where_color = torch.tensor(img_where_color, dtype=torch.float32) * f

            ret_lists.append((v_as_tokens, img_where_color))
    else:
        w, h = 512, 512

    if len(ret_lists) == 0:
        ret_lists.append(([-1], torch.zeros((w, h), dtype=torch.float32)))
    return ret_lists, w, h


def _tokens_img_attention_weight(
    img_context_seperated, tokenized_texts, ratio: int = 8, original_shape=False
):
    
    token_lis = tokenized_texts["input_ids"][0].tolist()
    w, h = img_context_seperated[0][1].shape

    w_r, h_r = always_round(w/ratio), always_round(h/ratio)
    ret_tensor = torch.zeros((w_r * h_r, len(token_lis)), dtype=torch.float32)
    
    for v_as_tokens, img_where_color in img_context_seperated:
        is_in = 0
        for idx, tok in enumerate(token_lis):
            if token_lis[idx : idx + len(v_as_tokens)] == v_as_tokens:
                is_in = 1

                # print(token_lis[idx : idx + len(v_as_tokens)], v_as_tokens)
                ret_tensor[:, idx : idx + len(v_as_tokens)] += (
                    _img_importance_flatten(img_where_color, w_r, h_r)
                    .reshape(-1, 1)
                    .repeat(1, len(v_as_tokens))
                )

        if not is_in == 1:
            print(f"Warning ratio {ratio} : tokens {v_as_tokens} not found in text")

    if original_shape:
        ret_tensor = ret_tensor.reshape((w_r, h_r, len(token_lis)))

    return ret_tensor


def _extract_seed_and_sigma_from_context(color_context, ignore_seed = -1):
    # Split seed and sigma from color_context if provided
    extra_seeds = {}
    extra_sigmas = {}
    for i, (k, _context) in enumerate(color_context.items()):
        _context_split = _context.split(',')
        if len(_context_split) > 2:
            try:
                seed = int(_context_split[-2])
                sigma = float(_context_split[-1])
                _context_split = _context_split[:-2]
                extra_sigmas[i] = sigma
            except ValueError:
                seed = int(_context_split[-1])
                _context_split = _context_split[:-1]
            if seed != ignore_seed:
                extra_seeds[i] = seed
        color_context[k] = ','.join(_context_split)
    return color_context, extra_seeds, extra_sigmas


def _encode_text_color_inputs(
        cond_embeddings, uncond_embeddings, 
        tokenizer, device, 
        color_map_image, color_context, 
        input_prompt, unconditional_input_prompt):
    # Process input prompt text
    text_input = tokenizer(
        [input_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    # Extract seed and sigma from color context
    color_context, extra_seeds, extra_sigmas = _extract_seed_and_sigma_from_context(color_context)
    is_extra_sigma = len(extra_sigmas) > 0
    
    # Process color map image and context
    seperated_word_contexts, width, height = _image_context_seperator(
        color_map_image, color_context, tokenizer
    )
    
    # Compute cross-attention weights
    cross_attention_weight_1 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=1, original_shape=True
    ).to(device)
    cross_attention_weight_8 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=8
    ).to(device)
    cross_attention_weight_16 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=16
    ).to(device)
    cross_attention_weight_32 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=32
    ).to(device)
    cross_attention_weight_64 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=64
    ).to(device)

    # Compute conditional and unconditional embeddings
    # cond_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [unconditional_input_prompt],
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    # uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    encoder_hidden_states = {
        # "CONTEXT_TENSOR": cond_embeddings,
        f"CROSS_ATTENTION_WEIGHT_ORIG": cross_attention_weight_1,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height/8)*always_round(width/8)}": cross_attention_weight_8,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height/16)*always_round(width/16)}": cross_attention_weight_16,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height/32)*always_round(width/32)}": cross_attention_weight_32,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height/64)*always_round(width/64)}": cross_attention_weight_64,
    }

    uncond_encoder_hidden_states = {
        # "CONTEXT_TENSOR": uncond_embeddings,
        f"CROSS_ATTENTION_WEIGHT_ORIG": 0,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height/8)*always_round(width/8)}": 0,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height/16)*always_round(width/16)}": 0,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height/32)*always_round(width/32)}": 0,
        f"CROSS_ATTENTION_WEIGHT_{always_round(height/64)*always_round(width/64)}": 0,
    }

    return extra_seeds, seperated_word_contexts, encoder_hidden_states, uncond_encoder_hidden_states


def encode_text_color_inputs(p, color_map_image, color_context):
    c = p.sd_model.get_learned_conditioning(p.prompt)
    # uc = p.sd_model.get_learned_conditioning(p.negative_prompt)
    uc = None
    _, _, encoder_hidden_states, _ = \
        _encode_text_color_inputs(c, uc, 
            p.sd_model.cond_stage_model.wrapped.tokenizer, 
            p.sd_model.device, color_map_image, color_context, p.prompt, p.negative_prompt)
    return encoder_hidden_states



def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def hijack_forward(self, 
        x: torch.Tensor, 
        context: Union[torch.Tensor, dict] = None, 
        mask=None):
    """hijack forward of CrossAttention in ldm module.
    """
    is_dict_format = isinstance(context, dict)
    if is_dict_format:
        context_dict = context
        context = context["CONTEXT_TENSOR"]
    
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k)

    attention_size_of_img = sim.shape[-2]
    cross_attention_weight = 0.0
    not_self_attn = sim.shape[1] != sim.shape[2]
    if is_dict_format and not_self_attn:
        f: Callable = context_dict["WEIGHT_FUNCTION"]
        try:
            w = context_dict[f"CROSS_ATTENTION_WEIGHT_{attention_size_of_img}"]
        except KeyError:
            w = context_dict[f"CROSS_ATTENTION_WEIGHT_ORIG"]
            if not isinstance(w, int):
                img_h, img_w, nc = w.shape
                ratio = math.sqrt(img_h * img_w / attention_size_of_img)
                w = F.interpolate(w.permute(2, 0, 1).unsqueeze(0), scale_factor=1/ratio, mode="bilinear", align_corners=True)
                w = F.interpolate(w.reshape(1, nc, -1), size=(attention_size_of_img,), mode='nearest').permute(2, 1, 0).squeeze()
            else:
                w = 0
        sigma = context_dict["SIGMA"]

        cross_attention_weight = torch.zeros_like(sim)
        cross_attn = f(w, sigma, sim)
        cross_attention_weight[0,:,:cross_attn.shape[-1]] = cross_attn
    else:
        cross_attention_weight = 0.0
    
    sim = (sim + cross_attention_weight) * self.scale

    del q, k

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


def hijack_CrossAttn(p):
    # 4. Hijack Cross Attention Module
    unet = p.sd_model.model.diffusion_model
    for _module in unet.modules():
        if _module.__class__.__name__ == "CrossAttention":
            _module.__class__.__call__ = hijack_forward

