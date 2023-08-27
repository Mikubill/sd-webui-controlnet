import torch
import os


# attention_channels of input, output, middle
SD_V12_CHANNELS = [320] * 4 + [640] * 4 + [1280] * 4 + [1280] * 6 + [640] * 6 + [320] * 6 + [1280] * 2
SD_XL_CHANNELS = [640] * 8 + [1280] * 40 + [1280] * 60 + [640] * 12 + [1280] * 20


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens,
                                                              self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


# Cross Attention to_k, to_v for IPAdapter
class To_KV(torch.nn.Module):
    def __init__(self, cross_attention_dim):
        super().__init__()

        channels = SD_XL_CHANNELS if cross_attention_dim == 2048 else SD_V12_CHANNELS
        self.to_kvs = torch.nn.ModuleList(
            [torch.nn.Linear(cross_attention_dim, channel, bias=False) for channel in channels])

    def load_state_dict(self, state_dict):
        # input -> output -> middle
        for i, key in enumerate(state_dict.keys()):
            self.to_kvs[i].weight.data = state_dict[key]


class IPAdapterModel(torch.nn.Module):
    def __init__(self, state_dict, clip_embeddings_dim):
        super().__init__()
        self.device = "cpu"

        # cross_attention_dim is equal to text_encoder output
        self.cross_attention_dim = state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]

        # number of tokens of ip_adapter embedding
        self.clip_extra_context_tokens = state_dict["image_proj"]["proj.weight"].shape[0] // self.cross_attention_dim

        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens
        )

        self.load_ip_adapter(state_dict)

    def load_ip_adapter(self, state_dict):
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = To_KV(self.cross_attention_dim)
        self.ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, clip_image_embeds):
        '''
        clip_image_embeds: clip_vision_output, size: batch_size, 1024(sdv1-2) or 1280(sdxl)
        return: cond, uncond for CFG, size: batch_size, num_tokens, 768(sdv1) or 1024(sdv2) or 2048(sdxl)
        '''

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        # input zero vector for unconditional.
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds


class PlugableIPAdapter(torch.nn.Module):
    def __init__(self, state_dict, clip_embeddings_dim):
        super().__init__()
        self.sdxl = clip_embeddings_dim == 1280
        self.ipadapter = IPAdapterModel(state_dict, clip_embeddings_dim=clip_embeddings_dim)
        self.control_model = self.ipadapter
        return

    def reset(self):
        return

    def hook(self, model, clip_vision_output, weight, dtype):
        return


class IPAdapter:
    def adapter(self, model, clip_vision_output, weight, model_name, dtype):
        self.dtype = torch.float32 if dtype == "fp32" else torch.float16
        device = "cuda"
        self.weight = weight  # ip_adapter scale

        clip_vision_emb = clip_vision_output.image_embeds.to(device, dtype=self.dtype)
        clip_embeddings_dim = clip_vision_emb.shape[1]

        # sd_v1-2: 1024, sd_xl: 1280 at the moment...
        self.sdxl = clip_embeddings_dim == 1280

        self.ipadapter = IPAdapterModel(
            os.path.join(CURRENT_DIR, os.path.join(CURRENT_DIR, "models", model_name)),
            clip_embeddings_dim=clip_embeddings_dim
        )

        self.ipadapter.to(device, dtype=self.dtype)

        self.image_emb, self.uncond_image_emb = self.ipadapter.get_image_embeds(clip_vision_emb)
        self.image_emb = self.image_emb.to(device, dtype=self.dtype)
        self.uncond_image_emb = self.uncond_image_emb.to(device, dtype=self.dtype)
        # Not sure of batch size at this point.
        self.cond_uncond_image_emb = None

        new_model = model.clone()

        '''
        patch_name of sdv1-2: ("input" or "output" or "middle", block_id)
        patch_name of sdxl: ("input" or "output" or "middle", block_id, transformer_index)
        '''
        if not self.sdxl:
            number = 0  # index of to_kvs
            for id in [1, 2, 4, 5, 7, 8]:  # id of input_blocks that have cross attention
                new_model.set_model_attn2_replace(self.patch_forward(number), "input", id)
                number += 1
            for id in [3, 4, 5, 6, 7, 8, 9, 10, 11]:  # id of output_blocks that have cross attention
                new_model.set_model_attn2_replace(self.patch_forward(number), "output", id)
                number += 1
            new_model.set_model_attn2_replace(self.patch_forward(number), "middle", 0)
        else:
            number = 0
            for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.patch_forward(number), "input", id, index)
                    number += 1
            for id in range(6):  # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.patch_forward(number), "output", id, index)
                    number += 1
            for index in range(10):
                set_model_patch_replace(new_model, self.patch_forward(number), "middle", 0, index)
                number += 1

        return (new_model,)

    # forward for patching
    def patch_forward(self, number):
        def forward(n, context_attn2, value_attn2, extra_options):
            org_dtype = n.dtype
            with torch.autocast("cuda", dtype=self.dtype):
                q = n
                k = context_attn2
                v = value_attn2
                b, _, _ = q.shape

                if self.cond_uncond_image_emb is None or self.cond_uncond_image_emb.shape[0] != b:
                    self.cond_uncond_image_emb = torch.cat(
                        [self.uncond_image_emb.repeat(b // 2, 1, 1), self.image_emb.repeat(b // 2, 1, 1)], dim=0)

                # k, v for ip_adapter
                ip_k = self.ipadapter.ip_layers.to_kvs[number * 2](self.cond_uncond_image_emb)
                ip_v = self.ipadapter.ip_layers.to_kvs[number * 2 + 1](self.cond_uncond_image_emb)

                q, k, v, ip_k, ip_v = map(
                    lambda t: t.view(b, -1, extra_options["n_heads"], extra_options["dim_head"]).transpose(1, 2),
                    (q, k, v, ip_k, ip_v),
                )

                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                                                       is_causal=False)
                out = out.transpose(1, 2).reshape(b, -1, extra_options["n_heads"] * extra_options["dim_head"])

                # output of ip_adapter
                ip_out = torch.nn.functional.scaled_dot_product_attention(q, ip_k, ip_v, attn_mask=None, dropout_p=0.0,
                                                                          is_causal=False)
                ip_out = ip_out.transpose(1, 2).reshape(b, -1, extra_options["n_heads"] * extra_options["dim_head"])

                out = out + ip_out * self.weight

            return out.to(dtype=org_dtype)

        return forward