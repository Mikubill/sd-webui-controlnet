import os
import copy

from modules import shared
from scripts import global_state
from scripts.cldm import PlugableControlModel
from scripts.adapter import PlugableAdapter, Adapter, StyleAdapter, Adapter_light
from scripts.logging import logger
from pathlib import Path


controlnet_default_config = {
    'in_channels': 4,
    'hint_channels': 3,
    'model_channels': 320,
    'attention_resolutions': [4, 2, 1],
    'num_res_blocks': 2,
    'channel_mult': [1, 2, 4, 4],
    'num_heads': 8,
    'use_spatial_transformer': True,
    'transformer_depth': 1,
    'context_dim': 768,
    'use_checkpoint': True,
    'legacy': False,
    'global_average_pooling': False
}

t2i_adapter_config = {
    'channels': [320, 640, 1280, 1280],
    'nums_rb': 2,
    'ksize': 1,
    'sk': True,
    'cin': 192,
    'use_conv': False
}

t2i_adapter_light_config = {
    'channels': [320, 640, 1280, 1280],
    'nums_rb': 4,
    'cin': 192,
}

t2i_adapter_style_config = {
    'width': 1024,
    'context_dim': 768,
    'num_head': 8,
    'n_layes': 3,
    'num_token': 8,
}


def build_model_by_guess(state_dict, unet, model_path):
    model_has_shuffle_in_filename = 'shuffle' in Path(os.path.abspath(model_path)).stem.lower()
    state_dict = {k.replace("control_model.", ""): v for k, v in state_dict.items()}

    network = None

    if 'input_hint_block.0.weight' in state_dict:
        config = copy.deepcopy(controlnet_default_config)
        config['global_average_pooling'] = model_has_shuffle_in_filename
        config['hint_channels'] = int(state_dict['input_hint_block.0.weight'].shape[1])
        config['context_dim'] = int(state_dict['input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight'].shape[1])

        if 'difference' in state_dict and unet is not None:
            unet_state_dict = unet.state_dict()
            unet_state_dict_keys = unet_state_dict.keys()
            final_state_dict = {}
            for key in state_dict.keys():
                p = state_dict[key]
                if key in unet_state_dict_keys:
                    p_new = p + unet_state_dict[key].clone().cpu()
                else:
                    p_new = p
                final_state_dict[key] = p_new
            state_dict = final_state_dict

        for key in state_dict.keys():
            p = state_dict[key]
            if 'proj_in.weight' in key or 'proj_out.weight' in key:
                if len(p.shape) == 2:
                    p = p[..., None, None]
            state_dict[key] = p

        network = PlugableControlModel(config, state_dict)

    if 'conv_in.weight' in state_dict:
        config = copy.deepcopy(t2i_adapter_config)
        config['cin'] = int(state_dict['conv_in.weight'].shape[1])
        adapter = Adapter(**config).cpu()
        adapter.load_state_dict(state_dict, strict=False)
        network = PlugableAdapter(adapter)

    if 'style_embedding' in state_dict:
        config = copy.deepcopy(t2i_adapter_style_config)
        adapter = StyleAdapter(**config).cpu()
        adapter.load_state_dict(state_dict, strict=False)
        network = PlugableAdapter(adapter)

    if 'body.0.in_conv.weight' in state_dict:
        config = copy.deepcopy(t2i_adapter_light_config)
        config['cin'] = int(state_dict['body.0.in_conv.weight'].shape[1])
        adapter = Adapter_light(**config).cpu()
        adapter.load_state_dict(state_dict, strict=False)
        network = PlugableAdapter(adapter)

    assert network is not None, '[ControlNet Error] Cannot recognize the ControlModel!'
    return network
