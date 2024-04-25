from .processor import *  # noqa
import functools

legacy_preprocessors = {
    # "none": {
    #     "label": "none",
    #     "call_function": lambda x, *args, **kwargs: (x, True),
    #     "unload_function": None,
    #     "managed_model": None,
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": None,
    #     "slider_1": None,
    #     "slider_2": None,
    #     "slider_3": None,
    #     "priority": 100,
    #     "tags": []
    # },
    # "invert": {
    #     "label": "invert (from white bg & black line)",
    #     "call_function": invert,
    #     "unload_function": None,
    #     "managed_model": None,
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": None,
    #     "slider_1": None,
    #     "slider_2": None,
    #     "slider_3": None,
    #     "priority": 20,
    #     "tags": [
    #         "Canny", "Lineart", "Scribble", "MLSD",
    #     ]
    # },
    "animal_openpose": {
        "label": "animal_openpose",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=False, include_face=False, use_animal_pose=True),
        "unload_function": g_openpose_model.unload,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    # "blur_gaussian": {
    #     "label": "blur_gaussian",
    #     "call_function": blur_gaussian,
    #     "unload_function": None,
    #     "managed_model": None,
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": {
    #         "label": "Resolution",
    #         "value": 512,
    #         "minimum": 64,
    #         "maximum": 2048
    #     },
    #     "slider_1": {
    #         "label": "Sigma",
    #         "minimum": 0.01,
    #         "maximum": 64.0,
    #         "value": 9.0
    #     },
    #     "slider_2": None,
    #     "slider_3": None,
    #     "priority": 0,
    #     "tags": [
    #         "Tile",
    #     ]
    # },
    # "canny": {
    #     "label": "canny",
    #     "call_function": canny,
    #     "unload_function": None,
    #     "managed_model": "model_canny",
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": {
    #         "label": "Resolution",
    #         "value": 512,
    #         "minimum": 64,
    #         "maximum": 2048
    #     },
    #     "slider_1": {
    #         "label": "Canny Low Threshold",
    #         "value": 100,
    #         "minimum": 1,
    #         "maximum": 255
    #     },
    #     "slider_2": {
    #         "label": "Canny High Threshold",
    #         "value": 200,
    #         "minimum": 1,
    #         "maximum": 255
    #     },
    #     "slider_3": None,
    #     "priority": 100,
    #     "tags": [
    #         "Canny"
    #     ]
    # },
    "densepose": {
        "label": "densepose (pruple bg & purple torso)",
        "call_function": functools.partial(densepose, cmap="viridis"),
        "unload_function": unload_densepose,
        "managed_model": "unknown",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "densepose_parula": {
        "label": "densepose_parula (black bg & blue torso)",
        "call_function": functools.partial(densepose, cmap="parula"),
        "unload_function": unload_densepose,
        "managed_model": "unknown",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "depth_anything": {
        "label": "depth_anything",
        "call_function": functools.partial(depth_anything, colored=False),
        "unload_function": unload_depth_anything,
        "managed_model": "model_depth_anything",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Depth"
        ]
    },
    "depth_hand_refiner": {
        "label": "depth_hand_refiner",
        "call_function": g_hand_refiner_model.run_model,
        "unload_function": g_hand_refiner_model.unload,
        "managed_model": "g_hand_refiner_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "value": 512,
            "minimum": 64,
            "maximum": 2048
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Depth"
        ]
    },
    "depth_leres": {
        "label": "depth_leres",
        "call_function": functools.partial(leres, boost=False),
        "unload_function": unload_leres,
        "managed_model": "model_leres",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": {
            "label": "Remove Near %",
            "minimum": 0,
            "maximum": 100,
            "value": 0,
            "step": 0.1
        },
        "slider_2": {
            "label": "Remove Background %",
            "minimum": 0,
            "maximum": 100,
            "value": 0,
            "step": 0.1
        },
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Depth"
        ]
    },
    "depth_leres++": {
        "label": "depth_leres++",
        "call_function": functools.partial(leres, boost=True),
        "unload_function": unload_leres,
        "managed_model": "model_leres",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": {
            "label": "Remove Near %",
            "minimum": 0,
            "maximum": 100,
            "value": 0,
            "step": 0.1
        },
        "slider_2": {
            "label": "Remove Background %",
            "minimum": 0,
            "maximum": 100,
            "value": 0,
            "step": 0.1
        },
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Depth"
        ]
    },
    "depth": {
        "label": "depth_midas",
        "call_function": midas,
        "unload_function": unload_midas,
        "managed_model": "model_midas",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Depth"
        ]
    },
    "depth_zoe": {
        "label": "depth_zoe",
        "call_function": zoe_depth,
        "unload_function": unload_zoe_depth,
        "managed_model": "model_zoe_depth",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Depth"
        ]
    },
    "dw_openpose_full": {
        "label": "dw_openpose_full",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=True, include_face=True, use_dw_pose=True),
        "unload_function": g_openpose_model.unload,
        "managed_model": 'g_openpose_model',
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    # "inpaint": {
    #     "label": "inpaint_global_harmonious",
    #     "call_function": identity,
    #     "unload_function": None,
    #     "managed_model": None,
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": None,
    #     "slider_1": None,
    #     "slider_2": None,
    #     "slider_3": None,
    #     "priority": 0,
    #     "tags": [
    #         "Inpaint"
    #     ]
    # },
    # "inpaint_only": {
    #     "label": "inpaint_only",
    #     "call_function": identity,
    #     "unload_function": None,
    #     "managed_model": None,
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": None,
    #     "slider_1": None,
    #     "slider_2": None,
    #     "slider_3": None,
    #     "priority": 100,
    #     "tags": [
    #         "Inpaint"
    #     ]
    # },
    # "inpaint_only+lama": {
    #     "label": "inpaint_only+lama",
    #     "call_function": lama_inpaint,
    #     "unload_function": unload_lama_inpaint,
    #     "managed_model": "model_lama",
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": None,
    #     "slider_1": None,
    #     "slider_2": None,
    #     "slider_3": None,
    #     "priority": 0,
    #     "tags": [
    #         "Inpaint"
    #     ]
    # },
    "instant_id_face_embedding": {
        "label": "instant_id_face_embedding",
        "call_function": functools.partial(g_insight_face_instant_id_model.run_model_instant_id, return_keypoints=False),
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Instant-ID"
        ],
        "returns_image": False,
    },
    "instant_id_face_keypoints": {
        "label": "instant_id_face_keypoints",
        "call_function": functools.partial(g_insight_face_instant_id_model.run_model_instant_id, return_keypoints=True),
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Instant-ID"
        ]
    },
    "ip-adapter_clip_sd15": {
        "label": "ip-adapter_clip_h",
        "call_function": functools.partial(clip, config='clip_h'),
        "unload_function": functools.partial(unload_clip, config='clip_h'),
        "managed_model": "unknown",
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "accepts_mask": True,  # CLIP mask
        "tags": [
            "IP-Adapter"
        ],
        "returns_image": False,
    },
    "ip-adapter_clip_sdxl": {
        "label": "ip-adapter_clip_g",
        "call_function": functools.partial(clip, config='clip_g'),
        "unload_function": functools.partial(unload_clip, config='clip_g'),
        "managed_model": "unknown",
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "accepts_mask": True,  # CLIP mask
        "tags": [
            "IP-Adapter"
        ],
        "returns_image": False,
    },
    "ip-adapter_clip_sdxl_plus_vith": {
        "label": "ip-adapter_clip_sdxl_plus_vith",
        "call_function": functools.partial(clip, config='clip_h'),
        "unload_function": functools.partial(unload_clip, config='clip_h'),
        "managed_model": "unknown",
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "accepts_mask": True,  # CLIP mask
        "tags": [
            "IP-Adapter"
        ],
        "returns_image": False,
    },
    "ip-adapter_face_id": {
        "label": "ip-adapter_face_id",
        "call_function": g_insight_face_model.run_model,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "IP-Adapter"
        ],
        "returns_image": False,
    },
    "ip-adapter_face_id_plus": {
        "label": "ip-adapter_face_id_plus",
        "call_function": face_id_plus,
        "unload_function": functools.partial(unload_clip, config='clip_h'),
        "managed_model": "unknown",
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "IP-Adapter"
        ],
        "returns_image": False,
    },
    "lineart_anime": {
        "label": "lineart_anime",
        "call_function": lineart_anime,
        "unload_function": unload_lineart_anime,
        "managed_model": "model_lineart_anime",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Lineart"
        ]
    },
    "lineart_anime_denoise": {
        "label": "lineart_anime_denoise",
        "call_function": lineart_anime_denoise,
        "unload_function": unload_lineart_anime_denoise,
        "managed_model": "model_manga_line",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Lineart"
        ]
    },
    "lineart_coarse": {
        "label": "lineart_coarse",
        "call_function": lineart_coarse,
        "unload_function": unload_lineart_coarse,
        "managed_model": "model_lineart_coarse",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Lineart"
        ]
    },
    "lineart": {
        "label": "lineart_realistic",
        "call_function": lineart,
        "unload_function": unload_lineart,
        "managed_model": "model_lineart",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Lineart"
        ]
    },
    "lineart_standard": {
        "label": "lineart_standard (from white bg & black line)",
        "call_function": lineart_standard,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Lineart"
        ]
    },
    "mediapipe_face": {
        "label": "mediapipe_face",
        "call_function": mediapipe_face,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "value": 512,
            "minimum": 64,
            "maximum": 2048
        },
        "slider_1": {
            "label": "Max Faces",
            "value": 1,
            "minimum": 1,
            "maximum": 10,
            "step": 1
        },
        "slider_2": {
            "label": "Min Face Confidence",
            "value": 0.5,
            "minimum": 0.01,
            "maximum": 1.0,
            "step": 0.01
        },
        "slider_3": None,
        "priority": 0,
        "tags": []
    },
    "mlsd": {
        "label": "mlsd",
        "call_function": mlsd,
        "unload_function": unload_mlsd,
        "managed_model": "model_mlsd",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": {
            "label": "MLSD Value Threshold",
            "minimum": 0.01,
            "maximum": 2.0,
            "value": 0.1,
            "step": 0.01
        },
        "slider_2": {
            "label": "MLSD Distance Threshold",
            "minimum": 0.01,
            "maximum": 20.0,
            "value": 0.1,
            "step": 0.01
        },
        "slider_3": None,
        "priority": 100,
        "tags": [
            "MLSD"
        ],
        "use_soft_projection_in_hr_fix": True
    },
    "normal_bae": {
        "label": "normal_bae",
        "call_function": normal_bae,
        "unload_function": unload_normal_bae,
        "managed_model": "model_normal_bae",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "NormalMap"
        ]
    },
    "normal_map": {
        "label": "normal_midas",
        "call_function": midas_normal,
        "unload_function": unload_midas,
        "managed_model": "model_midas",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": {
            "label": "Normal Background Threshold",
            "minimum": 0.0,
            "maximum": 1.0,
            "value": 0.4,
            "step": 0.01
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "NormalMap"
        ]
    },
    "openpose": {
        "label": "openpose",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=False, include_face=False),
        "unload_function": g_openpose_model.unload,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "openpose_face": {
        "label": "openpose_face",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=False, include_face=True),
        "unload_function": g_openpose_model.unload,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "openpose_faceonly": {
        "label": "openpose_faceonly",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=False, include_hand=False, include_face=True),
        "unload_function": g_openpose_model.unload,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "openpose_full": {
        "label": "openpose_full",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=True, include_face=True),
        "unload_function": g_openpose_model.unload,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "OpenPose"
        ]
    },
    "openpose_hand": {
        "label": "openpose_hand",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=True, include_face=False),
        "unload_function": g_openpose_model.unload,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "recolor_intensity": {
        "label": "recolor_intensity",
        "call_function": recolor_intensity,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "label": "Gamma Correction",
            "value": 1.0,
            "minimum": 0.1,
            "maximum": 2.0,
            "step": 0.001
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Recolor"
        ]
    },
    "recolor_luminance": {
        "label": "recolor_luminance",
        "call_function": recolor_luminance,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "label": "Gamma Correction",
            "value": 1.0,
            "minimum": 0.1,
            "maximum": 2.0,
            "step": 0.001
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Recolor"
        ]
    },
    "reference_adain": {
        "label": "reference_adain",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": True,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "label": "Style Fidelity (only for Balanced mode)",
            "value": 0.5,
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Reference"
        ]
    },
    "reference_adain+attn": {
        "label": "reference_adain+attn",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": True,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "label": "Style Fidelity (only for Balanced mode)",
            "value": 0.5,
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Reference"
        ]
    },
    "reference_only": {
        "label": "reference_only",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": True,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "label": "Style Fidelity (only for Balanced mode)",
            "value": 0.5,
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Reference"
        ]
    },
    "revision_clipvision": {
        "label": "revision_clipvision",
        "call_function": functools.partial(clip, config='clip_g'),
        "unload_function": functools.partial(unload_clip, config='clip_g'),
        "managed_model": None,
        "model_free": True,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": {
            "label": "Noise Augmentation",
            "value": 0.0,
            "minimum": 0.0,
            "maximum": 1.0
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Revision"
        ],
        "returns_image": False,
    },
    "revision_ignore_prompt": {
        "label": "revision_ignore_prompt",
        "call_function": functools.partial(clip, config='clip_g'),
        "unload_function": functools.partial(unload_clip, config='clip_g'),
        "managed_model": None,
        "model_free": True,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": {
            "label": "Noise Augmentation",
            "value": 0.0,
            "minimum": 0.0,
            "maximum": 1.0
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Revision"
        ],
        "returns_image": False,
    },
    "scribble_hed": {
        "label": "scribble_hed",
        "call_function": scribble_hed,
        "unload_function": unload_hed,
        "managed_model": "model_hed",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Scribble", "SparseCtrl",
        ]
    },
    "pidinet_scribble": {
        "label": "scribble_pidinet",
        "call_function": scribble_pidinet,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Scribble", "SparseCtrl",
        ]
    },
    # "scribble_xdog": {
    #     "label": "scribble_xdog",
    #     "call_function": scribble_xdog,
    #     "unload_function": None,
    #     "managed_model": None,
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": {
    #         "label": "Resolution",
    #         "value": 512,
    #         "minimum": 64,
    #         "maximum": 2048
    #     },
    #     "slider_1": {
    #         "label": "XDoG Threshold",
    #         "minimum": 1,
    #         "maximum": 64,
    #         "value": 32
    #     },
    #     "slider_2": None,
    #     "slider_3": None,
    #     "priority": 0,
    #     "tags": [
    #         "Scribble",
    #     ]
    # },
    "anime_face_segment": {
        "label": "seg_anime_face",
        "call_function": anime_face_segment,
        "unload_function": unload_anime_face_segment,
        "managed_model": "model_anime_face_segment",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "value": 512,
            "minimum": 64,
            "maximum": 2048
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Segmentation"
        ]
    },
    "oneformer_ade20k": {
        "label": "seg_ofade20k",
        "call_function": oneformer_ade20k,
        "unload_function": unload_oneformer_ade20k,
        "managed_model": "model_oneformer_ade20k",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Segmentation"
        ]
    },
    "oneformer_coco": {
        "label": "seg_ofcoco",
        "call_function": oneformer_coco,
        "unload_function": unload_oneformer_coco,
        "managed_model": "model_oneformer_coco",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Segmentation"
        ]
    },
    "segmentation": {
        "label": "seg_ufade20k",
        "call_function": uniformer,
        "unload_function": unload_uniformer,
        "managed_model": "model_uniformer",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Segmentation"
        ]
    },
    # "shuffle": {
    #     "label": "shuffle",
    #     "call_function": shuffle,
    #     "unload_function": None,
    #     "managed_model": None,
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": None,
    #     "slider_1": None,
    #     "slider_2": None,
    #     "slider_3": None,
    #     "priority": 100,
    #     "tags": [
    #         "Shuffle"
    #     ]
    # },
    "hed": {
        "label": "softedge_hed",
        "call_function": hed,
        "unload_function": unload_hed,
        "managed_model": "model_hed",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "SoftEdge"
        ]
    },
    "hed_safe": {
        "label": "softedge_hedsafe",
        "call_function": hed_safe,
        "unload_function": unload_hed,
        "managed_model": "model_hed",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "minimum": 64,
            "maximum": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "SoftEdge"
        ]
    },
    "pidinet": {
        "label": "softedge_pidinet",
        "call_function": pidinet,
        "unload_function": unload_pidinet,
        "managed_model": "model_pidinet",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "SoftEdge"
        ]
    },
    "pidinet_safe": {
        "label": "softedge_pidisafe",
        "call_function": pidinet_safe,
        "unload_function": unload_pidinet,
        "managed_model": "model_pidinet",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "SoftEdge"
        ]
    },
    "te_hed": {
        "label": "softedge_teed",
        "call_function": te_hed,
        "unload_function": unload_te_hed,
        "managed_model": "model_te_hed",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "value": 512,
            "minimum": 64,
            "maximum": 2048
        },
        "slider_1": {
            "label": "Safe Steps",
            "minimum": 0,
            "maximum": 10,
            "value": 2,
            "step": 1
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "SoftEdge"
        ]
    },
    "color": {
        "label": "t2ia_color_grid",
        "call_function": color,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "value": 512,
            "minimum": 64,
            "maximum": 2048
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "T2I-Adapter"
        ]
    },
    "pidinet_sketch": {
        "label": "t2ia_sketch_pidi",
        "call_function": pidinet_ts,
        "unload_function": unload_pidinet,
        "managed_model": "model_pidinet",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "T2I-Adapter"
        ]
    },
    "clip_vision": {
        "label": "t2ia_style_clipvision",
        "call_function": functools.partial(clip, config='clip_vitl'),
        "unload_function": functools.partial(unload_clip, config='clip_vitl'),
        "managed_model": "unknown",
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "T2I-Adapter"
        ],
        "returns_image": False,
    },
    "threshold": {
        "label": "threshold",
        "call_function": threshold,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "label": "Resolution",
            "value": 512,
            "minimum": 64,
            "maximum": 2048
        },
        "slider_1": {
            "label": "Binarization Threshold",
            "minimum": 0,
            "maximum": 255,
            "value": 127
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": []
    },
    "tile_colorfix": {
        "label": "tile_colorfix",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "label": "Variation",
            "value": 8.0,
            "minimum": 3.0,
            "maximum": 32.0,
            "step": 1.0
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Tile",
        ]
    },
    "tile_colorfix+sharp": {
        "label": "tile_colorfix+sharp",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "label": "Variation",
            "value": 8.0,
            "minimum": 3.0,
            "maximum": 32.0,
            "step": 1.0
        },
        "slider_2": {
            "label": "Sharpness",
            "value": 1.0,
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.01
        },
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Tile",
        ]
    },
    "tile_resample": {
        "label": "tile_resample",
        "call_function": tile_resample,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "label": "Down Sampling Rate",
            "value": 1.0,
            "minimum": 1.0,
            "maximum": 8.0,
            "step": 0.01
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Tile",
        ]
    }
}
