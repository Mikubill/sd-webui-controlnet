import pytest
import requests

from .template import APITestTemplate


expected_module_names = {
    "animal_openpose",
    "anime_face_segment",
    "blur_gaussian",
    "canny",
    "clip_vision",
    "color",
    "densepose",
    "densepose_parula",
    "depth",
    "depth_anything",
    "depth_hand_refiner",
    "depth_leres",
    "depth_leres++",
    "depth_zoe",
    "dw_openpose_full",
    "hed",
    "hed_safe",
    "inpaint",
    "inpaint_only",
    "inpaint_only+lama",
    "instant_id_face_embedding",
    "instant_id_face_keypoints",
    "invert",
    "ip-adapter-auto",
    "ip-adapter_clip_sd15",
    "ip-adapter_clip_sdxl",
    "ip-adapter_clip_sdxl_plus_vith",
    "ip-adapter_face_id",
    "ip-adapter_face_id_plus",
    "lineart",
    "lineart_anime",
    "lineart_anime_denoise",
    "lineart_coarse",
    "lineart_standard",
    "mediapipe_face",
    "mlsd",
    "none",
    "normal_bae",
    "normal_dsine",
    "normal_map",
    "oneformer_ade20k",
    "oneformer_coco",
    "openpose",
    "openpose_face",
    "openpose_faceonly",
    "openpose_full",
    "openpose_hand",
    "pidinet",
    "pidinet_safe",
    "pidinet_scribble",
    "pidinet_sketch",
    "recolor_intensity",
    "recolor_luminance",
    "reference_adain",
    "reference_adain+attn",
    "reference_only",
    "revision_clipvision",
    "revision_ignore_prompt",
    "scribble_hed",
    "scribble_xdog",
    "segmentation",
    "shuffle",
    "te_hed",
    "threshold",
    "tile_colorfix",
    "tile_colorfix+sharp",
    "tile_resample",
}

# Display name (label)
expected_module_alias = {
    "animal_openpose",
    "blur_gaussian",
    "canny",
    "densepose (pruple bg & purple torso)",
    "densepose_parula (black bg & blue torso)",
    "depth_anything",
    "depth_hand_refiner",
    "depth_leres",
    "depth_leres++",
    "depth_midas",
    "depth_zoe",
    "dw_openpose_full",
    "inpaint_global_harmonious",
    "inpaint_only",
    "inpaint_only+lama",
    "instant_id_face_embedding",
    "instant_id_face_keypoints",
    "invert (from white bg & black line)",
    "ip-adapter-auto",
    "ip-adapter_clip_g",
    "ip-adapter_clip_h",
    "ip-adapter_clip_sdxl_plus_vith",
    "ip-adapter_face_id",
    "ip-adapter_face_id_plus",
    "lineart_anime",
    "lineart_anime_denoise",
    "lineart_coarse",
    "lineart_realistic",
    "lineart_standard (from white bg & black line)",
    "mediapipe_face",
    "mlsd",
    "none",
    "normal_bae",
    "normal_dsine",
    "normal_midas",
    "openpose",
    "openpose_face",
    "openpose_faceonly",
    "openpose_full",
    "openpose_hand",
    "recolor_intensity",
    "recolor_luminance",
    "reference_adain",
    "reference_adain+attn",
    "reference_only",
    "revision_clipvision",
    "revision_ignore_prompt",
    "scribble_hed",
    "scribble_pidinet",
    "scribble_xdog",
    "seg_anime_face",
    "seg_ofade20k",
    "seg_ofcoco",
    "seg_ufade20k",
    "shuffle",
    "softedge_hed",
    "softedge_hedsafe",
    "softedge_pidinet",
    "softedge_pidisafe",
    "softedge_teed",
    "t2ia_color_grid",
    "t2ia_sketch_pidi",
    "t2ia_style_clipvision",
    "threshold",
    "tile_colorfix",
    "tile_colorfix+sharp",
    "tile_resample",
}


@pytest.mark.parametrize("alias", ("true", "false"))
def test_module_list(alias):
    json_resp = requests.get(
        APITestTemplate.BASE_URL + f"controlnet/module_list?alias_names={alias}"
    ).json()
    module_list = json_resp["module_list"]
    module_detail: dict = json_resp["module_detail"]
    expected_list = expected_module_alias if alias == "true" else expected_module_names
    assert set(module_list).issuperset(expected_list), expected_list - set(module_list)
    assert set(module_list) == set(module_detail.keys())
    assert module_detail["canny"] == dict(
        model_free=False,
        sliders=[
            {
                "name": "Resolution",
                "value": 512,
                "min": 64,
                "max": 2048,
                "step": 8,
            },
            {"name": "Low Threshold", "value": 100, "min": 1, "max": 255, "step": 1},
            {"name": "High Threshold", "value": 200, "min": 1, "max": 255, "step": 1},
        ],
    )


def test_control_types():
    json_resp = requests.get(
        APITestTemplate.BASE_URL + f"controlnet/control_types"
    ).json()
    assert "control_types" in json_resp
    actual_control_types = set(json_resp["control_types"].keys())
    expected_control_types = {
        "Canny",
        "Revision",
        "Inpaint",
        "Instant-ID",
        "InstructP2P",
        "Shuffle",
        "Scribble",
        "SparseCtrl",
        "MLSD",
        "OpenPose",
        "Depth",
        "All",
        "Lineart",
        "SoftEdge",
        "NormalMap",
        "IP-Adapter",
        "Reference",
        "T2I-Adapter",
        "Segmentation",
        "Tile",
        "Recolor",
    }
    assert actual_control_types.issuperset(expected_control_types), (
        expected_control_types - actual_control_types
    )
