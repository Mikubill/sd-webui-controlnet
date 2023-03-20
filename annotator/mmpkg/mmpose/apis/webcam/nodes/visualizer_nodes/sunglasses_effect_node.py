# Copyright (c) OpenMMLab. All rights reserved.
from itertools import groupby
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from ...utils import get_eye_keypoint_ids, load_image_from_disk_or_url
from ..base_visualizer_node import BaseVisualizerNode
from ..registry import NODES


@NODES.register_module()
class SunglassesEffectNode(BaseVisualizerNode):
    """Apply sunglasses effect (draw sunglasses at the facial area)to the
    objects with eye keypoints in the frame.

    Args:
        name (str): The node name (also thread name)
        input_buffer (str): The name of the input buffer
        output_buffer (str|list): The name(s) of the output buffer(s)
        enable_key (str|int, optional): Set a hot-key to toggle enable/disable
            of the node. If an int value is given, it will be treated as an
            ascii code of a key. Please note:
                1. If enable_key is set, the bypass method need to be
                    overridden to define the node behavior when disabled
                2. Some hot-key has been use for particular use. For example:
                    'q', 'Q' and 27 are used for quit
            Default: ``None``
        enable (bool): Default enable/disable status. Default: ``True``.
        kpt_thr (float): The score threshold of valid keypoints. Default: 0.5
        resource_img_path (str, optional): The resource image path or url.
            The image should be a pair of sunglasses with white background.
            If not specified, the url of a default image will be used. See
            ``SunglassesNode.default_resource_img_path``. Default: ``None``

    Example::
        >>> cfg = dict(
        ...    type='SunglassesEffectNode',
        ...    name='sunglasses',
        ...    enable_key='s',
        ...    enable=False,
        ...    input_buffer='vis',
        ...    output_buffer='vis_sunglasses')

        >>> from annotator.mmpkg.mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    # The image attributes to:
    # "https://www.vecteezy.com/vector-art/1932353-summer-sunglasses-
    # accessory-isolated-icon" by Vecteezy
    default_resource_img_path = (
        'https://user-images.githubusercontent.com/15977946/'
        '170850839-acc59e26-c6b3-48c9-a9ec-87556edb99ed.jpg')

    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 kpt_thr: float = 0.5,
                 resource_img_path: Optional[str] = None):

        super().__init__(
            name=name,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            enable_key=enable_key,
            enable=enable)

        if resource_img_path is None:
            resource_img_path = self.default_resource_img_path

        self.resource_img = load_image_from_disk_or_url(resource_img_path)
        self.kpt_thr = kpt_thr

    def draw(self, input_msg):
        canvas = input_msg.get_image()

        objects = input_msg.get_objects(lambda x: 'keypoints' in x)

        for model_cfg, group in groupby(objects,
                                        lambda x: x['pose_model_cfg']):
            left_eye_index, right_eye_index = get_eye_keypoint_ids(model_cfg)
            canvas = self.apply_sunglasses_effect(canvas, group,
                                                  left_eye_index,
                                                  right_eye_index)
        return canvas

    def apply_sunglasses_effect(self, canvas: np.ndarray, objects: List[Dict],
                                left_eye_index: int,
                                right_eye_index: int) -> np.ndarray:
        """Apply sunglasses effect.

        Args:
            canvas (np.ndarray): The image to apply the effect
            objects (list[dict]): The object list with keypoints
                - "keypoints" ([K,3]): keypoints in [x, y, score]
            left_eye_index (int): Keypoint index of the left eye
            right_eye_index (int): Keypoint index of the right eye

        Returns:
            np.ndarray: Processed image
        """

        hm, wm = self.resource_img.shape[:2]
        # anchor points in the sunglasses image
        pts_src = np.array([[0.3 * wm, 0.3 * hm], [0.3 * wm, 0.7 * hm],
                            [0.7 * wm, 0.3 * hm], [0.7 * wm, 0.7 * hm]],
                           dtype=np.float32)

        for obj in objects:
            kpts = obj['keypoints']

            if kpts[left_eye_index,
                    2] < self.kpt_thr or kpts[right_eye_index,
                                              2] < self.kpt_thr:
                continue

            kpt_leye = kpts[left_eye_index, :2]
            kpt_reye = kpts[right_eye_index, :2]
            # orthogonal vector to the left-to-right eyes
            vo = 0.5 * (kpt_reye - kpt_leye)[::-1] * [-1, 1]

            # anchor points in the image by eye positions
            pts_tar = np.vstack(
                [kpt_reye + vo, kpt_reye - vo, kpt_leye + vo, kpt_leye - vo])

            h_mat, _ = cv2.findHomography(pts_src, pts_tar)
            patch = cv2.warpPerspective(
                self.resource_img,
                h_mat,
                dsize=(canvas.shape[1], canvas.shape[0]),
                borderValue=(255, 255, 255))
            #  mask the white background area in the patch with a threshold 200
            mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            mask = (mask < 200).astype(np.uint8)
            canvas = cv2.copyTo(patch, mask, canvas)

        return canvas
