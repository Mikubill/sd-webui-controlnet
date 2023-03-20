# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from annotator.mmpkg.mmpose.datasets.builder import PIPELINES
from .top_down_transform import TopDownRandomFlip


@PIPELINES.register_module()
class HandRandomFlip(TopDownRandomFlip):
    """Data augmentation with random image flip. A child class of
    TopDownRandomFlip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center',
    'hand_type', 'rel_root_depth' and 'ann_info'.

    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center',
    'hand_type', 'rel_root_depth'.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        # base flip augmentation
        super().__call__(results)

        # flip hand type and root depth
        hand_type = results['hand_type']
        rel_root_depth = results['rel_root_depth']
        flipped = results['flipped']
        if flipped:
            hand_type[0], hand_type[1] = hand_type[1], hand_type[0]
            rel_root_depth = -rel_root_depth
        results['hand_type'] = hand_type
        results['rel_root_depth'] = rel_root_depth
        return results


@PIPELINES.register_module()
class HandGenerateRelDepthTarget:
    """Generate the target relative root depth.

    Required keys: 'rel_root_depth', 'rel_root_valid', 'ann_info'.

    Modified keys: 'target', 'target_weight'.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Generate the target heatmap."""
        rel_root_depth = results['rel_root_depth']
        rel_root_valid = results['rel_root_valid']
        cfg = results['ann_info']
        D = cfg['heatmap_size_root']
        root_depth_bound = cfg['root_depth_bound']
        target = (rel_root_depth / root_depth_bound + 0.5) * D
        target_weight = rel_root_valid * (target >= 0) * (target <= D)
        results['target'] = target * np.ones(1, dtype=np.float32)
        results['target_weight'] = target_weight * np.ones(1, dtype=np.float32)
        return results
