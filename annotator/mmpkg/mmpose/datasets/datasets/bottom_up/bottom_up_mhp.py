# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import json_tricks as json
from annotator.mmpkg.mmcv import Config
from xtcocotools.cocoeval import COCOeval

from annotator.mmpkg.mmpose.datasets.builder import DATASETS
from .bottom_up_coco import BottomUpCocoDataset


@DATASETS.register_module()
class BottomUpMhpDataset(BottomUpCocoDataset):
    """MHPv2.0 dataset for top-down pose estimation.

    "Understanding Humans in Crowded Scenes: Deep Nested Adversarial
    Learning and A New Benchmark for Multi-Human Parsing", ACM MM'2018.
    More details can be found in the `paper
    <https://arxiv.org/abs/1804.03287>`__

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    MHP keypoint indexes::

        0: "right ankle",
        1: "right knee",
        2: "right hip",
        3: "left hip",
        4: "left knee",
        5: "left ankle",
        6: "pelvis",
        7: "thorax",
        8: "upper neck",
        9: "head top",
        10: "right wrist",
        11: "right elbow",
        12: "right shoulder",
        13: "left shoulder",
        14: "left elbow",
        15: "left wrist",

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/mhp.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super(BottomUpCocoDataset, self).__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.ann_info['use_different_joint_weights'] = False
        print(f'=> num_images: {self.num_images}')

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        with open(res_file, 'r') as file:
            res_json = json.load(file)
            if not res_json:
                info_str = list(zip(stats_names, [
                    0,
                ] * len(stats_names)))
                return info_str

        coco_det = self.coco.loadRes(res_file)

        coco_eval = COCOeval(
            self.coco, coco_det, 'keypoints', self.sigmas, use_area=False)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str
