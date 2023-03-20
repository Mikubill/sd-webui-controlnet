# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from annotator.mmpkg.mmcv import Config

from ...builder import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset


@DATASETS.register_module()
class TopDownOCHumanDataset(TopDownCocoDataset):
    """OChuman dataset for top-down pose estimation.

    "Pose2Seg: Detection Free Human Instance Segmentation", CVPR'2019.
    More details can be found in the `paper
    <https://arxiv.org/abs/1803.10683>`__ .

    "Occluded Human (OCHuman)" dataset contains 8110 heavily occluded
    human instances within 4731 images. OCHuman dataset is designed for
    validation and testing. To evaluate on OCHuman, the model should be
    trained on COCO training set, and then test the robustness of the
    model to occlusion using OCHuman.

    OCHuman keypoint indexes (same as COCO)::

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'

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
            cfg = Config.fromfile('configs/_base_/datasets/ochuman.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super(TopDownCocoDataset, self).__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        assert self.use_gt_bbox
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db
