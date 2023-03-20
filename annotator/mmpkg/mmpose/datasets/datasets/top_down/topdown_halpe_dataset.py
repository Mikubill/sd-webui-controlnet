# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from annotator.mmpkg.mmcv import Config

from ...builder import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset


@DATASETS.register_module()
class TopDownHalpeDataset(TopDownCocoDataset):
    """HalpeDataset for top-down pose estimation.

    'https://github.com/Fang-Haoshu/Halpe-FullBody'

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Halpe keypoint indexes::

        0-19: 20 body keypoints,
        20-25: 6 foot keypoints,
        26-93: 68 face keypoints,
        94-135: 42 hand keypoints

        In total, we have 136 keypoints for wholebody pose estimation.

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
            cfg = Config.fromfile('configs/_base_/datasets/halpe.py')
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

        self.ann_info['use_different_joint_weights'] = False

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')
