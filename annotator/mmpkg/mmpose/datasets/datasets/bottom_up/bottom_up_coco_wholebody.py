# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
from annotator.mmpkg.mmcv import Config
from xtcocotools.cocoeval import COCOeval

from annotator.mmpkg.mmpose.datasets.builder import DATASETS
from .bottom_up_coco import BottomUpCocoDataset


@DATASETS.register_module()
class BottomUpCocoWholeBodyDataset(BottomUpCocoDataset):
    """CocoWholeBodyDataset dataset for bottom-up pose estimation.

    `Whole-Body Human Pose Estimation in the Wild', ECCV'2020.
    More details can be found in the `paper
    <https://arxiv.org/abs/2007.11858>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    In total, we have 133 keypoints for wholebody pose estimation.

    COCO-WholeBody keypoint indexes::

        0-16: 17 body keypoints,
        17-22: 6 foot keypoints,
        23-90: 68 face keypoints,
        91-132: 42 hand keypoints

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
            cfg = Config.fromfile('configs/_base_/datasets/coco_wholebody.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super(BottomUpCocoDataset, self).__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.ann_info['use_different_joint_weights'] = False

        self.body_num = 17
        self.foot_num = 6
        self.face_num = 68
        self.left_hand_num = 21
        self.right_hand_num = 21

        print(f'=> num_images: {self.num_images}')

    def _get_joints(self, anno):
        """Get joints for all people in an image."""
        num_people = len(anno)

        if self.ann_info['scale_aware_sigma']:
            joints = np.zeros((num_people, self.ann_info['num_joints'], 4),
                              dtype=np.float32)
        else:
            joints = np.zeros((num_people, self.ann_info['num_joints'], 3),
                              dtype=np.float32)

        for i, obj in enumerate(anno):
            keypoints = np.array(obj['keypoints'] + obj['foot_kpts'] +
                                 obj['face_kpts'] + obj['lefthand_kpts'] +
                                 obj['righthand_kpts']).reshape(-1, 3)

            joints[i, :self.ann_info['num_joints'], :3] = keypoints
            if self.ann_info['scale_aware_sigma']:
                # get person box
                box = obj['bbox']
                size = max(box[2], box[3])
                sigma = size / self.base_size * self.base_sigma
                if self.int_sigma:
                    sigma = int(np.ceil(sigma))
                assert sigma > 0, sigma
                joints[i, :, 3] = sigma

        return joints

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            cuts = np.cumsum([
                0, self.body_num, self.foot_num, self.face_num,
                self.left_hand_num, self.right_hand_num
            ]) * 3

            for img_kpt, key_point in zip(img_kpts, key_points):
                kpt = key_point.reshape((self.ann_info['num_joints'], 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                cat_results.append({
                    'image_id':
                    img_kpt['image_id'],
                    'category_id':
                    cat_id,
                    'keypoints':
                    key_point[cuts[0]:cuts[1]].tolist(),
                    'foot_kpts':
                    key_point[cuts[1]:cuts[2]].tolist(),
                    'face_kpts':
                    key_point[cuts[2]:cuts[3]].tolist(),
                    'lefthand_kpts':
                    key_point[cuts[3]:cuts[4]].tolist(),
                    'righthand_kpts':
                    key_point[cuts[4]:cuts[5]].tolist(),
                    'score':
                    img_kpt['score'],
                    'bbox': [left_top[0], left_top[1], w, h]
                })

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)

        cuts = np.cumsum([
            0, self.body_num, self.foot_num, self.face_num, self.left_hand_num,
            self.right_hand_num
        ])

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_body',
            self.sigmas[cuts[0]:cuts[1]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_foot',
            self.sigmas[cuts[1]:cuts[2]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_face',
            self.sigmas[cuts[2]:cuts[3]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_lefthand',
            self.sigmas[cuts[3]:cuts[4]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_righthand',
            self.sigmas[cuts[4]:cuts[5]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_wholebody',
            self.sigmas,
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str
