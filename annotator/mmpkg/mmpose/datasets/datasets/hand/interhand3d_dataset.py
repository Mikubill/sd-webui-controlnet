# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import json_tricks as json
import numpy as np
from annotator.mmpkg.mmcv import Config, deprecated_api_warning

from annotator.mmpkg.mmpose.core.evaluation.top_down_eval import keypoint_epe
from annotator.mmpkg.mmpose.datasets.builder import DATASETS
from ..base import Kpt3dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class InterHand3DDataset(Kpt3dSviewRgbImgTopDownDataset):
    """InterHand2.6M 3D dataset for top-down hand pose estimation.

    "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose
    Estimation from a Single RGB Image", ECCV'2020.
    More details can be found in the `paper
    <https://arxiv.org/pdf/2008.09309.pdf>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    InterHand2.6M keypoint indexes::

        0: 'r_thumb4',
        1: 'r_thumb3',
        2: 'r_thumb2',
        3: 'r_thumb1',
        4: 'r_index4',
        5: 'r_index3',
        6: 'r_index2',
        7: 'r_index1',
        8: 'r_middle4',
        9: 'r_middle3',
        10: 'r_middle2',
        11: 'r_middle1',
        12: 'r_ring4',
        13: 'r_ring3',
        14: 'r_ring2',
        15: 'r_ring1',
        16: 'r_pinky4',
        17: 'r_pinky3',
        18: 'r_pinky2',
        19: 'r_pinky1',
        20: 'r_wrist',
        21: 'l_thumb4',
        22: 'l_thumb3',
        23: 'l_thumb2',
        24: 'l_thumb1',
        25: 'l_index4',
        26: 'l_index3',
        27: 'l_index2',
        28: 'l_index1',
        29: 'l_middle4',
        30: 'l_middle3',
        31: 'l_middle2',
        32: 'l_middle1',
        33: 'l_ring4',
        34: 'l_ring3',
        35: 'l_ring2',
        36: 'l_ring1',
        37: 'l_pinky4',
        38: 'l_pinky3',
        39: 'l_pinky2',
        40: 'l_pinky1',
        41: 'l_wrist'

    Args:
        ann_file (str): Path to the annotation file.
        camera_file (str): Path to the camera file.
        joint_file (str): Path to the joint file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        use_gt_root_depth (bool): Using the ground truth depth of the wrist
            or given depth from rootnet_result_file.
        rootnet_result_file (str): Path to the wrist depth file.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (str): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 camera_file,
                 joint_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 use_gt_root_depth=True,
                 rootnet_result_file=None,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/interhand3d.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.ann_info['heatmap3d_depth_bound'] = data_cfg[
            'heatmap3d_depth_bound']
        self.ann_info['heatmap_size_root'] = data_cfg['heatmap_size_root']
        self.ann_info['root_depth_bound'] = data_cfg['root_depth_bound']
        self.ann_info['use_different_joint_weights'] = False

        self.camera_file = camera_file
        self.joint_file = joint_file

        self.use_gt_root_depth = use_gt_root_depth
        if not self.use_gt_root_depth:
            assert rootnet_result_file is not None
            self.rootnet_result_file = rootnet_result_file

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    @staticmethod
    def _encode_handtype(hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, f'Not support hand type: {hand_type}'

    def _get_db(self):
        """Load dataset.

        Adapted from 'https://github.com/facebookresearch/InterHand2.6M/'
            'blob/master/data/InterHand2.6M/dataset.py'
        Copyright (c) FaceBook Research, under CC-BY-NC 4.0 license.
        """
        with open(self.camera_file, 'r') as f:
            cameras = json.load(f)
        with open(self.joint_file, 'r') as f:
            joints = json.load(f)

        if not self.use_gt_root_depth:
            rootnet_result = {}
            with open(self.rootnet_result_file, 'r') as f:
                rootnet_annot = json.load(f)
            for i in range(len(rootnet_annot)):
                rootnet_result[str(
                    rootnet_annot[i]['annot_id'])] = rootnet_annot[i]

        gt_db = []
        bbox_id = 0
        for img_id in self.img_ids:
            num_joints = self.ann_info['num_joints']

            ann_id = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            ann = self.coco.loadAnns(ann_id)[0]
            img = self.coco.loadImgs(img_id)[0]

            capture_id = str(img['capture'])
            camera_name = img['camera']
            frame_idx = str(img['frame_idx'])
            image_file = osp.join(self.img_prefix, self.id2name[img_id])

            camera_pos = np.array(
                cameras[capture_id]['campos'][camera_name], dtype=np.float32)
            camera_rot = np.array(
                cameras[capture_id]['camrot'][camera_name], dtype=np.float32)
            focal = np.array(
                cameras[capture_id]['focal'][camera_name], dtype=np.float32)
            principal_pt = np.array(
                cameras[capture_id]['princpt'][camera_name], dtype=np.float32)
            joint_world = np.array(
                joints[capture_id][frame_idx]['world_coord'], dtype=np.float32)
            joint_cam = self._world2cam(
                joint_world.transpose(1, 0), camera_rot,
                camera_pos.reshape(3, 1)).transpose(1, 0)
            joint_img = self._cam2pixel(joint_cam, focal, principal_pt)[:, :2]

            joint_valid = np.array(
                ann['joint_valid'], dtype=np.float32).flatten()
            hand_type = self._encode_handtype(ann['hand_type'])
            hand_type_valid = ann['hand_type_valid']

            if self.use_gt_root_depth:
                bbox = np.array(ann['bbox'], dtype=np.float32)
                abs_depth = [joint_cam[20, 2], joint_cam[41, 2]]
            else:
                rootnet_ann_data = rootnet_result[str(ann_id[0])]
                bbox = np.array(rootnet_ann_data['bbox'], dtype=np.float32)
                abs_depth = rootnet_ann_data['abs_depth']
            # 41: 'l_wrist', left hand root
            # 20: 'r_wrist', right hand root
            rel_root_depth = joint_cam[41, 2] - joint_cam[20, 2]
            # if root is not valid, root-relative 3D depth is also invalid.
            rel_root_valid = joint_valid[20] * joint_valid[41]

            # if root is not valid -> root-relative 3D pose is also not valid.
            # Therefore, mark all joints as invalid
            joint_valid[:20] *= joint_valid[20]
            joint_valid[21:] *= joint_valid[41]

            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d[:, :2] = joint_img
            joints_3d[:21, 2] = joint_cam[:21, 2] - joint_cam[20, 2]
            joints_3d[21:, 2] = joint_cam[21:, 2] - joint_cam[41, 2]
            joints_3d_visible[...] = np.minimum(1, joint_valid.reshape(-1, 1))

            gt_db.append({
                'image_file': image_file,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'hand_type': hand_type,
                'hand_type_valid': hand_type_valid,
                'rel_root_depth': rel_root_depth,
                'rel_root_valid': rel_root_valid,
                'abs_depth': abs_depth,
                'joints_cam': joint_cam,
                'focal': focal,
                'princpt': principal_pt,
                'dataset': self.dataset_name,
                'bbox': bbox,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='MPJPE', **kwargs):
        """Evaluate interhand2d keypoint results. The pose prediction results
        will be saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - hand_type (np.ndarray[N, 4]): The first two dimensions are \
                    hand type, scores is the last two dimensions.
                - rel_root_depth (np.ndarray[N]): The relative depth of left \
                    wrist and right wrist.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['Capture6/\
                    0012_aokay_upright/cam410061/image4996.jpg']
                - output_heatmap (np.ndarray[N, K, H, W]): model outputs.
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed.
                Options: 'MRRPE', 'MPJPE', 'Handedness_acc'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['MRRPE', 'MPJPE', 'Handedness_acc']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = []
        for result in results:
            preds = result.get('preds')
            if preds is None and 'MPJPE' in metrics:
                raise KeyError('metric MPJPE is not supported')

            hand_type = result.get('hand_type')
            if hand_type is None and 'Handedness_acc' in metrics:
                raise KeyError('metric Handedness_acc is not supported')

            rel_root_depth = result.get('rel_root_depth')
            if rel_root_depth is None and 'MRRPE' in metrics:
                raise KeyError('metric MRRPE is not supported')

            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpt = {
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                }

                if preds is not None:
                    kpt['keypoints'] = preds[i, :, :3].tolist()
                if hand_type is not None:
                    kpt['hand_type'] = hand_type[i][0:2].tolist()
                    kpt['hand_type_score'] = hand_type[i][2:4].tolist()
                if rel_root_depth is not None:
                    kpt['rel_root_depth'] = float(rel_root_depth[i])

                kpts.append(kpt)
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value

    @staticmethod
    def _get_accuracy(outputs, gts, masks):
        """Get accuracy of multi-label classification.

        Note:
            - batch_size: N
            - label_num: C

        Args:
            outputs (np.array[N, C]): predicted multi-label.
            gts (np.array[N, C]): Groundtruth muti-label.
            masks (np.array[N, ]): masked outputs will be ignored for
                accuracy calculation.

        Returns:
            float: mean accuracy
        """
        acc = (outputs == gts).all(axis=1)
        return np.mean(acc[masks])

    def _report_metric(self, res_file, metrics):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'MRRPE', 'MPJPE', 'Handedness_acc'.

        Returns:
            list: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        gts_rel_root = []
        preds_rel_root = []
        rel_root_masks = []
        gts_joint_coord_cam = []
        preds_joint_coord_cam = []
        single_masks = []
        interacting_masks = []
        all_masks = []
        gts_hand_type = []
        preds_hand_type = []
        hand_type_masks = []

        for pred, item in zip(preds, self.db):
            # mrrpe
            if 'MRRPE' in metrics:
                if item['hand_type'].all() and item['joints_3d_visible'][
                        20, 0] and item['joints_3d_visible'][41, 0]:
                    rel_root_masks.append(True)

                    pred_left_root_img = np.array(
                        pred['keypoints'][41], dtype=np.float32)[None, :]
                    pred_left_root_img[:, 2] += item['abs_depth'][0] + pred[
                        'rel_root_depth']
                    pred_left_root_cam = self._pixel2cam(
                        pred_left_root_img, item['focal'], item['princpt'])

                    pred_right_root_img = np.array(
                        pred['keypoints'][20], dtype=np.float32)[None, :]
                    pred_right_root_img[:, 2] += item['abs_depth'][0]
                    pred_right_root_cam = self._pixel2cam(
                        pred_right_root_img, item['focal'], item['princpt'])

                    preds_rel_root.append(pred_left_root_cam -
                                          pred_right_root_cam)
                    gts_rel_root.append(
                        [item['joints_cam'][41] - item['joints_cam'][20]])
                else:
                    rel_root_masks.append(False)
                    preds_rel_root.append([[0., 0., 0.]])
                    gts_rel_root.append([[0., 0., 0.]])

            if 'MPJPE' in metrics:
                pred_joint_coord_img = np.array(
                    pred['keypoints'], dtype=np.float32)
                gt_joint_coord_cam = item['joints_cam'].copy()

                pred_joint_coord_img[:21, 2] += item['abs_depth'][0]
                pred_joint_coord_img[21:, 2] += item['abs_depth'][1]
                pred_joint_coord_cam = self._pixel2cam(pred_joint_coord_img,
                                                       item['focal'],
                                                       item['princpt'])

                pred_joint_coord_cam[:21] -= pred_joint_coord_cam[20]
                pred_joint_coord_cam[21:] -= pred_joint_coord_cam[41]
                gt_joint_coord_cam[:21] -= gt_joint_coord_cam[20]
                gt_joint_coord_cam[21:] -= gt_joint_coord_cam[41]

                preds_joint_coord_cam.append(pred_joint_coord_cam)
                gts_joint_coord_cam.append(gt_joint_coord_cam)

                mask = (np.array(item['joints_3d_visible'])[:, 0]) > 0

                if item['hand_type'].all():
                    single_masks.append(
                        np.zeros(self.ann_info['num_joints'], dtype=bool))
                    interacting_masks.append(mask)
                    all_masks.append(mask)
                else:
                    single_masks.append(mask)
                    interacting_masks.append(
                        np.zeros(self.ann_info['num_joints'], dtype=bool))
                    all_masks.append(mask)

            if 'Handedness_acc' in metrics:
                pred_hand_type = np.array(pred['hand_type'], dtype=int)
                preds_hand_type.append(pred_hand_type)
                gts_hand_type.append(item['hand_type'])
                hand_type_masks.append(item['hand_type_valid'] > 0)

        gts_rel_root = np.array(gts_rel_root, dtype=np.float32)
        preds_rel_root = np.array(preds_rel_root, dtype=np.float32)
        rel_root_masks = np.array(rel_root_masks, dtype=bool)[:, None]
        gts_joint_coord_cam = np.array(gts_joint_coord_cam, dtype=np.float32)
        preds_joint_coord_cam = np.array(
            preds_joint_coord_cam, dtype=np.float32)
        single_masks = np.array(single_masks, dtype=bool)
        interacting_masks = np.array(interacting_masks, dtype=bool)
        all_masks = np.array(all_masks, dtype=bool)
        gts_hand_type = np.array(gts_hand_type, dtype=int)
        preds_hand_type = np.array(preds_hand_type, dtype=int)
        hand_type_masks = np.array(hand_type_masks, dtype=bool)

        if 'MRRPE' in metrics:
            info_str.append(('MRRPE',
                             keypoint_epe(preds_rel_root, gts_rel_root,
                                          rel_root_masks)))

        if 'MPJPE' in metrics:
            info_str.append(('MPJPE_all',
                             keypoint_epe(preds_joint_coord_cam,
                                          gts_joint_coord_cam, all_masks)))
            info_str.append(('MPJPE_single',
                             keypoint_epe(preds_joint_coord_cam,
                                          gts_joint_coord_cam, single_masks)))
            info_str.append(
                ('MPJPE_interacting',
                 keypoint_epe(preds_joint_coord_cam, gts_joint_coord_cam,
                              interacting_masks)))

        if 'Handedness_acc' in metrics:
            info_str.append(('Handedness_acc',
                             self._get_accuracy(preds_hand_type, gts_hand_type,
                                                hand_type_masks)))

        return info_str
