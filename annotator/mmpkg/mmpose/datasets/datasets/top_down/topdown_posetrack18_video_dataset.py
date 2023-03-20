# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from annotator.mmpkg.mmcv import deprecated_api_warning

from ....core.post_processing import oks_nms, soft_oks_nms
from ...builder import DATASETS
from ..base import Kpt2dSviewRgbVidTopDownDataset

try:
    from poseval import eval_helpers
    from poseval.evaluateAP import evaluateAP
    has_poseval = True
except (ImportError, ModuleNotFoundError):
    has_poseval = False


@DATASETS.register_module()
class TopDownPoseTrack18VideoDataset(Kpt2dSviewRgbVidTopDownDataset):
    """PoseTrack18 dataset for top-down pose estimation.

    "Posetrack: A benchmark for human pose estimation and tracking", CVPR'2018.
    More details can be found in the `paper
    <https://arxiv.org/abs/1710.10000>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    PoseTrack2018 keypoint indexes::

        0: 'nose',
        1: 'head_bottom',
        2: 'head_top',
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
        img_prefix (str): Path to a directory where videos/images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
        ph_fill_len (int): The length of the placeholder to fill in the
            image filenames, default: 6 in PoseTrack18.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False,
                 ph_fill_len=6):
        super().__init__(
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
        self.frame_weight_train = data_cfg['frame_weight_train']
        self.frame_weight_test = data_cfg['frame_weight_test']
        self.frame_weight = self.frame_weight_test \
            if self.test_mode else self.frame_weight_train

        self.ph_fill_len = ph_fill_len

        # select the frame indices
        self.frame_index_rand = data_cfg.get('frame_index_rand', True)
        self.frame_index_range = data_cfg.get('frame_index_range', [-2, 2])
        self.num_adj_frames = data_cfg.get('num_adj_frames', 1)
        self.frame_indices_train = data_cfg.get('frame_indices_train', None)
        self.frame_indices_test = data_cfg.get('frame_indices_test',
                                               [-2, -1, 0, 1, 2])

        if self.frame_indices_train is not None:
            self.frame_indices_train.sort()
        self.frame_indices_test.sort()

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        if (not self.test_mode) or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_posetrack_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        file_name = img_ann['file_name']
        nframes = int(img_ann['nframes'])
        frame_id = int(img_ann['frame_id'])

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w))
            y2 = min(height - 1, y1 + max(0, h))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            image_files = []
            cur_image_file = osp.join(self.img_prefix, self.id2name[img_id])
            image_files.append(cur_image_file)

            # "images/val/012834_mpii_test/000000.jpg" -->> "000000.jpg"
            cur_image_name = file_name.split('/')[-1]
            ref_idx = int(cur_image_name.replace('.jpg', ''))

            # select the frame indices
            if not self.test_mode and self.frame_indices_train is not None:
                indices = self.frame_indices_train
            elif not self.test_mode and self.frame_index_rand:
                low, high = self.frame_index_range
                indices = np.random.randint(low, high + 1, self.num_adj_frames)
            else:
                indices = self.frame_indices_test

            for index in indices:
                if self.test_mode and index == 0:
                    continue
                # the supporting frame index
                support_idx = ref_idx + index
                # clip the frame index to make sure that it does not exceed
                # the boundings of frame indices
                support_idx = np.clip(support_idx, 0, nframes - 1)
                sup_image_file = cur_image_file.replace(
                    cur_image_name,
                    str(support_idx).zfill(self.ph_fill_len) + '.jpg')

                image_files.append(sup_image_file)

            rec.append({
                'image_file': image_files,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id,
                'nframes': nframes,
                'frame_id': frame_id,
                'frame_weight': self.frame_weight
            })
            bbox_id = bbox_id + 1

        return rec

    def _load_posetrack_person_detection_results(self):
        """Load Posetrack person detection results.

        Only in test mode.
        """
        num_joints = self.ann_info['num_joints']
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError('=> Load %s fail!' % self.bbox_file)

        print(f'=> Total boxes: {len(all_boxes)}')

        kpt_db = []
        bbox_id = 0
        for det_res in all_boxes:
            if det_res['category_id'] != 1:
                continue

            score = det_res['score']
            if score < self.det_bbox_thr:
                continue

            box = det_res['bbox']

            # deal with different bbox file formats
            if 'nframes' in det_res and 'frame_id' in det_res:
                nframes = int(det_res['nframes'])
                frame_id = int(det_res['frame_id'])
            elif 'image_name' in det_res:
                img_id = self.name2id[det_res['image_name']]
                img_ann = self.coco.loadImgs(img_id)[0]
                nframes = int(img_ann['nframes'])
                frame_id = int(img_ann['frame_id'])
            else:
                img_id = det_res['image_id']
                img_ann = self.coco.loadImgs(img_id)[0]
                nframes = int(img_ann['nframes'])
                frame_id = int(img_ann['frame_id'])

            image_files = []
            if 'image_name' in det_res:
                file_name = det_res['image_name']
            else:
                file_name = self.id2name[det_res['image_id']]

            cur_image_file = osp.join(self.img_prefix, file_name)
            image_files.append(cur_image_file)

            # "images/val/012834_mpii_test/000000.jpg" -->> "000000.jpg"
            cur_image_name = file_name.split('/')[-1]
            ref_idx = int(cur_image_name.replace('.jpg', ''))

            indices = self.frame_indices_test
            for index in indices:
                if self.test_mode and index == 0:
                    continue
                # the supporting frame index
                support_idx = ref_idx + index
                # clip the frame index to make sure that it does not exceed
                # the boundings of frame indices
                support_idx = np.clip(support_idx, 0, nframes - 1)
                sup_image_file = cur_image_file.replace(
                    cur_image_name,
                    str(support_idx).zfill(self.ph_fill_len) + '.jpg')

                image_files.append(sup_image_file)

            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)
            kpt_db.append({
                'image_file': image_files,
                'rotation': 0,
                'bbox': box[:4],
                'bbox_score': score,
                'dataset': self.dataset_name,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox_id': bbox_id,
                'nframes': nframes,
                'frame_id': frame_id,
                'frame_weight': self.frame_weight
            })
            bbox_id = bbox_id + 1
        print(f'=> Total boxes after filter '
              f'low score@{self.det_bbox_thr}: {bbox_id}')
        return kpt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mAP', **kwargs):
        """Evaluate posetrack keypoint results. The pose prediction results
        will be saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - num_keypoints: K

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['val/010016_mpii_test\
                    /000024.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap.
                - bbox_id (list(int))
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_folder = tmp_folder.name

        gt_folder = osp.join(
            osp.dirname(self.ann_file),
            osp.splitext(self.ann_file.split('_')[-1])[0])

        kpts = defaultdict(list)

        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                if not isinstance(image_paths[i], list):
                    image_id = self.name2id[image_paths[i]
                                            [len(self.img_prefix):]]
                else:
                    image_id = self.name2id[image_paths[i][0]
                                            [len(self.img_prefix):]]

                kpts[image_id].append({
                    'keypoints': preds[i],
                    'center': boxes[i][0:2],
                    'scale': boxes[i][2:4],
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        num_joints = self.ann_info['num_joints']
        vis_thr = self.vis_thr
        oks_thr = self.oks_thr
        valid_kpts = defaultdict(list)
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(img_kpts, oks_thr, sigmas=self.sigmas)
                valid_kpts[image_id].append(
                    [img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts[image_id].append(img_kpts)

        self._write_keypoint_results(valid_kpts, gt_folder, res_folder)

        info_str = self._do_python_keypoint_eval(gt_folder, res_folder)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value

    @staticmethod
    def _write_keypoint_results(keypoint_results, gt_folder, pred_folder):
        """Write results into a json file.

        Args:
            keypoint_results (dict): keypoint results organized by image_id.
            gt_folder (str): Path of directory for official gt files.
            pred_folder (str): Path of directory to save the results.
        """
        categories = []

        cat = {}
        cat['supercategory'] = 'person'
        cat['id'] = 1
        cat['name'] = 'person'
        cat['keypoints'] = [
            'nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
            'right_knee', 'left_ankle', 'right_ankle'
        ]
        cat['skeleton'] = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                           [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                           [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
                           [4, 6], [5, 7]]
        categories.append(cat)

        json_files = [
            pos for pos in os.listdir(gt_folder) if pos.endswith('.json')
        ]
        for json_file in json_files:

            with open(osp.join(gt_folder, json_file), 'r') as f:
                gt = json.load(f)

            annotations = []
            images = []

            for image in gt['images']:
                im = {}
                im['id'] = image['id']
                im['file_name'] = image['file_name']
                images.append(im)

                img_kpts = keypoint_results[im['id']]

                if len(img_kpts) == 0:
                    continue
                for track_id, img_kpt in enumerate(img_kpts[0]):
                    ann = {}
                    ann['image_id'] = img_kpt['image_id']
                    ann['keypoints'] = np.array(
                        img_kpt['keypoints']).reshape(-1).tolist()
                    ann['scores'] = np.array(ann['keypoints']).reshape(
                        [-1, 3])[:, 2].tolist()
                    ann['score'] = float(img_kpt['score'])
                    ann['track_id'] = track_id
                    annotations.append(ann)

            info = {}
            info['images'] = images
            info['categories'] = categories
            info['annotations'] = annotations

            with open(osp.join(pred_folder, json_file), 'w') as f:
                json.dump(info, f, sort_keys=True, indent=4)

    def _do_python_keypoint_eval(self, gt_folder, pred_folder):
        """Keypoint evaluation using poseval.

        Args:
            gt_folder (str): The folder of the json files storing
                ground truth keypoint annotations.
            pred_folder (str): The folder of the json files storing
                prediction results.

        Returns:
            List: Evaluation results for evaluation metric.
        """

        if not has_poseval:
            raise ImportError('Please install poseval package for evaluation'
                              'on PoseTrack dataset '
                              '(see requirements/optional.txt)')

        argv = ['', gt_folder + '/', pred_folder + '/']

        print('Loading data')
        gtFramesAll, prFramesAll = eval_helpers.load_data_dir(argv)

        print('# gt frames  :', len(gtFramesAll))
        print('# pred frames:', len(prFramesAll))

        # evaluate per-frame multi-person pose estimation (AP)
        # compute AP
        print('Evaluation of per-frame multi-person pose estimation')
        apAll, _, _ = evaluateAP(gtFramesAll, prFramesAll, None, False, False)

        # print AP
        print('Average Precision (AP) metric:')
        eval_helpers.printTable(apAll)

        stats = eval_helpers.getCum(apAll)

        stats_names = [
            'Head AP', 'Shou AP', 'Elb AP', 'Wri AP', 'Hip AP', 'Knee AP',
            'Ankl AP', 'Total AP'
        ]

        info_str = list(zip(stats_names, stats))

        return info_str
