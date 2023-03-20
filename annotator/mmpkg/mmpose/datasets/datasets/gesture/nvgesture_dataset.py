# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
import warnings
from collections import defaultdict

import json_tricks as json
import numpy as np
from annotator.mmpkg.mmcv import Config

from ...builder import DATASETS
from .gesture_base_dataset import GestureBaseDataset


@DATASETS.register_module()
class NVGestureDataset(GestureBaseDataset):
    """NVGesture dataset for gesture recognition.

    "Online Detection and Classification of Dynamic Hand Gestures
    With Recurrent 3D Convolutional Neural Network",
    Conference on Computer Vision and Pattern Recognition (CVPR) 2016.

    The dataset loads raw videos and apply specified transforms
    to return a dict containing the image tensors and other information.

    Args:
        ann_file (str): Path to the annotation file.
        vid_prefix (str): Path to a directory where videos are held.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 vid_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/nvgesture.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            vid_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.db = self._get_db()
        self.vid_ids = list(range(len(self.db)))
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        db = []
        with open(self.ann_file, 'r') as f:
            samples = f.readlines()

        use_bbox = bool(self.bbox_file)
        if use_bbox:
            with open(self.bbox_file, 'r') as f:
                bboxes = json.load(f)

        for sample in samples:
            sample = sample.strip().split()
            sample = {
                item.split(':', 1)[0]: item.split(':', 1)[1]
                for item in sample
            }
            path = sample['path'][2:]
            for key in ('depth', 'color'):
                fname, start, end = sample[key].split(':')
                sample[key] = {
                    'path': os.path.join(path, fname + '.avi'),
                    'valid_frames': (eval(start), eval(end))
                }
            sample['flow'] = {
                'path': sample['color']['path'].replace('color', 'flow'),
                'valid_frames': sample['color']['valid_frames']
            }
            sample['rgb'] = sample['color']
            sample['label'] = eval(sample['label']) - 1

            if use_bbox:
                sample['bbox'] = bboxes[path]

            del sample['path'], sample['duo_left'], sample['color']
            db.append(sample)

        return db

    def _get_single(self, idx):
        """Get anno for a single video."""
        anno = defaultdict(list)
        sample = self.db[self.vid_ids[idx]]

        anno['label'] = sample['label']
        anno['modality'] = self.modality
        if 'bbox' in sample:
            anno['bbox'] = sample['bbox']

        for modal in self.modality:
            anno['video_file'].append(
                os.path.join(self.vid_prefix, sample[modal]['path']))
            anno['valid_frames'].append(sample[modal]['valid_frames'])

        return anno

    def evaluate(self, results, res_folder=None, metric='AP', **kwargs):
        """Evaluate nvgesture recognition results. The gesture prediction
        results will be saved in ``${res_folder}/result_gesture.json``.

        Note:
            - batch_size: N
            - heatmap length: L

        Args:
            results (dict): Testing results containing the following
                items:
                - logits (dict[str, torch.tensor[N,25,L]]): For each item, \
                    the key represents the modality of input video, while \
                    the value represents the prediction of gesture. Three \
                    dimensions represent batch, category and temporal \
                    length, respectively.
                - label (np.ndarray[N]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed.
                Options: 'AP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['AP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_gesture.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_gesture.json')

        predictions = defaultdict(list)
        label = []
        for result in results:
            label.append(result['label'].cpu().numpy())
            for modal in result['logits']:
                logit = result['logits'][modal].mean(dim=2)
                pred = logit.argmax(dim=1).cpu().numpy()
                predictions[modal].append(pred)

        label = np.concatenate(label, axis=0)
        for modal in predictions:
            predictions[modal] = np.concatenate(predictions[modal], axis=0)

        with open(res_file, 'w') as f:
            json.dump(predictions, f, indent=4)

        results = dict()
        if 'AP' in metrics:
            APs = []
            for modal in predictions:
                results[f'AP_{modal}'] = (predictions[modal] == label).mean()
                APs.append(results[f'AP_{modal}'])
            results['AP_mean'] = sum(APs) / len(APs)

        return results
