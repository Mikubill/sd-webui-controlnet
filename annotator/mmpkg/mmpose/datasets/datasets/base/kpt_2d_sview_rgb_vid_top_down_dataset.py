# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset
from xtcocotools.coco import COCO

from annotator.mmpkg.mmpose.datasets import DatasetInfo
from annotator.mmpkg.mmpose.datasets.pipelines import Compose


class Kpt2dSviewRgbVidTopDownDataset(Dataset, metaclass=ABCMeta):
    """Base class for keypoint 2D top-down pose estimation with single-view RGB
    video as the input.

    All fashion datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_db`, 'evaluate'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where videos/images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        coco_style (bool): Whether the annotation json is coco-style.
            Default: True
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 coco_style=True,
                 test_mode=False):

        self.image_info = {}
        self.ann_info = {}

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.ann_info['use_different_joint_weights'] = data_cfg.get(
            'use_different_joint_weights', False)

        if dataset_info is None:
            raise ValueError(
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.')

        dataset_info = DatasetInfo(dataset_info)

        assert self.ann_info['num_joints'] == dataset_info.keypoint_num
        self.ann_info['flip_pairs'] = dataset_info.flip_pairs
        self.ann_info['flip_index'] = dataset_info.flip_index
        self.ann_info['upper_body_ids'] = dataset_info.upper_body_ids
        self.ann_info['lower_body_ids'] = dataset_info.lower_body_ids
        self.ann_info['joint_weights'] = dataset_info.joint_weights
        self.ann_info['skeleton'] = dataset_info.skeleton
        self.sigmas = dataset_info.sigmas
        self.dataset_name = dataset_info.dataset_name

        if coco_style:
            self.coco = COCO(ann_file)
            if 'categories' in self.coco.dataset:
                cats = [
                    cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())
                ]
                self.classes = ['__background__'] + cats
                self.num_classes = len(self.classes)
                self._class_to_ind = dict(
                    zip(self.classes, range(self.num_classes)))
                self._class_to_coco_ind = dict(
                    zip(cats, self.coco.getCatIds()))
                self._coco_ind_to_class_ind = dict(
                    (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                    for cls in self.classes[1:])
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id = self._get_mapping_id_name(
                self.coco.imgs)

        self.db = []

        self.pipeline = Compose(self.pipeline)

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _xywh2cs(self, x, y, w, h, padding=1.25):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor

        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """

        warnings.warn(
            'The ``_xywh2cs`` method will be deprecated and removed from '
            f'{self.__class__.__name__} in the future. Please use data '
            'transforms ``TopDownGetBboxCenterScale`` and '
            '``TopDownRandomShiftBboxCenter`` in the pipeline instead.',
            DeprecationWarning)

        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
            'image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if (not self.test_mode) and np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * padding

        return center, scale

    @abstractmethod
    def _get_db(self):
        """Load dataset."""

    @abstractmethod
    def evaluate(self, results, *args, **kwargs):
        """Evaluate keypoint results."""

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts
