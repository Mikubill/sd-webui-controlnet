# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod

import numpy as np
import xtcocotools
from torch.utils.data import Dataset
from xtcocotools.coco import COCO

from annotator.mmpkg.mmpose.datasets import DatasetInfo
from annotator.mmpkg.mmpose.datasets.pipelines import Compose


class Kpt2dSviewRgbImgBottomUpDataset(Dataset, metaclass=ABCMeta):
    """Base class for bottom-up datasets.

    All datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_single`

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
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

        # bottom-up
        self.base_size = data_cfg['base_size']
        self.base_sigma = data_cfg['base_sigma']
        self.int_sigma = False

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']
        self.ann_info['num_scales'] = data_cfg['num_scales']
        self.ann_info['scale_aware_sigma'] = data_cfg['scale_aware_sigma']

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.with_bbox = data_cfg.get('with_bbox', False)
        self.use_nms = data_cfg.get('use_nms', False)
        self.soft_nms = data_cfg.get('soft_nms', True)
        self.oks_thr = data_cfg.get('oks_thr', 0.9)

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
            if not test_mode:
                self.img_ids = [
                    img_id for img_id in self.img_ids if
                    len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
                ]
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id = self._get_mapping_id_name(
                self.coco.imgs)

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

    def _get_mask(self, anno, idx):
        """Get ignore masks to mask out losses."""
        coco = self.coco
        img_info = coco.loadImgs(self.img_ids[idx])[0]

        m = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)

        for obj in anno:
            if 'segmentation' in obj:
                if obj['iscrowd']:
                    rle = xtcocotools.mask.frPyObjects(obj['segmentation'],
                                                       img_info['height'],
                                                       img_info['width'])
                    m += xtcocotools.mask.decode(rle)
                elif obj['num_keypoints'] == 0:
                    rles = xtcocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'],
                        img_info['width'])
                    for rle in rles:
                        m += xtcocotools.mask.decode(rle)

        return m < 0.5

    @abstractmethod
    def _get_single(self, idx):
        """Get anno for a single image."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, results, *args, **kwargs):
        """Evaluate keypoint results."""

    def prepare_train_img(self, idx):
        """Prepare image for training given the index."""
        results = copy.deepcopy(self._get_single(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Prepare image for testing given the index."""
        results = copy.deepcopy(self._get_single(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def __len__(self):
        """Get dataset length."""
        return len(self.img_ids)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_img(idx)

        return self.prepare_train_img(idx)
