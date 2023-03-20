# Copyright (c) OpenMMLab. All rights reserved.
from .face_300w_dataset import Face300WDataset
from .face_aflw_dataset import FaceAFLWDataset
from .face_coco_wholebody_dataset import FaceCocoWholeBodyDataset
from .face_cofw_dataset import FaceCOFWDataset
from .face_wflw_dataset import FaceWFLWDataset

__all__ = [
    'Face300WDataset', 'FaceAFLWDataset', 'FaceWFLWDataset', 'FaceCOFWDataset',
    'FaceCocoWholeBodyDataset'
]
