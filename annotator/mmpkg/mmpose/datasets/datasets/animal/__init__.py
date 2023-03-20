# Copyright (c) OpenMMLab. All rights reserved.
from .animal_ap10k_dataset import AnimalAP10KDataset
from .animal_atrw_dataset import AnimalATRWDataset
from .animal_fly_dataset import AnimalFlyDataset
from .animal_horse10_dataset import AnimalHorse10Dataset
from .animal_locust_dataset import AnimalLocustDataset
from .animal_macaque_dataset import AnimalMacaqueDataset
from .animal_pose_dataset import AnimalPoseDataset
from .animal_zebra_dataset import AnimalZebraDataset

__all__ = [
    'AnimalHorse10Dataset', 'AnimalMacaqueDataset', 'AnimalFlyDataset',
    'AnimalLocustDataset', 'AnimalZebraDataset', 'AnimalATRWDataset',
    'AnimalPoseDataset', 'AnimalAP10KDataset'
]
