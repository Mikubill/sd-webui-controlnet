# Copyright (c) OpenMMLab. All rights reserved.
from .body3d_h36m_dataset import Body3DH36MDataset
from .body3d_mpi_inf_3dhp_dataset import Body3DMpiInf3dhpDataset
from .body3d_mview_direct_campus_dataset import Body3DMviewDirectCampusDataset
from .body3d_mview_direct_panoptic_dataset import \
    Body3DMviewDirectPanopticDataset
from .body3d_mview_direct_shelf_dataset import Body3DMviewDirectShelfDataset
from .body3d_semi_supervision_dataset import Body3DSemiSupervisionDataset

__all__ = [
    'Body3DH36MDataset', 'Body3DSemiSupervisionDataset',
    'Body3DMpiInf3dhpDataset', 'Body3DMviewDirectPanopticDataset',
    'Body3DMviewDirectShelfDataset', 'Body3DMviewDirectCampusDataset'
]
