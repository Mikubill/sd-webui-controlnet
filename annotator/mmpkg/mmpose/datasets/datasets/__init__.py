# Copyright (c) OpenMMLab. All rights reserved.
from ...deprecated import (TopDownFreiHandDataset, TopDownOneHand10KDataset,
                           TopDownPanopticDataset)
from .animal import (AnimalATRWDataset, AnimalFlyDataset, AnimalHorse10Dataset,
                     AnimalLocustDataset, AnimalMacaqueDataset,
                     AnimalPoseDataset, AnimalZebraDataset)
from .body3d import (Body3DH36MDataset, Body3DMviewDirectCampusDataset,
                     Body3DMviewDirectPanopticDataset,
                     Body3DMviewDirectShelfDataset)
from .bottom_up import (BottomUpAicDataset, BottomUpCocoDataset,
                        BottomUpCocoWholeBodyDataset, BottomUpCrowdPoseDataset,
                        BottomUpMhpDataset)
from .face import (Face300WDataset, FaceAFLWDataset, FaceCocoWholeBodyDataset,
                   FaceCOFWDataset, FaceWFLWDataset)
from .fashion import DeepFashionDataset
from .gesture import NVGestureDataset
from .hand import (FreiHandDataset, HandCocoWholeBodyDataset,
                   InterHand2DDataset, InterHand3DDataset, OneHand10KDataset,
                   PanopticDataset)
from .mesh import (MeshAdversarialDataset, MeshH36MDataset, MeshMixDataset,
                   MoshDataset)
from .top_down import (TopDownAicDataset, TopDownCocoDataset,
                       TopDownCocoWholeBodyDataset, TopDownCrowdPoseDataset,
                       TopDownH36MDataset, TopDownHalpeDataset,
                       TopDownJhmdbDataset, TopDownMhpDataset,
                       TopDownMpiiDataset, TopDownMpiiTrbDataset,
                       TopDownOCHumanDataset, TopDownPoseTrack18Dataset,
                       TopDownPoseTrack18VideoDataset)

__all__ = [
    'TopDownCocoDataset', 'BottomUpCocoDataset', 'BottomUpMhpDataset',
    'BottomUpAicDataset', 'BottomUpCocoWholeBodyDataset', 'TopDownMpiiDataset',
    'TopDownMpiiTrbDataset', 'OneHand10KDataset', 'PanopticDataset',
    'HandCocoWholeBodyDataset', 'FreiHandDataset', 'InterHand2DDataset',
    'InterHand3DDataset', 'TopDownOCHumanDataset', 'TopDownAicDataset',
    'TopDownCocoWholeBodyDataset', 'MeshH36MDataset', 'MeshMixDataset',
    'MoshDataset', 'MeshAdversarialDataset', 'TopDownCrowdPoseDataset',
    'BottomUpCrowdPoseDataset', 'TopDownFreiHandDataset',
    'TopDownOneHand10KDataset', 'TopDownPanopticDataset',
    'TopDownPoseTrack18Dataset', 'TopDownJhmdbDataset', 'TopDownMhpDataset',
    'DeepFashionDataset', 'Face300WDataset', 'FaceAFLWDataset',
    'FaceWFLWDataset', 'FaceCOFWDataset', 'FaceCocoWholeBodyDataset',
    'Body3DH36MDataset', 'AnimalHorse10Dataset', 'AnimalMacaqueDataset',
    'AnimalFlyDataset', 'AnimalLocustDataset', 'AnimalZebraDataset',
    'AnimalATRWDataset', 'AnimalPoseDataset', 'TopDownH36MDataset',
    'TopDownHalpeDataset', 'TopDownPoseTrack18VideoDataset',
    'Body3DMviewDirectPanopticDataset', 'Body3DMviewDirectShelfDataset',
    'Body3DMviewDirectCampusDataset', 'NVGestureDataset'
]
