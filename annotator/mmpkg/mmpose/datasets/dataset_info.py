# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


class DatasetInfo:

    def __init__(self, dataset_info):
        self._dataset_info = dataset_info
        self.dataset_name = self._dataset_info['dataset_name']
        self.paper_info = self._dataset_info['paper_info']
        self.keypoint_info = self._dataset_info['keypoint_info']
        self.skeleton_info = self._dataset_info['skeleton_info']
        self.joint_weights = np.array(
            self._dataset_info['joint_weights'], dtype=np.float32)[:, None]

        self.sigmas = np.array(self._dataset_info['sigmas'])

        self._parse_keypoint_info()
        self._parse_skeleton_info()

    def _parse_skeleton_info(self):
        """Parse skeleton information.

        - link_num (int): number of links.
        - skeleton (list((2,))): list of links (id).
        - skeleton_name (list((2,))): list of links (name).
        - pose_link_color (np.ndarray): the color of the link for
            visualization.
        """
        self.link_num = len(self.skeleton_info.keys())
        self.pose_link_color = []

        self.skeleton_name = []
        self.skeleton = []
        for skid in self.skeleton_info.keys():
            link = self.skeleton_info[skid]['link']
            self.skeleton_name.append(link)
            self.skeleton.append([
                self.keypoint_name2id[link[0]], self.keypoint_name2id[link[1]]
            ])
            self.pose_link_color.append(self.skeleton_info[skid].get(
                'color', [255, 128, 0]))
        self.pose_link_color = np.array(self.pose_link_color)

    def _parse_keypoint_info(self):
        """Parse keypoint information.

        - keypoint_num (int): number of keypoints.
        - keypoint_id2name (dict): mapping keypoint id to keypoint name.
        - keypoint_name2id (dict): mapping keypoint name to keypoint id.
        - upper_body_ids (list): a list of keypoints that belong to the
            upper body.
        - lower_body_ids (list): a list of keypoints that belong to the
            lower body.
        - flip_index (list): list of flip index (id)
        - flip_pairs (list((2,))): list of flip pairs (id)
        - flip_index_name (list): list of flip index (name)
        - flip_pairs_name (list((2,))): list of flip pairs (name)
        - pose_kpt_color (np.ndarray): the color of the keypoint for
            visualization.
        """

        self.keypoint_num = len(self.keypoint_info.keys())
        self.keypoint_id2name = {}
        self.keypoint_name2id = {}

        self.pose_kpt_color = []
        self.upper_body_ids = []
        self.lower_body_ids = []

        self.flip_index_name = []
        self.flip_pairs_name = []

        for kid in self.keypoint_info.keys():

            keypoint_name = self.keypoint_info[kid]['name']
            self.keypoint_id2name[kid] = keypoint_name
            self.keypoint_name2id[keypoint_name] = kid
            self.pose_kpt_color.append(self.keypoint_info[kid].get(
                'color', [255, 128, 0]))

            type = self.keypoint_info[kid].get('type', '')
            if type == 'upper':
                self.upper_body_ids.append(kid)
            elif type == 'lower':
                self.lower_body_ids.append(kid)
            else:
                pass

            swap_keypoint = self.keypoint_info[kid].get('swap', '')
            if swap_keypoint == keypoint_name or swap_keypoint == '':
                self.flip_index_name.append(keypoint_name)
            else:
                self.flip_index_name.append(swap_keypoint)
                if [swap_keypoint, keypoint_name] not in self.flip_pairs_name:
                    self.flip_pairs_name.append([keypoint_name, swap_keypoint])

        self.flip_pairs = [[
            self.keypoint_name2id[pair[0]], self.keypoint_name2id[pair[1]]
        ] for pair in self.flip_pairs_name]
        self.flip_index = [
            self.keypoint_name2id[name] for name in self.flip_index_name
        ]
        self.pose_kpt_color = np.array(self.pose_kpt_color)
