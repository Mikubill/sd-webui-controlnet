# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import Dict, Union

import numpy as np
from annotator.mmpkg.mmcv import Config, is_seq_of

from annotator.mmpkg.mmpose.core.post_processing.temporal_filters import build_filter


class Smoother():
    """Smoother to apply temporal smoothing on pose estimation results with a
    filter.

    Note:
        T: The temporal length of the pose sequence
        K: The keypoint number of each target
        C: The keypoint coordinate dimension

    Args:
        filter_cfg (dict | str): The filter config. See example config files in
            `configs/_base_/filters/` for details. Alternatively a config file
            path can be accepted and the config will be loaded.
        keypoint_dim (int): The keypoint coordinate dimension, which is
            also indicated as C. Default: 2
        keypoint_key (str): The dict key of the keypoints in the pose results.
            Default: 'keypoints'
    Example:
        >>> import numpy as np
        >>> # Build dummy pose result
        >>> results = []
        >>> for t in range(10):
        >>>     results_t = []
        >>>     for track_id in range(2):
        >>>         result = {
        >>>             'track_id': track_id,
        >>>             'keypoints': np.random.rand(17, 3)
        >>>         }
        >>>         results_t.append(result)
        >>>     results.append(results_t)
        >>> # Example 1: Smooth multi-frame pose results offline.
        >>> filter_cfg = dict(type='GaussianFilter', window_size=3)
        >>> smoother = Smoother(filter_cfg, keypoint_dim=2)
        >>> smoothed_results = smoother.smooth(results)
        >>> # Example 2: Smooth pose results online frame-by-frame
        >>> filter_cfg = dict(type='GaussianFilter', window_size=3)
        >>> smoother = Smoother(filter_cfg, keypoint_dim=2)
        >>> for result_t in results:
        >>>     smoothed_result_t = smoother.smooth(result_t)
    """

    def __init__(self,
                 filter_cfg: Union[Dict, str],
                 keypoint_dim: int = 2,
                 keypoint_key: str = 'keypoints'):
        if isinstance(filter_cfg, str):
            filter_cfg = Config.fromfile(filter_cfg).filter_cfg
        self.filter_cfg = filter_cfg
        self._filter = build_filter(filter_cfg)
        self.keypoint_dim = keypoint_dim
        self.key = keypoint_key
        self.padding_size = self._filter.window_size - 1
        self.history = {}

    def _get_filter(self):
        fltr = self._filter
        if not fltr.shareable:
            # If the filter is not shareable, build a new filter for the next
            # requires
            self._filter = build_filter(self.filter_cfg)
        return fltr

    def _collate_pose(self, results):
        """Collate the pose results to pose sequences.

        Args:
            results (list[list[dict]]): The pose results of multiple frames.

        Returns:
            dict[str, np.ndarray]: A dict of collated pose sequences, where
            the key is the track_id (in untracked scenario, the target index
            will be used as the track_id), and the value is the pose sequence
            in an array of shape [T, K, C]
        """

        if self._has_track_id(results):
            # If the results have track_id, use it as the target indicator
            results = [{res['track_id']: res
                        for res in results_t} for results_t in results]
            track_ids = results[0].keys()

            for t, results_t in enumerate(results[1:]):
                if results_t.keys() != track_ids:
                    raise ValueError(f'Inconsistent track ids in frame {t+1}')

            collated = {
                id: np.stack([
                    results_t[id][self.key][:, :self.keypoint_dim]
                    for results_t in results
                ])
                for id in track_ids
            }
        else:
            # If the results don't have track_id, use the target index
            # as the target indicator
            n_target = len(results[0])
            for t, results_t in enumerate(results[1:]):
                if len(results_t) != n_target:
                    raise ValueError(
                        f'Inconsistent target number in frame {t+1}: '
                        f'{len(results_t)} vs {n_target}')

            collated = {
                id: np.stack([
                    results_t[id][self.key][:, :self.keypoint_dim]
                    for results_t in results
                ])
                for id in range(n_target)
            }

        return collated

    def _scatter_pose(self, results, poses):
        """Scatter the smoothed pose sequences and use them to update the pose
        results.

        Args:
            results (list[list[dict]]): The original pose results
            poses (dict[str, np.ndarray]): The smoothed pose sequences

        Returns:
            list[list[dict]]: The updated pose results
        """
        updated_results = []
        for t, results_t in enumerate(results):
            updated_results_t = []
            if self._has_track_id(results):
                id2result = ((result['track_id'], result)
                             for result in results_t)
            else:
                id2result = enumerate(results_t)

            for track_id, result in id2result:
                result = copy.deepcopy(result)
                result[self.key][:, :self.keypoint_dim] = poses[track_id][t]
                updated_results_t.append(result)

            updated_results.append(updated_results_t)
        return updated_results

    @staticmethod
    def _has_track_id(results):
        """Check if the pose results contain track_id."""
        return 'track_id' in results[0][0]

    def smooth(self, results):
        """Apply temporal smoothing on pose estimation sequences.

        Args:
            results (list[dict] | list[list[dict]]): The pose results of a
                single frame (non-nested list) or multiple frames (nested
                list). The result of each target is a dict, which should
                contains:

                - track_id (optional, Any): The track ID of the target
                - keypoints (np.ndarray): The keypoint coordinates in [K, C]

        Returns:
            (list[dict] | list[list[dict]]): Temporal smoothed pose results,
            which has the same data structure as the input's.
        """

        # Check if input is empty
        if not (results) or not (results[0]):
            warnings.warn('Smoother received empty result.')
            return results

        # Check input is single frame or sequence
        if is_seq_of(results, dict):
            single_frame = True
            results = [results]
        else:
            assert is_seq_of(results, list)
            single_frame = False

        # Get temporal length of input
        T = len(results)

        # Collate the input results to pose sequences
        poses = self._collate_pose(results)

        # Smooth the pose sequence of each target
        smoothed_poses = {}
        update_history = {}
        for track_id, pose in poses.items():
            if track_id in self.history:
                # For tracked target, get its filter and pose history
                pose_history, pose_filter = self.history[track_id]
                if self.padding_size > 0:
                    # Pad the pose sequence with pose history
                    pose = np.concatenate((pose_history, pose), axis=0)
            else:
                # For new target, build a new filter
                pose_filter = self._get_filter()

            # Update the history information
            if self.padding_size > 0:
                pose_history = pose[-self.padding_size:].copy()
            else:
                pose_history = None
            update_history[track_id] = (pose_history, pose_filter)

            # Smooth the pose sequence with the filter
            smoothed_pose = pose_filter(pose)
            smoothed_poses[track_id] = smoothed_pose[-T:]

        self.history = update_history

        # Scatter the pose sequences back to the format of results
        smoothed_results = self._scatter_pose(results, smoothed_poses)

        # If the input is single frame, remove the nested list to keep the
        # output structure consistent with the input's
        if single_frame:
            smoothed_results = smoothed_results[0]
        return smoothed_results
