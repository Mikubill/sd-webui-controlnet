# ------------------------------------------------------------------------------
# Adapted from https://github.com/princeton-vl/pose-ae-train/
# Original licence: Copyright (c) 2017, umich-vl, under BSD 3-Clause License.
# ------------------------------------------------------------------------------

import numpy as np
import torch
from munkres import Munkres

from annotator.mmpkg.mmpose.core.evaluation import post_dark_udp


def _py_max_match(scores):
    """Apply munkres algorithm to get the best match.

    Args:
        scores(np.ndarray): cost matrix.

    Returns:
        np.ndarray: best match.
    """
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(int)
    return tmp


def _match_by_tag(inp, params):
    """Match joints by tags. Use Munkres algorithm to calculate the best match
    for keypoints grouping.

    Note:
        number of keypoints: K
        max number of people in an image: M (M=30 by default)
        dim of tags: L
            If use flip testing, L=2; else L=1.

    Args:
        inp(tuple):
            tag_k (np.ndarray[KxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[KxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[KxM]): top k value of the
                feature maps per keypoint.
        params(Params): class Params().

    Returns:
        np.ndarray: result of pose groups.
    """
    assert isinstance(params, _Params), 'params should be class _Params()'

    tag_k, loc_k, val_k = inp

    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]),
                        dtype=np.float32)

    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]

        tags = tag_k[idx]
        joints = np.concatenate((loc_k[idx], val_k[idx, :, None], tags), 1)
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]  # shape: [M, L]
        joints = joints[mask]  # shape: [M, 3 + L], 3: x, y, val

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            # shape: [M]
            grouped_keys = list(joint_dict.keys())
            if params.ignore_too_much:
                grouped_keys = grouped_keys[:params.max_num_people]
            # shape: [M, L]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            # shape: [M, M, L]
            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            # shape: [M, M]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (diff_normed,
                     np.zeros((num_added, num_added - num_grouped),
                              dtype=np.float32) + 1e10),
                    axis=1)

            pairs = _py_max_match(diff_normed)
            for row, col in pairs:
                if (row < num_added and col < num_grouped
                        and diff_saved[row][col] < params.tag_threshold):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    joint_dict_keys = list(joint_dict.keys())
    if params.ignore_too_much:
        # The new person joints beyond the params.max_num_people will be
        # ignored, for the dict is in ordered when python > 3.6 version.
        joint_dict_keys = joint_dict_keys[:params.max_num_people]
    results = np.array([joint_dict[i]
                        for i in joint_dict_keys]).astype(np.float32)
    return results


class _Params:
    """A class of parameter.

    Args:
        cfg(Config): config.
    """

    def __init__(self, cfg):
        self.num_joints = cfg['num_joints']
        self.max_num_people = cfg['max_num_people']

        self.detection_threshold = cfg['detection_threshold']
        self.tag_threshold = cfg['tag_threshold']
        self.use_detection_val = cfg['use_detection_val']
        self.ignore_too_much = cfg['ignore_too_much']

        if self.num_joints == 17:
            self.joint_order = [
                i - 1 for i in
                [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]
        else:
            self.joint_order = list(np.arange(self.num_joints))


class HeatmapParser:
    """The heatmap parser for post processing."""

    def __init__(self, cfg):
        self.params = _Params(cfg)
        self.tag_per_joint = cfg['tag_per_joint']
        self.pool = torch.nn.MaxPool2d(cfg['nms_kernel'], 1,
                                       cfg['nms_padding'])
        self.use_udp = cfg.get('use_udp', False)
        self.score_per_joint = cfg.get('score_per_joint', False)

    def nms(self, heatmaps):
        """Non-Maximum Suppression for heatmaps.

        Args:
            heatmap(torch.Tensor): Heatmaps before nms.

        Returns:
            torch.Tensor: Heatmaps after nms.
        """

        maxm = self.pool(heatmaps)
        maxm = torch.eq(maxm, heatmaps).float()
        heatmaps = heatmaps * maxm

        return heatmaps

    def match(self, tag_k, loc_k, val_k):
        """Group keypoints to human poses in a batch.

        Args:
            tag_k (np.ndarray[NxKxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[NxKxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[NxKxM]): top k value of the
                feature maps per keypoint.

        Returns:
            list
        """

        def _match(x):
            return _match_by_tag(x, self.params)

        return list(map(_match, zip(tag_k, loc_k, val_k)))

    def top_k(self, heatmaps, tags):
        """Find top_k values in an image.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            max number of people: M
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW])
            tags (torch.Tensor[NxKxHxWxL])

        Returns:
            dict: A dict containing top_k values.

            - tag_k (np.ndarray[NxKxMxL]):
                tag corresponding to the top k values of
                feature map per keypoint.
            - loc_k (np.ndarray[NxKxMx2]):
                top k location of feature map per keypoint.
            - val_k (np.ndarray[NxKxM]):
                top k value of feature map per keypoint.
        """
        heatmaps = self.nms(heatmaps)
        N, K, H, W = heatmaps.size()
        heatmaps = heatmaps.view(N, K, -1)
        val_k, ind = heatmaps.topk(self.params.max_num_people, dim=2)

        tags = tags.view(tags.size(0), tags.size(1), W * H, -1)
        if not self.tag_per_joint:
            tags = tags.expand(-1, self.params.num_joints, -1, -1)

        tag_k = torch.stack(
            [torch.gather(tags[..., i], 2, ind) for i in range(tags.size(3))],
            dim=3)

        x = ind % W
        y = ind // W

        ind_k = torch.stack((x, y), dim=3)

        results = {
            'tag_k': tag_k.cpu().numpy(),
            'loc_k': ind_k.cpu().numpy(),
            'val_k': val_k.cpu().numpy()
        }

        return results

    @staticmethod
    def adjust(results, heatmaps):
        """Adjust the coordinates for better accuracy.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            results (list(np.ndarray)): Keypoint predictions.
            heatmaps (torch.Tensor[NxKxHxW]): Heatmaps.
        """
        _, _, H, W = heatmaps.shape
        for batch_id, people in enumerate(results):
            for people_id, people_i in enumerate(people):
                for joint_id, joint in enumerate(people_i):
                    if joint[2] > 0:
                        x, y = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = heatmaps[batch_id][joint_id]
                        if tmp[min(H - 1, yy + 1), xx] > tmp[max(0, yy - 1),
                                                             xx]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[yy, min(W - 1, xx + 1)] > tmp[yy,
                                                             max(0, xx - 1)]:
                            x += 0.25
                        else:
                            x -= 0.25
                        results[batch_id][people_id, joint_id,
                                          0:2] = (x + 0.5, y + 0.5)
        return results

    @staticmethod
    def refine(heatmap, tag, keypoints, use_udp=False):
        """Given initial keypoint predictions, we identify missing joints.

        Note:
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmap: np.ndarray(K, H, W).
            tag: np.ndarray(K, H, W) |  np.ndarray(K, H, W, L)
            keypoints: np.ndarray of size (K, 3 + L)
                        last dim is (x, y, score, tag).
            use_udp: bool-unbiased data processing

        Returns:
            np.ndarray: The refined keypoints.
        """

        K, H, W = heatmap.shape
        if len(tag.shape) == 3:
            tag = tag[..., None]

        tags = []
        for i in range(K):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(int)
                x = np.clip(x, 0, W - 1)
                y = np.clip(y, 0, H - 1)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        results = []

        for _heatmap, _tag in zip(heatmap, tag):
            # distance of all tag values with mean tag of
            # current detected people
            distance_tag = (((_tag -
                              prev_tag[None, None, :])**2).sum(axis=2)**0.5)
            norm_heatmap = _heatmap - np.round(distance_tag)

            # find maximum position
            y, x = np.unravel_index(np.argmax(norm_heatmap), _heatmap.shape)
            xx = x.copy()
            yy = y.copy()
            # detection score at maximum position
            val = _heatmap[y, x]
            if not use_udp:
                # offset by 0.5
                x += 0.5
                y += 0.5

            # add a quarter offset
            if _heatmap[yy, min(W - 1, xx + 1)] > _heatmap[yy, max(0, xx - 1)]:
                x += 0.25
            else:
                x -= 0.25

            if _heatmap[min(H - 1, yy + 1), xx] > _heatmap[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            results.append((x, y, val))
        results = np.array(results)

        if results is not None:
            for i in range(K):
                # add keypoint if it is not detected
                if results[i, 2] > 0 and keypoints[i, 2] == 0:
                    keypoints[i, :3] = results[i, :3]

        return keypoints

    def parse(self, heatmaps, tags, adjust=True, refine=True):
        """Group keypoints into poses given heatmap and tag.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW]): model output heatmaps.
            tags (torch.Tensor[NxKxHxWxL]): model output tagmaps.

        Returns:
            tuple: A tuple containing keypoint grouping results.

            - results (list(np.ndarray)): Pose results.
            - scores (list/list(np.ndarray)): Score of people.
        """
        results = self.match(**self.top_k(heatmaps, tags))

        if adjust:
            if self.use_udp:
                for i in range(len(results)):
                    if results[i].shape[0] > 0:
                        results[i][..., :2] = post_dark_udp(
                            results[i][..., :2].copy(), heatmaps[i:i + 1, :])
            else:
                results = self.adjust(results, heatmaps)

        if self.score_per_joint:
            scores = [i[:, 2] for i in results[0]]
        else:
            scores = [i[:, 2].mean() for i in results[0]]

        if refine:
            results = results[0]
            # for every detected person
            for i in range(len(results)):
                heatmap_numpy = heatmaps[0].cpu().numpy()
                tag_numpy = tags[0].cpu().numpy()
                if not self.tag_per_joint:
                    tag_numpy = np.tile(tag_numpy,
                                        (self.params.num_joints, 1, 1, 1))
                results[i] = self.refine(
                    heatmap_numpy, tag_numpy, results[i], use_udp=self.use_udp)
            results = [results]

        return results, scores


class HeatmapOffsetParser:
    """The heatmap&offset parser for post processing."""

    def __init__(self, cfg):
        super(HeatmapOffsetParser, self).__init__()

        self.num_joints = cfg['num_joints']
        self.keypoint_threshold = cfg['keypoint_threshold']
        self.max_num_people = cfg['max_num_people']

        # init pooling layer
        kernel_size = cfg.get('max_pool_kernel', 5)
        self.pool = torch.nn.MaxPool2d(kernel_size, 1, kernel_size // 2)

    def _offset_to_pose(self, offsets):
        """Convert offset maps to pose maps.

        Note:
            batch size: N
            number of keypoints: K
            offset maps height: H
            offset maps width: W

        Args:
            offsets (torch.Tensor[NxKxHxW]): model output offset maps.

        Returns:
            torch.Tensor[NxKxHxW]: A tensor containing pose for each pixel.
        """
        h, w = offsets.shape[-2:]
        offsets = offsets.view(self.num_joints, -1, h, w)

        # generate regular coordinates
        x = torch.arange(0, offsets.shape[-1]).float()
        y = torch.arange(0, offsets.shape[-2]).float()
        y, x = torch.meshgrid(y, x)
        regular_coords = torch.stack((x, y), dim=0).unsqueeze(0)

        posemaps = regular_coords.to(offsets) - offsets
        posemaps = posemaps.view(1, -1, h, w)
        return posemaps

    def _get_maximum_from_heatmap(self, heatmap):
        """Find local maximum of heatmap to localize instances.

        Note:
            batch size: N
            heatmap height: H
            heatmap width: W

        Args:
            heatmap (torch.Tensor[Nx1xHxW]): model output center heatmap.

        Returns:
            tuple: A tuple containing instances detection results.

            - pos_idx (torch.Tensor): Index of pixels which have detected
                instances.
            - score (torch.Tensor): Score of detected instances.
        """
        assert heatmap.size(0) == 1 and heatmap.size(1) == 1
        max_map = torch.eq(heatmap, self.pool(heatmap)).float()
        heatmap = heatmap * max_map
        score = heatmap.view(-1)

        score, pos_idx = score.topk(self.max_num_people)
        mask = score > self.keypoint_threshold
        score = score[mask]
        pos_idx = pos_idx[mask]
        return pos_idx, score

    def decode(self, heatmaps, offsets):
        """Convert center heatmaps and offset maps to poses.

        Note:
            batch size: N
            number of keypoints: K
            offset maps height: H
            offset maps width: W

        Args:
            heatmaps (torch.Tensor[Nx(1+K)xHxW]): model output heatmaps.
            offsets (torch.Tensor[NxKxHxW]): model output offset maps.

        Returns:
            torch.Tensor[NxKx4]: A tensor containing predicted pose and
                score for each instance.
        """

        posemap = self._offset_to_pose(offsets)
        inst_indexes, inst_scores = self._get_maximum_from_heatmap(
            heatmaps[:, :1])

        poses = posemap.view(posemap.size(1), -1)[..., inst_indexes]
        poses = poses.view(self.num_joints, 2, -1).permute(2, 0,
                                                           1).contiguous()
        inst_scores = inst_scores.unsqueeze(1).unsqueeze(2).expand(
            poses.size())
        poses = torch.cat((poses, inst_scores), dim=2)
        return poses.clone()

    def refine_score(self, heatmaps, poses):
        """Refine instance scores with keypoint heatmaps.

        Note:
            batch size: N
            number of keypoints: K
            offset maps height: H
            offset maps width: W

        Args:
            heatmaps (torch.Tensor[Nx(1+K)xHxW]): model output heatmaps.
            poses (torch.Tensor[NxKx4]): decoded pose and score for each
                instance.

        Returns:
            torch.Tensor[NxKx4]: poses with refined scores.
        """
        normed_poses = poses.unsqueeze(0).permute(2, 0, 1, 3).contiguous()
        normed_poses = torch.cat((
            normed_poses.narrow(3, 0, 1) / (heatmaps.size(3) - 1) * 2 - 1,
            normed_poses.narrow(3, 1, 1) / (heatmaps.size(2) - 1) * 2 - 1,
        ),
                                 dim=3)
        kpt_scores = torch.nn.functional.grid_sample(
            heatmaps[:, 1:].view(self.num_joints, 1, heatmaps.size(2),
                                 heatmaps.size(3)),
            normed_poses,
            padding_mode='border').view(self.num_joints, -1)
        kpt_scores = kpt_scores.transpose(0, 1).contiguous()

        # scores only from keypoint heatmaps
        poses[..., 3] = kpt_scores
        # combine center and keypoint heatmaps
        poses[..., 2] = poses[..., 2] * kpt_scores

        return poses
