# Copyright (c) OpenMMLab. All rights reserved.
import annotator.mmpkg.mmcv as mmcv
import numpy as np
import torch

from annotator.mmpkg.mmpose.core import bbox_xywh2xyxy, bbox_xyxy2xywh
from annotator.mmpkg.mmpose.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CropValidClip:
    """Generate the clip from complete video with valid frames.

    Required keys: 'video', 'modality', 'valid_frames', 'num_frames'.

    Modified keys: 'video', 'valid_frames', 'num_frames'.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Crop the valid part from the video."""
        if 'valid_frames' not in results:
            results['valid_frames'] = [[0, n - 1]
                                       for n in results['num_frames']]
        lengths = [(end - start) for start, end in results['valid_frames']]
        length = min(lengths)
        for i, modal in enumerate(results['modality']):
            start = results['valid_frames'][i][0]
            results['video'][i] = results['video'][i][start:start + length]
            results['num_frames'] = length
        del results['valid_frames']
        if 'bbox' in results:
            results['bbox'] = results['bbox'][start:start + length]
        return results


@PIPELINES.register_module()
class TemporalPooling:
    """Pick frames according to either stride or reference fps.

    Required keys: 'video', 'modality', 'num_frames', 'fps'.

    Modified keys: 'video', 'num_frames'.

    Args:
        length (int): output video length. If unset, the entire video will
            be pooled.
        stride (int): temporal pooling stride. If unset, the stride will be
            computed with video fps and `ref_fps`. If both `stride` and
            `ref_fps` are unset, the stride will be 1.
        ref_fps (int): expected fps of output video. If unset, the video will
            be pooling with `stride`.
    """

    def __init__(self, length: int = -1, stride: int = -1, ref_fps: int = -1):
        self.length = length
        if stride == -1 and ref_fps == -1:
            stride = 1
        elif stride != -1 and ref_fps != -1:
            raise ValueError('`stride` and `ref_fps` can not be assigned '
                             'simultaneously, as they might conflict.')
        self.stride = stride
        self.ref_fps = ref_fps

    def __call__(self, results):
        """Implement data aumentation with random temporal crop."""

        if self.ref_fps > 0 and 'fps' in results:
            assert len(set(results['fps'])) == 1, 'Videos of different '
            'modality have different rate. May be misaligned after pooling.'
            stride = results['fps'][0] // self.ref_fps
            if stride < 1:
                raise ValueError(f'`ref_fps` must be smaller than video '
                                 f"fps {results['fps'][0]}")
        else:
            stride = self.stride

        if self.length < 0:
            length = results['num_frames']
            num_frames = (results['num_frames'] - 1) // stride + 1
        else:
            length = (self.length - 1) * stride + 1
            num_frames = self.length

        diff = length - results['num_frames']
        start = np.random.randint(max(1 - diff, 1))

        for i, modal in enumerate(results['modality']):
            video = results['video'][i]
            if diff > 0:
                video = np.pad(video, ((diff // 2, diff - (diff // 2)),
                                       *(((0, 0), ) * (video.ndim - 1))),
                               'edge')
            results['video'][i] = video[start:start + length:stride]
            assert results['video'][i].shape[0] == num_frames

        results['num_frames'] = num_frames
        if 'bbox' in results:
            results['bbox'] = results['bbox'][start:start + length:stride]
        return results


@PIPELINES.register_module()
class ResizeGivenShortEdge:
    """Resize the video to make its short edge have given length.

    Required keys: 'video', 'modality', 'width', 'height'.

    Modified keys: 'video', 'width', 'height'.
    """

    def __init__(self, length: int = 256):
        self.length = length

    def __call__(self, results):
        """Implement data processing with resize given short edge."""
        for i, modal in enumerate(results['modality']):
            width, height = results['width'][i], results['height'][i]
            video = results['video'][i].transpose(1, 2, 3, 0)
            num_frames = video.shape[-1]
            video = video.reshape(height, width, -1)
            if width < height:
                width, height = self.length, int(self.length * height / width)
            else:
                width, height = int(self.length * width / height), self.length
            video = mmcv.imresize(video,
                                  (width,
                                   height)).reshape(height, width, -1,
                                                    num_frames)
            results['video'][i] = video.transpose(3, 0, 1, 2)
            results['width'][i], results['height'][i] = width, height
        return results


@PIPELINES.register_module()
class MultiFrameBBoxMerge:
    """Compute the union of bboxes in selected frames.

    Required keys: 'bbox'.

    Modified keys: 'bbox'.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        if 'bbox' not in results:
            return results

        bboxes = list(filter(lambda x: len(x), results['bbox']))
        if len(bboxes) == 0:
            bbox_xyxy = np.array(
                (0, 0, results['width'][0] - 1, results['height'][0] - 1))
        else:
            bboxes_xyxy = np.stack([b[0]['bbox'] for b in bboxes])
            bbox_xyxy = np.array((
                bboxes_xyxy[:, 0].min(),
                bboxes_xyxy[:, 1].min(),
                bboxes_xyxy[:, 2].max(),
                bboxes_xyxy[:, 3].max(),
            ))
        results['bbox'] = bbox_xyxy
        return results


@PIPELINES.register_module()
class ResizedCropByBBox:
    """Spatial crop for spatially aligned videos by bounding box.

    Required keys: 'video', 'modality', 'width', 'height', 'bbox'.

    Modified keys: 'video', 'width', 'height'.
    """

    def __init__(self, size, scale=(1, 1), ratio=(1, 1), shift=0):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
        self.scale = scale
        self.ratio = ratio
        self.shift = shift

    def __call__(self, results):
        bbox_xywh = bbox_xyxy2xywh(results['bbox'][None, :])[0]
        length = bbox_xywh[2:].max()
        length = length * np.random.uniform(*self.scale)
        x = bbox_xywh[0] + np.random.uniform(-self.shift, self.shift) * length
        y = bbox_xywh[1] + np.random.uniform(-self.shift, self.shift) * length
        w, h = length, length * np.random.uniform(*self.ratio)

        bbox_xyxy = bbox_xywh2xyxy(np.array([[x, y, w, h]]))[0]
        bbox_xyxy = bbox_xyxy.clip(min=0)
        bbox_xyxy[2] = min(bbox_xyxy[2], results['width'][0])
        bbox_xyxy[3] = min(bbox_xyxy[3], results['height'][0])
        bbox_xyxy = bbox_xyxy.astype(np.int32)

        for i in range(len(results['video'])):
            video = results['video'][i].transpose(1, 2, 3, 0)
            num_frames = video.shape[-1]
            video = video.reshape(video.shape[0], video.shape[1], -1)
            video = mmcv.imcrop(video, bbox_xyxy)
            video = mmcv.imresize(video, self.size)

            results['video'][i] = video.reshape(video.shape[0], video.shape[1],
                                                -1, num_frames)
            results['video'][i] = results['video'][i].transpose(3, 0, 1, 2)
            results['width'][i], results['height'][i] = video.shape[
                1], video.shape[0]

        return results


@PIPELINES.register_module()
class GestureRandomFlip:
    """Data augmentation by randomly horizontal flip the video. The label will
    be alternated simultaneously.

    Required keys: 'video', 'label', 'ann_info'.

    Modified keys: 'video', 'label'.
    """

    def __init__(self, prob=0.5):
        self.flip_prob = prob

    def __call__(self, results):
        flip = np.random.rand() < self.flip_prob
        if flip:
            for i in range(len(results['video'])):
                results['video'][i] = results['video'][i][:, :, ::-1, :]
            for flip_pairs in results['ann_info']['flip_pairs']:
                if results['label'] in flip_pairs:
                    results['label'] = sum(flip_pairs) - results['label']
                    break

        results['flipped'] = flip
        return results


@PIPELINES.register_module()
class VideoColorJitter:
    """Data augmentation with random color transformations.

    Required keys: 'video', 'modality'.

    Modified keys: 'video'.
    """

    def __init__(self, brightness=0, contrast=0):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, results):
        for i, modal in enumerate(results['modality']):
            if modal == 'rgb':
                video = results['video'][i]
                bright = np.random.uniform(
                    max(0, 1 - self.brightness), 1 + self.brightness)
                contrast = np.random.uniform(
                    max(0, 1 - self.contrast), 1 + self.contrast)
                video = mmcv.adjust_brightness(video.astype(np.int32), bright)
                num_frames = video.shape[0]
                video = video.astype(np.uint8).reshape(-1, video.shape[2], 3)
                video = mmcv.adjust_contrast(video, contrast).reshape(
                    num_frames, -1, video.shape[1], 3)
                results['video'][i] = video
        return results


@PIPELINES.register_module()
class RandomAlignedSpatialCrop:
    """Data augmentation with random spatial crop for spatially aligned videos.

    Required keys: 'video', 'modality', 'width', 'height'.

    Modified keys: 'video', 'width', 'height'.
    """

    def __init__(self, length: int = 224):
        self.length = length

    def __call__(self, results):
        """Implement data augmentation with random spatial crop."""
        assert len(set(results['height'])) == 1, \
            f"the heights {results['height']} are not identical."
        assert len(set(results['width'])) == 1, \
            f"the widths {results['width']} are not identical."
        height, width = results['height'][0], results['width'][0]
        for i, modal in enumerate(results['modality']):
            video = results['video'][i].transpose(1, 2, 3, 0)
            num_frames = video.shape[-1]
            video = video.reshape(height, width, -1)
            start_h, start_w = np.random.randint(
                height - self.length + 1), np.random.randint(width -
                                                             self.length + 1)
            video = mmcv.imcrop(
                video,
                np.array((start_w, start_h, start_w + self.length - 1,
                          start_h + self.length - 1)))
            results['video'][i] = video.reshape(self.length, self.length, -1,
                                                num_frames).transpose(
                                                    3, 0, 1, 2)
            results['width'][i], results['height'][
                i] = self.length, self.length
        return results


@PIPELINES.register_module()
class CenterSpatialCrop:
    """Data processing by crop the center region of a video.

    Required keys: 'video', 'modality', 'width', 'height'.

    Modified keys: 'video', 'width', 'height'.
    """

    def __init__(self, length: int = 224):
        self.length = length

    def __call__(self, results):
        """Implement data processing with center crop."""
        for i, modal in enumerate(results['modality']):
            height, width = results['height'][i], results['width'][i]
            video = results['video'][i].transpose(1, 2, 3, 0)
            num_frames = video.shape[-1]
            video = video.reshape(height, width, -1)
            start_h, start_w = (height - self.length) // 2, (width -
                                                             self.length) // 2
            video = mmcv.imcrop(
                video,
                np.array((start_w, start_h, start_w + self.length - 1,
                          start_h + self.length - 1)))
            results['video'][i] = video.reshape(self.length, self.length, -1,
                                                num_frames).transpose(
                                                    3, 0, 1, 2)
            results['width'][i], results['height'][
                i] = self.length, self.length
        return results


@PIPELINES.register_module()
class ModalWiseChannelProcess:
    """Video channel processing according to modality.

    Required keys: 'video', 'modality'.

    Modified keys: 'video'.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Implement channel processing for video array."""
        for i, modal in enumerate(results['modality']):
            if modal == 'rgb':
                results['video'][i] = results['video'][i][..., ::-1]
            elif modal == 'depth':
                if results['video'][i].ndim == 4:
                    results['video'][i] = results['video'][i][..., :1]
                elif results['video'][i].ndim == 3:
                    results['video'][i] = results['video'][i][..., None]
            elif modal == 'flow':
                results['video'][i] = results['video'][i][..., :2]
            else:
                raise ValueError(f'modality {modal} is invalid.')
        return results


@PIPELINES.register_module()
class MultiModalVideoToTensor:
    """Data processing by converting video arrays to pytorch tensors.

    Required keys: 'video', 'modality'.

    Modified keys: 'video'.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Implement data processing similar to ToTensor."""
        for i, modal in enumerate(results['modality']):
            video = results['video'][i].transpose(3, 0, 1, 2)
            results['video'][i] = torch.tensor(
                np.ascontiguousarray(video), dtype=torch.float) / 255.0
        return results


@PIPELINES.register_module()
class VideoNormalizeTensor:
    """Data processing by normalizing video tensors with mean and std.

    Required keys: 'video', 'modality'.

    Modified keys: 'video'.
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, results):
        """Implement data normalization."""
        for i, modal in enumerate(results['modality']):
            if modal == 'rgb':
                video = results['video'][i]
                dim = video.ndim - 1
                video = video - self.mean.view(3, *((1, ) * dim))
                video = video / self.std.view(3, *((1, ) * dim))
                results['video'][i] = video
        return results
