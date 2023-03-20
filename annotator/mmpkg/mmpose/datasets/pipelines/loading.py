# Copyright (c) OpenMMLab. All rights reserved.
import annotator.mmpkg.mmcv as mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Loading image(s) from file.

    Required key: "image_file".

    Added key: "img".

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _read_image(self, path):
        img_bytes = self.file_client.get(path)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if img is None:
            raise ValueError(f'Fail to read {path}')
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    @staticmethod
    def _bgr2rgb(img):
        if img.ndim == 3:
            return mmcv.bgr2rgb(img)
        elif img.ndim == 4:
            return np.concatenate([mmcv.bgr2rgb(img_) for img_ in img], axis=0)
        else:
            raise ValueError('results["img"] has invalid shape '
                             f'{img.shape}')

    def __call__(self, results):
        """Loading image(s) from file."""
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        image_file = results.get('image_file', None)

        if isinstance(image_file, (list, tuple)):
            # Load images from a list of paths
            results['img'] = [self._read_image(path) for path in image_file]
        elif image_file is not None:
            # Load single image from path
            results['img'] = self._read_image(image_file)
        else:
            if 'img' not in results:
                # If `image_file`` is not in results, check the `img` exists
                # and format the image. This for compatibility when the image
                # is manually set outside the pipeline.
                raise KeyError('Either `image_file` or `img` should exist in '
                               'results.')
            if isinstance(results['img'], (list, tuple)):
                assert isinstance(results['img'][0], np.ndarray)
            else:
                assert isinstance(results['img'], np.ndarray)
            if self.color_type == 'color' and self.channel_order == 'rgb':
                # The original results['img'] is assumed to be image(s) in BGR
                # order, so we convert the color according to the arguments.
                if isinstance(results['img'], (list, tuple)):
                    results['img'] = [
                        self._bgr2rgb(img) for img in results['img']
                    ]
                else:
                    results['img'] = self._bgr2rgb(results['img'])
            results['image_file'] = None

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadVideoFromFile:
    """Loading video(s) from file.

    Required key: "video_file".

    Added key: "video".

    Args:
        to_float32 (bool): Whether to convert the loaded video to a float32
            numpy array. If set to False, the loaded video is an uint8 array.
            Defaults to False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _read_video(self, path):
        container = mmcv.VideoReader(path)
        sample = dict(
            height=int(container.height),
            width=int(container.width),
            fps=int(container.fps),
            num_frames=int(container.frame_cnt),
            video=[])
        for _ in range(container.frame_cnt):
            sample['video'].append(container.read())
        sample['video'] = np.stack(sample['video'], axis=0)
        return sample

    def __call__(self, results):
        """Loading video(s) from file."""
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        video_file = results.get('video_file', None)

        if isinstance(video_file, (list, tuple)):
            # Load videos from a list of paths
            for path in video_file:
                video = self._read_video(path)
                for key in video:
                    results[key].append(video[key])
        elif video_file is not None:
            # Load single video from path
            results.update(self._read_video(video_file))
        else:
            if 'video' not in results:
                # If `video_file`` is not in results, check the `video` exists
                # and format the image. This for compatibility when the image
                # is manually set outside the pipeline.
                raise KeyError('Either `video_file` or `video` should exist '
                               'in results.')
            if isinstance(results['video'], (list, tuple)):
                assert isinstance(results['video'][0], np.ndarray)
            else:
                assert isinstance(results['video'], np.ndarray)
                results['video'] = [results['video']]

            results['num_frames'] = [v.shape[0] for v in results['video']]
            results['height'] = [v.shape[1] for v in results['video']]
            results['width'] = [v.shape[2] for v in results['video']]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'file_client_args={self.file_client_args})')
        return repr_str
