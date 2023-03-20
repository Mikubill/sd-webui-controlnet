# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import subprocess
import tempfile
from typing import List, Optional, Union

from annotator.mmpkg.mmcv.utils import requires_executable


@requires_executable('ffmpeg')
def convert_video(in_file: str,
                  out_file: str,
                  print_cmd: bool = False,
                  pre_options: str = '',
                  **kwargs) -> None:
    """Convert a video with ffmpeg.

    This provides a general api to ffmpeg, the executed command is::

        `ffmpeg -y <pre_options> -i <in_file> <options> <out_file>`

    Options(kwargs) are mapped to ffmpeg commands with the following rules:

    - key=val: "-key val"
    - key=True: "-key"
    - key=False: ""

    Args:
        in_file (str): Input video filename.
        out_file (str): Output video filename.
        pre_options (str): Options appears before "-i <in_file>".
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    options = []
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                options.append(f'-{k}')
        elif k == 'log_level':
            assert v in [
                'quiet', 'panic', 'fatal', 'error', 'warning', 'info',
                'verbose', 'debug', 'trace'
            ]
            options.append(f'-loglevel {v}')
        else:
            options.append(f'-{k} {v}')
    cmd = f'ffmpeg -y {pre_options} -i {in_file} {" ".join(options)} ' \
          f'{out_file}'
    if print_cmd:
        print(cmd)
    subprocess.call(cmd, shell=True)


@requires_executable('ffmpeg')
def resize_video(in_file: str,
                 out_file: str,
                 size: Optional[tuple] = None,
                 ratio: Union[tuple, float, None] = None,
                 keep_ar: bool = False,
                 log_level: str = 'info',
                 print_cmd: bool = False) -> None:
    """Resize a video.

    Args:
        in_file (str): Input video filename.
        out_file (str): Output video filename.
        size (tuple): Expected size (w, h), eg, (320, 240) or (320, -1).
        ratio (tuple or float): Expected resize ratio, (2, 0.5) means
            (w*2, h*0.5).
        keep_ar (bool): Whether to keep original aspect ratio.
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    if size is None and ratio is None:
        raise ValueError('expected size or ratio must be specified')
    if size is not None and ratio is not None:
        raise ValueError('size and ratio cannot be specified at the same time')
    options = {'log_level': log_level}
    if size:
        if not keep_ar:
            options['vf'] = f'scale={size[0]}:{size[1]}'
        else:
            options['vf'] = f'scale=w={size[0]}:h={size[1]}:' \
                            'force_original_aspect_ratio=decrease'
    else:
        if not isinstance(ratio, tuple):
            ratio = (ratio, ratio)
        options['vf'] = f'scale="trunc(iw*{ratio[0]}):trunc(ih*{ratio[1]})"'
    convert_video(in_file, out_file, print_cmd, **options)


@requires_executable('ffmpeg')
def cut_video(in_file: str,
              out_file: str,
              start: Optional[float] = None,
              end: Optional[float] = None,
              vcodec: Optional[str] = None,
              acodec: Optional[str] = None,
              log_level: str = 'info',
              print_cmd: bool = False) -> None:
    """Cut a clip from a video.

    Args:
        in_file (str): Input video filename.
        out_file (str): Output video filename.
        start (None or float): Start time (in seconds).
        end (None or float): End time (in seconds).
        vcodec (None or str): Output video codec, None for unchanged.
        acodec (None or str): Output audio codec, None for unchanged.
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    options = {'log_level': log_level}
    if vcodec is None:
        options['vcodec'] = 'copy'
    if acodec is None:
        options['acodec'] = 'copy'
    if start:
        options['ss'] = start  # type: ignore
    else:
        start = 0
    if end:
        options['t'] = end - start  # type: ignore
    convert_video(in_file, out_file, print_cmd, **options)


@requires_executable('ffmpeg')
def concat_video(video_list: List,
                 out_file: str,
                 vcodec: Optional[str] = None,
                 acodec: Optional[str] = None,
                 log_level: str = 'info',
                 print_cmd: bool = False) -> None:
    """Concatenate multiple videos into a single one.

    Args:
        video_list (list): A list of video filenames
        out_file (str): Output video filename
        vcodec (None or str): Output video codec, None for unchanged
        acodec (None or str): Output audio codec, None for unchanged
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    tmp_filehandler, tmp_filename = tempfile.mkstemp(suffix='.txt', text=True)
    with open(tmp_filename, 'w') as f:
        for filename in video_list:
            f.write(f'file {osp.abspath(filename)}\n')
    options = {'log_level': log_level}
    if vcodec is None:
        options['vcodec'] = 'copy'
    if acodec is None:
        options['acodec'] = 'copy'
    convert_video(
        tmp_filename,
        out_file,
        print_cmd,
        pre_options='-f concat -safe 0',
        **options)
    os.close(tmp_filehandler)
    os.remove(tmp_filename)
