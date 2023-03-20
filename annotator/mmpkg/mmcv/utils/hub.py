# Copyright (c) OpenMMLab. All rights reserved.
# The 1.6 release of PyTorch switched torch.save to use a new zipfile-based
# file format. It will cause RuntimeError when a checkpoint was saved in
# torch >= 1.6.0 but loaded in torch < 1.7.0.
# More details at https://github.com/open-mmlab/mmpose/issues/904
from .parrots_wrapper import TORCH_VERSION
from .path import mkdir_or_exist
from .version_utils import digit_version

if TORCH_VERSION != 'parrots' and digit_version(TORCH_VERSION) < digit_version(
        '1.7.0'):
    # Modified from https://github.com/pytorch/pytorch/blob/master/torch/hub.py
    import os
    import sys
    import warnings
    import zipfile
    from urllib.parse import urlparse

    import torch
    from torch.hub import HASH_REGEX, _get_torch_home, download_url_to_file

    # Hub used to support automatically extracts from zipfile manually
    # compressed by users. The legacy zip format expects only one file from
    # torch.save() < 1.6 in the zip. We should remove this support since
    # zipfile is now default zipfile format for torch.save().
    def _is_legacy_zip_format(filename):
        if zipfile.is_zipfile(filename):
            infolist = zipfile.ZipFile(filename).infolist()
            return len(infolist) == 1 and not infolist[0].is_dir()
        return False

    def _legacy_zip_load(filename, model_dir, map_location):
        warnings.warn(
            'Falling back to the old format < 1.6. This support will'
            ' be deprecated in favor of default zipfile format '
            'introduced in 1.6. Please redo torch.save() to save it '
            'in the new zipfile format.', DeprecationWarning)
        # Note: extractall() defaults to overwrite file if exists. No need to
        #       clean up beforehand. We deliberately don't handle tarfile here
        #       since our legacy serialization format was in tar.
        #       E.g. resnet18-5c106cde.pth which is widely used.
        with zipfile.ZipFile(filename) as f:
            members = f.infolist()
            if len(members) != 1:
                raise RuntimeError(
                    'Only one file(not dir) is allowed in the zipfile')
            f.extractall(model_dir)
            extraced_name = members[0].filename
            extracted_file = os.path.join(model_dir, extraced_name)
        return torch.load(extracted_file, map_location=map_location)

    def load_url(url,
                 model_dir=None,
                 map_location=None,
                 progress=True,
                 check_hash=False,
                 file_name=None):
        r"""Loads the Torch serialized object at the given URL.

        If downloaded file is a zip file, it will be automatically decompressed

        If the object is already present in `model_dir`, it's deserialized and
        returned.
        The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
        ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

        Args:
            url (str): URL of the object to download
            model_dir (str, optional): directory in which to save the object
            map_location (optional): a function or a dict specifying how to
                remap storage locations (see torch.load)
            progress (bool, optional): whether or not to display a progress bar
                to stderr. Default: True
            check_hash(bool, optional): If True, the filename part of the URL
                should follow the naming convention ``filename-<sha256>.ext``
                where ``<sha256>`` is the first eight or more digits of the
                SHA256 hash of the contents of the file. The hash is used to
                ensure unique names and to verify the contents of the file.
                Default: False
            file_name (str, optional): name for the downloaded file. Filename
                from ``url`` will be used if not set. Default: None.

        Example:
            >>> url = ('https://s3.amazonaws.com/pytorch/models/resnet18-5c106'
            ...        'cde.pth')
            >>> state_dict = torch.hub.load_state_dict_from_url(url)
        """
        # Issue warning to move data if old env is set
        if os.getenv('TORCH_MODEL_ZOO'):
            warnings.warn(
                'TORCH_MODEL_ZOO is deprecated, please use env '
                'TORCH_HOME instead', DeprecationWarning)

        if model_dir is None:
            torch_home = _get_torch_home()
            model_dir = os.path.join(torch_home, 'checkpoints')

        mkdir_or_exist(model_dir)

        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        if file_name is not None:
            filename = file_name
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(
                url, cached_file))
            hash_prefix = None
            if check_hash:
                r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
                hash_prefix = r.group(1) if r else None
            download_url_to_file(
                url, cached_file, hash_prefix, progress=progress)

        if _is_legacy_zip_format(cached_file):
            return _legacy_zip_load(cached_file, model_dir, map_location)

        try:
            return torch.load(cached_file, map_location=map_location)
        except RuntimeError as error:
            if digit_version(TORCH_VERSION) < digit_version('1.5.0'):
                warnings.warn(
                    f'If the error is the same as "{cached_file} is a zip '
                    'archive (did you mean to use torch.jit.load()?)", you can'
                    ' upgrade your torch to 1.5.0 or higher (current torch '
                    f'version is {TORCH_VERSION}). The error was raised '
                    ' because the checkpoint was saved in torch>=1.6.0 but '
                    'loaded in torch<1.5.')
            raise error
else:
    from torch.utils.model_zoo import load_url  # type: ignore # noqa: F401
