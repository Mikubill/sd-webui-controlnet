# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping, Sequence
from functools import partial

import poptorch
from torch.utils.data.dataloader import default_collate

from annotator.mmpkg.mmcv.parallel import DataContainer


def collate(batch, samples_per_gpu=1):
    """Put each data field into a tensor/DataContainer with outer dimension
    batch size.

    TODO support for
    :type:`~mmcv.parallel.DataContainer`. Currently, it will be ignored.
    There are 3 cases.

    1. cpu_only = True, e.g., meta data.
    2. cpu_only = False, stack = True, e.g., images tensors.
    3. cpu_only = False, stack = False, e.g., gt bboxes.
    """

    if not isinstance(batch, Sequence):
        raise TypeError(
            f'`batch` should be a sequence, but got {type(batch)}.')

    if isinstance(batch[0], DataContainer):
        # TODO `DataContainer` will be supported in the future.
        raise TypeError('DataContainer is not supported in ipu data loader.')
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        collated_batch = []
        for samples in transposed:
            if not isinstance(samples[0], DataContainer):
                # At present, we will skip the processing of datacontainer,
                # which will reduce the performance of IPU DataLoder
                collated_batch.append(collate(samples, samples_per_gpu))
        return collated_batch
    elif isinstance(batch[0], Mapping):
        collated_batch = {}
        for key in batch[0]:
            if not isinstance(batch[0][key], DataContainer):
                # At present, we will skip the processing of datacontainer,
                # which will reduce the performance of IPU DataLoder
                collated_batch[key] = collate([d[key] for d in batch])
        return collated_batch
    else:
        return default_collate(batch)


class IPUDataLoader(poptorch.DataLoader):
    """Thin wrapper of `torch.utils.data.DataLoader`.

    Compared with the pytorch DataLoder, this DataLoder changes the way of
    calculation of batch size and adds the AsynchronousDataAccessor to
    load and release data faster in cpu mode.

    If this data loader is used in a distributed execution environment, it will
    ensure that each process uses a different subset of the dataset, providing
    you first call ``options.randomSeed(N)`` with an integer N which is the
    same across all hosts.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to get the data from.
        options (poptorch.Options): Options that will be used to compile
            and run the model.
        batch_size (int, optional): This is the batch size in the conventional
            sense of being the size that runs through an operation in the model
            at any given time.
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main
            process. (default: ``0``)
        drop_last (bool, optional): If True and the number of elements in the
            dataset is not a multiple of the combined batch size then the
            incomplete batch at the end will be dropped.
        persistent_workers (bool, optional): Re-use workers between
            iterations if True.
        auto_distributed_partitioning (bool, optional): If True, partitions the
            dataset for distributed execution automatically. Otherwise, it is
            assumed that partitioning has been handled manually.
        mode (poptorch.DataLoaderMode, optional): If `DataLoaderMode.Async`,
            uses an :py:class:`~poptorch.AsynchronousDataAccessor` to access
            the dataset. If `DataLoaderMode.Sync`, accesses the dataset
            synchronously.
        async_options (Dict[str, Any], optional): Options to pass to
            :py:class:`~poptorch.AsynchronousDataAccessor`.
        rebatched_worker_size (int, optional): When using AsyncRebatched: batch
            size of the tensors loaded by the workers.
            Default to the combined batch size.
            If specified the ``rebatched_worker_size`` must be less than
            or equal to the combined batch size.
        kwargs (Dict[str, Any], optional): Other options to pass to PyTorch's
            ``DataLoader`` constructor.
    """

    def __init__(self,
                 dataset,
                 options,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0,
                 drop_last=True,
                 persistent_workers=True,
                 auto_distributed_partitioning=True,
                 mode='sync',
                 async_options=None,
                 rebatched_worker_size=None,
                 **kwargs):
        """Lazy init:

        In many frameworks, the dataloader will be constructed before the
        initialization of the ipu options, so the lazy init method is used
        here, and the real initialization will not be done until the dataloader
        needs to be used and the options are input.
        """
        # lazy init: sometimes, we cannot get IPU options when build data
        #            loader
        self.kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'drop_last': drop_last,
            'persistent_workers': persistent_workers,
            'auto_distributed_partitioning': auto_distributed_partitioning,
            'mode': mode,
            'collate_fn': partial(collate, samples_per_gpu=batch_size),
            'async_options': async_options,
            'rebatched_worker_size': rebatched_worker_size,
            **kwargs
        }
        self.dataset = dataset
        self.initialized = False
        if options:
            self.init(options=options)

    def init(self, options, **kwargs):
        if not self.initialized:
            kwargs = {**self.kwargs, **kwargs, 'options': options}
            if kwargs['mode'] == 'sync':
                kwargs['mode'] = poptorch.DataLoaderMode.Sync
            elif kwargs['mode'] == 'async':
                kwargs['mode'] = poptorch.DataLoaderMode.AsyncRebatched
                if kwargs['async_options'] is None:
                    kwargs['async_options'] = {
                        'load_indefinitely': True,
                        'buffer_size': 8
                    }
                if kwargs['rebatched_worker_size'] is None:
                    kwargs['rebatched_worker_size'] = 128
            super().__init__(**kwargs)
            self.initialized = True

        return self
