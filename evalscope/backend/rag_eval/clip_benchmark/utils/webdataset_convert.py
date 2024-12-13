# Convert datasets to webdataset format
import os
import torch
import torch.utils.data
import webdataset
from tqdm import tqdm

from evalscope.backend.rag_eval.clip_benchmark.dataset_builder import DatasetWrapper
from evalscope.backend.rag_eval.utils.tools import PIL_to_bytes, path_to_bytes
from evalscope.utils.logger import get_logger

logger = get_logger()


def convert_dataset(
    dataset,
    split,
    output_folder,
    *,
    transform=None,
    image_format='webp',
    max_count=10_000,
    max_size=1_000_000_000,
    multilabel=False,
    verbose=True,
):
    """
    Convert an iterable `dataset` of (image, label) pairs to webdataset (.tar) format, and store in
       `output_folder/split`.

    Images may be passed in as either:
    * File paths: pass in `transform=path_to_bytes`;
    * PIL images: pass in `transform=PIL_to_bytes(image_format)` where `image_format` is e.g. "webp"; or
    * Raw binary data: use a PyTorch `Dataset` that supports `transform=PIL_to_bytes(image_format)`, and
        pass in `transform=None` here.
        Be sure that the transform is not applied twice.

    Copying image files directly or writing raw binary data is fastest since it allows multiprocessing;
    passing in PIL images will be slower, but should work for any format of dataset.

    Labels must be zero-indexed integers (for multilabel datasets, labels must be arrays/tensors).

    Classnames and zero-shot classification templates can be provided as attributes of the dataset (`.classes`
    and `.templates`) or filled in manually afterward. `dataset.classes` should be a list of strings indexed by
    the labels, and `dataset.templates` should be a list of strings containing `{c}` to specify where classnames
    are to be inserted.
    """
    # Create output directory
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)
    # Multiprocessed dataloader, should work with Dataset or list
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=lambda batch: batch[0],  # No collate, only for multiprocessing
    )
    if verbose:
        try:
            logger.info(f'Dataset size: {len(dataset)}')
        except TypeError:
            logger.info('IterableDataset has no len()')
    # Save classnames
    if hasattr(dataset, 'classes') and dataset.classes:
        classnames_fname = os.path.join(output_folder, 'classnames.txt')
        with open(classnames_fname, 'w') as classnames_file:
            logger.info(*dataset.classes, sep='\n', end='\n', file=classnames_file)
        if verbose:
            logger.info("Saved class names to '%s'" % classnames_fname)
    elif verbose:
        logger.info('WARNING: No class names found')
    # Save zeroshot templates
    if hasattr(dataset, 'templates') and dataset.templates:
        templates_fname = os.path.join(output_folder, 'zeroshot_classification_templates.txt')
        with open(templates_fname, 'w') as templates_file:
            logger.info(*dataset.templates, sep='\n', end='\n', file=templates_file)
        if verbose:
            logger.info("Saved class names to '%s'" % templates_fname)
    elif verbose:
        logger.info('WARNING: No zeroshot classification templates found')
    # Save dataset type
    if multilabel:
        type_fname = os.path.join(output_folder, 'dataset_type.txt')
        with open(type_fname, 'w') as type_file:
            logger.info('multilabel', end='\n', file=type_file)
            if verbose:
                logger.info("Saved dataset type to '%s'" % type_fname)
    # Write to TAR files
    data_fname = os.path.join(output_folder, split, r'%d.tar')
    sink = webdataset.ShardWriter(data_fname, maxcount=max_count, maxsize=max_size)
    nsamples = 0
    label_type = 'npy' if multilabel else 'cls'
    for index, (input, output) in enumerate(tqdm(dataloader, desc='Converting')):
        nsamples += 1
        if isinstance(input, str) and transform is path_to_bytes:
            # If copying file, determine image format from extension
            extension = (os.path.splitext(input)[1].replace('.', '').lower().replace('jpeg', 'jpg') or image_format)
        else:
            extension = image_format
        # Convert label if necessary
        if isinstance(output, torch.Tensor):
            if multilabel:
                output = output.detach().cpu().numpy()
            else:
                output = output.item()
        # Write example
        sink.write({
            '__key__': 's%07d' % index,
            extension: transform(input) if transform else input,
            label_type: output,
        })
    num_shards = sink.shard
    sink.close()
    if verbose:
        logger.info("Saved dataset to '%s'" % data_fname.replace(r'%d', '{0..%d}' % (num_shards - 1)))
    # Save number of shards
    nshards_fname = os.path.join(output_folder, split, 'nshards.txt')
    with open(nshards_fname, 'w') as nshards_file:
        logger.info(num_shards, end='\n', file=nshards_file)
    if verbose:
        logger.info("Saved number of shards = %d to '%s'" % (num_shards, nshards_fname))
    logger.info('Final dataset size:', nsamples)


def convert_retrieval_dataset(
    dataset,
    split,
    output_folder,
    *,
    transform=None,
    image_format='webp',
    max_count=10_000,
    max_size=1_000_000_000,
    verbose=True,
):
    """
    Convert an iterable `dataset` of (image, [caption1, caption2, ...]) pairs to webdataset (.tar) format,
    and store in `output_folder/split`.

    Labels must be lists of strings, with no newlines.

    Read the documentation of `convert_dataset` for more information.
    """
    # Create output directory
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)
    # Multiprocessed dataloader, should work with Dataset or list
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=lambda batch: batch[0],  # No collate, only for multiprocessing
    )
    if verbose:
        try:
            logger.info(f'Dataset size: {len(dataset)}')
        except TypeError:
            logger.info('IterableDataset has no len()')
    # No classnames
    # No zeroshot templates
    # Save dataset type
    type_fname = os.path.join(output_folder, 'dataset_type.txt')
    with open(type_fname, 'w') as type_file:
        logger.info('retrieval', end='\n', file=type_file)
    if verbose:
        logger.info("Saved dataset type to '%s'" % type_fname)
    # Write to TAR files
    data_fname = os.path.join(output_folder, split, r'%d.tar')
    sink = webdataset.ShardWriter(data_fname, maxcount=max_count, maxsize=max_size)
    nsamples = 0
    for index, (input, output) in enumerate(tqdm(dataloader, desc='Converting')):
        nsamples += 1
        if isinstance(input, str) and transform is path_to_bytes:
            # If copying file, determine image format from extension
            extension = (os.path.splitext(input)[1].replace('.', '').lower().replace('jpeg', 'jpg') or image_format)
        else:
            extension = image_format
        sink.write({
            '__key__': 's%07d' % index,
            extension: transform(input) if transform else input,
            'txt': '\n'.join(caption.replace('\n', r'\n') for caption in output),
        })
    num_shards = sink.shard
    sink.close()
    if verbose:
        logger.info("Saved dataset to '%s'" % data_fname.replace(r'%d', '{0..%d}' % (num_shards - 1)))
    # Save number of shards
    nshards_fname = os.path.join(output_folder, split, 'nshards.txt')
    with open(nshards_fname, 'w') as nshards_file:
        logger.info(num_shards, end='\n', file=nshards_file)
    if verbose:
        logger.info("Saved number of shards = %d to '%s'" % (num_shards, nshards_fname))
    logger.info('Final dataset size:', nsamples)


if __name__ == '__main__':
    from modelscope.msdatasets import MsDataset

    splits = ['train', 'validation']
    for split in splits:
        ds = MsDataset.load('modelscope/muge', split=split)
        hf_dataset = ds.to_hf_dataset()
        pytorch_dataset = DatasetWrapper(hf_dataset, image_key='image', text_key='query')
        convert_retrieval_dataset(
            pytorch_dataset,
            split,
            'data/muge',
            transform=PIL_to_bytes('jpg'),
            image_format='jpg',
            max_count=50_000,
        )
