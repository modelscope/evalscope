import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from evalscope.utils.logger import get_logger

logger = get_logger()


def build_dataset(
    dataset_name,
    root=None,
    transform=None,
    split='test',
    wds_cache_dir=None,
    **kwargs,
):
    """
    Main function to use in order to build a dataset instance,

    dataset_name: str
        name of the dataset

    root: str
        root folder where the dataset is downloaded and stored. can be shared among datasets.

    transform: torchvision transform applied to images

    split: str
        split to use, depending on the dataset can have different options.
        In general, `train` and `test` are available.
        For specific splits, please look at the corresponding dataset.

    custom_classname_file: str or None
        Custom classname file where keys are dataset names and values are list of classnames.

    custom_template_file: str or None
        Custom template file where keys are dataset names and values are list of prompts, or dicts
        where keys are classnames and values are class-specific prompts.

    """

    if dataset_name == 'dummy':
        ds = Dummy()
    elif dataset_name == 'custom':
        ds = build_custom_dataset(dataset_name, data_dir=root, transform=transform)
    else:
        # WebDataset support using `webdataset` library
        ds = build_wds_dataset(
            dataset_name,
            transform=transform,
            split=split,
            data_dir=root,
            cache_dir=wds_cache_dir,
        )

    return ds


class Dummy:

    def __init__(self):
        self.classes = ['blank image', 'noisy image']

    def __getitem__(self, i):
        return torch.zeros(3, 224, 224), 0

    def __len__(self):
        return 1


class DatasetWrapper(TorchDataset):

    def __init__(self, dataset, transform=None, image_key='image', text_key='query'):
        self.dataset = dataset
        self.transform = transform
        self.image_key = image_key
        self.text_key = text_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 加载图像
        image = item[self.image_key]
        if self.transform is not None:
            image = self.transform(image, return_tensors='pt')

        # 获取查询列表
        query = item[self.text_key]
        if isinstance(query, str):
            query = [query]

        return image, query


def get_dataset_default_task(dataset):
    if dataset in (
            'custom',
            'muge',
            'flickr30k',
            'flickr8k',
            'mscoco_captions',
            'mscoco_captions2017',
            'multilingual_mscoco_captions',
            'flickr30k-200',
            'crossmodal3600',
            'xtd200',
    ):
        return 'zeroshot_retrieval'
    else:
        return 'zeroshot_classification'


def get_dataloader(dataset_name, dataset, batch_size, num_workers):
    if dataset_name == 'custom':
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=image_captions_collate_fn,
        )
    else:
        dataloader = DataLoader(
            dataset.batched(batch_size),
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )
    return dataloader


def image_captions_collate_fn(batch):
    transposed = list(zip(*batch))
    imgs = transposed[0]
    texts = transposed[1]
    return imgs, texts


def build_custom_dataset(dataset_name, data_dir, transform=None):
    from datasets import Features, Image, Sequence, Value, load_dataset

    qrels_ds = load_dataset(
        'json',
        data_files=os.path.join(data_dir, 'image_queries.jsonl'),
        features=Features({
            'image_path': Image(decode=True),
            'query': Sequence(Value('string'))
        }),
        split='train',
    )

    dataset = DatasetWrapper(qrels_ds, transform, image_key='image_path', text_key='query')
    return dataset


def build_wds_dataset(dataset_name, transform, split='test', data_dir='root', cache_dir=None):
    """
    Load a dataset in WebDataset format. Either local paths or HTTP URLs can be specified.
    Expected file structure is:
    ```
    data_dir/
        train/
            nshards.txt
            0.tar
            1.tar
            ...
        test/
            nshards.txt
            0.tar
            1.tar
            ...
        classnames.txt
        zeroshot_classification_templates.txt
        dataset_type.txt
    ```
    Classnames and templates are required for zeroshot classification, while dataset type
    (equal to "retrieval") is required for zeroshot retrieval datasets.

    You can use the `clip_benchmark_export_wds` or corresponding API
    (`clip_benchmark.webdataset_builder.convert_dataset`) to convert datasets to this format.

    Set `cache_dir` to a path to cache the dataset, otherwise, no caching will occur.
    """
    import webdataset as wds

    def read_txt(fname):
        if '://' in fname:
            stream = os.popen("curl -L -s --fail '%s'" % fname, 'r')
            value = stream.read()
            if stream.close():
                raise FileNotFoundError('Failed to retreive data')
        else:
            with open(fname, 'r') as file:
                value = file.read()
        return value

    if not data_dir:
        data_dir = f'https://modelscope.cn/datasets/clip-benchmark/wds_{dataset_name}/resolve/master'

    # Git LFS files have a different file path to access the raw data than other files
    if data_dir.startswith('https://modelscope.cn/datasets'):
        *split_url_head, _, url_path = data_dir.split('/', 7)
        url_head = '/'.join(split_url_head)
        metadata_dir = '/'.join([url_head, 'resolve', url_path])
        tardata_dir = '/'.join([url_head, 'resolve', url_path])
    else:
        metadata_dir = tardata_dir = data_dir
    # Get number of shards
    nshards_fname = os.path.join(metadata_dir, split, 'nshards.txt')
    nshards = int(read_txt(nshards_fname))  # Do not catch FileNotFound, nshards.txt should be mandatory

    # Get dataset type (classification or retrieval)
    type_fname = os.path.join(metadata_dir, 'dataset_type.txt')
    try:
        dataset_type = read_txt(type_fname).strip().lower()
    except FileNotFoundError:
        dataset_type = 'classification'

    filepattern = os.path.join(tardata_dir, split, '{0..%d}.tar' % (nshards - 1))
    # Load webdataset (support WEBP, PNG, and JPG for now)
    if not cache_dir or not isinstance(cache_dir, str):
        cache_dir = None
    else:
        os.makedirs(cache_dir, exist_ok=True)
    dataset = wds.WebDataset(
        filepattern,
        cache_dir=cache_dir,
        nodesplitter=lambda src: src,
        shardshuffle=False,
        verbose=True,
    ).decode(wds.autodecode.ImageHandler('pil', extensions=['webp', 'png', 'jpg', 'jpeg']))

    # Load based on classification or retrieval task
    if dataset_type == 'retrieval':
        dataset = dataset.to_tuple(['webp', 'png', 'jpg', 'jpeg'], 'txt').map_tuple(transform, str.splitlines)
        dataset.classes = dataset.templates = None
    else:
        label_type = ('npy' if dataset_type == 'multilabel' else 'cls')  # Special case for multilabel
        dataset = dataset.to_tuple(['webp', 'png', 'jpg', 'jpeg'], label_type).map_tuple(transform, None)
        # Get class names if present
        classnames_fname = os.path.join(metadata_dir, 'classnames.txt')
        try:
            dataset.classes = [line.strip() for line in read_txt(classnames_fname).splitlines()]
        except FileNotFoundError:
            logger.warning('WARNING: classnames.txt not found')
            dataset.classes = None
        # Get zeroshot classification templates if present
        templates_fname = os.path.join(metadata_dir, 'zeroshot_classification_templates.txt')
        try:
            dataset.templates = [line.strip() for line in read_txt(templates_fname).splitlines()]
        except FileNotFoundError:
            logger.warning('WARNING: zeroshot_classification_templates.txt not found')
            dataset.templates = None

    return dataset
