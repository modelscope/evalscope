from torch.utils.data import Dataset as TorchDataset


class DatasetWrapper(TorchDataset):
    def __init__(self, dataset, transform=None, image_key="image", text_key="query"):
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
            image = self.transform(image, return_tensors="pt")

        # 获取查询列表
        query = item[self.text_key]
        if isinstance(query, str):
            query = [query]

        return image, query
