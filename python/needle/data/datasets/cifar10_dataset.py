import os
import pickle
from typing import Any, Iterable, Iterator, List, Optional, Sized, Union

import numpy as np

from ..data_basic import Dataset


class CIFAR10Dataset(Dataset):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """

        ### BEGIN YOUR SOLUTION
        super().__init__(transforms=transforms)
        if isinstance(base_folder, str): # 有时候，base_folder 是一个byte字符串
            base_folder = os.path.expanduser(base_folder)
        self.base_folder = base_folder
        self.train = train  # training set or test set
        self.p = p

        # 加载cifar 图片
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.targets = np.asarray(self.targets)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.base_folder, self.meta["filename"])
        
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}    
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img, target = self.data[index], self.targets[index]
    
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = self.apply_transforms(img)
        if len(img.shape) == 4:
            img = img.transpose((0, 3, 1, 2))
        elif len(img.shape) == 3:
            img = img.transpose((2, 0, 1))
        return img, target
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.data)
        ### END YOUR SOLUTION
