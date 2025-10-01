import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO
import os


class AVM_Data(Dataset):
    def __init__(self, root, split, fold, transform=None, train=True):
        super().__init__()
        """
        GLvsL_median_shape Dataset
        """
        self.img_dir = Path(root) / "nnsslPlans_onemmiso/Dataset001_AVM/Dataset001_AVM"
        label_file = Path(root) / "cls_labelsTr.json"
        split_file = Path(root) / "cls_splits.json"

        with open(split_file) as f:
            splits = json.load(f)
            if split in splits:
                self.img_files = splits[split]
            else:
                raise ValueError(f"Unknown split name: {split}")

        with open(label_file) as f:
            labels = json.load(f)
        #self.img_files = [x for x in self.img_files if x in labels.keys()]
        #self.labels = np.array([labels[i] for i in self.img_files]).astype(np.int8)
        self.labels = torch.tensor([labels[i] for i in self.img_files], dtype=torch.long)

        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        # filename from splits, e.g. '579_MRI_19990812_T1.nii.gz' or '579_19990812_MRI_T1.nii.gz'
        fname = os.path.basename(self.img_files[idx])

        # strip extension robustly (supports .nii.gz and others)
        if fname.endswith(".nii.gz"):
            stem = fname[:-7]
        else:
            stem = os.path.splitext(fname)[0]

        # split and locate the 8-digit date
        sid, *rest = stem.split("_")
        date_idx = next((i for i, p in enumerate(rest) if len(p) == 8 and p.isdigit()), None)
        if date_idx is None:
            raise ValueError(f"Cannot find YYYYMMDD in filename: {fname}")

        date = rest[date_idx]

        # modality: if date is first in 'rest', modality follows; otherwise it's before the date
        if date_idx == 0:
            if len(rest) < 2:
                raise ValueError(f"Cannot parse modality after date in: {fname}")
            modality = rest[1]
            exclude = {0, 1}
        else:
            modality = rest[0]
            exclude = {date_idx, 0}

        b2name = f"{sid}_{date}_{modality}" + (f"_T1.b2nd")
        b2path = self.img_dir / sid / "ses-DEFAULT" / b2name

        if not b2path.exists():
            raise FileNotFoundError(f"Preprocessed file not found: {b2path}")

        img, _ = Blosc2IO.load(str(b2path), mode="r")

        if self.train:
            img = self.transform(**{"image": torch.from_numpy(img[...])})["image"]
        else:
            img = self.transform.transforms[0](**{"image": torch.from_numpy(img[...])})["image"]

        return img, self.labels[idx], sid


    def __len__(self):
        return len(self.img_files)


class AVM_DataModule(BaseDataModule):
    def __init__(self, **params):
        super(AVM_DataModule, self).__init__(**params)

    def setup(self, stage: str):

        self.train_dataset = AVM_Data(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = AVM_Data(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
            train=False,
        )
        self.test_dataset = AVM_Data(
            self.data_path,
            split="test",
            transform=self.test_transforms,
            fold=self.fold,
            train=False,
        )