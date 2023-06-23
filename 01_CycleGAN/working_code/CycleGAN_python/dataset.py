from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class DryWetDataset(Dataset):
    def __init__(self, root_wet, root_dry, transform=None, **kwargs):
        self.root_wet = root_wet
        self.root_dry = root_dry
        self.transform = transform

        self.wet_images = os.listdir(root_wet)
        self.dry_images = os.listdir(root_dry)
        self.length_dataset = max(len(self.wet_images), len(self.dry_images)) # 1000, 1500
        self.wet_len = len(self.wet_images)
        self.dry_len = len(self.dry_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        wet_img = self.wet_images[index % self.wet_len]
        dry_img = self.dry_images[index % self.dry_len]

        wet_path = os.path.join(self.root_wet, wet_img)
        dry_path = os.path.join(self.root_dry, dry_img)

        wet_img = np.array(Image.open(wet_path).convert("RGB"))
        dry_img = np.array(Image.open(dry_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=wet_img, image0=dry_img)
            wet_img = augmentations["image"]
            dry_img = augmentations["image0"]

        return wet_img, dry_img





