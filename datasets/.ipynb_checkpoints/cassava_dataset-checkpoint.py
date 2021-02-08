from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

class CassavaDataset(Dataset):
    def __init__(self, image_dir, df, transform=None):
        self.image_dir = image_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        label = row.label
        image_name = row.image_id

        image = Image.open(os.path.join(self.image_dir, image_name))
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label