import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from PIL import Image


class TestDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super().__init__()
        
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob("{}/*.jpg".format(root_dir))


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()

        image = Image.open(self.image_paths[indx])
        if self.transform:
            image = self.transform(image)

        return {
            'image' : image,
            'image_id' : self.image_paths[indx]
        }
