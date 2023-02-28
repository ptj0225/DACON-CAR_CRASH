
from torch.utils.data import Dataset
import cv2
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, path_list, label_list, tfms):
        self.path_list = path_list
        self.label_list = label_list
        self.tfms = tfms
        
    def __getitem__(self, index):
        image = cv2.imread(self.path_list[index])
        if self.tfms is not None:
            image = self.tfms(image = image)['image']

        if self.label_list is not None:
            label = np.zeros(5)
            label[int(self.label_list[index][0])] = 1
            label[3 + int(self.label_list[index][1])] = 1
            label = np.float32(label)
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.path_list)