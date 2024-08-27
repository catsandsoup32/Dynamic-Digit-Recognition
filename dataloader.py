import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchmetrics import Accuracy
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image

# Data visualization --------------------------------------------------------------------------------------------------
def class_Labels_Length(folder_name):
    df = pd.DataFrame()
    class_list = os.listdir(folder_name)
    df.insert(0, "Class_Names", class_list, allow_duplicates=True)
    class_length_list = []
    totalLen = 0
    for class_name in class_list:
        class_path = os.path.join(folder_name, class_name) 
        class_length_list.append(len(os.listdir(class_path)))
        totalLen += len(os.listdir(class_path))
    df.insert(1, "Length of Class", class_length_list, allow_duplicates=True)        
    # print(f"Total Examples: {totalLen}\n")       
    # print(df.to_string())
    return df

# df = class_Labels_Length('data/extracted_images') 
# print(df.to_string())
    
transformForVisualization = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)) # subtract 0.5 and then divide 0.5 (z-score)
    # between -1 and 1 so centered at 0 for activation functions
])
imageForVisualization = Image.open('data/extracted_images/!/!_7731.jpg')
transformedImForVisualization = transformForVisualization(imageForVisualization)
arrayForVisualization = np.squeeze(np.array(transformedImForVisualization, dtype=int)) # squeeze gets rid of (1, 45 45)

fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.matshow(arrayForVisualization, cmap = 'magma', interpolation='none')
for (i, j), val in np.ndenumerate(arrayForVisualization):
    ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, color='red')
ax.set_xticks([])
ax.set_yticks([])

# imageForVisualization.show()
# plt.show()


# Define dataset ------------------------------------------------------------------------------------------------------
class MathSymbolDataset(Dataset): 
    def __init__(self, data_dir, mode = None, transform = None, seed = None):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.seed = seed # for sklearn train_test_split
        self.image_paths_list = []
        self.labels_list = []
        self._load_data()
    
    def _load_data(self):
        class_list = os.listdir(self.data_dir) # returns list of all folders within
        label_dict = {class_name: idx for idx, class_name in enumerate(class_list)}
        for class_name in class_list: 
            class_path = os.path.join(self.data_dir, class_name) # path to extracted_images and then specific folder
            for img_name in os.listdir(class_path): 
                self.image_paths_list.append(os.path.join(class_path, img_name)) # path to each example within everything
                self.labels_list.append(label_dict[class_name]) # this works because each img_path has corresponding label (same size)

        train_indices, test_indices = train_test_split(np.arange(len(self.image_paths_list)), test_size=0.2, random_state=self.seed)
        # length is entire dataset
        self.train_indices = train_indices
        self.test_indices = test_indices

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_indices) 
        elif self.mode == 'test':
            return len(self.test_indices)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_path = self.image_paths_list[self.train_indices[idx]]
            label = self.labels_list[self.train_indices[idx]]
        elif self.mode =='test':
            img_path = self.image_paths_list[self.test_indices[idx]]
            label = self.labels_list[self.test_indices[idx]]

        image = Image.open(img_path) 
        if self.transform:
            image = self.transform(image)
            # image = torch.squeeze(image, dim=0)

        return image, label
    




