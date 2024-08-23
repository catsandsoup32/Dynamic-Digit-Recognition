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
    print(f"Total Examples: {totalLen}\n")       
    print(df.to_string())
# class_Labels_Length('data/extracted_images') 
    
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
    
    
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)) # subtract 0.5 and then divide 0.5 (z-score)
])

train_dataset = MathSymbolDataset(data_dir='data/extracted_images', mode = 'train', transform=transform, seed=42)
train_size = int(0.9 * len(train_dataset)) # split into train and val set
val_size = len(train_dataset) - train_size
new_train_dataset, val_dataset = random_split(dataset=train_dataset, lengths=[train_size, val_size])
test_dataset = MathSymbolDataset(data_dir='data/extracted_images', mode = 'test', transform=transform, seed=42)

train_loader = DataLoader(new_train_dataset, batch_size=32, shuffle=True) # not using num_workers yet, can do it with if name
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# About 270,000 elements in new_train_dataset
# Each element is a length two tuple containing 1: a 1x45x45 tensor and 2: the label number



  
# ------------------------------------------------------------------------------------------------------------------
# Define model

class LeNet5V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   
            nn.Tanh(),  
            nn.AvgPool2d(kernel_size=2, stride=2),  
            
            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*9*9, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=81),
        )
        
    def forward(self, x):
        return self.classifier(self.feature(x))
    

class SimpleYOLO(nn.Module):
    def __init__(self):
        super(SimpleYOLO, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 81) 
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



model = SimpleYOLO()
criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
accuracy = Accuracy(task='multiclass', num_classes=81)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
accuracy = accuracy.to(device)
model = model.to(device)


# ------------------------------------------------------------------------------------------------------------------
# Train and val loops

num_epochs = 5
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    running_acc, running_loss = 0.0, 0.0
    for images, labels in tqdm(train_loader, desc="training loop"):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, labels)

    train_acc = running_acc / len(train_loader.dataset)
    train_accs.append(train_acc)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # valid phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="val loop"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, val loss: {val_loss}")
    torch.save(model.state_dict(), 'model.pth')  # Save the trained model
