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



# transform and init data
from dataloader import MathSymbolDataset

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

for test_im, test_label in train_loader:
    # torch.set_printoptions(threshold=torch.inf)
    # print(test_im)
    break

# About 270,000 elements in new_train_dataset
# Each element is a length two tuple containing 1: a 1x45x45 tensor and 2: the label number

# ------------------------------------------------------------------------------------------------------------------
# Define model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Size_out = [(size_in + 2*pad - kernel_size)/stride] + 1 floored

        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, stride=1, padding=2, dilation=2) 
        self.conv2 = nn.Conv2d(12, 24, kernel_size=5, stride=2, padding=2, dilation=1) 
        self.conv3 = nn.Conv2d(24, 28, kernel_size=3, stride=2, padding=1, dilation=1) 
        self.conv4 = nn.Conv2d(28, 32, kernel_size=3, stride=2, padding=1, dilation=1) 
    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.batch_norm1 = nn.BatchNorm2d(num_features=12) # C from (N, C, H, W) filter for more complex feature
        self.batch_norm2 = nn.BatchNorm2d(24)
        self.batch_norm3 = nn.BatchNorm2d(28)
        self.batch_norm4 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 81) 

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.softmax = nn.Softmax2d() 

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x))))
        x = self.softmax(self.leaky_relu((self.batch_norm4(self.conv4(x)))))
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
accuracy = Accuracy(task='multiclass', num_classes=81)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
accuracy = accuracy.to(device)
model = model.to(device)

# ------------------------------------------------------------------------------------------------------------------
# Train and val loops
def main(num_epochs, experimentNum, loadFromSaved):
    num_epochs = num_epochs
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    if loadFromSaved is not None:
        model.load_state_dict(torch.load(loadFromSaved, map_location=device, weights_only=True))
    
    for epoch in range(num_epochs):
        model.train()
        running_acc, running_loss = 0.0, 0.0
        for images, labels in tqdm(train_loader, desc="training loop"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0) # loss tensor times batch size
            running_acc += accuracy(outputs, labels)

        train_acc = running_acc / len(train_loader.dataset)
        train_accs.append(train_acc)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # valid phase
        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="val loop"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                running_acc += accuracy(outputs, labels)

            val_loss = running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            val_acc = running_acc / len(val_loader.dataset)
            val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss} train acc: {train_acc}, val loss: {val_loss}, val acc: {val_acc}")

        if num_epochs % 10 == 0:
            torch.save(model.state_dict(), f'save_states/CNNmodel{experimentNum}Epoch{epoch}.pt')  # Save the trained model

if __name__ == '__main__':
    main(num_epochs=15, experimentNum=2, loadFromSaved='save_states/CNNmodel1.pt')

# Experiment 1: Test, loss at around 0.3, did not print accuracy
# Experiment 2: 