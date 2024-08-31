import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image

# Key changes: change data_dir path

# ------------------------------------------------------------------------------------------------------------------
# Define model
from NEW_models import CNN_9, CNN_16

# transform and init data
from NEW_dataloader import MathSymbolDataset

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)) # subtract 0.5 and then divide 0.5 (z-score)
])


train_dataset = MathSymbolDataset(data_dir='data/extracted_images_new', mode = 'train', transform=transform, seed=21)
train_size = int(0.9 * len(train_dataset)) # split into train and val set
val_size = len(train_dataset) - train_size
new_train_dataset, val_dataset = random_split(dataset=train_dataset, lengths=[train_size, val_size])
test_dataset = MathSymbolDataset(data_dir='data/extracted_images_new', mode = 'test', transform=transform, seed=21)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Train and val loops
def main(num_epochs, experimentNum, use_model, use_dataset_train, use_dataset_val, use_dataset_test, num_classes, loadFromSaved, test_loop, LR):

    train_loader = DataLoader(use_dataset_train, batch_size=32, shuffle=True, num_workers=5) # num_workers must be with if name
    val_loader = DataLoader(use_dataset_val, batch_size=32, shuffle=False, num_workers=1)
    test_loader = DataLoader(use_dataset_test, batch_size=32, shuffle=False, num_workers=1)

    # About 270,000 elements in new_train_dataset
    # Each element is a length two tuple containing 1: a 1x45x45 tensor and 2: the label number

    # start loop -----------------------------------------------------------------------------------------------------
    num_epochs = num_epochs
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    model = use_model
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.Adam(model.parameters(), lr = LR)
    scheduler = StepLR(optimizer, step_size = 10, gamma = 0.8)
    accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accuracy = accuracy.to(device)
    model = model.to(device)

    if loadFromSaved is not None:
        model.load_state_dict(torch.load(loadFromSaved, map_location=device, weights_only=True))
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        accuracy.reset()
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}: training loop"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images) # forward pass
            loss = criterion(outputs, labels) 

            running_loss += loss.item() * images.size(0) # loss tensor times batch size
            running_acc += accuracy(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        train_acc = running_acc / len(train_loader.dataset) * 32 * 100
        train_accs.append(train_acc)
        train_loss = running_loss / len(train_loader.dataset) 
        train_losses.append(train_loss)

        # valid phase
        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}: val loop"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                running_acc += accuracy(outputs, labels)

            val_loss = running_loss / len(val_loader.dataset) 
            val_losses.append(val_loss)
            val_acc = running_acc / len(val_loader.dataset) * 32 * 100
            val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss} train acc: {train_acc}, val loss: {val_loss}, val acc: {val_acc}")
    

        scheduler.step()

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'NEW_save_states/CNNmodel{experimentNum}Epoch{epoch+1}.pt')  # Save the trained model\

    torch.save(model.state_dict(), f'NEW_save_states/CNNmodel{experimentNum}Epoch{epoch+1}.pt')  # Save the trained model\

    plt.plot(np.arange(num_epochs), list(train_losses))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over time')
    plt.show()

    if test_loop:
        model.eval()
        running_acc = 0.0
        for images, labels in tqdm(test_loader, desc="testing"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            running_acc += accuracy(outputs, images)
        
        test_loss = running_acc / len(test_loader.dataset) * 32 * 100
        print(f"TEST LOSS: {test_loss}")

if __name__ == '__main__':
    main(num_epochs = 120, 
         experimentNum = 18,
         use_model = CNN_16(),
         use_dataset_train = new_train_dataset,
         use_dataset_val = val_dataset,
         use_dataset_test = test_dataset,
         num_classes = 72,
         loadFromSaved = None,
         test_loop = True,
         LR = 0.0007)
    
    # TEMP ADD 45


# Duplicate of everything with NEW_ prefix, start with CNN_9 and adjust

# Experiment 15: 0.0035 LR from 0.003, same as CNN_9 but just increase LR
# Epoch 40- Train loss: 0.06470691475075424 train acc: 98.91757202148438, val loss: 0.09432324577125029, val acc: 98.84830474853516

# Experiment 16, changed test and train seed to 21, dropout last FC layer, learning rate to 0.0025
# peaked at 97.4 percent acc

# Experiment 17, same as before but decrease LR to 0.001 and 60 epochs (uses CNN_16)
# Epoch 45/60 - Train loss: 1.7970874412371167 train acc: 54.42210006713867, val loss: 0.022417195288928885, val acc: 99.59793090820312'''

# Experiment 18, same but decrease LR to 0.0007
'''
Epoch 40/120 - Train loss: 1.796863111263619 train acc: 54.47850799560547, val loss: 0.020148418561522927, val acc: 99.67630004882812
Epoch 50/120 - Train loss: 1.8003004343653521 train acc: 54.43686294555664, val loss: 0.021955023554910174, val acc: 99.70014953613281
Epoch 70/120 - Train loss: 1.7920134145669913 train acc: 54.63449478149414, val loss: 0.018674188788209548, val acc: 99.8364486694336
Epoch 90/120 - Train loss: 1.790238317859528 train acc: 54.67576217651367, val loss: 0.01661837912969268, val acc: 99.8841552734375
Epoch 105/120 - Train loss: 1.7832430657807232 train acc: 54.809791564941406, val loss: 0.01611637465686386, val acc: 99.89778137207031
Epoch 115/120 - Train loss: 1.7967954847319871 train acc: 54.458065032958984, val loss: 0.016530913844644706, val acc: 99.89437103271484
'''