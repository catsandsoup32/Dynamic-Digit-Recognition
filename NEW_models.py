import torch
import torch.nn as nn

# 72 classes

class CNN_19(nn.Module):
    def __init__(self):
        super(CNN_19, self).__init__()
        # Size_out = [(size_in + 2*pad - kernel_size)/stride] floored + 1 

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, dilation=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, dilation=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1) 
    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.batch_norm1 = nn.BatchNorm2d(num_features=32) # C from (N, C, H, W) filter for more complex feature
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 11 * 11, 1024) # was at 128 * 11 * 11 for 45, for 28 is 7
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 72) 

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.2)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x)))) # size = (45 + 4 - 5)/1 + 1 = 45 -> 22 after pool
        x = self.leaky_relu(self.batch_norm2(self.conv2(x)))            # size = (22 + 4 - 5)/1 + 1 = 22
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x)))) # size = (22 + 2 - 3)/1 + 1 = 22 -> 11 after pool
        x = x.view(x.size(0), -1) # flattens the tensor into [batch_size x (128 * 11 * 11)]
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

class CNN_16(nn.Module):
    def __init__(self):
        super(CNN_16, self).__init__()
        # Size_out = [(size_in + 2*pad - kernel_size)/stride] floored + 1 

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, dilation=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, dilation=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1) 
    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.batch_norm1 = nn.BatchNorm2d(num_features=32) # C from (N, C, H, W) filter for more complex feature
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 11 * 11, 1024) # was at 128 * 11 * 11 for 45, for 28 is 7
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 72) 

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.5)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x)))) # size = (45 + 4 - 5)/1 + 1 = 45 -> 22 after pool
        x = self.leaky_relu(self.batch_norm2(self.conv2(x)))            # size = (22 + 4 - 5)/1 + 1 = 22
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x)))) # size = (22 + 2 - 3)/1 + 1 = 22 -> 11 after pool
        x = x.view(x.size(0), -1) # flattens the tensor into [batch_size x (128 * 11 * 11)]
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

class CNN_9(nn.Module):
    def __init__(self):
        super(CNN_9, self).__init__()
        # Size_out = [(size_in + 2*pad - kernel_size)/stride] floored + 1 

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, dilation=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, dilation=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1) 
    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.batch_norm1 = nn.BatchNorm2d(num_features=32) # C from (N, C, H, W) filter for more complex feature
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 11 * 11, 1024) # was at 128 * 11 * 11 for 45, for 28 is 7
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 72) 

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x)))) # size = (45 + 4 - 5)/1 + 1 = 45 -> 22 after pool
        x = self.leaky_relu(self.batch_norm2(self.conv2(x)))            # size = (22 + 4 - 5)/1 + 1 = 22
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x)))) # size = (22 + 2 - 3)/1 + 1 = 22 -> 11 after pool
        x = x.view(x.size(0), -1) # flattens the tensor into [batch_size x (128 * 11 * 11)]
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CNN_23(nn.Module):
    def __init__(self):
        super(CNN_23, self).__init__()
        # Size_out = [(size_in + 2*pad - kernel_size)/stride] floored + 1 

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, dilation=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, dilation=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1) 
    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.batch_norm1 = nn.BatchNorm2d(num_features=32) # C from (N, C, H, W) filter for more complex feature
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 11 * 11, 1024) # was at 128 * 11 * 11 for 45, for 28 is 7
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 72) 
        self.dropout = nn.Dropout2d(p=0.3)

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x)))) # size = (45 + 4 - 5)/1 + 1 = 45 -> 22 after pool
        x = self.dropout(self.leaky_relu(self.batch_norm2(self.conv2(x))))         # size = (22 + 4 - 5)/1 + 1 = 22
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x)))) # size = (22 + 2 - 3)/1 + 1 = 22 -> 11 after pool
        x = x.view(x.size(0), -1) # flattens the tensor into [batch_size x (128 * 11 * 11)]
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CNN_22(nn.Module):
    def __init__(self):
        super(CNN_22, self).__init__()
        # Size_out = [(size_in + 2*pad - kernel_size)/stride] floored + 1 

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, dilation=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, dilation=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1) 
    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.batch_norm1 = nn.BatchNorm2d(num_features=32) # C from (N, C, H, W) filter for more complex feature
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 11 * 11, 512) 
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 72) 

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.3)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x)))) # size = (45 + 4 - 5)/1 + 1 = 45 -> 22 after pool
        x = self.leaky_relu(self.batch_norm2(self.conv2(x)))            # size = (22 + 4 - 5)/1 + 1 = 22
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x)))) # size = (22 + 2 - 3)/1 + 1 = 22 -> 11 after pool
        x = x.view(x.size(0), -1) # flattens the tensor into [batch_size x (128 * 11 * 11)]
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

# maybe overparameterized, can drop some layers


        


