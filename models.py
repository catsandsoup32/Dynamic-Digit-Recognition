import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Size_out = [(size_in + 2*pad - kernel_size)/stride] floored + 1 

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0, dilation=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0, dilation=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, dilation=1) 
    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.batch_norm1 = nn.BatchNorm2d(num_features=32) # C from (N, C, H, W) filter for more complex feature
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 7 * 7, 256) 
        self.fc2 = nn.Linear(256, 81)
        self.fc3 = nn.Linear(256, 81)
        self.batch_normFC = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.2)

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax() # DONT use this because there is already cross entropy loss (?)
        

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x)))) # size = (45 + 0 - 5)/1 + 1 = 41 -> 20 after pool
        x = self.leaky_relu(self.batch_norm2(self.conv2(x)))            # size = (20 + 0 - 5)/1 + 1 = 16
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x)))) # size = (16 + 0 - 3)/1 + 1 = 14 -> 7 after pool
        x = x.view(x.size(0), -1) # flattens the tensor into [batch_size x (128 * 7 * 7)]
        x = self.leaky_relu(self.batch_normFC(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    
class VamsiNN(nn.Module):
    def __init__(self):
        super(VamsiNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, dilation=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, dilation=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, dilation=1) 
        # probably don't need padding, lots of white space around image
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, 81)
        self.relu = nn.ReLU()
        self.batch_norm_fc = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x)))) # size = (45 - 3) + 1 = 43 -> 22 after pool
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x)))) # size = (21 - 3) + 1 = 19 -> 9 after pool
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x)))) # size = (9 - 3) + 1 = 7 -> 3 after pool
        x = x.view(x.size(0), -1)
        x = self.relu(self.batch_norm_fc(self.fc1(x)))
        x = self.fc2(x)
        return x


class CNN_9(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        self.fc3 = nn.Linear(256, 81) # was at 81

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax() # DONT use this because there is already cross entropy loss

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x)))) # size = (45 + 4 - 5)/1 + 1 = 45 -> 22 after pool
        x = self.leaky_relu(self.batch_norm2(self.conv2(x)))            # size = (22 + 4 - 5)/1 + 1 = 22
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x)))) # size = (22 + 2 - 3)/1 + 1 = 22 -> 11 after pool
        x = x.view(x.size(0), -1) # flattens the tensor into [batch_size x (128 * 11 * 11)]
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x



        


