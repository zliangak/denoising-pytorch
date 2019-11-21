import torch
import torch.nn as nn
import torch.nn.functional as F

class DA(nn.Module):
    def __init__(self, patch_width):
        super(DA, self).__init__()
        self.pw = patch_width
        self.fc1 = nn.Linear(self.pw*self.pw, 2*self.pw*2*self.pw)
        self.fc2 = nn.Linear(2*self.pw*2*self.pw, self.pw*self.pw)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


class ConvDa(nn.Module):
    def __init__(self):
        super(ConvDa, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.relu3 = nn.ReLU()
#        self.pool3 = nn.MaxPool2d(2, 2)



        ## decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
#        self.t_conv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
#        self.relu4 = nn.ReLU()

        self.t_conv2 = nn.ConvTranspose2d(16, 32, 2, stride=2) # 8,16,2,2
        self.relu5 = nn.ReLU()

        self.t_conv3 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.relu6 = nn.ReLU()

        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # say x.shape = [1, 1, 32, 32]
        x = self.relu1(self.conv1(x))   # [1, 32, 32, 32]
        x = self.pool1(x)               # [1, 32, 16, 16]
        x = self.relu2(self.conv2(x))   # [1, 16, 16, 16]
        x = self.pool2(x)               # [1, 16, 8, 8]
        x = self.relu3(self.conv3(x))   # [1, 8, 8, 8]
#        x = self.pool3(x)               # [1, 8, 4, 4]

        ## decode ##
        # add transpose conv layers, with relu activation function
#        x = self.relu4(self.t_conv1(x)) # [1, 8, 8, 8]
        x = self.relu5(self.t_conv2(x)) # [1, 16, 16, 16]
        x = self.relu6(self.t_conv3(x)) # [1, 32, 32, 32]
        # transpose again, output should have a sigmoid applied
        x = self.sigmoid(self.conv_out(x))

        return x