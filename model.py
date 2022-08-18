import torch
import torch.nn as nn


class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.BN1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.BN2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.BN3a = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.BN3b = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.BN4a = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.BN4b = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.BN5a = nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.BN5b = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.sigmoid=nn.Sigmoid()




    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.BN1(x))
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(self.BN2(x))
        x = self.pool2(x)
        
        x = self.conv3a(x)
        x = self.relu(self.BN3a(x))
        x = self.conv3b(x)
        x = self.relu(self.BN3b(x))
        x = self.pool3(x)
        
        x = self.conv4a(x)
        x = self.relu(self.BN4a(x))
        x = self.conv4b(x)
        x = self.relu(self.BN4b(x))
        x = self.pool4(x)
        
        x = self.conv5a(x)
        x = self.relu(self.BN5a(x))
        x = self.conv5b(x)
        x = self.relu(self.BN5b(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        #print(x.size())
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        logits = self.fc3(x)
        return logits