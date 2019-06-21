import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(4,  32,  8, 4)
        self.conv2 = nn.Conv2d(32, 64,  4, 2)
        self.conv3 = nn.Conv2d(64, 64,  3, 1)
        self.conv4 = nn.Conv2d(64, 512, 7, 1)
        
        self.state_value = nn.Linear(256, 1)
        self.advantage_fn = nn.Linear(256, 800)
        self.output = nn.Linear(801, 800)
        
    def forward(self, x):
        x = x.double()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        
        value_input = x[:, :256, 0, 0]
        advantage_input = x[:,256:, 0, 0]
        
        value_output = self.state_value(value_input)
        advantage_output = self.advantage_fn(advantage_input)
        
        pre_output = torch.cat((value_output, advantage_output), 1)
        return self.output(pre_output)
