import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class JointDQN(nn.Module):
    '''
    Modify the kernel size and stride to obtain the desired feature space size
    '''
    def __init__(self,input_shape=(8,17,17),num_actions=6,logger=None):
        super(JointDQN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.logger = logger

        self.dqn_input_size = self.feature_size(input_shape)

        self.dqn = nn.Sequential(
            nn.Linear(self.dqn_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1) 
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        self.logger.info(f"Features shape: {features.shape}")
        action_distr = self.dqn(features)
        self.logger.info(f"Action distribution shape: {action_distr.shape}")
        return action_distr
    
    def feature_size(self, input_shape):
        # Helper function to calculate the flattened size of CNN output
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.feature_extractor(x)
            return x.view(1, -1).size(1)