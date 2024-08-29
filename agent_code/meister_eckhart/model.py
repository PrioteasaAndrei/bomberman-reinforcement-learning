import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():
    def __init__(self, maxlen: int):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, sample_size: int):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)



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
        
    def train(self, sampled_batch, optimizer, loss_fn=nn.MSELoss(), epochs=1):
        '''
        Train the model given a sample of the replay buffer
        sampled_batch: list of Transitions
        '''

        # NOTE: use https://medium.com/@hkabhi916/mastering-deep-q-learning-with-pytorch-a-comprehensive-guide-a7e690d644fc
        # as a reference for the training loop and the train_agent function from the project from last year

        # TODO: from the transitions sampled_batch extract the states, actions, next_states and rewards
        ...
        # TODO: use the policy network to predict the q values for the next states
        ...
        # TODO: use the target value to predict the target q values for the next states
        # target_q = reward + gamma * max_a' Q(s', a')
        ...
        # TODO: compute the loss (mse) between the predicted Q values and the target Q values
        # loss.backward()
        # self.optimizer.step()

        # TODO: repeat for the number of epochs


        for _ in range(epochs):
            optimizer.zero_grad()

            # TODO: compute the loss (mse) between the predicted Q values and the target Q values
            # target_q is already given
            
