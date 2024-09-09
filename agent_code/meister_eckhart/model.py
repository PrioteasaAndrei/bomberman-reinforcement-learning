import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from torch.utils.data import Dataset
from tqdm import tqdm
from .config import *


class ReplayMemory():
    def __init__(self, maxlen: int):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, sample_size: int):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
    
    def can_provide_sample(self, sample_size: int):
        return len(self.memory) >= sample_size

def create_model(input_shape, num_actions, logger, model_type: str):
    if model_type == "JointDQN":
        return JointDQN(input_shape=input_shape, num_actions=num_actions, logger=logger)
    elif model_type == "SimpleDQN":
        return SimpleDQN(input_shape=input_shape, num_actions=num_actions, logger=logger)
    else:
        raise ValueError("Invalid model type {}".format(model_type))

class SimpleDQN(nn.Module):
    '''
    Number of parameters: 384 + input_shape_vectorized * 64
    '''
    def __init__(self,input_shape=(8,17,17),num_actions=6,logger=None):
        super(JointDQN, self).__init__()

        self.vectorized_shape = math.prod(input_shape)
       
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.vectorized_shape, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
      return self.net(x)
    
    def number_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    

class JointDQN(nn.Module):
    '''
    Network dimensions:
    input: 8x17x17
    feature_space: 128
    total number of parameters in the feature extractor: 5808 
    total number of parameters in the dqn: 8646 
    total params: 14454

    NOTE: maverick has only 1740 parameters

    '''
    def __init__(self,input_shape=(8,17,17),num_actions=6,logger=None):
        super(JointDQN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Pooling to reduce spatial dimensions
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Pooling to reduce spatial dimensions
            nn.Flatten()
        )

        self.logger = logger

        self.dqn_input_size = self.feature_size(input_shape)

        self.dqn = nn.Sequential(
            nn.Linear(self.dqn_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=1) 
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        action_distr = self.dqn(features)
        return action_distr
    
    def feature_size(self, input_shape):
        # Helper function to calculate the flattened size of CNN output
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.feature_extractor(x)
            return x.view(1, -1).size(1)
        
    def number_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
def action_to_tensor(action: str):
    '''
    Convert the action string to a tensor
    '''

    ## get index of action in ACTION and return it as tensor

    try:
        # Find the index of the target in the list
        index = ACTIONS.index(action)
    except ValueError:
        # If the target is not found, handle the exception (optional)
        raise ValueError(f"The target '{action}' is not found in the list.")
    
    # Convert the index to a tensor
    index_tensor = torch.tensor(index)
    return index_tensor

# This function is based on the following tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
def train_step(self, batch_size: int, gamma: int, device: torch.device, memory: ReplayMemory):
    '''
    Perform a single training step on a batch of transitions.
    Args:
        batch_size: The number of transitions to sample from the replay memory
        gamma: The discount factor
        device: The device to run the training step on
    '''
 
    self.logger.info("Training step...")

    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # NOTE: remember that these are tuples not lists

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    non_final_next_states = torch.stack([torch.from_numpy(s) for s in batch.next_state
                                                if s is not None])
    

    state_batch = torch.stack(list(map(torch.from_numpy,list(batch.state)))) 
    self.logger.info(f"Actions: {batch.action}")
    action_batch = torch.stack(list(list(map(action_to_tensor,batch.action))))
    # self.logger.info(f"Action batch shape: {action_batch.shape}")
    reward_batch = torch.stack(list(map(lambda x: torch.tensor([x]), batch.reward)))
    # self.logger.info(f"Reward batch shape: {reward_batch.shape}")

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch.float().to(device)).gather(1, action_batch.to(device))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # Reason for using the target network : https://stackoverflow.com/questions/54237327/why-is-a-target-network-required
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states.float().to(device)).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.view(-1, 1) * gamma) + reward_batch.to(device)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()
        
    self.logger.info(f"Loss: {loss.item()}")
    self.logger.info("Training step done.")

    return loss.item()