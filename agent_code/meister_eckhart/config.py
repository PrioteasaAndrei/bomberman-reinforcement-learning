import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from torch.utils.data import Dataset
from tqdm import tqdm


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

WALKING_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

TRAIN_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "saved_models/my-saved-model-rule-based-coin-heaven.pt"

# Hyper parameters -- DO modify
RECORD_ENEMY_TRANSITIONS = 1.0 # record enemy transitions with probability ...
GAMMA = 0.99
MEMORY_SIZE = 10000
BATCH_SIZE = 64

# the number of training steps
TRAIN_EPOCHS = 10_000
ROUND_TO_PLOT = 15000 #default 200
SAVE_MODEL_EVERY = 1000
UPDATE_TARGET_EVERY = 10

CROP_SIZE = 7

SCENARIO = "coin_heaven"
MODEL_TYPE = "JointDQN"
