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

TRAIN_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
SCENARIO = "crates"

LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "saved-model-" + SCENARIO +  ".pth.tar"
MODEL_LOAD_PATH = 'saved_models/coin-heaven-400ep-1000ds.pth.tar'
# set to False if you don't want to train further the saved model
TRAIN_FROM_CHECKPOINT = True
# if you want to reinitialize EpsilonGreedyStrategy
REINITIALIZE_EPSILON = True

# Hyper parameters -- DO modify
RECORD_ENEMY_TRANSITIONS = 1.0 # record enemy transitions with probability ...
GAMMA = 0.99
MEMORY_SIZE = 10000
BATCH_SIZE = 64

# the number of training steps
TRAIN_EPOCHS = 10_000
ROUND_TO_PLOT = 2 #default 200
SAVE_MODEL_EVERY = 100
UPDATE_TARGET_EVERY = 100
DECAY_STEPS = 1000

CROP_SIZE = 7

MODEL_TYPE = "JointDQN"
