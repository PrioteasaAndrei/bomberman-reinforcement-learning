import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import os

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
WALKING_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

TRAIN_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
SCENARIO = "loot-crate" # NOTE: make sure this is the exact same name as the scenario in the main.py command
MODEL_SAVE_PATH = "saved-model-" + SCENARIO +  ".pth.tar"
MODEL_LOAD_PATH = 'saved_models/coin-heaven-400ep-1000ds.pth.tar'
# MODEL_LOAD_PATH = 'saved-model-loot-crate.pth.tar'

# set to False if you don't want to train further the saved model
TRAIN_FROM_CHECKPOINT = True
# if you want to reinitialize EpsilonGreedyStrategy
REINITIALIZE_EPSILON = True

RECORD_ENEMY_TRANSITIONS = 1.0 # record enemy transitions with probability ...
MEMORY_SIZE = 10000
BATCH_SIZE = 64

ROUND_TO_PLOT = 2 
SAVE_MODEL_EVERY = 100

CROP_SIZE = 7

MODEL_TYPE = "JointDQN"

# Hyper parameters -- load them from the config file
# print current path
with open("agent_code/meister_eckhart/hyperparams.json") as f:
    config = json.load(f)

LEARNING_RATE = config['lr']
GAMMA = config['gamma']
UPDATE_TARGET_EVERY = config['target_update']
DECAY_STEPS = config['decay_steps']
START_EPSILON = config['start_epsilon']