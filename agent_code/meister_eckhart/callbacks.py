import os
import pickle
import random
import numpy as np
from .model import JointDQN
import torch
import logging
import settings

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()

        self.policy_net = JointDQN(input_shape=(8, 17, 17), num_actions=6, logger=self.logger)
        self.target_net = JointDQN(input_shape=(8, 17, 17), num_actions=6, logger=self.logger)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net = pickle.load(file)
        self.target_net = JointDQN(input_shape=(8, 17, 17), num_actions=6, logger=self.logger)
        self.target_net.load_state_dict(self.policy_net.state_dict())


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    raw_features = state_to_features(game_state)
    expanded_tensor = torch.tensor(raw_features).unsqueeze(0)  # Shape becomes (1, 8, 17, 17)
    # Repeat the tensor along the batch dimension to create a shape of (64, 8, 17, 17)
    new_tensor = expanded_tensor.repeat(64, 1, 1, 1)  # Shape becomes (64, 8, 17, 17)

    outputs = self.policy_net(new_tensor.float())
    self.logger.info(f"Model outputs: {outputs.shape}")

    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    # Exploitation TODO: actually do exploitation now its just testing
    # return np.random.choice(ACTIONS, p=self.model)
    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])



def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    
    '''
    Transforms the game state dictionary into a multi-channel numpy array that will be fed to the 
    feature discovery network
    '''
    if game_state is None:
        return None
    
    game_map = game_state['field']
    crates_map = np.where(game_map == 1, 1, 0)
    walls_map = np.where(game_map == -1, 1, 0)

    explosion_map = game_state['explosion_map'] / settings.EXPLOSION_TIMER
    assert explosion_map.min() >= 0 and explosion_map.max() <= 1

    #  Create a map of the coins
    coin_map = np.zeros(game_map.shape)
    coin_coords = game_state['coins']
    if len(coin_coords) > 0:
        coin_rows, coin_cols = zip(*coin_coords)
        coin_map[coin_rows, coin_cols] = 1

    # Create a map of the bombs
    bomb_map = np.zeros(game_map.shape)
    bomb_coords = game_state['bombs']
    if len(bomb_coords) > 0:
        coords, bomb_times = zip(*bomb_coords)
        coords = list(coords)
        bomb_times = np.array(list(bomb_times))
        bomb_rows, bomb_cols = zip(*coords)
        # Normalize the bomb times
        # We want high values (close to 1) for bombs that are about to explode and low values (close to 0) for bombs that are just placed
        bomb_map[bomb_rows, bomb_cols] = (settings.BOMB_TIMER - bomb_times) / settings.BOMB_TIMER
        assert bomb_map.min() >= 0 and bomb_map.max() <= 1

    ## TODO: add info about their bomb possibility to the feature vector after cnn
    other_agents = game_state['others']
    other_agents_map = np.zeros_like(walls_map)
    for _,_,_,(row, col) in other_agents:
        other_agents_map[row, col] = 1

    ## TODO: do something with the info for the bomb possibility
    my_agent = game_state['self']
    my_agent_map = np.zeros_like(walls_map)
    my_agent_map[my_agent[3]] = 1

    freetiles_map = np.zeros_like(walls_map)

    cum_map = crates_map + walls_map + explosion_map + coin_map + bomb_map + freetiles_map + my_agent_map + other_agents_map
    for i in range(freetiles_map.shape[0]):
        for j in range(freetiles_map.shape[1]):
            if cum_map[i,j] == 0:
                freetiles_map[i,j] = 1

    raw_features = np.stack([crates_map, walls_map, explosion_map, coin_map, bomb_map, freetiles_map, my_agent_map, other_agents_map]).astype(float)

    # logger.info(f"Raw features shape: {raw_features.shape}")
    return raw_features
