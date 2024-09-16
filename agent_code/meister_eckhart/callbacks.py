import os
import pickle
import random
import numpy as np
from .model import *
import torch
import torch.optim as optim
import logging
import settings
from .exploration_strategies import *
from .config import *

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
    global TRAIN_FROM_CHECKPOINT
    if not self.train:
        TRAIN_FROM_CHECKPOINT = False
    
    if self.train and os.path.isfile(MODEL_SAVE_PATH) and TRAIN_FROM_CHECKPOINT:
        self.logger.info("Loading model from saved state for further training.")
        self.policy_net = create_model(input_shape=(8, 7, 7), num_actions=6, logger=self.logger, model_type=MODEL_TYPE).to(TRAIN_DEVICE)
        self.target_net = create_model(input_shape=(8, 7, 7), num_actions=6, logger=self.logger, model_type=MODEL_TYPE).to(TRAIN_DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        # load the checkpoint
        checkpoint = torch.load(MODEL_SAVE_PATH)
        # restore model state, optimizer state, loss and epoch (if needed)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # if necessary for plotting
        self.n_round = checkpoint['n_round']
        self.loss = checkpoint['loss']
        # set update strategy
        # could also be saved at checkpoint
        self.epsilon_update_strategy = checkpoint['epsilon_strategy']
    elif self.train or not os.path.isfile(MODEL_SAVE_PATH):
        self.logger.info("Setting up model from scratch.")
        self.policy_net = create_model(input_shape=(8, 7, 7), num_actions=6, logger=self.logger, model_type=MODEL_TYPE).to(TRAIN_DEVICE)
        self.target_net = create_model(input_shape=(8, 7, 7), num_actions=6, logger=self.logger, model_type=MODEL_TYPE).to(TRAIN_DEVICE)

        self.logger.info(f"Number of parameters in the model: {self.policy_net.number_of_params()}")
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        # self.epsilon_update_strategy = LinearDecayStrategy(start_epsilon=1.0, min_epsilon=0.1, decay_steps=1000)
        self.epsilon_update_strategy = ExponentialDecayStrategy(start_epsilon=1.0, min_epsilon=0.1, decay_rate=0.999)
    else:
        self.logger.info("Loading model from saved state for inference.")
        self.policy_net = create_model(input_shape=(8, 7, 7), num_actions=6, logger=self.logger, model_type=MODEL_TYPE).to(TRAIN_DEVICE)
        checkpoint = torch.load(MODEL_SAVE_PATH)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        # with open(MODEL_SAVE_PATH, "rb") as file:
        #     self.policy_net = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    raw_features = state_to_features(game_state,self.logger)
    raw_features = torch.tensor(raw_features).unsqueeze(0)  # Shape becomes (1, 8, 17, 17)
    self.logger.debug(f"Raw features shape: {raw_features.shape}")
    outputs = self.policy_net(raw_features.float().to(TRAIN_DEVICE))
    outputs_list = outputs.detach().cpu().numpy().flatten().tolist()
    # apply softmax to the outputs
    outputs_list = np.exp(outputs_list) / np.sum(np.exp(outputs_list)) # HACK: this shouldnt be the case
 
    self.logger.info(f"Number of parameters in the model: {self.policy_net.number_of_params()}")
    self.logger.info(f"Feature space size: {self.policy_net.dqn_input_size}")


    if self.train:
        random_prob = self.epsilon_update_strategy.epsilon
        self.epsilon_update_strategy.update_epsilon(3) # step is irelevant for linear decay
        self.logger.info(f"Epsilon: {random_prob}")
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])


    ## TODO: move inference here, there is no point in doing it if we chose a random action
    ## TODO: move score measuring here
    self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=outputs_list)
    self.logger.info(f"Chosen action: {ACTIONS[np.argmax(outputs_list)]}")
    return ACTIONS[np.argmax(outputs_list)]


# TODO: Check this
def crop_map(map, agent_pos, crop_size, logger=None):
    """
    Crop the map around the agent position. The agent is in the middle of the cropped map. The cropped map is a square.
    """

    x, y = agent_pos
    x_min = max(0, x - crop_size // 2 )
    x_max = min(map.shape[0] - 1, x_min + crop_size - 1)
    x_min = x_max + 1 - crop_size

    y_min = max(0, y - crop_size // 2)
    y_max = min(map.shape[1] - 1, y_min + crop_size - 1)
    y_min = y_max + 1 - crop_size

    #logger.info(f"Agent position: {agent_pos}")
    #logger.info(f"x min: {x_min}, x max: {x_max}, y min: {y_min}, y max: {y_max}")

    return map[x_min:x_max+1, y_min:y_max+1]

def state_to_features(game_state: dict, logger=None) -> np.array:
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

    #crop the maps around the agent position
    #logger.info("-----------------")
    crates_map = crop_map(crates_map, my_agent[3], CROP_SIZE,logger)
    #logger.info(f"Crates map shape: {crates_map.shape}")
    walls_map = crop_map(walls_map, my_agent[3], CROP_SIZE,logger)
    #logger.info(f"Walls map shape: {walls_map.shape}")
    explosion_map = crop_map(explosion_map, my_agent[3], CROP_SIZE,logger)
    #logger.info(f"Explosion map shape: {explosion_map.shape}")
    coin_map = crop_map(coin_map, my_agent[3], CROP_SIZE,logger)
    #logger.info(f"Coin map shape: {coin_map.shape}")
    bomb_map = crop_map(bomb_map, my_agent[3], CROP_SIZE,logger)
    #logger.info(f"Bomb map shape: {bomb_map.shape}")
    freetiles_map = crop_map(freetiles_map, my_agent[3], CROP_SIZE,logger)
    #logger.info(f"Freetiles map shape: {freetiles_map.shape}")
    my_agent_map = crop_map(my_agent_map, my_agent[3], CROP_SIZE,logger)
    #logger.info(f"My agent map shape: {my_agent_map.shape}")
    other_agents_map = crop_map(other_agents_map, my_agent[3], CROP_SIZE,logger)
    #logger.info(f"Other agents map shape: {other_agents_map.shape}")
    #logger.info("-----------------")


    raw_features = np.stack([crates_map, walls_map, explosion_map, coin_map, bomb_map, freetiles_map, my_agent_map, other_agents_map]).astype(float)
    # logger.info(f"Raw features shape: {raw_features.shape}")
    return raw_features
