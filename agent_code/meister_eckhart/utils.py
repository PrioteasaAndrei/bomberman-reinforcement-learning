import numpy as np
import events as e
import settings
import logging

'''
NOTE: they create a view size, a window of things that are visible to the agent, smaller than the entire game optimizaiton
purposes. We might consider doing the same

NOTE: try to keep all values normalized. They use -0.5 0.5, we try to use 0 1, but it may be a problem because we dont have
negative reinforcement. not sure

NOTE: would a CNN really be necessary? They just flatten all the channels and feed them to the DQN, maybe reducing the information even more may not be
worth the training effort
'''

def state_to_raw_features(logger: logging.Logger ,game_state: dict) -> np.array:
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

    logger.info(f"Raw features shape: {raw_features.shape}")
    return raw_features