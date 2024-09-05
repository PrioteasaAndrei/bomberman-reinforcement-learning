from typing import List, Tuple
import events as e
import settings as s
import numpy as np
from collections import Counter, deque
from items import Bomb

# Custom events
MOVED_CLOSER_TO_COIN = 'MOVED_CLOSER_TO_COIN'
MOVED_FURTHER_FROM_COIN = 'MOVED_FURTHER_FROM_COIN'
AVOIDING_BOMB = 'AVOIDING_SELF_BOMB'
GOING_TO_BOMB = 'GOING_TO_BOMB'
MOVING_AWAY_FROM_BOMB = 'MOVING_AWAY_FROM_BOMB'
MOVING_CLOSER_TO_BOMB = 'MOVING_CLOSER_TO_BOMB'

# Rewards
COIN_COLLECTION_REWARD = 1
KILLED_OPPONENT_REWARD = 5
INVALID_ACTION_REWARD = -50
KILLED_SELF_REWARD = -100
GOT_KILLED_REWARD = -50
CRATE_DESTROYED_REWARD = 0.5
SURVIVED_ROUND_REWARD = 0.2
MOVE_REWARD = -0.1
MOVED_CLOSER_TO_COIN_REWARD = 0.4
MOVED_FURTHER_FROM_COIN_REWARD = -0.6
AVOIDING_BOMB_REWARD = 0.6
GOING_TO_BOMB_REWARD = -0.6
MOVING_AWAY_FROM_BOMB_REWARD = 0.6
MOVING_CLOSER_TO_BOMB_REWARD = -0.8

GAME_REWARDS = {
        e.COIN_COLLECTED: COIN_COLLECTION_REWARD,
        e.KILLED_OPPONENT: KILLED_OPPONENT_REWARD,
        e.INVALID_ACTION: INVALID_ACTION_REWARD,
        e.KILLED_SELF: KILLED_SELF_REWARD,
        e.GOT_KILLED: GOT_KILLED_REWARD,
        e.CRATE_DESTROYED: CRATE_DESTROYED_REWARD,
        e.SURVIVED_ROUND: SURVIVED_ROUND_REWARD,
        e.MOVED_DOWN: MOVE_REWARD,
        e.MOVED_LEFT: MOVE_REWARD,
        e.MOVED_RIGHT: MOVE_REWARD,
        e.MOVED_UP: MOVE_REWARD,
        MOVED_CLOSER_TO_COIN: MOVED_CLOSER_TO_COIN_REWARD,
        MOVED_FURTHER_FROM_COIN: MOVED_FURTHER_FROM_COIN_REWARD,
        AVOIDING_BOMB: AVOIDING_BOMB_REWARD,
        GOING_TO_BOMB: GOING_TO_BOMB_REWARD,
        MOVING_AWAY_FROM_BOMB: MOVING_AWAY_FROM_BOMB_REWARD,
        MOVING_CLOSER_TO_BOMB: MOVING_CLOSER_TO_BOMB_REWARD
    }

def crate_destroyer_reward(self, game_state, events: List[str]) -> int:
    """
    Rewards the agent liniarly for destroying crates. Encourages strategic bomb placement.
    """

    # TODO: calculate the number of destroyed crates by the agent
    

    # return no_destoyed_crates * CRATE_DESTROYED_REWARD

def avoided_bomb_reward(self, old_game_state, game_state, events: List[str]):
    """
    Rewards the agent for escaping the future blast of a bomb.
    """
    agent_old_position = old_game_state['self'][3]
    agent_new_position = game_state['self'][3]
    bomb_old_blast_coords = bomb_blasts(old_game_state['bombs'], old_game_state['field'])
    bomb_new_blast_coords = bomb_blasts(game_state['bombs'], game_state['field'])

    if agent_old_position in bomb_old_blast_coords and agent_new_position not in bomb_new_blast_coords:
        events.append(AVOIDING_BOMB)
    elif agent_old_position not in bomb_old_blast_coords and agent_new_position in bomb_new_blast_coords:
        events.append(GOING_TO_BOMB)
    elif agent_old_position in bomb_old_blast_coords and agent_new_position in bomb_new_blast_coords:
        closest_old_bomb, old_dist = bfs_to_objective(agent_old_position, bomb_old_blast_coords, old_game_state['field'])
        closest_new_bomb, new_dist = bfs_to_objective(agent_new_position, bomb_new_blast_coords, game_state['field'])
        if new_dist < old_dist:
            events.append(MOVING_AWAY_FROM_BOMB)
        elif new_dist > old_dist:
            events.append(MOVING_CLOSER_TO_BOMB)

def bomb_blasts(bombs, game_map):
    """
    Returns a list of coordinates of the blast of all bombs.
    """
    blast_coords = []
    for bomb_coord, time in bombs:
        bomb = Bomb(bomb_coord, "", time, s.BOMB_POWER, "")
        blast_coords = bomb.get_blast_coords(game_map)
    return list(set(blast_coords))


def bfs_to_objective(current_position: Tuple[int, int], objective_coordinates: List[Tuple[int,int]], game_map) -> Tuple[int, int]:
    """
    Rewards the agent for moving towards the objective.
    returns: position of the closest objective as tuple
    """
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    visited_cells = np.zeros_like(game_map)
    bfs_queue = deque()
    bfs_queue.append(((current_position, 0)))

    while bfs_queue:
        current_position, distance = bfs_queue.popleft()
        visited_cells[current_position] = 1
        if current_position in objective_coordinates:
            return current_position, distance

        for move in moves:
            next_position = (current_position[0] + move[0], current_position[1] + move[1])
            if 0 <= next_position[0] < game_map.shape[0] and 0 <= next_position[1] < game_map.shape[1] and \
                game_map[next_position] == 0 and not visited_cells[next_position]:
                bfs_queue.append((next_position, distance + 1))

    return (-1,-1), -1

def moved_towards_coin_reward(self, old_game_state, game_state, events: List[str]):
    """
    Rewards the agent for moving towards coins.
    """

    # get position of the closest coin to the agent in the old state
    closest_coin_coord, _ = bfs_to_objective(old_game_state['self'][3], old_game_state['coins'], old_game_state['field'])
    # get position of the closest coin to the agent in the new game state
    new_closest_coin_coord, _ = bfs_to_objective(game_state['self'][3], game_state['coins'], game_state['field'])

    if closest_coin_coord == (-1,-1) or new_closest_coin_coord == (-1,-1):
        return

    # calculate the distance to the closest coin in the old state
    old_distance = np.linalg.norm(np.array(old_game_state['self'][3]) - np.array(closest_coin_coord), ord=1)
    # calculate the distance to the closest coin in the new state
    new_distance = np.linalg.norm(np.array(game_state['self'][3]) - np.array(new_closest_coin_coord), ord=1)

    if old_distance < new_distance and not e.COIN_COLLECTED in events:
        return events.append(MOVED_FURTHER_FROM_COIN)
    else:
        return events.append(MOVED_CLOSER_TO_COIN)

    