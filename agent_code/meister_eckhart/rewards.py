from typing import List, Tuple
import events as e
import numpy as np
from collections import Counter, deque
from items import Bomb
import settings
from .config import *

# Custom events
MOVED_CLOSER_TO_COIN = 'MOVED_CLOSER_TO_COIN'
MOVED_FURTHER_FROM_COIN = 'MOVED_FURTHER_FROM_COIN'
AVOIDED_SELF_BOMB = 'AVOIDED_SELF_BOMB'
OUT_OF_BLAST = 'OUT_OF_BLAST'
INTO_BLAST = 'INTO_BLAST'

# Rewards HACK: check ranges again some are too high

COIN_COLLECTION_REWARD = 1
KILLED_OPPONENT_REWARD = 200
INVALID_ACTION_REWARD = -40
KILLED_SELF_REWARD = -100
GOT_KILLED_REWARD = -50
CRATE_DESTROYED_REWARD = 0.5
SURVIVED_ROUND_REWARD = 0.2
MOVE_REWARD = -0.1
MOVED_CLOSER_TO_COIN_REWARD = 10
MOVED_FURTHER_FROM_COIN_REWARD = -15
AVOIDED_SELF_BOMB_REWARD = 0
OUT_OF_BLAST_REWARD = 20
INTO_BLAST_REWARD = -30
BOMB_REWARD = 0

GOOD_BOMB_PLACEMENT_REWARD = 5


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
        e.BOMB_DROPPED: BOMB_REWARD,
        MOVED_CLOSER_TO_COIN: MOVED_CLOSER_TO_COIN_REWARD,
        MOVED_FURTHER_FROM_COIN: MOVED_FURTHER_FROM_COIN_REWARD,
        AVOIDED_SELF_BOMB: AVOIDED_SELF_BOMB_REWARD,
        OUT_OF_BLAST: OUT_OF_BLAST_REWARD,
        INTO_BLAST: INTO_BLAST_REWARD
    }

def crate_destroyer_reward(self, game_state, events: List[str]) -> int:
    """
    Rewards the agent liniarly for destroying crates. Encourages strategic bomb placement.
    """
    return Counter(events)['CRATE_DESTROYED'] * CRATE_DESTROYED_REWARD

def get_blasts(bombs, field):
    """
    Auxiliary function for getting the fields which will be blasted by bombs
    """
    
    blasted_fields = []
    for bomb in bombs :
        bomb = Bomb(bomb[0], "", 2, settings.BOMB_POWER, "")            
        blasted_fields.append(bomb.get_blast_coords(field))
    return blasted_fields

def avoided_self_bomb_reward(self, game_state, events: List[str]):
    """
    Rewards the agent for avoiding its own bomb.
    """
    if(e.BOMB_EXPLODED in events and e.KILLED_SELF not in events):
        events.append(AVOIDED_SELF_BOMB)


def into_out_of_blast(self, old_game_state, new_game_state, events: List[str]):
    """
    Rewards the agent for getting out of the future blast of its own bomb and penalizes for getting into the future blast of its own bomb.
    """
    if(old_game_state['self'][3] in get_blasts(old_game_state['bombs'], old_game_state['field']) and new_game_state['self'][3] not in get_blasts(new_game_state['bombs'], new_game_state['field']) ):
        events.append(OUT_OF_BLAST)
    if(old_game_state['self'][3] not in get_blasts(old_game_state['bombs'], old_game_state['field']) and new_game_state['self'][3] in get_blasts(new_game_state['bombs'], new_game_state['field']) ):
        events.append(INTO_BLAST)


def bfs_to_objective(current_position: Tuple[int, int], objective_coordinates: List[Tuple[int,int]], game_map, explosion_map) -> Tuple[int, int]:
    """
    Rewards the agent for moving towards the objective.
    returns: position of the closest objective as tuple
    """
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    visited_cells = np.zeros_like(game_map) + np.abs(np.where(game_map == 1, 0, game_map))
    bfs_queue = deque()
    bfs_queue.append(((current_position, 0)))

    while bfs_queue:
        current_position, distance = bfs_queue.popleft()
        visited_cells[current_position] = 1
        if current_position in objective_coordinates:
            return current_position

        for move in moves:
            next_position = (current_position[0] + move[0], current_position[1] + move[1])
            if 0 <= next_position[0] < game_map.shape[0] and 0 <= next_position[1] < game_map.shape[1] and \
                game_map[next_position] == 0 and explosion_map[next_position] == 0 and not visited_cells[next_position]:
                bfs_queue.append((next_position, distance + 1))

    return (-1,-1)

def moved_towards_coin_reward(self, old_game_state, game_state, events: List[str]):
    """
    Rewards the agent for moving towards coins.
    """

    # get position of the closest coin to the agent in the old state
    closest_coin_coord = bfs_to_objective(old_game_state['self'][3], old_game_state['coins'], old_game_state['field'], old_game_state['explosion_map'])
    # get position of the closest coin to the agent in the new game state
    new_closest_coin_coord = bfs_to_objective(game_state['self'][3], game_state['coins'], game_state['field'], game_state['explosion_map'])  

    if closest_coin_coord == (-1,-1) or new_closest_coin_coord == (-1,-1):
        return

    # calculate the distance to the closest coin in the old state
    old_distance = np.linalg.norm(np.array(old_game_state['self'][3]) - np.array(closest_coin_coord), ord=1)
    # calculate the distance to the closest coin in the new state
    new_distance = np.linalg.norm(np.array(game_state['self'][3]) - np.array(new_closest_coin_coord), ord=1)

    if old_distance < new_distance and not e.COIN_COLLECTED in events:
        events.append(MOVED_FURTHER_FROM_COIN)
    elif (old_distance > new_distance and not e.COIN_COLLECTED in events):
        events.append(MOVED_CLOSER_TO_COIN)

    
def get_explosion_map(bombs: List[Tuple[Tuple[int, int], int]], field: np.array, explosion_map: np.array) -> np.array:
    """
    Creates explosion map from the placed bombs and updates the field with the explosion map.
    """

    existing_explosions = np.copy(explosion_map)

    # normalize the bomb times so that the smaller the more dangerous

    existing_explosions = existing_explosions * (settings.BOMB_TIMER * -1) + 1
    
    for bomb in bombs:
        bomb_position = bomb[0]
        bomb_timer = bomb[1]

        # normalize the bomb timer
        bomb_timer = bomb_timer - settings.BOMB_TIMER + 1

        # update the obstacles in the field
        field[bomb_position] = -77 # value doesnt matter ; assume you cannot walk over a bomb

        # update the explosion map
        for direction in WALKING_DIRECTIONS:
            for explosion_length in range(0,4):
                direction_np = np.array(direction)
                bomb_position_np = np.array(bomb_position)
                explosion_location = explosion_length * direction_np + bomb_position_np
                explosion_location = tuple(explosion_location)

                # if obstruction in the way stop updating the explosion map in that direction
                if field[explosion_location] == -1:
                    break

                # if a more recent bomb with more steps to live is there, keep that instead
                existing_explosions[explosion_location] = min(existing_explosions[explosion_location], bomb_timer)


    return existing_explosions
                    

def simulate_explosion_map(current_explosion_map: np.array, bomb_position: Tuple[int, int], field: np.array) -> np.array:
    '''
    We need this in the situations where we want to see if placing a bomb will kill us or not
    '''

    cur_map = np.copy(current_explosion_map)

    # simulate one step by decreasing the timer
    cur_map[cur_map < 1] -= 1

    # bomb timers that are < -3 have already exploded
    cur_map[cur_map < -3] = 1

    # current bomb time is 0 because it has just been placed (1 no bomb, 0 just placed)
    bomb_timer = 0

    # update the explosion map
    for direction in WALKING_DIRECTIONS:
        for explosion_length in range(0,4):
            explosion_location = explosion_length * direction + bomb_position

            # if obstruction in the way stop updating the explosion map in that direction
            if field[explosion_location] == -1:
                break

            # if a more recent bomb with more steps to live is there, keep that instead
            cur_map[explosion_location] = min(cur_map[explosion_location], bomb_timer)

    return cur_map


def check_death(explosion_map: np.array, position: Tuple[int, int], unreacheable_positions: List[Tuple[int,int]], field: np.array) -> bool:
    '''
    Returns True if the agent will die regardless of available moves or false if it can escape certain death.
    '''

    if unreacheable_positions is None:
        visited_pos = []
    else:
        visited_pos = unreacheable_positions.copy()

    queue = deque()
    queue.append((position, 0)) # takes 0 turns to reach the current position

    while queue:
        current_position, turns = queue.popleft()

        if current_position in visited_pos:
            continue

        visited_pos.append(current_position)

        # 1 means it is a safe cell
        if explosion_map[current_position] == 1:
            return False

        if explosion_map[current_position] == 4 - turns:
            continue

        # add neighbors to the queue
        for direction in WALKING_DIRECTIONS:
            next_position = current_position + direction

            # if it is an empty cell
            if field[next_position] == 0:
                queue.append((next_position, turns + 1))

    return True

def get_number_exploded_crates(position: Tuple[int, int], explosion_map: np.array, field: np.array) -> int:
    '''
    Returns the number of crates that will be exploded if a bomb is placed at the given position.
    '''
    exploded_crates = 0

    for direction in WALKING_DIRECTIONS:
        for explosion_length in range(0,4):
            explosion_location = explosion_length * direction + position

            # if obstruction in the way stop updating the explosion map in that direction
            if field[explosion_location] == -1:
                break

            # if a crate is there and no other explosion / bomb is gonna take that crate
            if field[explosion_location] == 1 and explosion_map[explosion_location] == 1:
                exploded_crates += 1

    return exploded_crates
    

