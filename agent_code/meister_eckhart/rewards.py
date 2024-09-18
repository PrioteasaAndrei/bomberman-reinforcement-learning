from typing import List, Tuple
import events as e
import numpy as np
from collections import Counter, deque
from items import Bomb
import settings

LONG_WAIT_LIMIT = 3
POSITION_HISTORY_SIZE = 6
REPEATED_POSITION_LIMIT = 3
FREQUENT_POSITION_LIMIT = 1


# Custom events
MOVED_CLOSER_TO_COIN = 'MOVED_CLOSER_TO_COIN'
MOVED_FURTHER_FROM_COIN = 'MOVED_FURTHER_FROM_COIN'
AVOIDED_SELF_BOMB = 'AVOIDED_SELF_BOMB'
OUT_OF_BLAST = 'OUT_OF_BLAST'
INTO_BLAST = 'INTO_BLAST'
LONG_WAIT = 'LONG_WAIT'
WIGGLING = 'WIGGLING'
PLACED_BOMB_IN_CORNER = 'PLACED_BOMB_IN_CORNER'

# Rewards
COIN_COLLECTION_REWARD = 10 #default 1
KILLED_OPPONENT_REWARD = 200
INVALID_ACTION_REWARD = -100
KILLED_SELF_REWARD = -100
GOT_KILLED_REWARD = -50
CRATE_DESTROYED_REWARD = 8
SURVIVED_ROUND_REWARD = 0.2
MOVE_REWARD = -0.1
MOVED_CLOSER_TO_COIN_REWARD = 0.8 #default 0.4
MOVED_FURTHER_FROM_COIN_REWARD = -1 #default -0.6
AVOIDED_SELF_BOMB_REWARD = 0
OUT_OF_BLAST_REWARD = 20
INTO_BLAST_REWARD = -30
BOMB_REWARD = -0.1
LONG_WAIT_REWARD = -100
WIGGLING_REWARD = -100
PLACED_BOMB_IN_CORNER_REWARD = -100


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
        # e.WAITED: MOVE_REWARD, # penalize just waiting
        e.BOMB_DROPPED: BOMB_REWARD,
        MOVED_CLOSER_TO_COIN: MOVED_CLOSER_TO_COIN_REWARD,
        MOVED_FURTHER_FROM_COIN: MOVED_FURTHER_FROM_COIN_REWARD,
        AVOIDED_SELF_BOMB: AVOIDED_SELF_BOMB_REWARD,
        OUT_OF_BLAST: OUT_OF_BLAST_REWARD,
        INTO_BLAST: INTO_BLAST_REWARD,
        LONG_WAIT: LONG_WAIT_REWARD,
        WIGGLING: WIGGLING_REWARD,
        PLACED_BOMB_IN_CORNER: PLACED_BOMB_IN_CORNER_REWARD
    }

def avoid_long_wait(self, events: List[str]):
    """
    Penalize too long waits.
    """
    if e.WAITED in events:
        self.waited_times += 1
    else:
        self.waited_times = 0

    if self.waited_times > LONG_WAIT_LIMIT:
        events.append(LONG_WAIT)

def avoid_wiggling(self, events: List[str]):

    position_freq = Counter(self.position_history)
    frequent_positions = 0

    for pos, count  in position_freq.items():
        if count > REPEATED_POSITION_LIMIT:
            frequent_positions += 1
    
    if frequent_positions > FREQUENT_POSITION_LIMIT:
        events.append(WIGGLING)

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


def bfs_to_objective(current_position: Tuple[int, int], objective_coordinates: List[Tuple[int,int]], game_map) -> Tuple[int, int]:
    #IDEA: crop the game_map to match the view of the agent
    """
    Rewards the agent for moving towards the objective.
    returns: position of the closest objective as tuple
    """
    #TODO EXPLOSIONS NOT CONSIDERED
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
                game_map[next_position] == 0 and not visited_cells[next_position]:
                bfs_queue.append((next_position, distance + 1))

    return (-1,-1)

def moved_towards_coin_reward(self, old_game_state, game_state, events: List[str]):
    """
    Rewards the agent for moving towards coins.
    """

    # get position of the closest coin to the agent in the old state
    closest_coin_coord = bfs_to_objective(old_game_state['self'][3], old_game_state['coins'], old_game_state['field'])
    # get position of the closest coin to the agent in the new game state
    new_closest_coin_coord = bfs_to_objective(game_state['self'][3], game_state['coins'], game_state['field'])

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

    
def placed_bomb_in_corner(self,old_game_state, events: List[str]):
    """
    Rewards the agent for placing a bomb in a corner.
    """

    if e.BOMB_DROPPED in events:
        bomb_position = old_game_state['self'][3]
        if bomb_position in [(1,1), (1,15), (15,1), (15,15)]:
            events.append('PLACED_BOMB_IN_CORNER')
            self.logger.info(f'Placed bomb in corner at {bomb_position}')
