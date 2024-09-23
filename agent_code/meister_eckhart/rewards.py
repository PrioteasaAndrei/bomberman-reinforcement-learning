
from typing import List, Tuple
import events as e
import numpy as np
from collections import Counter, deque
from items import Bomb
import settings
from .config import WALKING_DIRECTIONS

#Avoid waiting too long and wiggling parameters
LONG_WAIT_LIMIT = 3
POSITION_HISTORY_SIZE = 7
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
ENEMY_IN_RANGE_OF_BOMB = 'ENEMY_IN_RANGE_OF_BOMB'
CRATES_IN_RANGE_OF_BOMB = 'CRATES_IN_RANGE_OF_BOMB'
WILL_EXPLODE_MORE_THAN_3_CRATES = 'WILL_EXPLODE_MORE_THAN_3_CRATES'
INTO_EXPLOSION = 'INTO_EXPLOSION'
NOT_LEAVING_EXPLOSION = 'NOT_LEAVING_EXPLOSION'
MOVED_CLOSER_TO_BOMB = 'MOVED_CLOSER_TO_BOMB'
MOVED_FURTHER_FROM_BOMB = 'MOVED_FURTHER_FROM_BOMB'

# Movement rewards
INVALID_ACTION_REWARD = -100
MOVE_REWARD = -0.1
LONG_WAIT_REWARD = -100
WIGGLING_REWARD = -150

#Crate rewards
CRATE_DESTROYED_REWARD = 8

#Coin rewards
COIN_COLLECTION_REWARD = 10 
MOVED_CLOSER_TO_COIN_REWARD = 0.8 
MOVED_FURTHER_FROM_COIN_REWARD = -1 


#Bomb rewards
KILLED_OPPONENT_REWARD = 200
KILLED_SELF_REWARD = -100
GOT_KILLED_REWARD = -50
AVOIDED_SELF_BOMB_REWARD = 20
BOMB_REWARD = -0.1
OUT_OF_BLAST_REWARD = 50    #not in blast tiles anymore
INTO_BLAST_REWARD = -60     #went back into blast tiles
INTO_EXPLOSION_REWARD = -40       #went back into blast tiles as the bombs explodes, killing the agent
NOT_LEAVING_EXPLOSION_REWARD = -50    #waiting or invalid action in blast tiles
MOVED_CLOSER_TO_BOMB_REWARD = -5     #approaching bomb
MOVED_FURTHER_FROM_BOMB_REWARD = 4 #going away from bomb
PLACED_BOMB_IN_CORNER_REWARD = -100
WILL_EXPLODE_MORE_THAN_3_CRATES_REWARD = 30
CRATES_IN_RANGE_OF_BOMB_REWARD = 12
ENEMY_IN_RANGE_OF_BOMB_REWARD = 30
SURVIVED_ROUND_REWARD = 50


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
        e.WAITED: MOVE_REWARD, 
        e.BOMB_DROPPED: BOMB_REWARD,
        MOVED_CLOSER_TO_COIN: MOVED_CLOSER_TO_COIN_REWARD,
        MOVED_FURTHER_FROM_COIN: MOVED_FURTHER_FROM_COIN_REWARD,
        AVOIDED_SELF_BOMB: AVOIDED_SELF_BOMB_REWARD,
        OUT_OF_BLAST: OUT_OF_BLAST_REWARD,
        INTO_BLAST: INTO_BLAST_REWARD,
        LONG_WAIT: LONG_WAIT_REWARD,
        WIGGLING: WIGGLING_REWARD,
        PLACED_BOMB_IN_CORNER: PLACED_BOMB_IN_CORNER_REWARD,
        WILL_EXPLODE_MORE_THAN_3_CRATES: WILL_EXPLODE_MORE_THAN_3_CRATES_REWARD,
        CRATES_IN_RANGE_OF_BOMB: CRATES_IN_RANGE_OF_BOMB_REWARD,
        ENEMY_IN_RANGE_OF_BOMB: ENEMY_IN_RANGE_OF_BOMB_REWARD,
        INTO_EXPLOSION: INTO_EXPLOSION_REWARD,
        NOT_LEAVING_EXPLOSION: NOT_LEAVING_EXPLOSION_REWARD,
        MOVED_CLOSER_TO_BOMB: MOVED_CLOSER_TO_BOMB_REWARD,
        MOVED_FURTHER_FROM_BOMB: MOVED_FURTHER_FROM_BOMB_REWARD

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
        if count >= REPEATED_POSITION_LIMIT:
            frequent_positions += 1
    
    if frequent_positions > FREQUENT_POSITION_LIMIT:
        events.append(WIGGLING)


def get_blasts(bombs, field):
    """
    Auxiliary function for getting the fields which will be blasted by bombs
    """
    
    blasted_fields = []
    for bomb in bombs :
        bomb = Bomb(bomb[0], "", 2, settings.BOMB_POWER, "")            
        blasted_fields.append(bomb.get_blast_coords(field))
    if(blasted_fields == []):
        return []
    return blasted_fields[0]

def avoided_self_bomb_reward(self, game_state, events: List[str]):
    """
    Rewards the agent for avoiding its own bomb.
    """
    if(e.BOMB_EXPLODED in events and e.KILLED_SELF not in events):
        events.append(AVOIDED_SELF_BOMB)


def bfs_to_bombs(current_position: Tuple[int, int], objective_coordinates: List[Tuple[int,int]], game_map) -> Tuple[int, int]:
    """
    Rewards the agent for moving towards or away the objective.
    returns: position of the closest objective as tuple
    """
    if (current_position in objective_coordinates): #We are standing on a bomb
        return current_position
    
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    visited_cells = np.zeros_like(game_map) + np.abs(np.where(game_map == 1, 0, game_map))
    bfs_queue = deque()
    bfs_queue.append(((current_position, 0)))
    checked_fields = 0

    while bfs_queue and checked_fields <= 50:
        current_position, distance = bfs_queue.popleft()
        visited_cells[current_position] = 1
        checked_fields += 1
        if current_position in objective_coordinates:
            return current_position

        for move in moves:
            next_position = (current_position[0] + move[0], current_position[1] + move[1])
            if 0 <= next_position[0] < game_map.shape[0] and 0 <= next_position[1] < game_map.shape[1] and \
                game_map[next_position] == 0 and not visited_cells[next_position]:
                bfs_queue.append((next_position, distance + 1))

    return (-1,-1)  

def blast_events(self, old_game_state, new_game_state, events: List[str], logger):
    """
    Rewards the agent for getting out of the future blast of its own bomb and penalizes for getting into the future blast of its own bomb.
    """
    old_coords = old_game_state['self'][3]
    new_coords = new_game_state['self'][3]
    old_blasts = get_blasts(old_game_state['bombs'], old_game_state['field'])
    new_blasts = get_blasts(new_game_state['bombs'], new_game_state['field'])
    
    old_bombs = old_game_state['bombs']
    new_bombs = new_game_state['bombs']
    if (old_bombs != None and new_bombs != None):
        old_bombs = list(coords for coords,_ in old_bombs)
        new_bombs = list(coords for coords,_ in new_bombs)

        old_closest_bomb = bfs_to_bombs(old_coords, old_bombs, old_game_state['field'])
        new_closest_bomb = bfs_to_bombs(new_coords, new_bombs, new_game_state['field'])
        old_distance = np.linalg.norm(np.array(old_coords) - np.array(old_closest_bomb), ord=1)
        new_distance = np.linalg.norm(np.array(new_coords) - np.array(new_closest_bomb), ord=1)

        if(old_coords in old_blasts) and (old_distance < new_distance):
            events.append(MOVED_CLOSER_TO_BOMB)

        if(old_coords in old_blasts) and (old_distance > new_distance):
            events.append(MOVED_FURTHER_FROM_BOMB)
        

    if(old_coords in old_blasts) and (new_coords not in new_blasts):
        events.append(OUT_OF_BLAST)

    if(old_coords not in old_blasts) and (new_coords in new_blasts):
        events.append(INTO_BLAST)

    if(new_coords in new_blasts) and ((e.WAITED or e.INVALID_ACTION) in events):
        events.append(NOT_LEAVING_EXPLOSION)

    if(old_coords not in old_blasts) and ((e.KILLED_SELF or e.GOT_KILLED) in events):
        events.append(INTO_EXPLOSION)
    

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

    
def placed_bomb_in_corner(self,old_game_state, events: List[str]):
    """
    Rewards the agent for placing a bomb in a corner.
    """

    if e.BOMB_DROPPED in events:
        bomb_position = old_game_state['self'][3]
        if bomb_position in [(1,1), (1,15), (15,1), (15,15)]:
            events.append('PLACED_BOMB_IN_CORNER')
            self.logger.info(f'Placed bomb in corner at {bomb_position}')

    
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
        field[bomb_position] = -10 # assume you cannot walk over a bomb

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
    Check if placing a bomb will kill our agent or not
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

def get_bomb_reward(game_state: dict , events: List[str]) -> int:
    
    if game_state is None:
        return None
    
    agent_position = game_state['self'][3]
    other_agents_positions = [agent[3] for agent in game_state['others']]
    explosion_map = get_explosion_map(game_state['bombs'], game_state['field'], game_state['explosion_map'])
    field = game_state['field']

    potential_crate = False
    oponent_in_reach = False
    bomb_feature = 0 

    # free from bombs / explosion
    if explosion_map[agent_position] == 1: 
        for step in WALKING_DIRECTIONS:
            possible_position = (agent_position[0] + step[0], agent_position[1] + step[1])
            if possible_position in other_agents_positions:
                oponent_in_reach = True
                events.append(ENEMY_IN_RANGE_OF_BOMB)
            if field[possible_position] == 1 and explosion_map[possible_position] == 1:
                potential_crate = True
                events.append(CRATES_IN_RANGE_OF_BOMB)

    if check_death(simulate_explosion_map(explosion_map, agent_position, field), agent_position, None, field):
        events.append(e.KILLED_SELF)

    if game_state['self'][2]:
        bomb_feature = get_number_exploded_crates(agent_position, explosion_map, field)
        if bomb_feature >= 3:
            events.append(WILL_EXPLODE_MORE_THAN_3_CRATES)
