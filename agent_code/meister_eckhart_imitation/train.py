from collections import namedtuple, deque

import pickle
from typing import List
import os
from .callbacks import state_to_features
from .model import ReplayMemory
from .model import train_step
from .exploration_strategies import *
import matplotlib.pyplot as plt
from .rewards import *
from tqdm import tqdm
from .callbacks import MODEL_SAVE_PATH
import torch
from .rewards import *
from .config import *

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.memory = ReplayMemory(MEMORY_SIZE)
    self.losses = []
    self.scores = []
    self.round_scores = 0
    self.waited_times = 0

    self.position_history = deque(maxlen=POSITION_HISTORY_SIZE)

    self.round_custom_scores = []
    
    self.epsilon_values = []

    self.crates_destroyed_per_round = 0
    self.crates_destroyed_list = []

    self.round_reward = 0
    self.running_steps = 0
    self.running_reward = []
    
    self.survived_steps_per_round = []

    # TODO: problem with train from checkpoint
    if os.path.isfile(RULE_BASE_TRANSITIONS_PATH):
       self.rule_based_training_memory = load_transitions(self.logger, RULE_BASE_TRANSITIONS_PATH)

       # train on them

       for epoch in tqdm(range(RULE_BASED_TRAINING_EPOCHS)):
            loss = train_step(self, BATCH_SIZE, GAMMA, TRAIN_DEVICE,self.rule_based_training_memory)
            self.losses.append(loss)

            # update target net

            if epoch % 4 == 0:
                # set the target net weights to the ones of the policy net
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if epoch % SAVE_MODEL_EVERY == 0:
                checkpoint = {
                    'model_state_dict': self.policy_net.state_dict(),  # Save model weights
                    'optimizer_state_dict': self.optimizer.state_dict(),  # Save optimizer state
                }
            
                torch.save(checkpoint, MODEL_SAVE_PATH)

                plt.clf()

                plt.plot(self.losses)
                plt.xlabel("Training steps")
                plt.ylabel("Loss")
                plt.title("Training losses")
                plt.savefig("logs/policy_net_losses" +".png")

            
        
        # stop the program here

    else:
       # collect
       self.rule_based_training_memory = ReplayMemory(MEMORY_SIZE_RULE_BASED)

        

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    pass 

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    if len(self.rule_based_training_memory) >= MEMORY_SIZE_RULE_BASED - 1 and not os.path.exists(RULE_BASE_TRANSITIONS_PATH):
        save_transitions(self.rule_based_training_memory, RULE_BASE_TRANSITIONS_PATH)
        self.logger.info("Saved rule based training memory")


  
'''
# NOTE: run with continue without training to get all the transitions
'''
def enemy_game_events_occurred(self, enemy_name: str, old_enemy_game_state: dict, enemy_action: str, new_enemy_game_state: dict, enemy_events: List[str]):
   '''
   Function signature taken from the discord channel of the course.
   Use a random sampling to spread out the transition types so we can get some from further in the game

   '''
       
   if enemy_name == 'rule_based_agent' and not os.path.exists(RULE_BASE_TRANSITIONS_PATH): # NOTE: this has to be changed in coin heavn to rule_based_agent
        self.logger.debug(f'xxxxx Enemy {enemy_name} has events: {enemy_events} and has taken action {enemy_action}')
        if enemy_action is not None and random.random() < RECORD_ENEMY_TRANSITIONS: # when the enemy is dead
            self.rule_based_training_memory.append(Transition(state_to_features(old_enemy_game_state), enemy_action, state_to_features(new_enemy_game_state), reward_from_events(self, enemy_events)))


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = GAME_REWARDS
    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



def get_score(events: List[str]) -> int:
    '''
    tracks the true score we use for evaluating our agents

    :param events: events that occured in game step
    '''
    true_game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
    }
    score = 0
    for event in events:
        if event in true_game_rewards:
            score += true_game_rewards[event]
    return score



def save_transitions(transitions: List[Transition],path: str):
    '''
    Save transitions to the replay memory
    '''
    with open(path, 'wb') as file:
        pickle.dump(transitions, file)

def load_transitions(logger,path: str) -> ReplayMemory:
    '''
    Load transitions from the replay memory
    '''
    with open(path, 'rb') as file:
        buffer = pickle.load(file)
        logger.info(f"Loaded {len(buffer)} transitions. Type of buffer: {type(buffer)}")
        return buffer
    

    
