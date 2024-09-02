from collections import namedtuple, deque

import pickle
from typing import List

# import events as e
from .callbacks import state_to_features
from .model import ReplayMemory
from .model import train_step
from .exploration_strategies import *
import matplotlib.pyplot as plt
from .rewards import *

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
RECORD_ENEMY_TRANSITIONS = 1.0 # record enemy transitions with probability ...
GAMMA = 0.99
MEMORY_SIZE = 10000
BATCH_SIZE = 64

TRAIN_DEVICE = 'mps'
ROUND_TO_PLOT = 200


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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # check for custom events
    moved_towards_coin_reward(self, old_game_state, new_game_state, events)

    # state_to_features is defined in callbacks.py
    self.memory.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    self.round_scores += get_score(events)


    if self.memory.can_provide_sample(BATCH_SIZE):
        # Train your agent
        self.logger.info("Initiating one step of training...")
        loss = train_step(self, BATCH_SIZE, GAMMA, TRAIN_DEVICE)
        self.losses.append(loss)

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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.memory.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    self.scores.append(self.round_scores)
    self.round_scores = 0

    # update target net
    if last_game_state['round'] % 4 == 0:
        # set the target net weights to the ones of the policy net
        self.target_net.load_state_dict(self.policy_net.state_dict())


    if last_game_state['round'] == ROUND_TO_PLOT:
        # Plot the losses
        plt.plot(self.losses[::10])
        plt.xlabel("Training steps")
        plt.ylabel("Loss")
        plt.title("Training losses")
        plt.savefig("logs/policy_net_losses" +".png")
        plt.clf()

        ## number of scores higher that 4
        big_scores = len([score for score in self.scores if score > 4])
        # Plot the scores
        plt.plot(self.scores)
        plt.xlabel("Training steps")
        plt.ylabel("Score")
        plt.title("Scores. We have " + str(big_scores) + " scores higher than 4")
        plt.savefig("logs/scores" +".png")


    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.policy_net, file)

#TODO: we can define different strategies: aggresive vs defensive
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


## TODO: rewrite
def get_score(events: List[str]) -> int:
    '''
    tracks the true score

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
