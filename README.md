# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

# Project guidelines (taken from the pdf)

- You should develop two (or more) diferent models and submit your best performing model to the tournament
- one of the model needs to use a techinque from the lecture (so q-learning)
- You can share your trained agents (without training code) on #final-project-beat-my-agent and download other teams' agents to test your approach.


# Game instructions

## Bombs
- will detonate after 4 stepts
- creates explosion 3 tiles up, down, left and right (plus current cell, so at most 7 tiles in a row or column)
- the explosion in dangerous for 2 rounds
- agents can drop a new bomb after explosion is gone (so 2 + 4 = 6 rounds)

## Points
- blow agent: 5 points
- take coin: 1 point

## Moves
- move a tile, drop bomb, or just wait
- 0.5 s time limit per action, otherwise wait is selected. Make sure that inference of our model is under 0.5 on a cpu
- if time is overreached at one step, the excess is reduced from the thinking time of the next step


# Methodology

Use DQN built from scratch for one model, with replay buffer and Q target learning.
https://huggingface.co/learn/deep-rl-course/unit3/from-q-to-dqn

Steps:

- defining the network, the epsilon greedy strategy and replay buffer in pytorch
- define logging and plotting mechanism for the network and loading it
- in the setup function (callbacks), load the model and see if we retrain it or used the learned weights
- in act (callbacks), just do inference taking into account the exploration rate
- find a way to create features to train on (either learn them using an encoder or hand craft them)


# Todos

1. Find out if they will use different dimensions for the board. In this case using an encoder to learn the features may not work for different board sizes.

2. Do we want a higher dimensional feature space than the given game state? For the given code from last year, see if the game state liniarized dimension is bigger than the feature space dimension (i.e. 343).

3. Ask if we are allowed to use gymnasium or stable baseline for the project?

4. What is the difference between events and features? Why do we need both and why are not the events linked to the features 1:1?