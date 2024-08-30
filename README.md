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
    - create a policy network and a target network that is copied from the policy network at some interval
    - train when enough data is acquired in the replay buffer
    - they train after your owns agent move
- define logging and plotting mechanism for the network and loading it
- in the setup function (callbacks), load the model and see if we retrain it or used the learned weights
- in act (callbacks), just do inference taking into account the exploration rate
- consider dealing with invalid actions

# Questions

3. Ask if we are allowed to use gymnasium or stable baseline for the project?

4. What is the difference between events and features? Why do we need both and why are not the events linked to the features 1:1?
   A: events are used to map to rewards which contribute to the calculation of the target q in the bellman equation which is then compared with the predicted Q of the policy network and the loss is used for training.

# TODOS

- [ ] implement new useful events and create functions to check if they are fullfilled. See restrictions in pdf and in Architecture
- [ ] create a training script (bash) that train our agent with no gui trough the given 4 scenarios
- [ ] write a function that saves and loads the model at the end of a training cycle (.pth or pickle whatever works)
- [ ] in the train_setup and setup (callback) functions initialize and load the models, initialize the optimizer and loss function, initialize the ReplayMemory
- [ ] add the distance between updating the target newtork and the policy network as a hyperparam and plot the network performance based on this param (try per episodes and per steps)

# Architecture

setup_train is called after setup in callbacks when we are in train mode
they then train the policy net on the states from the replay experience buffer and the predicted target q values
https://medium.com/@hkabhi916/mastering-deep-q-learning-with-pytorch-a-comprehensive-guide-a7e690d644fc

We need to define events and check for them so that we can define more rewards so we can speed up the training in the bellman equation
theory says that auxilliary rewards should only depend on the game states and not 2 on the actions leading there.