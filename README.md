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

- [ ] improve feature space by using some of the defined features from maverick + Bomberbrains
- [ ] define a way to test our CNN network on gogle colab with batch_size 64 and 800 000 training steps
- [ ] find out why the agent chooses WAIT after certain number of steps
- [ ] add the distance between updating the target newtork and the policy network as a hyperparam and plot the network performance based on this param (try per episodes and per steps)
- [ ] why is linear decay numerically unstable even with decay_steps = 1_000?
- [ ] IMPORTANT: adjust linear decay steps to the number of steps we estimate to have for a number of training rounds; test the number of decay steps compared to training rounds
- [ ] IMPORTANT: we update target net every 100 rounds, isnt that too big? Test this maybe make it smaller
- [ ] merge actually training into master and start working on master
# Working on TOODOS
- [ ] implement new useful events and create functions to check if they are fullfilled. See restrictions in pdf and in Architecture


# Solved TODOs
- [X] why is linear decay not linear?
- [X] create a training script (bash) that train our agent with no gui trough the given 4 scenarios
- [X] in the train_setup and setup (callback) functions initialize and load the models, initialize the optimizer and loss function, initialize the ReplayMemory
- [X] write a function that saves and loads the model at the end of a training cycle (.pth or pickle whatever works)
- [X] create a buffer of actions of the rule based agent to feed to our network for initial training where the agent is too weak to gather enough moves to learn from them

# Running Shell Script
Before executing the script for the first time, run: `chmod 755 run_agent.sh`.
Execute the script with: `./run_agent.sh`.
Arguments can be directly changed in the script.

# Architecture

setup_train is called after setup in callbacks when we are in train mode
they then train the policy net on the states from the replay experience buffer and the predicted target q values
https://medium.com/@hkabhi916/mastering-deep-q-learning-with-pytorch-a-comprehensive-guide-a7e690d644fc

We need to define events and check for them so that we can define more rewards so we can speed up the training in the bellman equation
theory says that auxilliary rewards should only depend on the game states and not 2 on the actions leading there.


Mon Sep  9 15:55:50 EEST 2024

Plan for today:
- clean code and remove unnecesary parts
- look at the training step of other repos and implement a simpler agent that learns for the coin-heaven scenario (not on rule-based-agent-buffer)
- maybe redo training step function for our agent

Notes:
- 2 Linears work as well as seen in https://github.com/nickstr15/bomberman/blob/master/agent_code/maverick/Model.py
- training is the problem, we should look at at least 1 000 000 training steps


Mon Sep 16 19:03:18 CEST 2024

Epsilon estimates:

200 rounds is approximately 3500 steps so on average, 18 steps per round at the beginning
with 10 000 decay steps we reach epsilon 0.7

10 000 decay steps go to 0.1 30 0000 steps at 60% (in a 2000 rounds scenario)

Linear Decay on 400 games with 1000 decay steps produced better agents than Exponential decay with 0.999 decay param for 400 games

Running inference only (without 0.1 exploration) gets the agent stuck in wiggling.

We may have a problem with the update time for the target net. 100 rounds of 100 actions each may be too much.

Wed Sep 18 12:52:53 CEST 2024

When we use our trained model from coin heaven on loot crate we want to have again some exploration and exploitation

Today's todos:

- [ ] add number of steps in objectives to the functions