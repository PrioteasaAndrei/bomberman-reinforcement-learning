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
