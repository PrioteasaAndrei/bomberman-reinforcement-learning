# Reinforcement Learning Project for Bomberman Game

This is the repository containing all the code developed for the final project of the lecture Machine Learning Essentials SS24 at Heidelberg University.

## Installation

To install the project, run:
```bash
git clone https://github.com/PrioteasaAndrei/bomberman-reinforcement-learning.git
cd bomberman-reinforcement-learning
```
To install all necessary dependencies, make sure you have [conda](https://docs.anaconda.com/miniconda/) installed and then run the following commands:

```bash
conda create --name rl_project --file environment.yml
conda activate rl_project
```

## Usage

To run the game with our best performing agent, run:
```
python main.py play --my-agent meister_eckhart
```