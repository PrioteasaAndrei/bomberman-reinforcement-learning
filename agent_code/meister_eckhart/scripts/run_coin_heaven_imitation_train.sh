#!/bin/zsh

cd ../../..

n_rounds=1000

python main.py play --agents meister_eckhart rule_based_agent --n-rounds $n_rounds --scenario coin-heaven --train 1 --continue-without-training --no-gui --save-stats agent_code/meister_eckhart/stats.json