#!/bin/zsh
# Alternatively for bash use: #!/bin/bash

# Change directory to execute main.py
cd ../..

agent1="meister_eckhart"
agent2="peaceful_agent"
agent3="peaceful_agent"
agent4="peaceful_agent"
n_train=1
n_rounds=10
scenario="classic"

if [ "$n_train" -eq 1 ]; then
    mode_message="in training mode"
else
    mode_message="in inference mode"
fi

# Run game
echo "Running agent $agent1 $mode_message with $agent2, $agent3, $agent4 in scenario $scenario for $n_rounds episodes."
python main.py play --agents $agent1 $agent2 $agent3 $agent4 --train $n_train --n-rounds $n_rounds --scenario $scenario --no-gui