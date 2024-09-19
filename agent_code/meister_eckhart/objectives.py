import optuna
import os
from .config import *
import json

NO_ROUNDS = 1200
OPTIMIZATION_TRIALS = 300

def update_hyperparams(lr, gamma, target_update, decay_steps):
    """
    Update the hyperparameters in the hyperparams.json file.
    """
    # Hyper parameters -- load them from the config file
    print(os.getcwd())
    with open('agent_code/meister_eckhart/hyperparams.json') as f:
        config = json.load(f)
    
    config['lr'] = lr
    config['gamma'] = gamma
    config['target_update'] = target_update
    config['decay_steps'] = decay_steps

    # write the new hyperparameters to the config file
    with open('agent_code/meister_eckhart/hyperparams.json', 'w') as f:
        json.dump(config, f)

def run_training_scenario(no_rounds=NO_ROUNDS):
    
    train_cmd = f'python main.py play --agents meister_eckhart --train 1 --n-rounds {no_rounds} --scenario {SCENARIO} --no-gui'
    # os.chdir('../..')
    os.system(train_cmd)
    # os.chdir('agent_code/meister_eckhart')


def run_training(lr, gamma, target_update, decay_steps, no_rounds=NO_ROUNDS):
    '''
    python main.py play --agents meister_eckhart --train 1 --n-rounds 1200 --scenario loot-crate --no-gui
    '''

    update_hyperparams(lr, gamma, target_update, decay_steps)
    run_training_scenario(no_rounds)

    # save the running reward
    with open('agent_code/meister_eckhart/logs/running_reward.txt', 'r') as file:
        score = float(file.readline().strip())

    return score


def objective(trial):
    """
    Objective function for the optimization of the hyperparameters just for the crate scenario.
    """
    # Hyperparameters to optimize
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)
    gamma = trial.suggest_float("gamma", 0.6, 0.999) # modify lower bound here to 0.6
    target_update = trial.suggest_int("target_update", 2, 20) # modify upper bound here to 20
    decay_steps = trial.suggest_int("decay_steps", 1000, 20_000)
    
    return run_training(lr, gamma, target_update, decay_steps)

def run_hyperparam_optimization():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTIMIZATION_TRIALS)

    with open('agent_code/meister_eckhart/logs/best_trial.txt', 'w') as f:
        f.write('Best trial:\n')
        trial = study.best_trial
        f.write(f'Value: {trial.value}\n')
        f.write('Params:\n')
        for key, value in trial.params.items():
            f.write(f'    {key}: {value}\n')