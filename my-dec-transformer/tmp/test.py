import torch
print(f"cuda available: {torch.cuda.is_available()}")

import gym
import gym_examples

env= gym.make('gym_examples/contGrid-v5')
