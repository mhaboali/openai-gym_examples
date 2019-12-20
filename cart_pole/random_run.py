import gym
import random, os
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import scipy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

example_folderPath = 'D:\\mine\\Work\\OpenAI\\cart_pole'

## Learing rate for training the model
LR = 1e-3
## Create the OpenAI gym environment
env = gym.make("CartPole-v0")
## Make sure everything is reset at the beginning
env.reset()
## Number of steps per episode
goal_steps = 500
## Desired goal value to be used for filtering the outliers
score_requirement = 50
## Number of episodes for the training
episodes_num = 10000

def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(200):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()
            
            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()
            
            # this executes the environment with an action, 
            # and returns the observation of the environment, 
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break
                
some_random_games_first()