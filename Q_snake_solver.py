import os
from re import L
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from tensorflow.python.keras.backend_config import epsilon
from snakeQ import *

from tensorflow.keras.models import Sequential,  clone_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import SGD
import numpy as np

import random as r
import os

import time

def playOneGame(game, snake0, snake1):
    options = [[1,0],[0,1],[-1,0],[0,-1]]
    states = [[],[]]
    actions = [[],[]] 
    rewards = [[],[]]
    predictions = [[], []]
    terminal = False
    game.resetGame()
    state0, valids = game.getState()
    state1 = switchAgentContext(state0, game)

    length = 0
    
    while not terminal:
    
        length += 1
        state1 = switchAgentContext(state0, game)

        #t0 = time.time()

        action0, prediction0 = snake0.getAction(state0, valids[0])
        action1, prediction1 = snake1.getAction(state1, valids[1])
        
        #print(str((time.time()-t0)))

        if len(valids[0]) > 0: action0 = r.choice(valids[0])
        else: action0 = [0,0]
        if len(valids[1]) > 0: action1 = r.choice(valids[1])
        else: action1 = [0,0]
        
        if action0 in options:
            states[0].append(state0)
            actions[0].append(action0)
        predictions[0].append(prediction0)
        if action1 in options:
            states[1].append(state1)
            actions[1].append(action1)
        predictions[1].append(prediction1)

        
        state0, reward0, reward1, terminal, valids, wintype = game.step([action0, action1])
        

        rewards[0].append(reward0)
        rewards[1].append(reward1)
        
    
    return rewards, states, actions, wintype, predictions, length 

def train():
    epsilon = 0.1
    rewardScheme = [-2.5,10,-50,50] #[step, food, lose/tie, win]
    rewardNormFactor = 50
    replayBuffer = []
    gamesPerRetest = 512 
    gamesTestNewModel = 128

    currentTrainee = Policy(epsilon=0.0,snakeModel=Snake_Model_DQN)
    previousBest = Policy(epsilon=0.1,snakeModel=Snake_Model_DQN)

    game = SnakeGame(GUI=False)

    for x in range(gamesPerRetest):

                


if __name__ == '__main__':
    train()

