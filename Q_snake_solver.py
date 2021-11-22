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
    state0, valids = game.getState()
    state1 = switchAgentContext(state0, game)
    reward0 = None
    reward1 = None

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

        if length > 10000:
            break
        
    
    return rewards, states, actions, wintype, predictions, length 

def train():
    options = [[1,0],[0,1],[-1,0],[0,-1]]
    games_per_update = 128
    competitionGames = 100
    replayBufferLenth = 1024*32
    gamma = 1.0

    snake0 = Policy(epsilon=0.1)
    snake1 = Policy()
    #snake1s = [RandomOponant(), RandomOponant(), RandomOponant()] 
    game = SnakeGame(GUI=False)
    modelIndex = len(list(os.listdir('models/'))) 
    bestWinrate = 0
    losses = []
    gameLengths = []
    wintypes= []
    wins = []
    k = 1
    snake_names = list(os.listdir('models/'))
    snake_names.sort()

    snake_names = os.listdir('models/')
    if len (snake_names) > 0:
        snake1.model.load_weights('models/' + snake_names[-1])

    replayMemory = [[], [], []]


    while True:
        replayMemory = [[], [], []]

        ### Play the model Against it'self
        for _ in range(games_per_update):
            snake0.epsilon = 0.1 
            #rewards, states, actions, wintype, length =  playOneGame(game, snake0, randomOpp)
            rewards, states, actions, wintype, predictions, length =  playOneGame(game, snake0, snake0)
            
            if rewards[0][-1] > rewards[1][-1]: wins.append(1)
            elif rewards[1][-1] > rewards[0][-1]: wins.append(0) 

            for n in range(2):
                for x in range(len(rewards[n])):
                    replayMemory[0].append(states[n][x])
                    replayMemory[1].append(actions[n][x])
                    if x < len(rewards[n]) - 1:
                        replayMemory[2].append(rewards[n][x] + gamma*predictions[n][x+1])
                    else:
                        replayMemory[2].append(rewards[n][x])

            print("Done: " + str(_) +    "                                                  ", end='\r')

        # Train the model on the experince

        replayMemory[0] = replayMemory[0][-replayBufferLenth:] 
        replayMemory[1] = replayMemory[1][-replayBufferLenth:] 
        replayMemory[2] = replayMemory[2][-replayBufferLenth:] 

        for x in range(games_per_update):
            loss = snake0.trainModelPredictor(replayMemory[0], replayMemory[1], replayMemory[2])
            losses.append(loss)

        for x in range(competitionGames):
            snake0.epsilon = 0.0
            rewards, states, actions, wintype, predictions, length =  playOneGame(game, snake0, snake1)
            
            if rewards[0][-1] > rewards[1][-1]: wins.append(1)
            elif rewards[1][-1] > rewards[0][-1]: wins.append(0) 

            if wintype == 'cornered':
                wintypes.append(1)
            else:
                wintypes.append(0)
            gameLengths.append(length)
            print("Done Competition Games: " + str(_) +    "                                                  ", end='\r')

        meanLosses = np.mean(np.array(losses[-100:]))
        meanLength = np.mean(np.array(gameLengths[-100:]))
        meanWinType = np.mean(np.array(wintypes[-100:]))
        winRate = np.mean(np.array(wins[-100:]))

        print("# Games:" + str(len(gameLengths))+ " # Non-tie Games:" + str(len(wins)) + " Winrate:" + str(winRate)[0:5] + " Avg Loss:" + str(meanLosses)[0:5] + " Avg Length:" + str(meanLength)[0:5] + " Cornering Rate:" + str(meanWinType)[0:5] +   "                                                  ")


        #if winRate > bestWinrate and len(wins) > 100:
        #    bestWinrate = winRate
        if winRate > 0.55:
            print("Saving good Model")
            snake0.model.save('models/bestModel_rev' + str(modelIndex) + '.h5')
            modelIndex+= 1

        snake_names = os.listdir('models/')
        if len (snake_names) > 0:
            snake1.model.load_weights('models/' + snake_names[-1])

        losses = []
        gameLengths = []
        wintypes= []
        wins = []

                


if __name__ == '__main__':
    train()