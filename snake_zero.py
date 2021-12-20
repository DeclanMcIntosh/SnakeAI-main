import os
from re import L

from tensorflow.python.keras.backend_config import epsilon
from snakeQ import *

from tensorflow.keras.models import Sequential,  clone_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import SGD
import numpy as np

import random as r
import os
import sys

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
    reward0 = None
    reward1 = None

    length = 0
    
    while not terminal:
    
        if length > 128:
            terminal = True

        length += 1
        state1 = switchAgentContext(state0, game)

        action0, prediction0 = snake0.getAction(state0, valids[0])
        action1, prediction1 = snake1.getAction(state1, valids[1])
        

        if len(valids[0]) == 0: action0 = [0,0]
        if len(valids[1]) == 0: action1 = [0,0]
        
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

        if length > 128:
            terminal = True
        
    
    return rewards, states, actions, wintype, predictions, length 

def compete(snake0, snake1s, game, competitionGames, string_, f): 
    gameLengths = []
    wintypes= []
    wins = []
    for x in range(competitionGames):
        snake1 = r.choice(snake1s)
        snake0.epsilon = 0.0
        rewards, states, actions, wintype, predictions, length =  playOneGame(game, snake0, snake1)
        
        if rewards[0][-1] > rewards[1][-1]: wins.append(1)
        elif rewards[1][-1] > rewards[0][-1]: wins.append(0) 

        if wintype == 'cornered':
            wintypes.append(1)
        else:
            wintypes.append(0)
        gameLengths.append(length)
    print("Done Competition Games: " + str(x) +    "                                                  ", end='\r')

    meanLength = np.mean(np.array(gameLengths))
    meanWinType = np.mean(np.array(wintypes))
    winRate = np.mean(np.array(wins))

    print(string_ + str(len(gameLengths))+ " # Non-tie Games:" + str(len(wins)) + " Winrate:" + str(winRate)[0:5] + " Avg Length:" + str(meanLength)[0:5] + " Cornering Rate:" + str(meanWinType)[0:5] +   "                                                  ")

    with open(f, 'a+') as fileRef:
        fileRef.write(string_ + str(len(gameLengths))+ " # Non-tie Games:" + str(len(wins)) + " Winrate:" + str(winRate)[0:5] + " Avg Length:" + str(meanLength)[0:5] + " Cornering Rate:" + str(meanWinType)[0:5] + "\n")

    return winRate


def train(f):
    options = [[1,0],[0,1],[-1,0],[0,-1]]
    games_per_update = 2500
    competitionGames = 256
    replayBufferLenth = 1024*64*8
    gamma = 1
    eps= 0.1
    fails = 0

    snake0 = Policy(epsilon=eps)
    snake1 = Policy()
    prevSnakes = [Policy(),Policy(),Policy()]
    randomSnake = RandomOponant()
    game = SnakeGame(GUI=False)
    modelIndex = len(list(os.listdir('models/'))) 
    k = 1
    snake_names = list(os.listdir('models/'))
    snake_names.sort()

    snake_names = os.listdir('models/')
    if len (snake_names) > 0:
        snake1.model.load_weights('models/' + snake_names[-1])

        for x in range(min([len(prevSnakes), len(snake_names)])):
            print(snake_names[-(x+1)])
            prevSnakes[x].model.load_weights('models/' + snake_names[-(x+1)])

    replayMemory = [[], [], []]


    while True:

        ### Play the model Against it'self
        for _ in range(games_per_update):
            snake0.epsilon = eps
            if _ % 5 == 0:
                rewards, states, actions, wintype, predictions, length =  playOneGame(game, snake0, r.choice(prevSnakes))
            elif (_ + 1 ) % 10 == 0:
                rewards, states, actions, wintype, predictions, length =  playOneGame(game, snake0, randomSnake)
            else:
                rewards, states, actions, wintype, predictions, length =  playOneGame(game, snake0, snake0)


            k = len(rewards[0])
            if rewards[0][-1] > rewards[1][-1] or rewards[1][-1] > rewards[0][-1] or True: # Force to train only on wins or loses
                for n in range(2):
                    last = 0
                    for x in range(k):
                        replayMemory[0].append(states[n][k-x-1])
                        replayMemory[1].append(actions[n][k-x-1])
                        replayMemory[2].append(rewards[n][k-x-1] + gamma*last)
                        last = rewards[n][k-x-1] + gamma*last

            print("Done: " + str(_) +    "                                                  ", end='\r')

        snake0.trainModelPredictor(replayMemory[0], replayMemory[1], replayMemory[2])

        
        winRate1 = compete(snake0, [randomSnake], game, competitionGames, "VS RANDOM--> # Games ", f)
        winRate2 = compete(snake0, prevSnakes, game, competitionGames, "VS 3 PREVIOUS--> # Games ", f)
        winRate3 = compete(snake0, [snake1], game, competitionGames, "VS PREVIOUS Best--> # Games ", f)

        snake_names = os.listdir('models/')
        if winRate3 >= 0.55 and winRate2 >= 0.55:
            # Flush memory 
            replayMemory = [[], [], []]
            with open(f, 'a+') as fileRef:
                fileRef.write("Saving good Model\n")
            print("Saving good Model")
            snake0.model.save('models/bestModel_rev' + format(modelIndex, '03d') + '.h5')
            modelIndex+= 1
            fails = 0
        elif len (snake_names) > 0:
            snake0 = Policy()
            snake0.model.load_weights('models/' + snake_names[-1])
            fails += 1
        
        if fails > 4:
            with open(f, 'a+') as fileRef:
                fileRef.write("Model Seems Stuck Reseting to Random Weights\n")
            print("Model Seems Stuck Reseting to Random Weights")
            snake0 = Policy(epsilon=eps)
            fails = 0


        if len (snake_names) > 0:
            snake1.model.load_weights('models/' + snake_names[-1])
            for x in range(min([len(prevSnakes), len(snake_names)])):
                prevSnakes[x].model.load_weights('models/' +snake_names[-(x+1)])

        gameLengths = []
        wintypes= []
        wins = []

                


if __name__ == '__main__':
    f = 'logs0.txt'
    print("starting")
    with open(f, 'a+') as fileRef:
        fileRef.write("starting\n")
    train(f)