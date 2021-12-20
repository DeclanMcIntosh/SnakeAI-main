import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
from re import L
from tensorflow.python.keras.backend import random_normal

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

# {'bestModel_rev000.h5': 1406.5946896614125, 'bestModel_rev001.h5': 1338.1399723191828, 'bestModel_rev002.h5': 1441.3037611020495, 'bestModel_rev003.h5': 1377.0647318029162, 'bestModel_rev004.h5': 1292.3480780833836, 'bestModel_rev005.h5': 1413.5007143348962, 'bestModel_rev006.h5': 1363.5722668411051, 'bestModel_rev007.h5': 1382.6408964634802, 'bestModel_rev008.h5': 1397.099147821223, 'bestModel_rev009.h5': 1392.8710902296525, 'bestModel_rev010.h5': 1432.7492042238612, 'bestModel_rev011.h5': 1416.4600171648333, 'bestModel_rev012.h5': 1390.4573177157572, 'bestModel_rev013.h5': 1321.9831366038177, 'bestModel_rev014.h5': 1470.381305493068, 'bestModel_rev015.h5': 1466.2122927305604, 'bestModel_rev016.h5': 1411.241451800356, 'bestModel_rev017.h5': 1449.187574272581, 'bestModel_rev018.h5': 1427.4516207154034, 'bestModel_rev019.h5': 1460.8482966098293, 'bestModel_rev020.h5': 1424.7956234581713, 'bestModel_rev021.h5': 1436.0824413014575, 'bestModel_rev022.h5': 1431.317869500204, 'bestModel_rev023.h5': 1456.5586230726144, 'bestModel_rev024.h5': 1458.756501038899, 'bestModel_rev025.h5': 1429.1214016212443, 'bestModel_rev026.h5': 1332.8250179836552, 'bestModel_rev027.h5': 1492.0809164893337, 'bestModel_rev028.h5': 1493.6831062710853, 'bestModel_rev029.h5': 1352.5886240446146, 'bestModel_rev030.h5': 1380.4861273368108, 'bestModel_rev031.h5': 1461.838104229003, 'bestModel_rev032.h5': 1399.184210461206, 'bestModel_rev033.h5': 1500.3787793168121, 'bestModel_rev034.h5': 1355.5313425116667, 'bestModel_rev035.h5': 1328.7032491039154, 'Random': 1181.6381158990616}
    
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

        #t0 = time.time()

        action0, prediction0 = snake0.getAction(state0, valids[0])
        action1, prediction1 = snake1.getAction(state1, valids[1])
        
        #print(str((time.time()-t0)))

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
    #print("Done Competition Games: " + str(x) +    "                                                  ", end='\r')

    meanLength = np.mean(np.array(gameLengths))
    meanWinType = np.mean(np.array(wintypes))
    winRate = np.mean(np.array(wins))

    #print(string_ + str(len(gameLengths))+ " # Non-tie Games:" + str(len(wins)) + " Winrate:" + str(winRate)[0:5] + " Avg Length:" + str(meanLength)[0:5] + " Cornering Rate:" + str(meanWinType)[0:5] +   "                                                  ")

    return winRate

def main():
    gamesToPlay = 10000
    snakeDir = 'models/'
    #snakeDir = 'modelsBigModel/'
    eloKFactor = 8
    gamesPerMatch = 1

    randomSnake = RandomOponant()
    snake0 = Policy()
    snake1 = Policy()

    snakeNames = list(os.listdir(snakeDir))

    snakeNames.append('Random')

    game = SnakeGame(GUI=False)

    elos = {}

    for name in snakeNames:
        elos[name] = 1400

    while True:
        for _ in range(gamesToPlay//gamesPerMatch):
            print("Done: " + str(_) +    "                                                  ", end='\r')
            zero = None
            one = None
            players = []
            while zero == one:
                zero = r.choice(snakeNames)
                one = r.choice(snakeNames)
            
            if zero == 'Random':
                players.append(randomSnake)
            else:
                snake0.model.load_weights(snakeDir + zero)
                players.append(snake0)

            if one == 'Random':
                players.append(randomSnake)
            else:
                snake1.model.load_weights(snakeDir + one)
                players.append(snake1)

            rewards, states, actions, wintype, predictions, length = playOneGame(game, players[0], players[1])

            if rewards[0][-1] > rewards[1][-1]:
                winRateZero = 1
                winRateOne = 0 
            elif rewards[0][-1] < rewards[1][-1]:
                winRateZero = 0
                winRateOne = 1
            else:
                winRateZero = 0.5
                winRateOne = 0.5
            
            E_zero = 1/(1+10**((elos[one]-elos[zero])/400))
            E_one = 1/(1+10**((elos[zero]-elos[one])/400))

            zeroNewElo = elos[zero] + eloKFactor*(winRateZero-E_zero)
            oneNewElo = elos[one] + eloKFactor*(winRateOne-E_one)
            elos[zero] = zeroNewElo
            elos[one] = oneNewElo
        print(elos)








if __name__ == '__main__':
    print(format(3, '03d'))
    main()