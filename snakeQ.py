from tkinter import *
from tkinter import messagebox
#import Tkinter as tk
import random as r
import time
import numpy as np
from tensorflow.python.keras.backend import binary_crossentropy

from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.ops.gen_array_ops import pad, shape_eager_fallback
#from game import *
from policy import *

from tensorflow.keras.models import Sequential,  clone_model, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, LeakyReLU, InputLayer, Add
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np

import random as r

'''
TODO:
- make game logic # DONE 
- make random oponant (generic class interface) # DONE 
- move GUI to openCV to make it work threaded 
- average game length for two random actors is 154.42 moves.
'''

def switchAgentContext(state0, game):
    state1 = np.copy(state0)
    state1[:,:,0] = state0[:,:,1]
    state1[:,:,1] = state0[:,:,0]

    if len(game.snake1) > len(game.snake0):
        state1[:,:,7] = np.ones(shape=(game.boardSize,game.boardSize))
    else:
        state1[:,:,7] = np.zeros(shape=(game.boardSize,game.boardSize))

    return state1



class SnakeGame():
    def __init__(self,GUI=True, UserPlayable=True, player1=RandomOponant(), player2=None, animationTime=1000, rewardscheme = [0,0,0,1]):
        ###############   Main Program #################
        self.GUI=GUI
        self.UserPlayable=UserPlayable
        self.player1=player1
        self.player2=player2    
        self.animationTime=animationTime
        self.wait = False
        self.boardSize = 5
        self.channels = 8

        self.stepReward = rewardscheme[0]
        self.foodReward = rewardscheme[1]
        self.lossReward = rewardscheme[2]
        self.winsReward = rewardscheme[3]

        self.head0 = 0
        self.head1 = 1
        self.up = 2
        self.right = 3
        self.down = 4
        self.left = 5
        self.food = 6
        self.bigger = 7

        self.steps = 0

        self.state = np.zeros(shape=(self.boardSize,self.boardSize,self.channels))

        self.results = []

        self.rollString=""
        if self.GUI:
            self.root=Tk()                   #Window defined
            self.root.title("Modifed Battle Snake")         #Title given
            self.colour={1:"deep sky blue", 0:"lawn green"}
            self.b = []
            for x in range(self.boardSize):
                self.b.append([])
            for i in range(self.boardSize):
                for j in range(self.boardSize):
                    self.b[i].append(self.button(self.root))
                    self.b[i][j].config(command= lambda row=i,col=j:self.click(row,col))
                    self.b[i][j].grid(row=i,column=j)

            self.label=Label(text= '')
            self.label.grid(row=7,column=0,columnspan=6)

            self.resetGame()
            self.root.mainloop()

        self.state = np.zeros(shape=(self.boardSize,self.boardSize,self.channels)) 
        self.resetGame()

    def updateState(self):
        self.state[:,:,0:6] = np.zeros(shape=(self.boardSize,self.boardSize,6))

        if len(self.snake0) > len(self.snake1):
            self.state[:,:,self.bigger] = np.ones(shape=(self.boardSize,self.boardSize))

        def writeSnake(snake, state, head):
            for x in range(len(snake)):
                if x == 0:
                    state[snake[x][0],snake[x][1], head] = 1
                else:
                    if snake[x-1][0] - snake[x][0] == -1: state[snake[x][0],snake[x][1], self.down] = 1
                    if snake[x-1][0] - snake[x][0] == 1: state[snake[x][0],snake[x][1], self.up] = 1
                    if snake[x-1][1] - snake[x][1] == -1: state[snake[x][0],snake[x][1], self.left] = 1
                    if snake[x-1][1] - snake[x][1] == 1: state[snake[x][0],snake[x][1], self.right] = 1

        writeSnake(self.snake0, self.state, self.head0)
        writeSnake(self.snake1, self.state, self.head1)

        if np.max(self.state[:,:,6]) == 0:
            taken = np.max(self.state[:,:,:6], axis=2)
            proposed_food =(r.randint(0,self.boardSize-1),r.randint(0,self.boardSize-1))
            while taken[proposed_food[0], proposed_food[1]] != 0 :
                proposed_food =(r.randint(0,self.boardSize-1),r.randint(0,self.boardSize-1))
            self.state[proposed_food[0], proposed_food[1], 6] = 1

    def resetGame(self):

        self.steps = 0

        self.state = np.zeros(shape=(self.boardSize,self.boardSize,self.channels))

        self.snake0 = [[self.boardSize//2,1],[self.boardSize//2,0],[self.boardSize//2+1,0],[self.boardSize//2+2,0]]
        self.snake1 = [[self.boardSize//2,self.boardSize-2],[self.boardSize//2,self.boardSize-1],[self.boardSize//2-1,self.boardSize-1],[self.boardSize//2-2,self.boardSize-1]]

        self.updateState()

        if self.GUI:
            self.render()
            self.checkValidActions()

    def render(self):
        players = ['A', 'B']

        for x in range(self.boardSize):
            for y in range(self.boardSize):
                self.b[x][y]["text"]= ''

        for n in range(7):
            for x in range(self.boardSize):
                for y in range(self.boardSize): 
                    if n ==self.head0 and self.state[x,y,n]: self.b[x][y]["text"]= 'A'
                    if n ==self.head1 and self.state[x,y,n]: self.b[x][y]["text"]= 'B'
                    if n ==self.up    and self.state[x,y,n]: self.b[x][y]["text"]= '⇓'
                    if n ==self.right and self.state[x,y,n]: self.b[x][y]["text"]= '⇒'
                    if n ==self.down  and self.state[x,y,n]: self.b[x][y]["text"]= '⇑'
                    if n ==self.left  and self.state[x,y,n]: self.b[x][y]["text"]= '⇐'
                    if n ==self.food  and self.state[x,y,n]: self.b[x][y]["text"]= '❤'


        # Render amounts on benches

    def checkValidActions(self):

        def checkSnake(state, snake):
            valid = []
            taken = np.max(state[:,:,0:6], axis=2)

            okValues = list(np.arange(self.boardSize))

            for k in [[1,0],[-1,0],[0,1],[0,-1]]: # up down left right
                tail0 = self.snake0[-1]
                tail1 = self.snake1[-1]
                index0 = snake[0][0]+k[0]
                index1 = snake[0][1]+k[1]
                
                if index0 in okValues and index1 in okValues:
                    if not taken[index0,index1] or [index0,index1] in [tail0,tail1]: valid.append(k)
            return valid

        valids0 = checkSnake(self.state,self.snake0)
        valids1 = checkSnake(self.state,self.snake1)
        
        #invalid0 = []
        #invalid1 = []
        #for valid0 in valids0:
        #    for valid1 in valids1:
        #        if self.snake0[0][0] + valid0[0] == self.snake1[0][0] + valid1[0]:
        #            if self.snake0[0][1] + valid0[1] == self.snake1[0][1] + valid1[1]:
        #                invalid0.append(valid0)
        #                invalid1.append(valid1)
        #
        #if len(self.snake0) <= len(self.snake1):
        #    if len(valids0) != len(invalid0):
        #        valids0 = [x for x in valids0 if x not in invalid0]
        #elif len(self.snake1) <= len(self.snake0):
        #    if len(valids1) != len(invalid1):
        #        valids1 = [x for x in valids1 if x not in invalid1]


        valids = [valids0,valids1]

        #print(valids)

        return valids

    def step(self, actions):

        self.steps+=1
        if self.GUI:
            self.root.title("Modifed Battle Snake")
        valids = self.checkValidActions()
        terminal = False

        reward = self.stepReward
        rewardopp = self.stepReward

        # move heads        
        self.snake0.insert(0, [self.snake0[0][0]+actions[0][0],self.snake0[0][1]+actions[0][1]])
        self.snake1.insert(0, [self.snake1[0][0]+actions[1][0],self.snake1[0][1]+actions[1][1]])

        # check eating and remove tails
        if not self.state[self.snake0[0][0],self.snake0[0][1],self.food] == 1:
            self.snake0.pop()
        else:
            self.state[self.snake0[0][0],self.snake0[0][1],self.food] = 0
            reward = self.foodReward
        if not self.state[self.snake1[0][0],self.snake1[0][1],self.food] == 1:
            self.snake1.pop()
        else:
            self.state[self.snake1[0][0],self.snake1[0][1],self.food] = 0
            rewardopp = self.foodReward

        snake0_dead = self.snake0[0] in self.snake1
        snake1_dead = self.snake1[0] in self.snake0

        # handel headbutts
        if snake0_dead and snake1_dead:
            if len(self.snake0) > len(self.snake1):
                snake0_dead = False
            elif len(self.snake0) < len(self.snake1):
                snake1_dead = False

        self.updateState()

        valids = self.checkValidActions()

        if len(valids[0]) == 0:   snake0_dead = True
        if len(valids[1]) == 0:  snake1_dead = True

        if self.steps > 128:
            snake0_dead = True
            snake1_dead = True

        terminal = snake0_dead or snake1_dead

        if snake0_dead and not snake1_dead: 
            reward = self.lossReward
            rewardopp = self.winsReward
        if snake1_dead and not snake0_dead: 
            reward = self.winsReward
            rewardopp = self.lossReward
        if snake0_dead and snake1_dead: 
            reward = self.lossReward
            rewardopp = self.lossReward

        if terminal and (len(valids[0])==0 or len(valids[1])==0):
            wintype = 'cornered'
        else:
            wintype = 'headbut'
 
        if self.GUI:
            self.render()
        if terminal:
            if self.GUI:
                print("Game over!")
                if not snake0_dead:
                    self.root.title("SNAKE 'A' WINS!")
                    print("SNAKE 'A' WINS!")
                if not snake1_dead:
                    self.root.title("SNAKE 'B' WINS!")
                    print("SNAKE 'B' WINS")
                if snake0_dead and snake1_dead:
                    self.root.title("Draw!")
                    print("DRAW!")
                for x in range(4):
                    self.waithere()
            self.resetGame()

        valids = self.checkValidActions()


        return self.state, reward, rewardopp, terminal, valids, wintype

    def convertPositionToAction(self,row,col):
        return [row-self.snake0[0][0],col-self.snake0[0][1]]

    def getState(self):
        return  self.state, self.checkValidActions()

    def waithere(self):
        self.wait = True
        var = IntVar()
        self.root.after(self.animationTime, var.set, 1)
        #print("waiting...")
        self.root.wait_variable(var)
        self.wait = False

    def button(self, frame, color=None):#Function to define a button
        if color == None:
            button_=Button(frame,padx=1,bg="papaya whip",width=5,height=1,text="   ",font=('arial',80,'bold'),relief="sunken",bd=5)
        else:
            button_=Button(frame,padx=1,bg=color,width=5,height=1,text="   ",font=('arial',80,'bold'),relief="sunken",bd=5)
        return button_
    
    def click(self,row,col):
        #self.b[row][col].config(text= str(self.toAct),state=DISABLED,disabledforeground=self.colour[ self.toAct])
        action = self.convertPositionToAction(row,col)
        valids = self.checkValidActions()[0]
        secondState= switchAgentContext(self.state, self)
        if action in valids:
            if not self.player1 is None and not self.player2 is None:
                while True:
                    self.waithere()
                    ok_moves = self.checkValidActions()
                    #print(ok_moves)
                    #print("now getting actions")
                    action0, _ = self.player1.getAction(self.state, ok_moves[0])
                    action1, _ = self.player2.getAction(secondState, ok_moves[1])

                    #print(action0, action1)
                    self.step([action0,action1])
            else:
                actionmodel, _ = self.player1.getAction(secondState, self.checkValidActions()[1])
                self.step([action, actionmodel])
                self.waithere()
    
if __name__ == '__main__':

    opp = Policy(0.0)
    opp.model = load_model('models/bestModel_rev13.h5')
    opp.compile()

    opp1 = Policy(0.0)
    opp1.model = load_model('models/bestModel_rev13.h5')
    opp1.compile()

    test = SnakeGame(animationTime=100, player2=RandomOponant(), player1=opp)
    #test = SnakeGame(animationTime=750, player2=RandomOponant(), player1=RandomOponant())
    #test = SnakeGame(animationTime=1000)
    #test.checkValidActions()


'''

Notes on size of problem 

You have 7 pieces total, so your bench can be any combination from 7-n to zero where n is the number of pieces on the board for any configuration 

'''