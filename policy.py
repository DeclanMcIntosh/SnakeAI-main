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

from tensorflow.keras.models import Sequential,  clone_model, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, LeakyReLU, InputLayer, Add
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np

import random as r


def Snake_Model():

    def residual(x, channels):
        x1 = Conv2D(channels, 3, padding='same')(x)
        #x1 = BatchNormalization()(x1)
        x1 = ReLU()(x1)     
        x1 = Conv2D(channels, 3, padding='same')(x1)
        #x1 = BatchNormalization()(x1)
        x1 = ReLU()(x1)       
        x = Add()([x,x1])
        x = ReLU()(x)
        return x


    input_ = Input(shape=(5,5,8))

    x = Conv2D(128,1,padding='same')(input_)
    #x = BatchNormalization()(x)
    x = ReLU()(x)     
    x = residual(x, 128)
    x = residual(x, 128)
    x = residual(x, 128)
    x = residual(x, 128)

    x1 = Conv2D(2,1,padding='same')(x)
    x1 = Flatten()(x1)
    x1 = Dense(128)(x1)
    x1 = ReLU()(x1) 
    x1 = Dense(1)(x1)
    x1 = Activation('tanh')(x1)

    #x2 = Conv2D(2,1,padding='same')(x)
    #x2 = Flatten()(x2)
    #x2 = Dense(128)(x2)
    #x2 = ReLU()(x2) 
    #x2 = Dense(1)(x2)
    #x2 = Activation('sigmoid')(x2)

    model = Model(inputs=input_,outputs=x1)
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3))
    return model

class Policy():
    def __init__(self, epsilon=0.0, gamma=0.95, board_size=5):
        self.epsilon = epsilon
        self.gamma = gamma
        self.board_size = board_size
        self.model = Snake_Model()

        self.historyStates = []
        self.historyActions = []
        self.historyRewards = []

        self.model.build(input_shape = (None, 5,5,10))

    def createModelInput(self, state, action):
        # make sure head you are evaluating for is at depth 0...
        options = [[1,0],[0,1],[-1,0],[0,-1]]

        input_mod = state
        #print(input_mod[:,:,0])

        input_mod = np.rot90(input_mod,options.index(action), axes=(0,1))

        #print(input_mod[:,:,0])

        if np.isnan(np.sum(input_mod)):
            print("what the fuck inputs") 

        input_mod = np.clip(input_mod, 0, 1)     
        return input_mod

    def getAction(self, state, valids):
        if len(valids) == 0:
            return [0,0]
        if len(valids) == 1:
            return valids[0]

        #t0 = time.time()

        input_mods = [np.expand_dims(self.createModelInput(state, valids[0]),axis=0)]

        for action in valids[1:]:
            
            input_mods.append(np.expand_dims(self.createModelInput(state, action),axis=0))

        input_mods = np.concatenate(input_mods, axis=0)
            
        values = self.model(input_mods, training=False)

        estimates =  np.squeeze(values.numpy(),axis=1)

        if np.isnan(np.sum(estimates)):
            print("what the fuck")

        if r.random() > self.epsilon:
            index = np.argmax(estimates)
            action = valids[index]
            prediction = estimates[index]
        else:
            action = r.choice(valids)
            index = valids.index(action)
            prediction = estimates[index]

        #print(str((time.time()-t0)))
        return action, prediction

        

        #weighted roullet wheel of choices
        #estimates = [x/sum(estimates) for x in estimates]
        #return r.choices(valids, k=1, weights=estimates)[0]

    def compile(self):
        self.model.compile(loss='mse', optimizer=SGD(learning_rate=1e-3, momentum=0.9))
        self.model.build(input_shape = (None, 5,5,10))

    def trainModelPredictor(self, states, actions, rewards):

        c = list(zip(states, actions, rewards))
        r.shuffle(c)   
        states, actions, rewards = zip(*c)

        states  = states[-32:]
        actions = actions[-32:]
        rewards = rewards[-32:]

        rewards = np.expand_dims(np.array(rewards,dtype=np.float32),1)/25.

        rewards = np.clip(rewards, -1, 1)

        inputs = []

        for x in range(len(states)):
            inputs.append(self.createModelInput(states[x],actions[x]))

        inputs = np.array(inputs)

        loss = self.model.train_on_batch(inputs, rewards)

        return loss

class RandomOponant():
    def __init__(self):
        pass
    def getAction(self, state, valids):
        #print(valids)
        if len(valids) > 0:
            x = r.choice(valids)
            return x, 0
        else: 
            return [0,1], 0