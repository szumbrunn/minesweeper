from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import sys
import random
from datetime import datetime 

# define board size and number of bombs
BOARD_SIZE_X = 5
BOARD_SIZE_Y = 5
NUMBER_OF_BOMBS = 3
FIRST_CHOICE_FREE = False ## make sure this is False! there's a bug somewhere otherwise

# define individual rewards after each step/game
REWARD_GAME_WON = 10
REWARD_GAME_LOST = -10

REWARD_ZERO_FIELD = 0
REWARD_NUMBER_FIELD = 10

# calculate actual input vector size
BOARD_VECTOR_LENGTH = BOARD_SIZE_X*BOARD_SIZE_Y

# enable debug print
DEBUG_PRINT = False

SAVE_MODEL = True
FILE_INPUT = 'my_model.h5'
LOAD_MODEL = True 
FILE_OUTPUT= 'my_model.h5'

def printParams():
    print("Reward game WON:", REWARD_GAME_WON)
    print("Reward game LOST:", REWARD_GAME_LOST)
    print("Reward ZERO field:", REWARD_ZERO_FIELD)
    print("Reward NUMBER field:", REWARD_NUMBER_FIELD)
    print("====================================")

class RMPlayer(object):
    def __init__(self):
        super().__init__()

    def get_action(self, state):
        action = np.random.choice(list(range(0,BOARD_VECTOR_LENGTH)))
        return action

    def update(self,new_state,reward):
        #do nothing
        i = 0

class DQNLearner(object):
    def __init__(self):
        super().__init__()
        self._last_target = None
        self._learning = True
        self._learning_rate = .01
        self._discount = .2
        self._epsilon = .9 #set to 0.1 and then change during iterations
        self._last_action = None
        self._last_state = None


        # Create Model
        model = Sequential()

        model.add(Dense(100, kernel_initializer='lecun_uniform', input_shape=(BOARD_VECTOR_LENGTH,)))
        model.add(Activation('relu'))

        model.add(Dense(50, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))

        model.add(Dense(BOARD_VECTOR_LENGTH, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

        self._model = model

        if LOAD_MODEL:
            self.load_model()


    def get_action(self, state):
        state = state.flatten()
        rewards = self._model.predict([np.array([state])], batch_size=1)
        #if np.random.uniform(0,1) < self._epsilon:
        #    action = rewards[0]
        #else:
        #    action = np.random.choice(list(range(0,BOARD_VECTOR_LENGTH)))

        #print(self._epsilon)
        #self._last_target = rewards
        self._last_state = state
        return rewards[0]

    def update(self,new_state,rewards):
        new_state = new_state.flatten()
        if self._learning:
            rewards = self._model.predict([np.array([new_state])], batch_size=1)
            maxQ = np.max(rewards[0])
            new = self._discount * maxQ
    
            # Update model
            self._model.fit(np.array([self._last_state]), 
                            rewards, 
                            batch_size=1, 
                            epochs=1, 
                            verbose=0)

    def save_model(self):
        print("Saving to: "+FILE_OUTPUT)
        self._model.save(FILE_OUTPUT)  # creates a HDF5 file 'my_model.h5'

    def load_model(self):
        # returns a compiled model
        # identical to the previous one
        print("Loading from: "+FILE_INPUT)
        self._model = load_model(FILE_INPUT)

# Basic class for main sweeper game, currently supports inputs by command line
class MineSweeper(object):
    BOMB = 9
    EMPTY = 0
    FLAGGED = 10 #not implemented
    COVERED = -1 #change to -1, it was 99
    fieldsToPick = 0
    dimX = 0
    dimY = 0
    game = 0
    win = 0
    loss = 0
    _first_move = True
    _report_every = 0
    _save_every = 100000
    _num_learning_rounds = 0
    over = False
    round = 0

    # Creates two fields, one containing bombs and one that is hidden
    def __init__(self, num_learn_rounds, dimX=3, dimY=3, bombs=1, learner=None, report_every=100, save_every=1000, debug=False):
        self.dimX = dimX
        self.dimY = dimY
        self.bombs = bombs
        self.p = learner
        self.loss = 0
        self.win = 0
        self._report_every = report_every
        self._save_every = save_every
        self._num_learning_rounds = num_learn_rounds
        self._debug = debug

        if self.p is None:
            self.p = DQNLearner()

        random.seed()

        self.reset()

    def setField(self, field):
        self.reset()
        self.field = field
    
    def reset(self, choiceX=-1, choiceY=-1):
        random.seed()
        self.over = False
        self._first_move = True

        self.fieldsToPick = self.dimX*self.dimY - self.bombs
        self.field = np.zeros((self.dimX,self.dimY))
        self.visibleField = np.ones((self.dimX,self.dimY))*self.COVERED
        i = self.bombs
        # place bombs
        while i>0:
            x = random.randint(0,self.dimX-1)
            y = random.randint(0,self.dimY-1)
            if self.field[x][y] != self.BOMB:
                if choiceX > 0 and choiceY > 0:
                    if choiceX!=x and choiceY!=y:
                        self.field[x][y] = self.BOMB
                        i -= 1
                else:
                    self.field[x][y] = self.BOMB
                    i -= 1
        #self.field[0][1] = self.BOMB

        # calc nearby bomb fields
        self.calcBombFields()

        if self._debug:
            self.showVisibleField()
    
    def calcBombFields(self):
        for x in range(0,self.dimX):
            for y in range(0,self.dimY):
                self.field[x][y] = self.fieldValue(x,y)
    
    def fieldValue(self, x,y):
        if self.hasBomb(x,y):
            return self.BOMB
        count = 0
        for a in range(max(x-1,0), min(x+2,self.dimX)):
            for b in range(max(y-1,0),min(y+2,self.dimY)):
                count += self.hasBomb(a,b)
        return count

    def hasBomb(self, x,y):
        if self.field[x][y]==self.BOMB:
            return True
        else:
            return False

    def showField(self):
        print(self.field)

    def showVisibleField(self):
        print(self.visibleField)

    def updateVisibleField(self, x, y):
        self.showFieldsRecursively(x,y)
    
    def showFieldsRecursively(self,x,y):
        # if field has value
        self.visibleField[x][y] = self.field[x][y]
        if self.field[x][y] > 0:
            self.fieldsToPick -= 1
            return
        else:
            # if field is empty
            for a in range(max(x-1,0), min(x+2,self.dimX)):
                for b in range(max(y-1,0),min(y+2,self.dimY)):
                    #if a in range(0,self.dimX) and b in range(0,self.dimY) and a!=x and b!=y:
                    if self.visibleField[a][b]== self.COVERED:
                        if self.field[a][b]<9 and self.field[a][b]>0:
                            self.visibleField[a][b] = self.field[a][b]
                            self.fieldsToPick -= 1
                        elif self.field[a][b]==0 :
                            self.showFieldsRecursively(a,b)
    
    def isRunning(self):
        if self.fieldsToPick>0 and self.over==False:
            return True
        else:
            return False

    def pickField(self, x, y):
        if self._debug:
            print("choice: {},{}".format(x,y))
        if self.hasBomb(x,y):
            if self._debug:
                print("You loose!")
                self.showField()
            self.over = True
            self.loss += 1
            return REWARD_GAME_LOST
        else:
            self.updateVisibleField(x,y)
            if self.bombs==self.countUncovered():
                if self._debug:
                    print("You win!")
                    self.showField()
                self.over = True
                self.win += 1
                return REWARD_GAME_WON
            else:
                if self.field[x][y] == 0:
                    return REWARD_ZERO_FIELD
                else:
                    return REWARD_NUMBER_FIELD-self.field[x][y]

    def pickFieldByVector(self, v):
        v = np.reshape(v,(self.dimX,self.dimY))
        coodX = -1
        coodY = -1
        for x in range(0,self.dimX):
            for y in range(0,self.dimY):
                if v[x][y] == 1:
                    coodX = x
                    coodY = y
        return self.pickField(coodX,coodY)


    def countUncovered(self):
        unique, counts = np.unique(self.visibleField, return_counts=True)
        return dict(zip(unique, counts))[self.COVERED]

    def play(self):
        while self.isRunning():
            x = int(input("Enter a number for x: "))
            y = int(input("Enter a number for y: "))
            self.pickField(x,y)

    def removeInvalidActions(self, actions):
        actions = np.reshape(actions,(self.dimX,self.dimY))
        for x in range(0,self.dimX):
            for y in range(0,self.dimY):
                if self.visibleField[x][y] != self.COVERED:
                    actions[x][y] = -10e10
        return actions
    
    def run(self):
        self.reset()
        if self._debug:
            print("game: {}".format(self.game))  #num of current game
        while True:
            state = self.visibleField
            # Determine hit/stay
            rewards = self.p.get_action(state)
            actions = self.removeInvalidActions(rewards)
            #print(actions)
            v = np.zeros(BOARD_VECTOR_LENGTH) 
            if np.random.random() > self.p._epsilon:
                lastaction = np.random.choice(np.where(actions > -10e10)[0])
            else:    
                lastaction = np.argmax(actions)
            v[lastaction] = 1
            reward = self.pickFieldByVector(v)
            rewards[lastaction] += reward 
            self.p.update(state,rewards) # Update the learner with a reward of 0 (No change)
            # If game is over
            if not self.isRunning():
                break

        if self.countUncovered() == self.bombs:
            print(self.showField())
        self.game += 1

        self.report()
    
    def test(self, field):
        self.setField(field)
        self.calcBombFields()
        if self._debug:
            print(self.field)
        state = self.visibleField

        if self._debug:
            print("game: {}".format(self.game))  #num of current game
    
        while True:
            # Determine hit/stay
            actions = self.p.get_action(state)
            # Apply the action if hit
            actions = self.removeInvalidActions(actions)
            #print(actions)
            v = np.zeros(BOARD_VECTOR_LENGTH) 
            lastaction = np.argmax(actions)
            v[lastaction] = 1
            reward = self.pickFieldByVector(v)
            #self.p.update(self.visibleField,reward) # Update the learner with a reward of 0 (No change)
            print(self.visibleField)
            # If game is over
            if not self.isRunning():
                break

        self.report()

    def report(self):
        if self.game % self._num_learning_rounds == 0:
            print("Learning ended")
            print("After " + str(self.game) + " games : {0:.4}".format(self.win / (self.win + self.loss)))
            printParams()
            print("Test starts")
        elif self.game % self._report_every == 0:
            print("After " + str(self.game) + " games : {0:.4}".format(self.win / (self.win + self.loss)))

        if SAVE_MODEL and self.game % self._save_every == 0:
            self.p.save_model()
