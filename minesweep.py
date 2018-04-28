from keras import Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import sys
import random

# define board size and number of bombs
BOARD_SIZE_X = 5
BOARD_SIZE_Y = 5
NUMBER_OF_BOMBS = 3
FIRST_CHOICE_FREE = False

# define individual rewards after each step/game
REWARD_GAME_WON = 10
REWARD_GAME_LOST = -10

REWARD_ZERO_FIELD = 5
REWARD_NUMBER_FIELD = 2
REWARD_ALREADY_SHOWN_FIELD = -100

# calculate actual input vector size
BOARD_VECTOR_LENGTH = BOARD_SIZE_X*BOARD_SIZE_Y

# enable debug print
DEBUG_PRINT = False

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
        self._epsilon = .9
        self._last_action_x = None
        self._last_action_y = None
        self._last_state = None


        # Create Model
        model = Sequential()

        model.add(Conv2D(5, 3, strides=2, activation="relu", input_shape=(BOARD_SIZE_X, BOARD_SIZE_Y,1)))

        model.add(Conv2D(1, 2, strides=1, activation="relu"))

        model.add(Conv2DTranspose(1, 5, strides=1))

        model.add(Dense(1,activation="relu"))





#        model.add(Conv2D(5,3,2, activation="relu", input_shape=(BOARD_SIZE_X, BOARD_SIZE_Y, 1)))



#        model.add(Dense(100, kernel_initializer='lecun_uniform'))
#        model.add(Activation('relu'))
#
#        model.add(Dense(50, kernel_initializer='lecun_uniform'))
#        model.add(Activation('relu'))
#
#        model.add(Dense(BOARD_VECTOR_LENGTH, kernel_initializer='lecun_uniform'))
#        model.add(Activation('linear'))

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

        self._model = model

    def get_action(self, state):
        # state = state.flatten()
        state = np.expand_dims(state, axis=2)
        rewards = self._model.predict(np.expand_dims(state, axis=0), batch_size=1)
        if np.random.uniform(0,1) < self._epsilon:
            #action = np.argmax(rewards[0])
            action_x = int(int(np.argmax(rewards[0])) / int(BOARD_SIZE_X))
            action_y = int(np.argmax(rewards[0]) % BOARD_SIZE_Y)

        else:
            action_x = np.random.choice(list(range(0, BOARD_SIZE_X)))
            action_y = np.random.choice(list(range(0, BOARD_SIZE_Y)))

        self._last_target = rewards
        self._last_state = state
        self._last_action_x = action_x
        self._last_action_y = action_y
        return action_x, action_y

    def update(self,new_state,reward):
#        new_state = new_state.flatten()
        if self._learning:
            new_state = np.expand_dims(new_state, axis=2)
            rewards = self._model.predict(np.expand_dims(new_state, axis=0), batch_size=1)
            maxQ = np.max(rewards[0])
            new = self._discount * maxQ
            
            self._last_target[0][self._last_action_x][self._last_action_y] = reward+new
    
            # Update model
            self._model.fit(np.array([self._last_state]), 
                            self._last_target, 
                            batch_size=1, 
                            epochs=1, 
                            verbose=0)

# Basic class for main sweeper game, currently supports inputs by command line
class MineSweeper(object):
    BOMB = 9
    EMPTY = 0
    FLAGGED = 10
    COVERED = 99
    fieldsToPick = 0
    dimX = 0
    dimY = 0
    game = 0
    win = 0
    loss = 0
    _first_move = True
    _report_every = 0
    _num_learning_rounds = 0
    over = False

    # Creates two fields, one containing bombs and one that is hidden
    def __init__(self, num_learn_rounds, dimX=3, dimY=3, bombs=1, learner=None, report_every=100):
        self.dimX = dimX
        self.dimY = dimY
        self.bombs = bombs
        self.p = learner
        self.loss = 0
        self.win = 0
        self._report_every = report_every
        self._num_learning_rounds = num_learn_rounds

        if self.p is None:
            self.p = DQNLearner()

        random.seed()

        self.reset()
    
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
        for x in range(0,self.dimX):
            for y in range(0,self.dimY):
                self.field[x][y] = self.fieldValue(x,y)

        if DEBUG_PRINT:
            self.showVisibleField()
    
    def fieldValue(self, x,y):
        if self.hasBomb(x,y):
            return 9
        count = 0
        for a in range(max(x-1,0), min(x+2,self.dimX)):
            for b in range(max(y-1,0),min(y+2,self.dimY)):
                count += self.hasBomb(a,b)
        return count

    def hasBomb(self, x,y):
        if self.field[x][y]==9:
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
                    if self.visibleField[a][b]==99:
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
        if DEBUG_PRINT:
            print("choice: {},{}".format(x,y))
        if FIRST_CHOICE_FREE and self._first_move==True:
            self.reset(x,y)
            self._first_move = False
        if self.hasBomb(x,y):
            if DEBUG_PRINT:
                print("You loose!")
                self.showField()
            self.over = True
            self.loss += 1
            return REWARD_GAME_LOST
        else:
            show = False
            if self.visibleField[x][y] == 99:
                show = True
            else:
                return REWARD_ALREADY_SHOWN_FIELD
            self.updateVisibleField(x,y)
            #if show and DEBUG_PRINT:
            #    self.showVisibleField()
            if self.bombs==self.countUncovered():
                if DEBUG_PRINT:
                    print("You win!")
                    self.showField()
                self.over = True
                self.win += 1
                return REWARD_GAME_WON
            else:
                if self.field[x][y] == 0:
                    return REWARD_ZERO_FIELD
                else:
                    return REWARD_NUMBER_FIELD

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
        return dict(zip(unique, counts))[99]

    def play(self):
        while game.isRunning():
            x = int(input("Enter a number for x: "))
            y = int(input("Enter a number for y: "))
            game.pickField(x,y)
    
    def run(self):
        self.reset()
        state = self.visibleField

        if DEBUG_PRINT:
            print("game: {}".format(self.game))
    
        while True:
            # Determine hit/stay
            p1_action_x, p1_action_y = self.p.get_action(state)
        
            # Apply the action if hit
            v = np.zeros((BOARD_SIZE_X,BOARD_SIZE_Y))    #(BOARD_VECTOR_LENGTH)
            v[p1_action_x][p1_action_y] = 1
            reward = self.pickFieldByVector(v)
            self.p.update(self.visibleField,reward) # Update the learner with a reward of 0 (No change)

            # If game is over
            if not self.isRunning():
                break

        self.game += 1

        self.report()

    def report(self):
        if self.game % self._num_learning_rounds == 0:
            print(str(self.game) +" : "  +str(self.win / (self.win + self.loss)))
        elif self.game % self._report_every == 0:
            print(str(self.win / (self.win + self.loss)))

num_learning_rounds = 50000
number_of_test_rounds = 1000

game = MineSweeper(num_learning_rounds, BOARD_SIZE_X, BOARD_SIZE_Y ,NUMBER_OF_BOMBS,report_every=10)
#game = MineSweeper(num_learning_rounds, BOARD_SIZE_X, BOARD_SIZE_Y ,NUMBER_OF_BOMBS, RMPlayer())
total = num_learning_rounds + number_of_test_rounds
for k in range(0,total):
    game.run()

#game.play()