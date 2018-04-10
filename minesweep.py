from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import sys
import random

BOARD_VECTOR_LENGTH = 100

class DQNLearner(object):
    def __init__(self):
        super().__init__()
        self._last_target = None
        self._learning = True
        self._learning_rate = .1
        self._discount = .1
        self._epsilon = .9
        self._last_action = None
        self._last_state = None


        # Create Model
        model = Sequential()

        model.add(Dense(2, init='lecun_uniform', input_shape=(BOARD_VECTOR_LENGTH,)))
        model.add(Activation('relu'))

        model.add(Dense(10, init='lecun_uniform'))
        model.add(Activation('relu'))

        model.add(Dense(4, init='lecun_uniform'))
        model.add(Activation('linear'))

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

        self._model = model

    def get_action(self, state):
        state = state.flatten()
        rewards = self._model.predict([np.array([state])], batch_size=1)
        if np.random.uniform(0,1) < self._epsilon:
            action = np.argmax(rewards[0])
        else:
            action = np.random.choice(list(range(0,BOARD_VECTOR_LENGTH)))
        
        self._last_target = rewards
        self._last_state = state
        return action

    def update(self,new_state,reward):
        new_state = new_state.flatten()
        if self._learning:
            rewards = self._model.predict([np.array([new_state])], batch_size=1)
            maxQ = np.max(rewards[0])
            new = self._discount * maxQ
            
            self._last_target[0][self._last_action] = reward+new
    
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
    over = False

    # Creates two fields, one containing bombs and one that is hidden
    def __init__(self, dimX=3, dimY=3, bombs=1, learner=None):
        self.dimX = dimX
        self.dimY = dimY
        self.bombs = bombs
        self.p = learner

        if self.p is None:
            self.p = DQNLearner()

        random.seed()

        self.reset()
    
    def reset(self):
        random.seed()
        self.over = False

        self.fieldsToPick = self.dimX*self.dimY - self.bombs
        self.field = np.zeros((self.dimX,self.dimY))
        self.visibleField = np.ones((self.dimX,self.dimY))*self.COVERED
        i = self.bombs
        # place bombs
        while i>0:
            x = random.randint(0,self.dimX-1)
            y = random.randint(0,self.dimY-1)

            if self.field[x][y] != self.BOMB:
                self.field[x][y] = self.BOMB
                i -= 1

        # calc nearby bomb fields
        for x in range(0,self.dimX):
            for y in range(0,self.dimY):
                self.field[x][y] = self.fieldValue(x,y)

        #self.showVisibleField()
    
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
        if self.hasBomb(x,y):
            print("You loose!")
            self.over = True
            return -10
        else:
            if self.visibleField[x][y] == 99:
                self.showVisibleField()
            self.updateVisibleField(x,y)
            if self.bombs==self.countUncovered():
                print("You win!")
                self.over = True
                return 10
            else:
                if self.field[x][y] == 0:
                    return 5
                else:
                    return 1

    def pickFieldByVector(self, v):
        v = np.reshape(v,(self.dimX,self.dimY))
        coodX = -1
        coodY = -1
        count = 0
        for x in range(0,self.dimX):
            for y in range(0,self.dimY):
                if v[x][y] == 1 and self.visibleField[x][y] == 99:
                    count += 1
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

        print("game: {}".format(self.game))
    
        while True:
            # Determine hit/stay
            p1_action = self.p.get_action(state)
        
            # Apply the action if hit
            v = np.zeros(BOARD_VECTOR_LENGTH)
            v[p1_action] = 1
            reward = self.pickFieldByVector(v)
        
            # If game is over
            if not self.isRunning():
                break
        
            self.p.update(self.visibleField,reward) # Update the learner with a reward of 0 (No change)

        self.game += 1


game = MineSweeper(10,10,6)
#game.play()
# while game.isRunning():
#     v = np.zeros(9)
#     v[random.randint(0,8)] = 1
#     print(np.reshape(v,(3,3)))
#     game.pickFieldByVector(v)

num_learning_rounds = 2
#game = Game(num_learning_rounds, Learner()) #Q learner
number_of_test_rounds = 1
for k in range(0,num_learning_rounds + number_of_test_rounds):
    game.run()