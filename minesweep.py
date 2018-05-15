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
REWARD_ALREADY_SHOWN_FIELD = -10

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
    print("Reward ALREADY SHOWN filed:", REWARD_ALREADY_SHOWN_FIELD)
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
        if np.random.uniform(0,1) < self._epsilon:
            action = np.argmax(rewards[0])
        else:
            action = np.random.choice(list(range(0,BOARD_VECTOR_LENGTH)))
        
        self._last_target = rewards
        self._last_state = state
        self._last_action = action
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
    _save_every = 1000
    _num_learning_rounds = 0
    over = False

    # Creates two fields, one containing bombs and one that is hidden
    def __init__(self, num_learn_rounds, dimX=3, dimY=3, bombs=1, learner=None, report_every=100, save_every=1000):
        self.dimX = dimX
        self.dimY = dimY
        self.bombs = bombs
        self.p = learner
        self.loss = 0
        self.win = 0
        self._report_every = report_every
        self._save_every = save_every
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
        if DEBUG_PRINT:
            print("choice: {},{}".format(x,y))
        if self.hasBomb(x,y):
            if DEBUG_PRINT:
                print("You loose!")
                self.showField()
            self.over = True
            self.loss += 1
            return REWARD_GAME_LOST
        else:
            if self.visibleField[x][y] != self.COVERED:
                return REWARD_ALREADY_SHOWN_FIELD
            self.updateVisibleField(x,y)
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
        while game.isRunning():
            x = int(input("Enter a number for x: "))
            y = int(input("Enter a number for y: "))
            game.pickField(x,y)
    
    def run(self):
        self.reset()
        state = self.visibleField

        if DEBUG_PRINT:
            print("game: {}".format(self.game))  #num of current game
    
        while True:
            # Determine hit/stay
            p1_action = self.p.get_action(state)
            # Apply the action if hit
            v = np.zeros(BOARD_VECTOR_LENGTH)
            v[p1_action] = 1
            reward = self.pickFieldByVector(v)
            self.p.update(self.visibleField,reward) # Update the learner with a reward of 0 (No change)

            # If game is over
            if not self.isRunning():
                break

        self.game += 1

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


num_learning_rounds = 10000    #50000
number_of_test_rounds = 200    #1000

game = MineSweeper(num_learning_rounds, BOARD_SIZE_X, BOARD_SIZE_Y ,NUMBER_OF_BOMBS,report_every=100, save_every=10000)
#game = MineSweeper(num_learning_rounds, BOARD_SIZE_X, BOARD_SIZE_Y ,NUMBER_OF_BOMBS, RMPlayer())
total = num_learning_rounds + number_of_test_rounds

#write in file
#orig_stdout = sys.stdout
#f = open('results5x5x3-combinations-Jovana.txt', 'w')
#sys.stdout = f

#measure time
start_time = datetime.now() 

#REWARD_GAME_WON_values = [0, 50, 150, 300]
#REWARD_GAME_LOST_values = [0, -50, -150, -300]

#for i in range(0, len(REWARD_GAME_WON_values)):
#    REWARD_GAME_WON = REWARD_GAME_WON_values[i]
#    for j in range(0, len(REWARD_GAME_LOST_values)):
#        REWARD_GAME_LOST = REWARD_GAME_LOST_values[j]
#CURRENT_GAME = 0

print("Learning starts")
for k in range(0,total):
    #CURRENT_GAME += 1
    game.run()
print("Test ended")
print("------------------------------------")

#game.play()

time_elapsed = datetime.now() - start_time 
print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

#sys.stdout = orig_stdout
#f.close()