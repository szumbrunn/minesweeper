import numpy as np
import random

# Basic class for main sweeper game, currently supports inputs by command line
class MineSweeper(object):
    BOMB = 9
    EMPTY = 0
    FLAGGED = 10
    COVERED = 99
    fieldsToPick = 0
    dimX = 0
    dimY = 0
    over = False

    # Creates two fields, one containing bombs and one that is hidden
    def __init__(self, dimX=3, dimY=3, bombs=1):
        self.dimX = dimX
        self.dimY = dimY
        self.bombs = bombs

        random.seed()

        self.fieldsToPick = dimX*dimY - bombs
        self.field = np.zeros((dimX,dimY))
        self.visibleField = np.ones((dimX,dimY))*self.COVERED
        i = bombs
        # place bombs
        while i>0:
            x = random.randint(0,dimX-1)
            y = random.randint(0,dimY-1)

            if self.field[x][y] != self.BOMB:
                self.field[x][y] = self.BOMB
                i -= 1

        # calc nearby bomb fields
        for x in range(0,dimX):
            for y in range(0,dimY):
                self.field[x][y] = self.fieldValue(x,y)

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
        if self.hasBomb(x,y):
            print("You loose!")
            self.over = True
            return
        else:
            self.updateVisibleField(x,y)
            self.showVisibleField()
            if self.bombs==self.countUncovered():
                print("You win!")
                self.over = True
                return

    def countUncovered(self):
        unique, counts = np.unique(self.visibleField, return_counts=True)
        return dict(zip(unique, counts))[99]

    def play(self):
        while game.isRunning():
            x = int(input("Enter a number for x: "))
            y = int(input("Enter a number for y: "))
            game.pickField(x,y)


game = MineSweeper(10,10,11)
game.play()

