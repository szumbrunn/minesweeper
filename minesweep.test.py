from minesweep import MineSweeper
import numpy as np

# define board size and number of bombs
BOARD_SIZE_X = 5
BOARD_SIZE_Y = 5
NUMBER_OF_BOMBS = 2

# calculate actual input vector size
BOARD_VECTOR_LENGTH = BOARD_SIZE_X*BOARD_SIZE_Y

# enable debug print
DEBUG_PRINT = False

FILE_OUTPUT= 'my_model.h5'

game = MineSweeper(10, BOARD_SIZE_X, BOARD_SIZE_Y ,NUMBER_OF_BOMBS,report_every=1, save_every=10000, debug=True)


field = np.zeros((BOARD_SIZE_X,BOARD_SIZE_Y))
field[0][0] = game.BOMB
field[2][1] = game.BOMB
field[4][3] = game.BOMB
game.test(field)