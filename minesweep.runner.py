from minesweep import MineSweeper
from datetime import datetime 

# define board size and number of bombs
BOARD_SIZE_X = 5
BOARD_SIZE_Y = 5
NUMBER_OF_BOMBS = 3

num_learning_rounds = 20000    #50000
number_of_test_rounds = 200    #1000

game = MineSweeper(num_learning_rounds, BOARD_SIZE_X, BOARD_SIZE_Y ,NUMBER_OF_BOMBS,report_every=100, save_every=10000, debug=False)
#game = MineSweeper(num_learning_rounds, BOARD_SIZE_X, BOARD_SIZE_Y ,NUMBER_OF_BOMBS, RMPlayer())
total = num_learning_rounds + number_of_test_rounds

start_time = datetime.now() 

print("Learning starts")
for k in range(0,total):
    game.run()
print("Test ended")
print("------------------------------------")
time_elapsed = datetime.now() - start_time 
print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))