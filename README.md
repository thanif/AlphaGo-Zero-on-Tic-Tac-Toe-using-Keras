# Alpha-Go-Zero
Mastering Tic Tac Toe Using Self Play and Reinforcement Learning (3X3 Board)

# Node Class
Node Class contains the following attributes

n is the number of times an action has been taken

w is the total value of the next state

q is the mean value of the next state

p is the prior probability of selecting action a 

state contains the current state of the game

actions contain the possible actions from current state

p contains the move probabilities

v is the value of state (for the current player)

# actions function
It returns the actions possible from current state in the form of indices

# children function
It returns the updated node with child array containing possible actions in the form of states and parent containing the current state

# conclusion function
It returns 1 if player wins, 2 if opponent wins, -1 if game has not finished, 0 if game has drawn

# simulation function
It performs a complete Monte Carlo simulation and returns move probabilities

# mcst function
It performs Monte Carlo Tree Search by performing multiple simulations and returns best move, current_move_probabilities

# print_maze function
It prints the output in the form of a board

# episode function
It performs a complete episode of the game using Monte Carlo Tree Search and returns the predictions from pi_model providing move probabilities, z_model providing value of the state and states providing different states during an episode with an additional value of 1 if it's player 1's move and 2 if it's player 2's move 

# References
https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188

https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0

http://tim.hibal.org/blog/alpha-zero-how-and-why-it-works/

https://applied-data.science/static/main/res/alpha_go_zero_cheat_sheet.png
