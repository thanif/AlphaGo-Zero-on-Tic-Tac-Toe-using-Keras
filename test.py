import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
import math
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, BatchNormalization, LeakyReLU, Flatten, merge, Input, ELU
from keras.models import load_model, Model
from keras.optimizers import SGD
from keras import regularizers
import os.path

def softmax_cross_entropy_with_logits(y_true, y_pred):

	p = y_pred
	pi = y_true

	zero = tf.zeros(shape = tf.shape(pi), dtype=tf.float32)
	where = tf.equal(pi, zero)

	negatives = tf.fill(tf.shape(pi), -100.0) 
	p = tf.where(where, negatives, p)

	loss = tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)

	return loss

def custom_loss_function(y_true, y_pred):

	p = y_pred
	pi = y_true

	zero = tf.zeros(shape = tf.shape(pi), dtype=tf.float32)
	where = tf.equal(pi, zero)

	negatives = tf.fill(tf.shape(pi), -100.0) 
	p = tf.where(where, negatives, p)

	loss = tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)

	return loss





class Node:


	def __init__(self,dim):
	
		self.state = np.zeros(dim*dim)
		self.actions = None
		self.n = 0
		self.w = 0
		self.q = 0
		self.v = 0
		self.p = 0
		self.parent = None
		self.child = None




def actions(n):

	n.actions = []
	
	

	for i in range(np.shape(n.state)[0]):
	
		
	
		if n.state[i] == 0:
		
			n.actions.append(i)
			
	return n
	
	
def children(n,player,opponent):

	dim = int((np.shape(n.state)[0])**(1/2.0))
	
	n.child = []

	for i in range(len(n.actions)):
	
		c = Node(dim)
		
		c.state[:] = n.state[:]
		
		c.state[n.actions[i]] = player
			
		c.parent = n	
		
		
		n.child.append(c)
		
	return n	


def conclusion(node, player, opponent, option):	

	
	
					 	
			
	diagonal_1 = int((np.shape(node.state)[0])**(1/2.0))
	
	if option == 0:
	
		k = 0
	
	
	
		for i in range(diagonal_1):
	
			sum_c = 0
	
			for j in range(diagonal_1):
		
				if node.state[k] == player:
			
					sum_c+=1
				
				k+=1	
		
			if sum_c == diagonal_1:
		
			
		
				return 1
			
		k = 0
	
	
	
		for i in range(diagonal_1):
	
			sum_c = 0
	
			for j in range(diagonal_1):
		
				if node.state[k] == opponent:
			
					sum_c+=1
				
				k+=1	
		
			if sum_c == diagonal_1:
		
				return 2		
						
	
	
	
	
	
	
		for i in range(diagonal_1):
	
			sum_r = 0
		
			k = i
	
			for j in range(diagonal_1):
		
				if node.state[k] == player:
			
					sum_r+=1
				
				k+=diagonal_1	
		
			if sum_r == diagonal_1:
		
			
		
				return 1
			

	
	
	
		for i in range(diagonal_1):
	
			sum_r = 0
		
			k = i
	
			for j in range(diagonal_1):
		
				if node.state[k] == opponent:
			
					sum_r+=1
				
				k+=diagonal_1
		
			if sum_r == diagonal_1:
		
				return 2		
						
		
		
			
		
		check  = 0
		
		sum_d1 = 0
		
		for d_1 in range(diagonal_1):
	
			if node.state[check] == player:
				
				sum_d1+=1
			check+=diagonal_1 + 1
		
		if sum_d1 == diagonal_1:
	
		
		
			return 1		
				
		
		check = 0
		
		sum_d1 = 0
		
		for d_1 in range(diagonal_1):
		
			if node.state[check] == opponent:
				
				sum_d1+=1
			check+=diagonal_1+1
		
		if sum_d1 == diagonal_1:
		
			return 2		
		
		
		check = diagonal_1-1
		
		sum_d2 = 0		
			
		for d_2 in range(diagonal_1):
		
			if node.state[check] == player:
			
				sum_d2+=1
				
			check+= diagonal_1 - 1
				
		if sum_d2 == diagonal_1:
	
		
		
			return 1
			
		
		check = diagonal_1 - 1
		
		sum_d2 = 0		
			
		for d_2 in range(diagonal_1):
		
			if node.state[check] == opponent:
			
				sum_d2+=1
				
			check+= diagonal_1 - 1
				
		if sum_d2 == diagonal_1:
		
			return 2	
			
		if 0 in node.state:
			
			return -1	
			
				 	
		
	return 0
	
			


		
def simulation(n,player,opponent,epsilon,current_move_p,pi_model,z_model):

	
	
	path = []
	ids = []
	
	
	#print "start state : ", n.state
	
	if n.child!=None:
	
	
		while n.child!=None:
	
	
		
	
	
	
	
	
			U_Q = []
		
		
			
			for i in range(len(n.child)):
		
				
		
				U_Q.append(epsilon*(n.child[i].p/float(n.child[i].n+1))+n.child[i].q)
				
			
			
					
			
			a_max = np.argmax(U_Q)
		
		
		
			ids.append(a_max)
			
			
		
			n = n.child[a_max]
		
			path.append(n)	
		
		
		
		#print "leaf state : ", n.state
		
		if conclusion(n,player,opponent,0) == -1:	
			n = actions(n)
	
			n = children(n, player, opponent)
			
            		
                
                	model = pi_model
            		
                		
			x_test = np.zeros((1,1,10))
			x_test[0,0,:9] = n.state	
			x_test[0,0,9] = player	
			x_test = x_test[...,np.newaxis]	
			p_ann = model.predict(x_test, batch_size=1)	
		
			allowed = []
		
			for i in range(len(n.state)):
		
				if n.state[i]==0:
			
					allowed.append(i)
				
				
			mask = np.ones(len(n.state),dtype=bool)
			mask[allowed] = False
		
		
			p_ann[0,mask] = -100

		

			#SOFTMAX
			odds = np.exp(p_ann)
			probs = odds / np.sum(odds) ###put this just before the for?
				
		
		
		
			model = z_model
			
			
			x_test = np.zeros((1,1,10))
			x_test[0,0,:9] = n.state	
			x_test[0,0,9] = player	
			x_test = x_test[...,np.newaxis]	
			v_ann = model.predict(x_test, batch_size=1)
	
		
	
	
			n.v = v_ann
		
		
			k = 0
		
			for i in range(len(n.state)):
		
				if i in allowed:
					n.child[k].p = probs[0,i]
				
					k+=1
				
			
		
			k = 0
		
			index = -1
			
			

			while n.parent!=None:
		
				if k==0:
	
		
					n = n.parent
					n.child[ids[index]].n+=1
					n.child[ids[index]].w+=v_ann
					n.child[ids[index]].q+=(n.child[ids[index]].w/float(n.child[ids[index]].n))
					
					
		
				
	
				else:
		
					n = n.parent
					n.child[ids[index]].n+=1
					n.child[ids[index]].w+=n.child[ids[index]].v
					n.child[ids[index]].q+=(n.child[ids[index]].w/float(n.child[ids[index]].n))
					
					
		
				k+=1
			
				index-=1
				
			#print "back up : ",n.state
	

		else:
		
			k = 0
		
			index = -1
			
			

			while n.parent!=None:
		
	
				n = n.parent
				n.child[ids[index]].n+=1
				n.child[ids[index]].w+=n.child[ids[index]].v
				n.child[ids[index]].q+=(n.child[ids[index]].w/float(n.child[ids[index]].n))
					
					
		
				k+=1
			
				index-=1
				
			#print "back up : ",n.state


		
	else:
	
		if conclusion(n,player,opponent,0) == -1:	
		
			n = actions(n)
	
			n = children(n, player, opponent)

			model = pi_model

			x_test = np.zeros((1,1,10))
			x_test[0,0,:9] = n.state	
			x_test[0,0,9] = player	
			x_test = x_test[...,np.newaxis]	
			
			p_ann = model.predict(x_test, batch_size=1)	
		
			allowed = []
		
			for i in range(len(n.state)):
		
				if n.state[i]==0:
			
					allowed.append(i)
				
				
			mask = np.ones(len(n.state),dtype=bool)
			mask[allowed] = False
		
		
			p_ann[0,mask] = -100

		

			#SOFTMAX
			odds = np.exp(p_ann)
			probs = odds / np.sum(odds) ###put this just before the for?
				
		
			model = z_model		

			x_test = np.zeros((1,1,10))
			x_test[0,0,:9] = n.state		
			x_test[0,0,9] = player
			x_test = x_test[...,np.newaxis]	
			v_ann = model.predict(x_test, batch_size=1)
	
		
	
	
		
		
			n.v = v_ann
			
			
			k = 0
		
			for i in range(len(n.state)):
		
				if i in allowed:
					n.child[k].p = probs[0,i]
				
					k+=1
					
			
			a = np.zeros(len(n.state))
				
			j = 0	
					
			for i in probs[0]:
			
				a[j] = i
				
				j+=1


			current_move_p.append(a)
				
			
							
						
			#print "current probs : ",current_move_p	
			
		
		
	return n,current_move_p

	
				

def conv_layer(_in):


	conv1 = Conv2D(256, (3, 3), activation='linear', kernel_regularizer = regularizers.l2(0.0001),padding = 'same')(_in)

    	bn1 = BatchNormalization()(conv1)
    	
    	lr1 = ELU()(bn1)

	return lr1
	
	
def residual_layer(_in):

    	conv1 = Conv2D(256, (3, 3), activation='linear', kernel_regularizer = regularizers.l2(0.0001),padding = 'same')(_in)

    	bn1 = BatchNormalization()(conv1)
    	
    	lr1 = ELU()(bn1)
    	
    	conv2 = Conv2D(256, (3, 3), activation='linear', kernel_regularizer = regularizers.l2(0.0001),padding = 'same')(lr1)

    	bn2 = BatchNormalization()(conv2)
    	
    	m1 = merge([_in, bn2], mode='sum')
    	
    	lr2 = ELU()(m1)
    	
    	
    	return lr2
	
	


def value_head(_in):


	conv1 = Conv2D(1, (1, 1), activation='linear', kernel_regularizer = regularizers.l2(0.0001),padding = 'same')(_in)
	
	bn1 = BatchNormalization()(conv1)
	
	lr1 = ELU()(bn1)	
	
	f1 = Flatten()(lr1)
	
	d1 = Dense(20, activation='linear', kernel_regularizer=regularizers.l2(0.0001))(f1)
	
	lr2 = ELU()(d1)
	
	d2 = Dense(1, use_bias=False, activation='tanh', kernel_regularizer=regularizers.l2(0.0001))(lr2)
	
	return d2

def policy_head(_in):


	conv1 = Conv2D(2, (1, 1), activation='linear', kernel_regularizer = regularizers.l2(0.0001),padding = 'same')(_in)

	bn1 = BatchNormalization()(conv1)
	
	lr1 = ELU()(bn1)
	
	f1 = Flatten()(lr1)
	
	d1 = Dense(9, activation='linear', kernel_regularizer=regularizers.l2(0.0001))(f1)
	
	return d1



def mcst(n, iterations,player,opponent,current_move_p,pi_model,z_model,p_val):


	
	
	z = Node(int(math.sqrt(len(n.state))))
	
	z.state = n.state
	
	
	epsilon = 1001
	
	for i in range(iterations):
	
	
		#print "Simulation : ", str(i+1), epsilon
		
		
	
		z,current_move_p = simulation(z,player,opponent,epsilon,current_move_p,pi_model,z_model)
		
		
		
		if (i+1)%10==0:
		
			epsilon-=100
		
		#for i in range(len(z.child)):
		
		#	print z.child[i].n
		
	
		
	best = []	
	
	
		
	for i in range(len(z.child)):
		
		
	
		best.append(z.child[i].n)
		
		
				
	
	
			
	p_val.append(best)
	
	return z.child[np.argmax(best)],current_move_p,p_val,np.argmax(best)
		
	
		
	

def print_maze(node):

    board = ""
    
    l = 0


    dim = int((np.shape(node.state)[0])**(1/2.0))	
   
    for i in range(int((np.shape(node.state)[0])**(1/2.0))*2):
        
        if i%2 == 0:
        
        	for k in range(int((np.shape(node.state)[0])**(1/2.0))):
            		board += "|"
            		
            		
            		
            		
            		if node.state[l]==1:
            		
            			
            		
            			board+=" A "
            			
            			board+=" "
            			
            		elif node.state[l]==2:	
            		
            			
            		
            			board+=" B "
            			
            			board+=" "
            			
            		else:	
            		
            			board+="    "
            			
            		l+=1	
            		
            		
        else:
        
        	board+= " --- "*dim
       
        board += "\n"

    return board
		
		
		
		
	 
def episode(dim,turn,pi_model,z_model):

	states = []
	vals = []
	current_move_p = []
	free = []
	p_val = []
	p_child = []

	

	check = 0

	
	if turn == 1:
		p = 1
		
	else:
		p = 0
			
			
	n = Node(int(dim))		
		
	
	
	print "Start State  \n"
			
			
			
	maze = print_maze(n)
			
	print maze
	
	
	while conclusion(n, 1, 2, 0) == -1:

		
	
		if p==1:
		
			
		
			print "Monte Carlo 1's Move"
			
			
			a = np.zeros(len(n.state)+1)
			
			a[:9] = n.state
			
			a[9] = 1
			
			states.append(a)
			
			
			
			
			
			
			
			n,probs,p_val,best = mcst(n,100,1,2,current_move_p,pi_model,z_model,p_val)

			

			

			#for i in range(len(n.child)):
			
			#	print "asd : ", probs
			
				
			
			print "After Monte Carlo 1's Move "
			
			#print "check : ", current_move_p
			
			maze = print_maze(n)
			
			print maze
		
				
		
			p=0
			
	
		elif p==0:
		
			
		
			print "Player's Move"
			

			ind = raw_input("Enter the index at which you want to place your move ")
			
			if n.state[int(ind)] == 0:
			
				n.state[int(ind)] = 2
				
			else:
			
				while n.state[int(ind)] != 0:
				
					ind = raw_input("Enter the index at which you want to place your move ")	 	
				
				n.state[int(ind)] = 2
			
			print "After Player's Move "
			
			#print "check : ", current_move_p
			
			maze = print_maze(n)
			
			print maze
		

			p=1
	
	if conclusion(n, 1, 2, 0) == 1:
	
		print "Monte Carlo 1 Wins "
		
		for i in range(len(states)):
		
			if states[i][9]==1:
		
				vals.append(1)
			else:
			
				vals.append(-1)	
		
	elif conclusion(n, 1, 2, 0) == 2:
	
		print "Player Wins"
		
		for i in range(len(states)):
			
			if states[i][9]==2:
		
				vals.append(1)
			else:
			
				vals.append(-1)	

	else:
	
		print "Game Drawn "	
		
		for i in range(len(states)):
		
			vals.append(0)
		
	
	
	
	#print "states : ",states
	
	#print "probs : ",probs
	
	#print "vals : ",vals
	
	
	#print np.shape(states)
	
	#print np.shape(probs)
	
	#print np.shape(vals)
	
	return states,probs,vals,p_val		

def main():



	pi_model = load_model('pi_model_ann.h5', custom_objects={'custom_loss_function': custom_loss_function})


	z_model  = load_model('z_model_ann.h5', custom_objects={'custom_loss_function': custom_loss_function})
	
	ch = raw_input("Enter 0 if you want to take the first turn ")
	
	if int(ch)==0:
	
		turn = 0
	
	else:
	
		turn = 1	
	
	
	
	

	for ep in range(2):
	
		print "Episode : ",str(ep+1)
	
		states,_,vals,p = episode(3,turn,pi_model,z_model)

		
		
		if turn == 0:
		
			turn = 1
			
		else:
		
			turn = 0	
		

	
if __name__ == "__main__":
    main()	
