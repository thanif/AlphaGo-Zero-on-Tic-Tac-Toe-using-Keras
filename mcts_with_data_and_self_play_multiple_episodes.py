import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import math
import numpy as np
import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, BatchNormalization, LeakyReLU, Flatten, concatenate, Input, ELU
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

    	m1 = concatenate([_in, bn2])

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

def model_pi():
    inputs = Input(shape=(1, 10, 1))

    conv1 = conv_layer(inputs)

    r1 = residual_layer(conv1)
    r2 = residual_layer(r1)
    r3 = residual_layer(r2)
    r4 = residual_layer(r3)
    r5 = residual_layer(r4)
    r6 = residual_layer(r5)
    r7 = residual_layer(r6)
    r8 = residual_layer(r7)
    r9 = residual_layer(r8)
    r10 = residual_layer(r9)

    r11 = residual_layer(r10)
    r12 = residual_layer(r11)
    r13 = residual_layer(r12)
    r14 = residual_layer(r13)
    r15 = residual_layer(r14)
    r16 = residual_layer(r15)
    r17 = residual_layer(r16)
    r18 = residual_layer(r17)
    r19 = residual_layer(r18)
    r20 = residual_layer(r19)

    r21 = residual_layer(r20)
    r22 = residual_layer(r21)
    r23 = residual_layer(r22)
    r24 = residual_layer(r23)
    r25 = residual_layer(r24)
    r26 = residual_layer(r25)
    r27 = residual_layer(r26)
    r28 = residual_layer(r27)
    r29 = residual_layer(r28)
    r30 = residual_layer(r29)


    r31 = residual_layer(r30)
    r32 = residual_layer(r31)
    r33 = residual_layer(r32)
    r34 = residual_layer(r33)
    r35 = residual_layer(r34)
    r36 = residual_layer(r35)
    r37 = residual_layer(r36)
    r38 = residual_layer(r37)
    r39 = residual_layer(r38)
    r40 = residual_layer(r39)

    output = policy_head(r40)



    model = Model(inputs=[inputs], outputs=[output])

    model.compile(loss=custom_loss_function, optimizer=SGD(lr=0.1, momentum = 0.9))

    return model


def model_z():
    inputs = Input(shape=(1, 10, 1))

    conv1 = conv_layer(inputs)

    r1 = residual_layer(conv1)
    r2 = residual_layer(r1)
    r3 = residual_layer(r2)
    r4 = residual_layer(r3)
    r5 = residual_layer(r4)
    r6 = residual_layer(r5)
    r7 = residual_layer(r6)
    r8 = residual_layer(r7)
    r9 = residual_layer(r8)
    r10 = residual_layer(r9)

    r11 = residual_layer(r10)
    r12 = residual_layer(r11)
    r13 = residual_layer(r12)
    r14 = residual_layer(r13)
    r15 = residual_layer(r14)
    r16 = residual_layer(r15)
    r17 = residual_layer(r16)
    r18 = residual_layer(r17)
    r19 = residual_layer(r18)
    r20 = residual_layer(r19)

    r21 = residual_layer(r20)
    r22 = residual_layer(r21)
    r23 = residual_layer(r22)
    r24 = residual_layer(r23)
    r25 = residual_layer(r24)
    r26 = residual_layer(r25)
    r27 = residual_layer(r26)
    r28 = residual_layer(r27)
    r29 = residual_layer(r28)
    r30 = residual_layer(r29)


    r31 = residual_layer(r30)
    r32 = residual_layer(r31)
    r33 = residual_layer(r32)
    r34 = residual_layer(r33)
    r35 = residual_layer(r34)
    r36 = residual_layer(r35)
    r37 = residual_layer(r36)
    r38 = residual_layer(r37)
    r39 = residual_layer(r38)
    r40 = residual_layer(r39)

    output = value_head(r40)



    model = Model(inputs=[inputs], outputs=[output])

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.1, momentum = 0.9))

    return model



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



		best.append(z.child[i].n/float(100))






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



	print ("Start State  \n")



	maze = print_maze(n)

	print (maze)


	while conclusion(n, 1, 2, 0) == -1:



		if p==1:



			print ("Monte Carlo 1's Move")


			a = np.zeros(len(n.state)+1)

			a[:9] = n.state

			a[9] = 1

			states.append(a)







			n,probs,p_val,best = mcst(n,101,1,2,current_move_p,pi_model,z_model,p_val)

			free.append(best)

			#print "free total : ",free

			#print "p_val : ",p_val
			#print "p_child : ",len(p_child)




			if check > 0:

				#print "p_val : ",p_val




				pi = p_val[-2][:]


				#print "pi : ",pi

				#print np.shape(pi),best

				#print "free : ",free[-2]



				v = -1

				for x in range(len(pi)):

					if pi[x]!=0:

						v+=1

					if v==free[-2]:

						pi[x] = 0




				index = 0

				#print "p_val : ",p_val

				#print pi,len(p_child[-1])

				temp = []

				for i in range(int(dim)*int(dim)):

					if pi[i]!=0:

						temp.append(p_val[-1][index])

						index+=1

					else:

						temp.append(0)




				new_p = []

				for z in range(len(p_val)-1):

					new_p.append(p_val[z])

				new_p.append(temp)

				p_val = new_p

				#print "p_val : ",p_val

				#print "pi : ",pi







			check+=1

			p_child.append(n.child)

			#print "len : ",np.shape(p_val)

			#for i in range(len(n.child)):

			#	print "asd : ", probs



			print ("After Monte Carlo 1's Move ")

			#print "check : ", current_move_p

			maze = print_maze(n)

			print (maze)



			p=0


		elif p==0:



			print ("Monte Carlo 2's Move")


			a = np.zeros(len(n.state)+1)

			a[:9] = n.state

			a[9] = 2

			states.append(a)











			n,probs,p_val,best = mcst(n,101,2,1,current_move_p,pi_model,z_model,p_val)

			free.append(best)

			#print "free total : ",free

			#print "p_val : ",p_val
			#print "p_child : ",len(p_child)




			if check > 0:

				#print "p_val : ",p_val




				pi = p_val[-2][:]


				#print "pi : ",pi

				#print np.shape(pi),best

				#print "free : ",free[-2]



				v = -1

				for x in range(len(pi)):

					if pi[x]!=0:

						v+=1

					if v==free[-2]:

						pi[x] = 0




				index = 0

				#print "p_val : ",p_val

				#print pi,len(p_child[-1])

				temp = []

				for i in range(int(dim)*int(dim)):

					if pi[i]!=0:

						temp.append(p_val[-1][index])

						index+=1

					else:

						temp.append(0)




				new_p = []

				for z in range(len(p_val)-1):

					new_p.append(p_val[z])

				new_p.append(temp)

				p_val = new_p

				#print "p_val : ",p_val

				#print "pi : ",pi







			check+=1

			p_child.append(n.child)

			#print "len : ",np.shape(p_val)



			#for i in range(len(n.child)):

			#	print n.child[i].p,n.child[i].v


			print ("After Monte Carlo 2's Move ")

			#print "check : ", current_move_p

			maze = print_maze(n)

			print (maze)


			p=1

	if conclusion(n, 1, 2, 0) == 1:

		print ("Monte Carlo 1 Wins ")

		for i in range(len(states)):

			if states[i][9]==1:

				vals.append(1)
			else:

				vals.append(-1)

	elif conclusion(n, 1, 2, 0) == 2:

		print ("Monte Carlo 2 Wins")

		for i in range(len(states)):

			if states[i][9]==2:

				vals.append(1)
			else:

				vals.append(-1)

	else:

		print ("Game Drawn ")

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




	if  os.path.isfile("best_z_model_ann.h5"):

		z_model = load_model('best_z_model_ann.h5')
	else:
		z_model = model_z()




	if  os.path.isfile("best_pi_model_ann.h5"):
		pi_model = load_model('best_pi_model_ann.h5', custom_objects={'custom_loss_function': softmax_cross_entropy_with_logits})
	else:
		pi_model = model_pi()




	turn = 0

	for ep in range(3000):

		print ("Episode : ",str(ep+1))

		states,_,vals,p = episode(3,turn,pi_model,z_model)

		#print "ppppp : ",p, np.shape(p)

		if turn == 0:

			turn = 1

		else:

			turn = 0

		if  os.path.isfile("z_train_data_ann.npy"):

			train_data = np.load("z_train_data_ann.npy")
			train_label = np.load("z_train_label_ann.npy")



			temp_x = np.zeros((np.shape(train_data)[0]+len(vals),1,10,1))
			temp_y = np.zeros((np.shape(train_data)[0]+len(vals),1))

			#print "shape : ",np.shape(train_data)[0]

			for i in range(np.shape(train_data)[0]):

				temp_x[i,0] = train_data[i]
				temp_y[i] = train_label[i]

			#print "iii : ",i



			k = 0

			for j in range(i+1,i+len(vals)+1):


				temp_x[j,0,:,0] = states[k]
				temp_y[j] = vals[k]

				k+=1

			#print "tx : ",temp_x
			#print "ty : ",temp_y

			np.save("z_train_data_ann.npy", temp_x)
			np.save("z_train_label_ann.npy", temp_y)


		else:

			temp_x = np.zeros((len(vals),1,10,1))
			temp_y = np.zeros((len(vals),1))

			for i in range(len(vals)):



				temp_x[i,0,:,0] = states[i]
				temp_y[i] = vals[i]




			np.save("z_train_data_ann.npy", temp_x)
			np.save("z_train_label_ann.npy", temp_y)

		if  os.path.isfile("pi_train_data_ann.npy"):

			train_data = np.load("pi_train_data_ann.npy")
			train_label = np.load("pi_train_label_ann.npy")



			temp_x = np.zeros((np.shape(train_data)[0]+len(vals),1,10,1))
			temp_y = np.zeros((np.shape(train_data)[0]+len(vals),9))

			for i in range(np.shape(train_data)[0]):

				temp_x[i,0] = train_data[i]
				temp_y[i] = train_label[i]

			k = 0

			for j in range(i+1,i+len(vals)+1):


				temp_x[j,0,:,0] = states[k]
				temp_y[j] = p[k]

				k+=1

			#print "tx : ",temp_x
			#print "ty : ",temp_y

			np.save("pi_train_data_ann.npy", temp_x)
			np.save("pi_train_label_ann.npy", temp_y)


		else:

			temp_x = np.zeros((len(vals),1,10,1))
			temp_y = np.zeros((len(vals),9))

			for i in range(len(vals)):



				temp_x[i,0,:,0] = states[i]
				temp_y[i] = p[i]




			np.save("pi_train_data_ann.npy", temp_x)
			np.save("pi_train_label_ann.npy", temp_y)



	#print states[:3]

	#print probs[:3]

	#print vals[:3]


if __name__ == "__main__":
    main()
