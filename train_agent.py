import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import tensorflow as tf
from tensorflow import initializers
import time

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable


### HYPERPARAMETERS

# E-greedy exploration [0 = no exploration, 1 = strict exploration]
epsilon = 1
epsilon_min = 0.1
epsilon_decay = 0.999995

# learning rate of the optimizer, here adaptive
learning_rate = 0.0005

# learning rate of the Q function [0 = no learning, 1 = only consider rewards]
# alpha = 0.5

# Q learning discount factor [0 = only weight current state, 1 = weight future reward only]
gamma = 0.8

training_start = 200  # total number of steps after which network training starts
training_interval = 5    # number of steps between subsequent training steps

save_interval = 5*10**4
print_interval = 500

print_goals = False


### HELPTER FUNCTIONS

# export state with history to file for debugging
def saveStateAsTxt(state_array):
    state_array[state_array > 200] = 4
    state_array[state_array > 100] = 3
    state_array[state_array >  50] = 2
    state_array[state_array >  10] = 1

    state_array = reshapeInputData(state_array, opt.minibatch_size)
    
    # append history, most recent state is last
    string = ''
    for i in range(opt.hist_len ):

        # consistent with visualization
        string += str(np.array(state_array[0,:,:,i], dtype=np.uint8)) + '\n\n'

    with open('test.txt', 'w') as textfile:
        print(string, file=textfile)

# color highlighting for the console
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# reformat data for network input
def reshapeInputData(input_batch, no_batches):
    input_batch = input_batch.reshape((no_batches, opt.hist_len, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz))
    # reformat input data if convolutions are used (consistent with visual map)
    input_batch = np.rot90(input_batch, axes=(1, 2))
    input_batch = np.rot90(input_batch, axes=(2, 3))
    # rotate mapview 180 degree
    input_batch = np.rot90(input_batch, axes=(1, 2))
    input_batch = np.rot90(input_batch, axes=(1, 2))

    # input_batch = input_batch.reshape((no_batches, opt.hist_len * opt.pob_siz * opt.cub_siz * opt.pob_siz * opt.cub_siz))
    return input_batch

# get one-hot encoding for next_action_batch
def prepareNextActionBatch(input):
    # value = np.argmax(input, axis=0)
    # print('value\n', value)
    return trans.one_hot_action(input)[0,:]


### the loss function
def Q_loss(Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=gamma):
    """
    All inputs should be tensorflow variables!
    We use the following notation:
       N : minibatch size
       A : number of actions
    Required inputs:
       Q_s: a NxA matrix containing the Q values for each action in the sampled states.
            This should be the output of your neural network.
            We assume that the network implments a function from the state and outputs the 
            Q value for each action, each output thus is Q(s,a) for one action 
            (this is easier to implement than adding the action as an additional input to your network)
       action_onehot: a NxA matrix with the one_hot encoded action that was selected in the state
                      (e.g. each row contains only one 1)
       Q_s_next: a NxA matrix containing the Q values for the next states.
       best_action_next: a NxA matrix with the best current action for the next state
       reward: a Nx1 matrix containing the reward for the transition
       terminal: a Nx1 matrix indicating whether the next state was a terminal state
       discount: the discount factor
    """
    # calculate: reward + discount * Q(s', a*),
    # where a* = arg max_a Q(s', a) is the best action for s' (the next state)
    target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
    # NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
    #       use it as the target for Q_s
    target_q = tf.stop_gradient(target_q)
    # calculate: Q(s, a) where a is simply the action taken to get from s to s'
    selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
    loss = tf.reduce_sum(tf.square(selected_q - target_q))    
    return loss

# add observation to the state
def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs


### TRAINING

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None



# setup placeholders for states (x) actions (u) and rewards and terminal values
x = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, opt.hist_len))
u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
xn = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, opt.hist_len))
r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))

### the network definition
with tf.variable_scope('DQN', reuse=tf.AUTO_REUSE):
    def network(x):
        conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[5, 5], padding='same', strides=2, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[5, 5], padding='same', strides=3, activation=tf.nn.relu)
        
        conv2_flat = tf.layers.flatten(conv2)
        dropout1 = tf.layers.dropout(inputs=conv2_flat, rate=0.3)
        
        fcon1 = tf.contrib.layers.fully_connected(dropout1, 128, tf.nn.relu, weights_initializer=initializers.random_normal, biases_initializer=tf.zeros_initializer)
        fcon2= tf.contrib.layers.fully_connected(fcon1, 256, tf.nn.relu)
        
        dropout2 = tf.layers.dropout(inputs=fcon2, rate=0.5)
        output_layer = tf.contrib.layers.fully_connected(dropout2, opt.act_num, tf.nn.relu)
        return output_layer


# get the output from your network

### TRAINING ROUTINE
with tf.Session() as sess:
    # declare the networks outputs symbolically
    Q = network(x)
    Qn = network(xn)

    # calculate the loss
    loss = Q_loss(Q, u, Qn, ustar, r, term)

    # sess = tf.Session()
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    #train_ops = tf.contrib.layers.optimize(loss, tf.train.get_global_step(),optimizer=tf.train.AdagradOptimizer, learning_rate = 0.01)


    # setup an optimizer in tensorflow to minimize the loss
    train_ops = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)

    # initialize all the variables, call after setting up the optimizer
    sess.run(tf.global_variables_initializer())

    # prepare to save the network weights
    saver = tf.train.Saver()

    # run for some steps
    steps = 1 * 10**5 * 5
    epi_step = 0
    nepisodes = 0

    # some statistics
    loss_value = 0
    performance_stats = []
    network_stats = []
    max_step = 0
    max_last = 0


    # initialize the environment
    state = sim.newGame(opt.tgt_y, opt.tgt_x)
    state_with_history = np.zeros((opt.hist_len, opt.state_siz))
    append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
    next_state_with_history = np.copy(state_with_history)

    # train for <steps> steps
    for step in range(steps+1):
        
        # goal check
        if state.terminal or epi_step >= opt.early_stop:
            epi_step = 0
            max_step = step
            nepisodes += 1

            if print_goals: print('episode: {:>4} | step:{:>7} | steps: {:>4} | epsilon {:>1.6f}'.format(nepisodes, step, max_step-max_last, epsilon))
            performance_stats.append(np.array([nepisodes, step, max_step-max_last, epsilon]))
            max_last = max_step

            # reset the game
            state = sim.newGame(opt.tgt_y, opt.tgt_x)
            # and reset the history
            state_with_history[:] = 0
            append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
            next_state_with_history = np.copy(state_with_history)
        
        epi_step += 1
        
        # format state for network input
        input_reshaped = reshapeInputData(state_with_history, 1)
        # create batch of input state
        input_batched = np.tile(input_reshaped, (opt.minibatch_size, 1, 1, 1))
        
        random = True  # for stats
        
        ### take one action per step
        if np.random.rand() <= epsilon:
            action = randrange(opt.act_num)
        else:
            qvalues = sess.run([Q], feed_dict={x: input_batched})[0]  # take the first batch entry
            action = np.argmax(qvalues)
            random = False

        action_onehot = trans.one_hot_action(action)
        # apply action
        next_state = sim.step(action)
        # append to history
        append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
        # add to the transition table
        trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
        # mark next state as current state
        state_with_history = np.copy(next_state_with_history)
        state = next_state
        
        
        ### OPTIMIZE NETWORK here
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()

        state_batch = reshapeInputData(state_batch, opt.minibatch_size)
        next_state_batch = reshapeInputData(next_state_batch, opt.minibatch_size)
        
        #qvalues = sess.run([Q], feed_dict={x: reshapeInputData(state_batch, opt.minibatch_size)})
        qnvalues = sess.run([Qn], feed_dict={xn: next_state_batch})
        # print('qn:\n', qnvalues[0])
        next_action_index = np.argmax(qnvalues[0], axis=1).reshape(32,1)
        # print('nai\n', next_action_index)
        # get the next best action, one-hot encoded
        action_batch_next = np.apply_along_axis(prepareNextActionBatch, 1, next_action_index)
        # print('abn\n', action_batch_next)

        # this calls the optimizer defined in train_ops once, which minimizes the loss function Q_loss by calculating [Q, Qn] with the data provided below
        sess.run(train_ops, feed_dict = {x : state_batch, u : action_batch, ustar : action_batch_next, xn : next_state_batch, r : reward_batch, term : terminal_batch})

        # calculate the loss after the training epoch
        lossVal = sess.run(loss, feed_dict = {x : state_batch, u : action_batch, ustar : action_batch_next, xn : next_state_batch, r : reward_batch, term : terminal_batch})
        
        # update epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # output
        if(step % print_interval == 0):
            network_stats.append(np.array([lossVal, random, step, epsilon]))
            if random:
                print('Loss: {:>15.3f} | random action : {:d} | Steps: {:>8d} | Epsilon: {:<1.6f}'.format(lossVal, random, step, epsilon))
            else:
                print('Loss: {:>15.3f} | random action : {:d} | Steps: {:>8d} | Epsilon: {:<1.6f}'.format(lossVal, random, step, epsilon))

        # save the network weights & stats
        if (step % save_interval == 0 and step > 0):
            i = str(round(step))
            filename = './stats/network_stats' + i + '.txt'
            np.savetxt(filename, np.array(network_stats), delimiter=',')
            filename = './stats/performance_stats' + i + '.txt'
            np.savetxt(filename, np.array(performance_stats), delimiter=',')
            saver.save(sess, './weights/checkpoint' + i + '.ckpt')
            print('> stats/weights saved')

        # plot
        if opt.disp_on:
            if win_all is None:
                plt.subplot(121)
                win_all = plt.imshow(state.screen)
                plt.subplot(122)
                win_pob = plt.imshow(state.pob)
            else:
                win_all.set_data(state.screen)
                win_pob.set_data(state.pob)
            plt.pause(opt.disp_interval)
            plt.draw()

    # final save
    filename = 'network_stats_final' + '.txt'
    np.savetxt(filename, np.array(network_stats), delimiter=',')
    filename = 'performance_stats_final' + '.txt'
    np.savetxt(filename, np.array(performance_stats), delimiter=',')
    saver.save(sess,"./weights/weights_final.ckpt")
    print('> final stats/weights saved')

# 2. perform a final test of your model and save it
# TODO

