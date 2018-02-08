import numpy as np
import matplotlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import initializers
import time
# custom modules
from utils import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable

# sample some noise
def sample_noise(shape):
    noise = tf.random_normal(shape)
    return noise

def noisy_dense(x, size, name, bias=True, activation_fn=tf.identity):
    # noise function
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))

    # initialize mu and sigma as in the paper
    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),
                                                maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))
    sigma_init = tf.constant_initializer(0.4/np.power(x.get_shape().as_list()[1], 0.5))

    # make epsilon a callable variable
    p = sample_noise([x.get_shape().as_list()[1], 1])
    p_var = tf.get_variable(trainable=False, name=name+"p", initializer=p)

    q = sample_noise([1, size])
    q_var = tf.get_variable(initializer=q,trainable=False, name=name+"q")

    # apply factorized noise
    f_p = f(p_var)
    f_q = f(q_var)
    w_epsilon = f_p*f_q
    b_epsilon = tf.squeeze(f_q)

    # w = w_mu + w_sigma*w_epsilon
    w_mu = tf.get_variable(name + "/wmu", [x.get_shape()[1], size], initializer=mu_init)
    w_sigma = tf.get_variable(name + "/wsigma", [x.get_shape()[1], size], initializer=sigma_init)
    w = w_mu + tf.multiply(w_sigma, w_epsilon)
    ret = tf.matmul(x, w)
    if bias:
        # b = b_mu + b_sigma*b_epsilon
        b_mu = tf.get_variable(name + "/bmu", [size], initializer=mu_init)
        b_sigma = tf.get_variable(name + "/bsigma", [size], initializer=sigma_init)
        b = b_mu + tf.multiply(b_sigma, b_epsilon)
        return activation_fn(ret + b)
    else:
        return activation_fn(ret)


### HYPERPARAMETERS

# activate target network
use_target_net = False

# frequency of target weights update
tau = 1000

# E-greedy exploration [0 = no exploration, 1 = strict exploration]  / epsilon currently fixed below!
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
training_interval = 5  # number of steps between subsequent training steps

save_interval = 5 * 10 ** 4
print_interval = 500

print_goals = False


### HELPER FUNCTIONS

# export state with history to file for debugging
def saveStateAsTxt(state_array):
    state_array[state_array > 200] = 4
    state_array[state_array > 100] = 3
    state_array[state_array > 50] = 2
    state_array[state_array > 10] = 1

    state_array = reshapeInputData(state_array, opt.minibatch_size)

    # append history, most recent state is last
    string = ''
    for i in range(opt.hist_len):
        # consistent with visualization
        string += str(np.array(state_array[0, :, :, i], dtype=np.uint8)) + '\n\n'

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
    return trans.one_hot_action(input)[0, :]


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
    # target_q = tf.Print(target_q, [target_q],message="target q")

    # calculate: Q(s, a) where a is simply the action taken to get from s to s'
    selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
    # action_onehot = tf.Print(action_onehot, [action_onehot],message="action_onehot")
    # selected_q = tf.Print(selected_q, [selected_q],message="selected_q")

    # print("Selected q",sess.run(selected_q))
    loss = tf.reduce_sum(tf.square(selected_q - target_q))
    return loss


# add observation to the state
def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0] - 1):
        state[i, :] = state[i + 1, :]
    state[-1, :] = obs

# generate ops to copy the weight variables from src to target scope (network)
def get_weight_copy_ops(src, target):
    # get the relevant variables first
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src)
    # print('src\n', src_vars)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target)
    # print('dest\n', dest_vars)

    # store all copy operations here
    ops = []
    for src_var, dest_var in zip(src_vars, dest_vars):
        ops.append(dest_var.assign(src_var.value()))
    return ops


### the network definition
# define networks and scope
def network(inputs, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        return network_structure(inputs)

def shuffle_noise(scope):
    #print(tf.get_collection(tf.GraphKeys.VARIABLES, scope=trainNet_scope))
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        np_op = tf.get_variable(name='noisy1p', trainable=False).assign(sample_noise(tf.get_variable('noisy1p').get_shape()))
        nq_op = tf.get_variable(name='noisy1q', trainable=False).assign(sample_noise(tf.get_variable('noisy1q').get_shape()))

        np_op2 = tf.get_variable(name='noisy2p', trainable=False).assign(
            sample_noise(tf.get_variable('noisy2p').get_shape()))
        nq_op2 = tf.get_variable(name='noisy2q', trainable=False).assign(
            sample_noise(tf.get_variable('noisy2q').get_shape()))
        sess.run([nq_op, np_op, np_op2, nq_op2])

        # print(sess.run(tf.get_variable('noisy2q'))[0,0])


# define the network structure
def network_structure(x):
    # input_layer = tf.reshape(x, [-1, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, opt.hist_len])
    conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[5, 5], padding='valid',
                             kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01), strides=2,
                             activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[5, 5], padding='valid',
                             kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01), strides=2,
                             activation=tf.nn.relu)

    conv2_flat = tf.layers.flatten(conv2)

    fcon1 = tf.contrib.layers.fully_connected(conv2_flat, 128, tf.nn.relu,
                                              weights_initializer=initializers.random_normal(mean=0.0, stddev=0.01),
                                              biases_initializer=tf.zeros_initializer)
    dropout1 = tf.layers.dropout(inputs=fcon1, rate=0.3)

    # fcon2= tf.contrib.layers.fully_connected(fcon1, 256, tf.nn.relu)

    # dropout2 = tf.layers.dropout(inputs=fcon2, rate=0.5)
    output_layer = tf.contrib.layers.fully_connected(dropout1, opt.act_num,
                                                     weights_initializer=initializers.random_normal(mean=0.0,
                                                                                                    stddev=0.01),
                                                     activation_fn=None)
    return output_layer

def actNoisy(input_batched):
    # shuffle_noise(trainNet_scope)

    param_noise_scale = tf.get_variable("param_noise_scale", (), initializer=tf.constant_initializer(0.01), trainable=False)
    param_noise_threshold = tf.get_variable("param_noise_threshold", (), initializer=tf.constant_initializer(0.05), trainable=False)
    
    # compute the q values for the different nets first
    [qvalues, qvalues_noisy, qvalues_adaptive] = sess.run([Q, Q_noisy, Q_adaptive], feed_dict={x: input_batched})[0]
    # qvalues_noisy = sess.run([Q_noisy], feed_dict={x: input_batched})[0]
    # qvalues_adaptive = sess.run([Q_adaptive], feed_dict={x: input_batched})[0]
    
    # perturb variables of noisyNet
    perturb_vars(trainNet_scope, noisyNet_scope)

    # calculate Kullback-Leibler Divergenz 
    kl = tf.reduce_sum(tf.nn.softmax(qvalues) * (tf.log(tf.nn.softmax(qvalues)) - tf.log(tf.nn.softmax(qvalues_adaptive))), axis=-1)  # axis right? just copied (should be however)
    mean_kl = tf.reduce_mean(kl)

    # update the noise scale -> stddev
    if mean_kl < param_noise_threshold:
        param_noise_scale.assign(param_noise_scale * 1.01)
    else:
        param_noise_scale.assign(param_noise_scale / 1.01)

    # bis hierher gekommen: https://github.com/openai/baselines/blob/master/baselines/deepq/build_graph.py#L256
    
    
    action = None
    # keep epsilon of 0.01 / maybe delete
    if np.random.rand() <= epsilon:
        action = randrange(opt.act_num)
    else:
        action = np.argmax(qvalues_noisy)
    
    return action

def perturb_vars(original_scope, perturbed_scope):
    # grep all variables in the fully_connected layers
    noisyLayerSelector = 'fully_connected'

    all_training_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=original_scope)
    all_perturbable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=perturbed_scope)

    # !!! TODO !!! this function should return all those variables from all_perturbable_vars which contain noisyLayerSelector (also 'fully_connected') -> alle vars der fully connected layers
    vars_to_be_perturbed = # to implement

    assert len(all_training_vars) == len(all_perturbable_vars)  # this basically just checks if the networks are of the same structure
    # collect all ops to update the weights with noise
    perturb_ops = []
    for var, perturbed_var in zip(all_training_vars, all_perturbable_vars):
        if (perturbed_var in vars_to_be_perturbed):
            # Perturb this variable
            op = tf.assign(perturbed_var, var + tf.random_normal(shape=tf.shape(var), mean=0., stddev=param_noise_scale))
        else:
            # Do not perturb, just assign
            op = tf.assign(perturbed_var, var)
        perturb_ops.append(op)
    assert len(perturb_ops) == len(all_vars)
    
    # apply noise to the necessary variables here
    sess.run(perturb_ops)


### TRAINING

# initialization
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
x = tf.placeholder(tf.float32, shape=(None, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, opt.hist_len), name="x")
u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
xn = tf.placeholder(tf.float32,
                    shape=(opt.minibatch_size, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, opt.hist_len))
r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))


### TRAINING ROUTINE
with tf.Session() as sess:
    # define the network scopes
    trainNet_scope = 'trainDQN'
    targetNet_scope = 'targetDQN'
    noisyNet_scope = 'noisyDQN'
    adaptiveNet_scope = 'adaptiveDQN'

    # declare the networks outputs symbolically
    Q = network(x, trainNet_scope)
    Qn_target = network(xn, trainNet_scope)
    if use_target_net: Qn_target = network(xn, targetNet_scope)
    
    # add a couple of networks for NoisyNets
    Q_noisy = network(x, noisyNet_scope)
    Q_adaptive = network(x, adaptiveNet_scope)
    
    # calculate the loss using the target network
    loss = Q_loss(Q, u, Qn_target, ustar, r, term)

    # setup an optimizer in tensorflow to minimize the loss
    training_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, trainNet_scope)
    train_ops = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss, var_list=training_variables)

    # initialize all the variables, call after setting up the optimizer
    sess.run(tf.global_variables_initializer())

    # get the copy operations to update the target network weights
    copy_ops = []
    if use_target_net: copy_ops = get_weight_copy_ops(trainNet_scope, targetNet_scope)

    # prepare to save the network weights
    saver = tf.train.Saver()

    # run for some steps
    steps = 1 * 10 ** 5 * 5
    epi_step = 0
    nepisodes = 0
    solved_episodes = 0

    # some statistics
    loss_value = 0
    performance_stats = []
    network_stats = []
    max_step = 0
    max_last = 0
    epsilon = 0.01

    # initialize the environment
    state = sim.newGame(opt.tgt_y, opt.tgt_x)
    state_with_history = np.zeros((opt.hist_len, opt.state_siz))
    append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
    next_state_with_history = np.copy(state_with_history)

    # train for <steps> steps
    for step in range(steps + 1):

        # goal check
        if state.terminal or epi_step >= opt.early_stop:

            max_step = step
            nepisodes += 1
            if epi_step < opt.early_stop:
                solved_episodes += 1

            if print_goals: print(
                'episode: {:>4} | step:{:>7} | steps: {:>4} | epsilon {:>1.6f}'.format(nepisodes, step,
                                                                                       max_step - max_last, epsilon))
            performance_stats.append(np.array([nepisodes, solved_episodes, epi_step, epsilon]))
            max_last = max_step
            print('Episodes:\t{:>6} | Solved:\t{:>4} | Steps:\t{:>7}'.format(nepisodes, solved_episodes, epi_step))
            epi_step = 0
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

        action = actNoisy(input_batched)
        '''
        # TODO: Hier wird die Noise auf dem Noisy Layer neu gesamplet
        shuffle_noise(trainNet_scope)

        qvalues = sess.run([Q], feed_dict={x: input_batched})[0]  # take the first batch entry
        action = np.argmax(qvalues)
        '''

        random = False
        action_onehot = trans.one_hot_action(action)
        # apply action
        next_state = sim.step(action)
        # append to history
        append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
        # add to the transition table
        trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward,
                  next_state.terminal)
        # mark next state as current state
        state_with_history = np.copy(next_state_with_history)
        state = next_state

        # refresh the target network weights every <tau> steps
        if use_target_net and (step % tau) == 0:
            sess.run(copy_ops)
            print('> weights updated from [{}] to [{}]'.format(trainNet_scope, targetNet_scope))


        ### OPTIMIZE TRAINING NETWORK here
        if step >= training_start:
            state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()

            state_batch = reshapeInputData(state_batch, opt.minibatch_size)
            next_state_batch = reshapeInputData(next_state_batch, opt.minibatch_size)

            qvalues = sess.run([Q], feed_dict={x: state_batch})
            # qnvalues = sess.run([Qn], feed_dict={xn: next_state_batch})
            # sample qn values using the target network
            qnvalues = sess.run([Qn_target], feed_dict={xn: next_state_batch})
            # print('q:\n', qvalues[0])
            # print('qn:\n', qnvalues[0])
            next_action_index = np.argmax(qnvalues[0], axis=1).reshape(opt.minibatch_size, 1)
            # get the next best action, one-hot encoded
            action_batch_next = np.apply_along_axis(prepareNextActionBatch, 1, next_action_index)

            # this calls the optimizer defined in train_ops once, which minimizes the loss function Q_loss by calculating [Q, Qn] with the data provided below
            sess.run(train_ops,
                     feed_dict={x: state_batch, u: action_batch, ustar: action_batch_next, xn: next_state_batch,
                                r: reward_batch, term: terminal_batch})

            # calculate the loss after the training epoch
            lossVal = sess.run(loss, feed_dict={x: state_batch, u: action_batch, ustar: action_batch_next,
                                            xn: next_state_batch, r: reward_batch, term: terminal_batch})

            if (step % print_interval == 0):
                network_stats.append(np.array([step, lossVal, epsilon]))
                print('> Training step: {:>7} \t Loss: {:>5.3f} \t Epsilon: {:<1.1f}'.format(step, lossVal, epsilon))

            # save the network weights & stats
            if (step % save_interval == 0 and step > 0):
                i = str(round(step))
                tf.add_to_collection('Q', Q)
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
    tf.add_to_collection('Q', Q)
    filename = 'network_stats_final' + '.txt'
    np.savetxt(filename, np.array(network_stats), delimiter=',')
    filename = 'performance_stats_final' + '.txt'
    np.savetxt(filename, np.array(performance_stats), delimiter=',')
    saver.save(sess, "./weights/weights_final.ckpt")
    print('> final stats/weights saved')

