import tensorflow as tf
from tensorflow import initializers
from utils     import Options, rgb2gray
import numpy as np
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable
import matplotlib.pyplot as plt

#    tf.add_to_collection("Q", Q)

def human_printable(inp_img):
    ret = np.zeros((3,3))
    ret[0,0] = inp_img [5,5]
    ret[0,1] = inp_img [5,15]
    ret[0,2] = inp_img [5,25]
    ret[1,0] = inp_img [15,5]
    ret[1,1] = inp_img [15,15]
    ret[1,2] = inp_img [15,25]
    ret[2,0] = inp_img [25,5]
    ret[2,1] = inp_img [25,15]
    ret[2,2] = inp_img [25,25]
    return ret


def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs
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

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("./weights/checkpoint50000.ckpt.meta")

win_all = None
win_pob = None
'''
TODO: Try to use Placeholder as input for x and fully_connected_2/Relu as output 
'''



def_graph = tf.get_default_graph()

with tf.Session() as sess:
    x = sess.graph.get_tensor_by_name('x:0')
    Q = tf.get_collection("Q")[0]

    imported_meta.restore(sess, tf.train.latest_checkpoint('./weights/'))

    maxlen = 100000

    # initialize the environment
    state = sim.newGame(opt.tgt_y, opt.tgt_x)
    state_with_history = np.zeros((opt.hist_len, opt.state_siz))
    append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
    next_state_with_history = np.copy(state_with_history)
    trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)
    epi_step = 0


    # train for <steps> steps
    for step in range(100 + 1):

        # goal check
        if state.terminal or epi_step >= opt.early_stop:
            epi_step = 0
            max_step = step


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
        #print(input_reshaped[0][0])
        print(input_reshaped.shape)
        print(human_printable(input_reshaped[0,:,:,3]))

        ### take one action per step
        qvalues = sess.run(Q,feed_dict={x: input_batched})[0]  # take the first batch entry
        action = np.argmax(qvalues)
        print(action)
        action_onehot = trans.one_hot_action(action)
        print(action_onehot)
        # apply action
        next_state = sim.step(action)
        print(next_state.reward)
        # append to history
        append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
        # add to the transition table
        trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward,
                  next_state.terminal)
        # mark next state as current state
        state_with_history = np.copy(next_state_with_history)
        state = next_state

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
