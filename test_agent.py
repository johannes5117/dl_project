# basic imports
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import time

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable


display_on = True

win_all = None
win_pob = None

opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

def reshapeInputData(input_batch, no_batches):
    return input_batch.reshape((no_batches, historyLength * opt.pob_siz * opt.cub_siz * opt.pob_siz * opt.cub_siz))

historyLength = opt.hist_len
num_classes = 5  # 0 = no action / 1 = up / 2 = down / 3 = left / 4 = right

dqn = load_model('qnet.h5')

test_episodes = 5

stats = np.zeros((opt.eval_nepisodes, 4))
step = 0
max_step = 0  # used for statistics
max_last = 0

for e in range(test_episodes):
    # reset the environment
    # reset the game
    state = sim.newGame(opt.tgt_y, opt.tgt_x)
    # and reset the history
    state_with_history[:] = 0
    append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
    next_state_with_history = np.copy(state_with_history)

    while True:
        ### goal check or max_step
        if state.terminal or (step - max_last) >= opt.early_stop:
            max_step = step
            break
        
        # increase the counter in each iteration globally
        step += 1

        # make a Qnet prediction here based on the current state s
        input_state = reshapeInputData(state_with_history, 1)
        q_actions = dqn.predict(input_state, verbose=0)
        # get the action which corresponds to the max Q value (action = argmax)
        action = np.argmax(q_actions)
        print('q actions:\t', q_actions)
        action_onehot = trans.one_hot_action(action)

        next_state = sim.step(action)

        # append to history
        append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
        # add to the transition table
        trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
        # mark next state as current state
        state_with_history = np.copy(next_state_with_history)
        state = next_state

        ### print map here
        if display_on:
            # plt.ion()
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
            time.sleep(0.2)


    steps = max_step - max_last
    max_last = max_step
    print('episode: {:>4}/{} | steps: {:>4}'.format(e, test_episodes, steps))
