import tensorflow as tf
from tensorflow import initializers
from utils     import Options, rgb2gray
import numpy as np
from utils     import Options, rgb2gray
from simulator_deterministic_start import SimulatorDeterministicStart
from transitionTable import TransitionTable
import matplotlib.pyplot as plt

class AgentTester():

    def append_to_hist(self, state, obs):
        """
        Add observation to the state.
        """
        for i in range(state.shape[0]-1):
            state[i, :] = state[i+1, :]
        state[-1, :] = obs
    # reformat data for network input
    def reshapeInputData(self, input_batch, no_batches):
        input_batch = input_batch.reshape((no_batches, self.opt.hist_len, self.opt.pob_siz * self.opt.cub_siz, self.opt.pob_siz * self.opt.cub_siz))
        # reformat input data if convolutions are used (consistent with visual map)
        input_batch = np.rot90(input_batch, axes=(1, 2))
        input_batch = np.rot90(input_batch, axes=(2, 3))
        # rotate mapview 180 degree
        input_batch = np.rot90(input_batch, axes=(1, 2))
        input_batch = np.rot90(input_batch, axes=(1, 2))

        # input_batch = input_batch.reshape((no_batches, opt.hist_len * opt.pob_siz * opt.cub_siz * opt.pob_siz * opt.cub_siz))
        return input_batch

    def start_eval(self, speicherort, display):
        # 0. initialization
        self.opt = Options()
        sim = SimulatorDeterministicStart(self.opt.map_ind, self.opt.cub_siz, self.opt.pob_siz, self.opt.act_num)
        imported_meta = tf.train.import_meta_graph(speicherort)

        win_all = None
        win_pob = None

        def_graph = tf.get_default_graph()

        with tf.Session() as sess:
            with tf.variable_scope("new_testing_scope", reuse=tf.AUTO_REUSE):

                x = sess.graph.get_tensor_by_name('x:0')
                Q = tf.get_collection("Q")[0]

                imported_meta.restore(sess, tf.train.latest_checkpoint('./weights/'))

                maxlen = 100000

                # initialize the environment
                state = sim.newGame(self.opt.tgt_y, self.opt.tgt_x, 0)
                state_with_history = np.zeros((self.opt.hist_len, self.opt.state_siz))
                self.append_to_hist(state_with_history, rgb2gray(state.pob).reshape(self.opt.state_siz))
                next_state_with_history = np.copy(state_with_history)
                trans = TransitionTable(self.opt.state_siz, self.opt.act_num, self.opt.hist_len,
                                    self.opt.minibatch_size, maxlen)
                epi_step = 0

                episodes = 0

                solved_epoisodes = 0

                step_sum = 0
                # train for <steps> steps
                while True:

                    # goal check
                    if state.terminal or epi_step >= self.opt.early_stop:
                        if state.terminal:
                            solved_epoisodes += 1
                        episodes += 1
                        step_sum = step_sum + epi_step
                        epi_step = 0

                        # reset the game
                        try:
                            state = sim.newGame(self.opt.tgt_y, self.opt.tgt_x, episodes)
                        except:
                            return (step_sum,solved_epoisodes)

                        # and reset the history
                        state_with_history[:] = 0
                        self.append_to_hist(state_with_history, rgb2gray(state.pob).reshape(self.opt.state_siz))
                        next_state_with_history = np.copy(state_with_history)

                        if display:
                            if win_all is None:
                                plt.subplot(121)
                                win_all = plt.imshow(state.screen)
                                plt.subplot(122)
                                win_pob = plt.imshow(state.pob)
                            else:
                                win_all.set_data(state.screen)
                                win_pob.set_data(state.pob)
                            plt.pause(self.opt.disp_interval)
                            plt.draw()

                    epi_step += 1

                    # format state for network input
                    input_reshaped = self.reshapeInputData(state_with_history, 1)
                    # create batch of input state
                    input_batched = np.tile(input_reshaped, (self.opt.minibatch_size, 1, 1, 1))

                    ### take one action per step
                    qvalues = sess.run(Q,feed_dict={x: input_batched})[0]  # take the first batch entry
                    action = np.argmax(qvalues)
                    action_onehot = trans.one_hot_action(action)
                    # apply action
                    next_state = sim.step(action)
                    # append to history
                    self.append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(self.opt.state_siz))
                    # add to the transition table
                    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward,
                              next_state.terminal)
                    # mark next state as current state
                    state_with_history = np.copy(next_state_with_history)
                    state = next_state

                    if display:
                        if win_all is None:
                            plt.subplot(121)
                            win_all = plt.imshow(state.screen)
                            plt.subplot(122)
                            win_pob = plt.imshow(state.pob)
                        else:
                            win_all.set_data(state.screen)
                            win_pob.set_data(state.pob)
                        plt.pause(self.opt.disp_interval)
                        plt.draw()

