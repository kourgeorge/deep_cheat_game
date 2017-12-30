__author__ = 'gkour'
import numpy as np


class PlayerTrainer:
    def __init__(self, agent_network, sess):
        self._agent_network = agent_network
        self._sess = sess
        self._gradBuffer_act = {}

        self._gradBuffer_act = sess.run(self._agent_network.action_model_trainable_vars())
        for ix, grad in enumerate(self._gradBuffer_act):
            self._gradBuffer_act[ix] = grad * 0

    def accumulate_action_gradients(self, episode_states, episode_actions, episode_rewards):
        feed_dict = {self._agent_network.reward_holder: episode_rewards,
                     self._agent_network.action_holder: episode_actions,
                     self._agent_network.state_in: np.vstack(episode_states)}

        grads_act = self._sess.run(self._agent_network.gradients_act, feed_dict=feed_dict)
        for idx, grad in enumerate(grads_act):
            self._gradBuffer_act[idx] += grad

    def update_action_model(self):
        feed_dict = dict(zip(self._agent_network.gradient_holders_act, self._gradBuffer_act))
        _ = self._sess.run(self._agent_network.update_batch_act, feed_dict=feed_dict)
        for ix, grad in enumerate(self._gradBuffer_act):
            self._gradBuffer_act[ix] = grad * 0
