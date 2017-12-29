__author__ = 'gkour'
import numpy as np


class PlayerTrainer:
    def __init__(self, agent, sess):
        self._agent = agent
        self._sess = sess
        self._gradBuffer_act = {}
        self._gradBuffer_arg = {}

        self._gradBuffer_act = sess.run(self._agent.action_model_trainable_vars())
        for ix, grad in enumerate(self._gradBuffer_act):
            self._gradBuffer_act[ix] = grad * 0

        self._gradBuffer_arg = sess.run(self._agent.argument_model_trainable_vars())
        for ix, grad in enumerate(self._gradBuffer_arg):
            self._gradBuffer_arg[ix] = grad * 0

    def accumulate_action_gradients(self, episode_states, episode_actions, episode_args, episode_rewards):
        feed_dict = {self._agent.reward_holder: episode_rewards,
                     self._agent.action_holder: episode_actions,
                     self._agent.argument_holder: episode_args,
                     self._agent.state_in: np.vstack(episode_states)}

        grads_act = self._sess.run(self._agent.gradients_act, feed_dict=feed_dict)
        for idx, grad in enumerate(grads_act):
            self._gradBuffer_act[idx] += grad


    def update_action_model(self):
        feed_dict = dict(zip(self._agent.gradient_holders_act, self._gradBuffer_act))
        _ = self._sess.run(self._agent.update_batch_act, feed_dict=feed_dict)
        for ix, grad in enumerate(self._gradBuffer_act):
            self._gradBuffer_act[ix] = grad * 0
