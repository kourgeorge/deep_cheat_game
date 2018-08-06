import tensorflow as tf
import tensorflow.contrib.slim as slim
import Config as config

class PlayerNetwork():

    def __init__(self, lr, s_size, action_size, h_size):
        self._s_size = s_size
        self._action_size = action_size
        self._h_size = h_size
        self._regularization_param = 0.001

        # Implementing F(state)=action
        self.state_in = tf.placeholder(shape=[None, self._s_size], dtype=tf.float32)
        self.action_distribution = self._construct_action_model()
        self.cards_selection = self._construct_cards_model()

        # Collect the loss given the episode info.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.cards_holder = tf.placeholder(shape=[None, config.num_ranks], dtype=tf.int32)

        taken_action_probability = PlayerNetwork.get_decision_probability(self.action_holder, self.action_distribution)

        self.loss_act = -tf.reduce_mean(tf.log(taken_action_probability) * self.reward_holder)
        self.loss_cards = -tf.reduce_mean(tf.log(self.cards_selection) * self.reward_holder)


        # Calculate the gradients for each variable given the episode performance
        self.gradients_act = tf.gradients(self.loss_act, self.action_model_trainable_vars())
        self.gradients_cards = tf.gradients(self.loss_cards, self.cards_model_trainable_vars())

        # Initialize the placeholders for all
        self.gradient_holders_act = []
        for idx, var in enumerate(self.action_model_trainable_vars()):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder_act')
            self.gradient_holders_act.append(placeholder)

        self.gradient_holders_cards = []
        for idx, var in enumerate(self.cards_model_trainable_vars()):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder_cards')
            self.gradient_holders_cards.append(placeholder)

        # optimize given the gradients holder
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch_act = optimizer.apply_gradients(
            zip(self.gradient_holders_act, self.action_model_trainable_vars()))

        self.update_batch_cards = optimizer.apply_gradients(
            zip(self.gradient_holders_cards, self.cards_model_trainable_vars()))

        self.saver = tf.train.Saver()


    def _construct_action_model(self):
        hidden_layers_stack_action = slim.stack(self.state_in, slim.fully_connected, [self._h_size, 2 * self._h_size],
                                                activation_fn=tf.nn.tanh,
                                                weights_regularizer=slim.l2_regularizer(self._regularization_param),
                                                scope="action")

        action_output = slim.fully_connected(hidden_layers_stack_action, self._action_size, activation_fn=tf.nn.softmax,
                                             biases_initializer=None,
                                             weights_regularizer=slim.l2_regularizer(self._regularization_param),
                                             scope="action")

        return action_output


    def _construct_cards_model(self):

        hidden_layers_stack_action = slim.stack(self.state_in, slim.fully_connected,
                                                [self._h_size, 2 * self._h_size],
                                                activation_fn=tf.nn.tanh,
                                                weights_regularizer=slim.l2_regularizer(self._regularization_param),
                                                scope="cards")

        action_output = slim.fully_connected(hidden_layers_stack_action, config.num_ranks,
                                             activation_fn=tf.nn.relu,
                                             biases_initializer=None,
                                             weights_regularizer=slim.l2_regularizer(self._regularization_param),
                                             scope="cards")

        return action_output

    def action_model_trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="action")

    def cards_model_trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cards")


    @staticmethod
    def get_decision_probability(actual_decision, decisions_probabilities):
        action_indexes = tf.range(0, tf.shape(decisions_probabilities)[0]) * tf.shape(decisions_probabilities)[
            1] + actual_decision
        return tf.gather(tf.reshape(decisions_probabilities, [-1]), action_indexes)


    def save_model(self, sess, path):
        self.saver.save(sess, path)


    def load_model(self, sess, path):
        self.saver.restore(sess, path)
