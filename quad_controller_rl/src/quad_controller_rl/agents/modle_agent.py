from quad_controller_rl.agents.base_agent import BaseAgent
import tensorflow as tf
import numpy as np

class MODLE(BaseAgent):
    def __init__(self, task):
        self.sess = tf.Session()
        self.task = task
        print("MODLE Init")

        self.modle = tf.train.import_meta_graph(
            meta_graph_or_file=('./off/policy-0.meta')
        )
        self.modle.restore(self.sess, tf.train.latest_checkpoint('./off/'))
        print("MODLE Load")

        self.graph = tf.get_default_graph()
        print("GRAPH Load")

    def step(self, state, reward, done):

        action = self.sess.run(
            'base_actor/action_net_out:0',
            feed_dict={
                'action_state_in:0':self.preprocess(state)
            }
        )

        return self.posprocess( action )

    def preprocess(self, dateIn):
        dateOut = dateIn.reshape(1, -1)
        return dateOut

    def posprocess(self, dateIn):
        action_range = np.divide(self.task.action_space.high - self.task.action_space.low, 2)
        action_mid = action_range + self.task.action_space.low

        dateOut = np.array([0, 0, 0, 0, 0, 0])
        if dateIn is not None:
            if False:
                dateOut      = dateIn[0] * action_range + action_mid
            else:
                dateOut[0:3] = dateIn[0] * action_range[0:3] + action_mid[0:3]

        return dateOut
