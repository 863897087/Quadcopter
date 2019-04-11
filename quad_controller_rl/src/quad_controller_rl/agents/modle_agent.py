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
        state = self.preprocess(state)
        state = state.reshape(1, -1)

        action = self.sess.run(
            'base_actor/action_net_out:0', feed_dict={'action_state_in:0':state}
        )

        return self.posprocess(action)

    def preprocess(self, date):
        return date

    def posprocess(self, date):
        if False:
            tempdate = np.array([0, 0, 0, 0, 0, 0])
            if date is not None:
                tempdate[0:3] = date[0][0:3]
            return tempdate
        else:
            return date
