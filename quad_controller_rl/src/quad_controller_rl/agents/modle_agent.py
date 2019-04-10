from quad_controller_rl.agents.base_agent import BaseAgent
import tensorflow as tf
import numpy as np

class MODLE(BaseAgent):
    def __init__(self, task):
        self.sess = tf.Session()
        self.task = task

        print("MODLE Init")

        self.modle = tf.train.import_meta_graph(
            meta_graph_or_file=('./hover/policy-163931.meta')
        )
        self.modle.restore(self.sess, tf.train.latest_checkpoint('./hover/'))

        print("MODLE Load")

        self.graph = tf.get_default_graph()

        print("GRAPH Load")

        self.action_state_in = self.graph.get_operation_by_name('action_state_in')
        self.action_net_out = self.graph.get_operation_by_name('action_net_out')
        print("GRAPH Interface")

        for operation in self.graph.get_operations():
            print(operation)
        key_names = self.graph.get_all_collection_keys()
        print(key_names)
        for name in key_names:
            print(name)
            for key in self.graph.get_collection(name):
                print(key)

        print("Init END")

    def step(self, state, reward, done):
        state = self.preprocess(state)

        action = self.sess.run(
            self.action_net_out,
            feed_dict={self.action_state_in:state}
        )

        return self.posprocess(action)

    def preprocess(self, date):
        return date

    def posprocess(self, date):
        if True:
            tempdate = np.array([0, 0, 0, 0, 0, 0])
            if date is not None:
                tempdate[0:3] = date[0][0:3]
            return tempdate
        else:
            return date

        return action
