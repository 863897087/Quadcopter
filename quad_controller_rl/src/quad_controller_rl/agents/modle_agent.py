from quad_controller_rl.agents.base_agent import BaseAgent
import tensorflow as tf
import numpy as np

class MODLE(BaseAgent):
    def __init__(self, task):
        self.task = task
        self.offsess = None
        self.hoversess = None
        self.landingsess = None
        print("MODLE Init")

    def Takeoff(self):
        if self.offsess is not None:
            return
        self.offgraph = tf.Graph()
        with self.offgraph.as_default():
            self.offmodle = tf.train.import_meta_graph(meta_graph_or_file=('./lastoff/policy-0.meta'))
        print("Takeoff Model Load")
        self.offsess = tf.Session(graph=self.offgraph)
        with self.offsess.as_default():
            with self.offgraph.as_default():
                self.offmodle.restore(self.offsess, tf.train.latest_checkpoint('./lastoff/'))
        print("Takeoff Varible Load")

    def HOVER(self):
        if self.hoversess is not None:
            return
        self.hovergraph = tf.Graph()
        with self.hovergraph.as_default():
            self.hovermodle = tf.train.import_meta_graph( meta_graph_or_file=('./lasthover/policy-0.meta') )
        print("HOVER Modle Load")
        self.hoversess = tf.Session(graph=self.hovergraph)
        with self.hoversess.as_default():
            with self.hovergraph.as_default():
                self.hovermodle.restore(self.hoversess, tf.train.latest_checkpoint('./lasthover/'))
        print("HOVER Varible Load")

    def LANDING(self):
        if self.landingsess is not None:
            return
        self.landinggraph = tf.Graph()
        with self.landinggraph.as_default():
            self.landingmodle = tf.train.import_meta_graph(meta_graph_or_file=('./lastlanding/policy-0.meta'))
        print("LANDING Modle Load")
        self.landingsess = tf.Session(graph=self.landinggraph)
        with self.landingsess.as_default():
            with self.landinggraph.as_default():
                self.landingmodle.restore(self.landingsess, tf.train.latest_checkpoint('./lastlanding/'))
        print("LANDING Varible Load")

    def SetTakeoff(self):
        self.En_Off = True
        self.En_Hover = False
        self.En_Landing = False

    def SetTakehover(self):
        self.En_Off = False
        self.En_Hover = True
        self.En_Landing = False

    def SetTakelanding(self):
        self.En_Off = False
        self.En_Hover = False
        self.En_Landing = True

    def step(self, state, reward, done):
        if self.En_Off:
            with self.offsess.as_default():
                with self.offgraph.as_default():
                    action = self.offsess.run(
                        'base_actor/action_net_out:0',
                        feed_dict={
                            'action_state_in:0':self.preprocess(state)
                        }
                    )
        if self.En_Hover:
            with self.hoversess.as_default():
                with self.hovergraph.as_default():
                    action = self.hoversess.run(
                        'base_actor/action_net_out:0',
                        feed_dict={
                            'action_state_in:0':self.preprocess(state)
                        }
                    )
        if self.En_Landing:
            with self.landingsess.as_default():
                with self.landinggraph.as_default():
                    action = self.landingsess.run(
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
