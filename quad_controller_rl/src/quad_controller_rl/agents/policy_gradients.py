from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl import util
from collections import namedtuple
from random import sample
import tensorflow as tf
import math
import os
import numpy as np
import pandas as pd


class DDPG(BaseAgent):

    def __init__(self, task):
        self.state_dim = 3#7
        self.action_dim = 3#6
        self.task = task
        self.train = Train(
            self.action_dim,
            self.state_dim,
            self.preprocess(self.task.action_space.low),
            self.preprocess(self.task.action_space.high),
            self.preprocess(self.task.observation_space.low),
            self.preprocess(self.task.observation_space.high)
        )
        self.noise = Noise(self.action_dim)
        self.reset()
        print("DDPG init")
        self.status_filename = os.path.join(util.get_param('out'), "status_{}.csv".format(util.get_timestamp()))
        self.status_columns = ['episode', 'total_reward']
        self.episode_num = 1
        self.total_reward = 0
        print("Saving status {} to {}".format(self.status_columns, self.status_filename))

    def preprocess(self, date):
        return date[0:3]
    def posprocess(self, date):
        temp_array = np.zeros(self.task.action_space.shape)
        temp_array[0:3] = date
        return temp_array

    def reset(self):
        self.next_state = None
        self.state = None
        self.action = None
        self.reward = None
        self.done = None
        self.noise.reset()
        print("DDPG reset")
        self.total_reward = 0

    def write_status(self, status):
        df_status = pd.DataFrame([status], columns=self.status_columns)
        df_status.to_csv(self.status_filename, mode='a', index=False, header=not os.path.isfile(self.status_filename))

    def step(self, state, reward, done):
        self.next_state = self.preprocess(
            (state - self.task.observation_space.low) / (self.task.observation_space.high - self.task.observation_space.low)
            )
        self.next_state = self.next_state.reshape(1, -1)
        self.reward = reward
        self.done = done

        if self.state is not None and self.action is not None:
            self.train.experience.push(self.state, self.action, self.reward, self.next_state, self.done)
            self.total_reward += self.reward

        self.state = self.next_state
        self.action = self.train.actor.operation_AB_out(self.state)
        self.action = self.action + self.noise.sample()

        if 1000 < len(self.train.experience) and 0 == (len(self.train.experience) % 10):
            print("train {}".format(len(self.train.experience)))
            for times in range(25):
                exp_state, exp_action, exp_reward, exp_next_state, exp_done = self.train.experience.pop(
                    self.train.batch_size)

                cost_critic_summary, QB_summary = self.train.operation_train_critic(
                    exp_state, exp_action, exp_reward, exp_next_state, exp_done
                )
                gradient_summary, cost_actor_summary = self.train.operation_train_actor(exp_state)

                self.train.operation_update_net()

                if times % 5 == 0:
                    self.train.operation_write_summary(
                        cost_critic_summary,
                        QB_summary,
                        gradient_summary,
                        cost_actor_summary,
                        self.train.iterations
                    )
                self.train.iterations += 1

        if done:
            print("done {}".format(len(self.train.experience)))
            self.write_status([self.episode_num, self.total_reward])
            self.episode_num += 1

            self.reset()

        return self.posprocess(self.action)

class Noise:
    def __init__(self, size, theta=0.15, sigma=0.3, mu=None):
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        self.state = self.mu

    def sample(self):

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn( len(x) )
        self.state = x + dx

        return self.state

class Train:
    def __init__(self, action_dim, state_dim, aclow, achigh, stlow, sthigh):
        self.sess = tf.Session()
        self.action_low = aclow
        self.action_high = achigh
        self.action_dim = action_dim

        self.state_low = stlow
        self.state_high = sthigh
        self.state_dim = state_dim

        self.batch_size = 64
        self.iterations = 0

        self.experience = Experience(self.action_dim, self.state_dim, 1000000)
        self.critic = Critic(self.sess, self.batch_size, self.action_dim, self.state_dim, 1, 1)
        self.actor = Actor(self.sess, self.batch_size, self.action_dim, self.state_dim, self.action_low, self.action_high)

        self.sess.run(tf.global_variables_initializer())
        self.critic.operation_assign_QT_net()
        self.actor.operation_assign_AT_net()

    def operation_train_actor(self, state):
        action = self.actor.operation_AB_out(state)
        gradient_summary, gradient = self.critic.operation_gradient(action, state)
        cost_summary = self.actor.operation_learn(gradient, state)
        return gradient_summary, cost_summary

    def operation_train_critic(self, state, action, reward, next_state, done):
        next_action = self.actor.operation_AT_out(next_state)
        target = self.critic.operation_target(next_action, next_state, reward, done)
        cost_summary, QB_summary = self.critic.operation_learn(target, action, state)
        return cost_summary, QB_summary

    def operation_write_summary(self, cost_critic_summary, QB_critic_summary, gradient_critic_summary, cost_actor_summary, iters):
        self.critic.writer.add_summary(cost_critic_summary, iters)
        self.critic.writer.add_summary(QB_critic_summary, iters)
        self.critic.writer.add_summary(gradient_critic_summary, iters)

        self.actor.writer.add_summary(cost_actor_summary, iters)

    def operation_update_net(self):
        self.actor.operation_update_AT_net()
        self.critic.operation_update_QT_net()

Experience_format = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
class Experience:
    def __init__(self, action_dim, state_dim, size=1000):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.size = size
        self.idx = 0
        self.list = []

    def __len__(self):
        return len(self.list)

    def push(self, state, action, reward, next_state, done):
        BufHand = Experience_format(state, action, reward, next_state, done)
        if len(self.list) < self.size:
            self.list.append(BufHand)
        else:
            self.list[self.idx] = BufHand
            self.idx = (self.idx + 1) % self.size

    def pop(self, size):
        list = sample(self.list, size)

        state = np.asarray([line.state for line in list if line is not None], dtype=np.float32).reshape(size, self.state_dim)
        action = np.asarray([line.action for line in list if line is not None], dtype=np.float32).reshape(size, self.action_dim)
        reward = np.asarray([line.reward for line in list if line is not None], dtype=np.float32).reshape(size, 1)
        next_state = np.asarray([line.next_state for line in list if line is not None], dtype=np.float32).reshape(size, self.state_dim)
        done = np.asarray([line.done for line in list if line is not None], dtype=np.float32).reshape(size, 1)

        return state, action, reward, next_state, done

class Actor:
    def __init__(self, sess, batch_size, action_dim, state_dim, aclow, achigh):
        self.sess = sess
        self.batch_size = batch_size

        self.action_low = aclow
        self.action_high = achigh
        self.action_range = self.action_high - self.action_low
        
        self.state_dim = state_dim
        self.state_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.state_dim],
            name='StateInput'
            )

        self.action_dim = action_dim
        self.gradient_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.action_dim],
            name='GradientInput'
            )

        with tf.variable_scope('actor_Bnet'):
            self.AB_out, self.AB_var = self.CreateNet(True)
        with tf.variable_scope('actor_Tnet'):
            self.AT_out, self.AT_var = self.CreateNet(False)

        self.BuildUpdateGraph(0.001)
        self.BuildCostGraph()
        self.BuildSummaryGraph()

    def CreateNet(self, OptimizerEn=False):
       l1 = tf.layers.dense(
            inputs=self.state_in,
            units=30,
            activation=tf.nn.tanh,
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(-1/math.sqrt(self.state_dim), 1/math.sqrt(self.state_dim)),
            bias_initializer=tf.random_uniform_initializer(-0.01, 0.01),
            trainable=OptimizerEn,
            name='l1',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.006)
            )
       print(l1.name, os.path.split(l1.name)[0] + '/kernel:0')
       l1_w = tf.get_default_graph().get_tensor_by_name(os.path.split(l1.name)[0] + '/kernel:0')
       l1_b = tf.get_default_graph().get_tensor_by_name(os.path.split(l1.name)[0] + '/bias:0')

       l2 = tf.layers.dense(
            inputs=l1,
            units=20,
            activation=tf.nn.tanh,
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(-1/math.sqrt(30), 1/math.sqrt(30)),
            bias_initializer=tf.random_uniform_initializer(-0.01, 0.01),
            trainable=OptimizerEn,
            name='l2',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.006)        
            )
       print(l2.name, os.path.split(l2.name)[0] + '/kernel:0')
       l2_w = tf.get_default_graph().get_tensor_by_name(os.path.split(l2.name)[0] + '/kernel:0')
       l2_b = tf.get_default_graph().get_tensor_by_name(os.path.split(l2.name)[0] + '/bias:0')

       l3 = tf.layers.dense(
            inputs=l2,
            units=self.action_dim,
            activation=tf.nn.tanh,
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(-1/math.sqrt(20), 1/math.sqrt(20)),
            bias_initializer=tf.random_uniform_initializer(-0.01, 0.01),
            trainable=OptimizerEn,
            name='l3',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.006)
            )
       print(l3.name, os.path.split(l3.name)[0] + '/kernel:0')
       l3_w = tf.get_default_graph().get_tensor_by_name(os.path.split(l3.name)[0] + '/kernel:0')
       l3_b = tf.get_default_graph().get_tensor_by_name(os.path.split(l3.name)[0] + '/bias:0')

       return (l3 * self.action_range + self.action_low), [l1_w, l1_b, l2_w, l2_b, l3_w, l3_b]

    def BuildUpdateGraph(self, LearnRate=0.5):
        self.assign_AT_net = [tf.assign(AT_w, AB_w) for AB_w, AT_w in zip(self.AB_var, self.AT_var)]
        self.update_AT_net = [tf.assign(AT_w, (AB_w * LearnRate + AT_w * (1 - LearnRate))) for AB_w, AT_w in zip(self.AB_var, self.AT_var)]

    def BuildCostGraph(self):
        self.cost = tf.reduce_mean(self.gradient_in * self.AB_out)
        self.train = tf.train.AdamOptimizer(-0.001).minimize(self.cost)

    def BuildSummaryGraph(self):
        self.cost_summary = tf.summary.scalar('actor_cost', self.cost)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('/home/robond/Desktop/summary_actor')

    def operation_AB_out(self, state):
        return self.sess.run(self.AB_out, feed_dict={self.state_in:state})

    def operation_AT_out(self, state):
        return self.sess.run(self.AT_out, feed_dict={self.state_in:state})
    
    def operation_learn(self, gradient, state):
        cost_summary, _ = self.sess.run([self.cost_summary, self.train], feed_dict={self.gradient_in:gradient, self.state_in:state})
        return cost_summary

    def operation_assign_AT_net(self):
        self.sess.run(self.assign_AT_net)

    def operation_update_AT_net(self):
        self.sess.run(self.update_AT_net)
        
class Critic:
    def __init__(self, sess, batch_size, action_dim, state_dim, reward_dim, done_dim):
        self.sess = sess
        self.batch_size = batch_size

        self.state_dim = state_dim
        self.state_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.state_dim],
            name='StateInput'
            )

        self.action_dim = action_dim
        self.actor_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.action_dim],
            name='ActorInput'            
            )
    
        self.reward_dim = reward_dim
        self.reward_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.reward_dim],
            name='RewardInput'
            )

        self.done_dim = done_dim
        self.done_in = tf.placeholder(
            dtype=tf.bool,
            shape=[None, self.done_dim],
            name='DoneInput'
        )

        self.target_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, 1],
            name='TargetInput'
            )

        with tf.variable_scope('critic_Bnet'):
            self.QB_out, self.QB_var = self.CreateNet(True)
        with tf.variable_scope('critic_Tnet'):
            self.QT_out, self.QT_var = self.CreateNet(False) 

        self.BuildUpdateGraph(0.0001)
        self.BuildTargetGraph(0.99)
        self.BuildCostGraph()
        self.BuildGradientGraph()
        self.BuildSummaryGraph()

    def CreateNet(self, OptimizerEn=False):
        l1_s_w = tf.get_variable(
                name='11_s_w',
                shape=[self.state_dim, 50],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(-1/math.sqrt(self.state_dim), 1/math.sqrt(self.state_dim)),
                regularizer=tf.contrib.layers.l2_regularizer(0.006),
                trainable=OptimizerEn
                )
        l1_a_w = tf.get_variable(
                name='l1_a_w',
                shape=[self.action_dim, 50],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(-1/math.sqrt(self.action_dim), 1/math.sqrt(self.action_dim)),
                regularizer=tf.contrib.layers.l2_regularizer(0.006),
                trainable=OptimizerEn
                )
        l1_b = tf.get_variable(
                name='l1_b',
                shape=[50],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.01, 0.01),
                trainable=OptimizerEn
                )
        l1 = tf.nn.tanh(tf.matmul(self.actor_in, l1_a_w) + tf.matmul(self.state_in, l1_s_w) + l1_b)

        l2_w = tf.get_variable(
                name='l2_w',
                shape=[50, 40],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(-1/math.sqrt(50), 1/math.sqrt(50)),
                regularizer=tf.contrib.layers.l2_regularizer(0.006),
                trainable=OptimizerEn                
                )
        l2_b = tf.get_variable(
                name='l2_b',
                shape=[40],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.01, 0.01),
                trainable=OptimizerEn
                )
        l2 = tf.nn.tanh(tf.matmul(l1, l2_w) + l2_b)

        l3_w = tf.get_variable(
                name='l3_w',
                shape=[40, 1],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(-1/math.sqrt(40), 1/math.sqrt(40)),
                regularizer=tf.contrib.layers.l2_regularizer(0.006),
                trainable=OptimizerEn
                ) 
        l3_b = tf.get_variable(
                name='l3_b',
                shape=[1],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.01, 0.01),
                trainable=OptimizerEn
                )  
        l3 = tf.matmul(l2, l3_w) + l3_b

        return l3, [l1_s_w, l1_a_w, l1_b, l2_w, l2_b, l3_w, l3_b]

    def BuildUpdateGraph(self, LearnRate=0.5):
        self.assign_QT_net = [tf.assign(QT_w, QB_w) for QB_w, QT_w in zip(self.QB_var, self.QT_var)]
        self.update_QT_net = [tf.assign(QT_w, (QB_w * LearnRate + QT_w * (1 - LearnRate))) for QB_w, QT_w in zip(self.QB_var, self.QT_var)]

    def BuildTargetGraph(self, GAMMA=0.5):
        self.target = tf.where(
            self.done_in,
            tf.constant(0, dtype=tf.float32, shape=[self.batch_size, 1]),
            self.QT_out * GAMMA
        ) + self.reward_in

    def BuildCostGraph(self):
        self.cost = tf.reduce_mean(tf.square(self.target_in - self.QB_out))
        self.train = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    def BuildGradientGraph(self):
         gradient = tf.gradients(self.QB_out, self.actor_in)
         self.gradient = tf.reshape(gradient, (self.batch_size, self.action_dim))

    def BuildSummaryGraph(self):
        self.cost_summary = tf.summary.scalar('critic_cost', self.cost)
        self.QB_summary = tf.summary.scalar('critic_Q', tf.reduce_mean(self.QB_out))
        self.gradient_summary = tf.summary.scalar('critic_gradient', tf.reduce_mean(self.gradient))
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('/home/robond/Desktop/summary_critic')
    
    def operation_target(self, action, state, reward, done):
        return self.sess.run(
                    self.target, 
                    feed_dict={
                            self.actor_in:action,
                            self.state_in:state,
                            self.reward_in:reward,
                            self.done_in:done
                    }
                )

    def operation_learn(self, target, action, state):
        cost_summary, QB_summary, _ = self.sess.run(
                                            [self.cost_summary, self.QB_summary, self.train],
                                            feed_dict={
                                                self.target_in:target,
                                                self.actor_in:action,
                                                self.state_in:state            
                                            }            
                                      )
        return cost_summary, QB_summary

    def operation_gradient(self, action, state):
        gradient_summary, gradient = self.sess.run(
                    [self.gradient_summary, self.gradient],
                    feed_dict={
                        self.actor_in:action,
                        self.state_in:state    
                    }
                )
        return gradient_summary, gradient

    def operation_update_QT_net(self):
        self.sess.run(self.update_QT_net)

    def operation_assign_QT_net(self):
        self.sess.run(self.assign_QT_net)

