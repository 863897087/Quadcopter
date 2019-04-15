from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl import util
from collections import namedtuple
from random import sample
from copy import copy
import tensorflow as tf
import math
import os
import numpy as np
import pandas as pd


class DDPG(BaseAgent):

    def __init__(self, task):
        self.task = task

        self.state_dim = self.task.observation_space.shape[0]
        if False:
            self.action_dim = self.task.action_space.shape[0]
        else:
            self.action_dim = 3
        self.reward_dim = 1
        self.done_dim = 1

        self.experience = Experience(self.action_dim, self.state_dim, 1000000)
        self.train = Train(self.action_dim, self.state_dim, self.reward_dim, self.done_dim)
        self.noise = Noise(self.action_dim)
        self.adaptive_noise = AdaptiveParamNoise(initial_stddev=float(0.3), desired_action_stddev=float(0.4), adoption_coefficient=float(1.0001))

        self.reset()
        print("DDPG init")
        self.status_filename = os.path.join(util.get_param('out'), "status_{}.csv".format(util.get_timestamp()))
        self.status_columns = ['episode', 'total_reward']
        self.reward_num = 1
        self.total_reward = 0
        self.max_total_reward = -1000
        print("Saving status {} to {}".format(self.status_columns, self.status_filename))

        self.action_num = 1
        self.action_filename = os.path.join(util.get_param('out'), "action_{}.csv".format(util.get_timestamp()))
        print("Saving action")

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

    def reset(self):
        self.next_state = None
        self.state = None
        self.action = None
        self.reward = None
        self.done = None
        self.noise.reset()
        print("DDPG reset")
        self.total_reward = 0

    def write_reward(self):
        status = [self.reward_num, self.total_reward]
        df_status = pd.DataFrame([status], columns=self.status_columns)
        df_status.to_csv(self.status_filename, mode='a', index=False, header=not os.path.isfile(self.status_filename))
        self.reward_num += 1

    def write_action(self):
        if self.action is None:
            return
        temp = np.asarray([self.action_num, self.adaptive_noise.current_stddev, 0, 0, 0, 0, 0, 0])
        temp[2:] = self.posprocess( self.action )

        df_status = pd.DataFrame([temp], columns=['episode', 'noise',' l1', 'l2', 'l3', 'l4', 'l5', 'l6'])
        df_status.to_csv(self.action_filename, mode='a', index=False, header=not os.path.isfile(self.action_filename))
        self.action_num += 1

    def step(self, state, reward, done):
        self.reward = reward
        self.done = done
        self.next_state = self.preprocess( state )

        if self.state is not None and self.action is not None:
            self.total_reward += self.reward
            self.experience.push(self.state, self.action, self.reward,  self.next_state, self.done)

        self.state = self.next_state
        self.train.operation_param_noise_reset(self.adaptive_noise.current_stddev)
        self.action = self.train.operation_param_noise_action(self.state)
        #self.action = self.action + self.noise.sample()

        if 1000 < len(self.experience) and 0 == (len(self.experience) % 1):
            for times in range(1):
                exp_state, exp_action, exp_reward, exp_next_state, exp_done = self.experience.pop(self.train.batch_size)

                cost_critic_summary, QB_summary = self.train.operation_train_critic(
                    exp_state, exp_action, exp_reward, exp_next_state, exp_done
                )
                gradient_summary, cost_actor_summary = self.train.operation_train_actor(exp_state)

                self.train.operation_update_net()

                self.adaptive_noise.adapt(
                    self.train.operation_param_noise_adaptive(exp_state, self.adaptive_noise.current_stddev)
                )

                if times % 5 == 0:
                    self.train.operation_write_summary(
                        cost_critic_summary,
                        QB_summary,
                        gradient_summary,
                        cost_actor_summary,
                    )

        if done:
            print("done {} max reward {} reward {}".format(len(self.experience), self.max_total_reward, self.total_reward))
            self.write_reward()
            if self.max_total_reward < self.total_reward:
                self.max_total_reward = self.total_reward
                self.train.saver.save(sess=self.train.sess, save_path="./modle/policy", global_step=0, write_meta_graph=True)
            self.reset()
        self.write_action()
        return self.posprocess( self.action )

class AdaptiveParamNoise:
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            self.current_stddev /= self.adoption_coefficient
        else:
            self.current_stddev *= self.adoption_coefficient

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

class Train:
    def __init__(self, action_dim, state_dim, reward_dim, done_dim):

        self.batch_size = 64
        self.iterations = 0
        self.sess = tf.Session()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.done_dim = done_dim

        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        self.critic = Critic(
            self.sess, self.batch_size, self.action_dim, self.state_dim, self.reward_dim, self.done_dim
        )

        self.actor = Actor(
            self.sess, self.batch_size, self.action_dim, self.state_dim
        )

        self.setup_actor_base()
        self.setup_actor_copy()
        self.setup_critic_base()
        self.setup_critic_copy()
        self.setup_param_noise()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.actor_copy_policy)
        self.sess.run(self.critic_copy_policy)

        self.graph_interface()

    def graph_interface(self):
        print("ALL COLLECTION KEYS LIST: ")
        self.keys_names = tf.get_default_graph().get_all_collection_keys()
        print(self.keys_names)

        for name in self.keys_names:
            print("KEY", name)
            for variable in tf.get_default_graph().get_collection(name):
                print(variable)

        self.saver = tf.train.Saver( max_to_keep=100 )

        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./modle/'))
        except:
            print("Nathing Variable Value Can Load")
        finally:
            print("Load Variable Value OK")


    def perturb_actor_updates(self, actor, actor_param_noise):
        updates = []
        for var, perturb_var in zip(actor.vars, actor_param_noise.vars):
            if var in actor.perturb_vars:
                print(perturb_var.name, var.name)
                updates.append(tf.assign(
                        perturb_var,
                        var + tf.random_normal(tf.shape(var), mean=0., stddev=self.param_noise_stddev)
                    )
                )
            else:
                updates.append(tf.assign(perturb_var, var))
        return updates

    def copy_updates(self, actor, copy_param):
        updates = []
        for base_var, copy_var in zip(actor.vars, copy_param.vars):
            updates.append(
                tf.assign(copy_var, base_var)
            )
        return updates

    def upgrade_updates(self, actor, upgrade_param, learn_rate):
        updates = []
        for base_var, upgrade_var in zip(actor.vars, upgrade_param.vars):
            updates.append(
                tf.assign(
                    upgrade_var,
                    base_var * learn_rate + upgrade_var * (1 - learn_rate)))
        return updates

    def setup_param_noise(self):
        self.actor_param_noise = copy(self.actor)
        self.actor_param_noise.name = "param_noise_actor"
        self.actor_param_noise_tf = self.actor_param_noise(OptimizerEn=False, layer_norm=True, reuse=False)
        self.actor_param_noise_policy = tf.group(* self.perturb_actor_updates(self.actor, self.actor_param_noise), name='actor_param_noise_policy' )

        self.actor_adaptive_param_noise = copy(self.actor)
        self.actor_adaptive_param_noise.name = "adaptive_param_noise_actor"
        self.actor_adaptive_param_noise_tf = self.actor_adaptive_param_noise(OptimizerEn=False, layer_norm=True, reuse=False)
        self.actor_adaptive_param_noise_policy = tf.group(* self.perturb_actor_updates(self.actor, self.actor_adaptive_param_noise), name='actor_adaptive_param_noise_policy' )
        self.actor_adaptive_param_noise_distance = tf.sqrt(
            tf.reduce_mean(
                tf.square(
                    self.actor_base_tf - self.actor_adaptive_param_noise_tf
                )
            ),
            name='actor_adaptive_param_noise_distance'
        )

    def setup_actor_base(self):
        self.actor.name = "base_actor"
        self.actor_base_tf = self.actor(OptimizerEn=True, layer_norm=True, reuse=False)
        self.actor.BuildCostGraph(self.actor_base_tf)
        self.actor.BuildSummaryGraph()

    def setup_actor_copy(self):
        self.actor_copy = copy(self.actor)
        self.actor_copy.name = "copy_actor"
        self.actor_copy_ty = self.actor_copy(OptimizerEn=False, layer_norm=True, reuse=False)
        self.actor_copy_policy = tf.group(* self.copy_updates(self.actor, self.actor_copy), name='actor_copy_policy' )
        self.actor_upgrade_policy = tf.group(* self.upgrade_updates(self.actor, self.actor_copy, 0.0001), name='actor_upgrade_policy' )

    def setup_critic_base(self):
        self.critic.name = "base_critic"
        self.critic_base_tf = self.critic(OptimizerEn=True, reuse=False)
        self.critic.BuildCostGraph(self.critic_base_tf)
        self.critic.BuildGradientGraph(self.critic_base_tf)
        self.critic.BuildSummaryGraph(self.critic_base_tf)

    def setup_critic_copy(self):
        self.critic_copy = copy(self.critic)
        self.critic_copy.name = "copy_critic"
        self.critic_copy_ty = self.critic_copy(OptimizerEn=False, reuse=False)
        self.critic_copy_policy = tf.group(* self.copy_updates(self.critic, self.critic_copy), name='critic_copy_policy' )
        self.critic_upgrade_policy = tf.group(* self.upgrade_updates(self.critic, self.critic_copy, 0.000001), name='critic_upgrade_policy' )

        self.critic.BuildTargetGraph(self.critic_copy_ty, 0.99)

    def operation_update_net(self):
        self.sess.run(self.actor_upgrade_policy)
        self.sess.run(self.critic_upgrade_policy)

    def operation_param_noise_adaptive(self, state, current_stddev):

        self.sess.run(
            self.actor_adaptive_param_noise_policy,
            feed_dict={
                self.param_noise_stddev:current_stddev
            }
        )

        distance = self.sess.run(
            self.actor_adaptive_param_noise_distance,
            feed_dict={
                self.param_noise_stddev:current_stddev,
                self.actor.state_in:state,
                self.actor_adaptive_param_noise.state_in:state
            }
        )

        mean_distance = distance

        return mean_distance

    def operation_param_noise_action(self, state):
        return self.sess.run(self.actor_param_noise_tf, feed_dict={self.actor_param_noise.state_in:state})

    def operation_param_noise_reset(self, param_noise_stddev):
        self.sess.run(self.actor_param_noise_policy, feed_dict={self.param_noise_stddev: param_noise_stddev})

    def operation_train_actor(self, state):
        action = self.sess.run(self.actor_base_tf, feed_dict={self.actor.state_in:state})
        gradient_summary, gradient = self.critic.operation_gradient(action, state)
        cost_summary = self.actor.operation_learn(gradient, state)
        return gradient_summary, cost_summary

    def operation_train_critic(self, state, action, reward, next_state, done):
        next_action = self.sess.run(self.actor_copy_ty, feed_dict={self.actor_copy.state_in: state})
        target = self.critic.operation_target(next_action, next_state, reward, done)
        cost_summary, QB_summary = self.critic.operation_learn(target, action, state)
        return cost_summary, QB_summary

    def operation_write_summary(self, cost_critic_summary, QB_critic_summary, gradient_critic_summary, cost_actor_summary):
        self.critic.writer.add_summary(cost_critic_summary,     self.iterations)
        self.critic.writer.add_summary(QB_critic_summary,       self.iterations)
        self.critic.writer.add_summary(gradient_critic_summary, self.iterations)
        self.actor.writer.add_summary(cost_actor_summary,       self.iterations)

        self.iterations += 1

def ortho_init( scale=1.0 ):
    def _ortho_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:
            flat_shape = ( np.prod(shape[:1]), shape[-1] )
        else:
            raise NotImplementedError

        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        if u.shape == flat_shape:
            q = u
        else:
            q = v

        q = q.reshape(shape)
        return (scale * q[ :shape[0], :shape[1] ]).astype(np.float32)

class Model(object):
    def __init__(self):
        self.name = None

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    @property
    def train_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    @property
    def perturb_vars(self):
        return [var for var in self.train_vars if 'LayerNorm' not in var.name]

class Actor(Model):
    def __init__(self, sess, batch_size, action_dim, state_dim):
        super().__init__()

        self.sess = sess
        self.batch_size = batch_size

        self.state_dim = state_dim
        self.state_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.state_dim],
            name='action_state_in'
            )

        self.action_dim = action_dim
        self.gradient_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.action_dim],
            name='action_gradient_in'
            )

    def __call__(self, OptimizerEn, layer_norm, reuse):

        with tf.variable_scope(name_or_scope=self.name, dtype=tf.float32, reuse=reuse):
            NetHandle = self.CreateNet(OptimizerEn=OptimizerEn, layer_norm=layer_norm)
        return NetHandle

    def CreateNet(self, OptimizerEn=False, layer_norm=False):
       l0 = self.state_in

       l1 = tf.layers.dense(
            inputs=l0,
            units=30,
            use_bias=True,
            kernel_initializer=ortho_init( np.sqrt(2) ),
            bias_initializer=tf.constant_initializer(0.0),
            trainable=OptimizerEn,
            name='l1',
            )

       if layer_norm:
           l1 = tf.contrib.layers.layer_norm(l1,
                                             center=True,
                                             scale=True,
                                             trainable=OptimizerEn
            )
       l1 = tf.nn.tanh(l1)

       l2 = tf.layers.dense(
            inputs=l1,
            units=20,
            use_bias=True,
            kernel_initializer=ortho_init( np.sqrt(2) ),
            bias_initializer=tf.constant_initializer(0.0),
            trainable=OptimizerEn,
            name='l2',
            )

       if layer_norm:
           l2 = tf.contrib.layers.layer_norm(l2,
                                             center=True,
                                             scale=True,
                                             trainable=OptimizerEn
            )
       l2 = tf.nn.tanh(l2)

       l3 = tf.layers.dense(
            inputs=l2,
            units=self.action_dim,
            kernel_initializer=tf.random_uniform_initializer(minval=-(3e-3), maxval=(3e-3)),
            trainable=OptimizerEn,
            name='l3',
            )

       return tf.nn.tanh(l3, name='action_net_out')

    def BuildCostGraph(self, NetOut):
        self.cost = tf.reduce_mean( tf.multiply( self.gradient_in, NetOut ), name='action_cost')
        self.train = tf.train.AdamOptimizer(-0.001).minimize( self.cost, name='action_train')

    def BuildSummaryGraph(self):
        self.cost_summary = tf.summary.scalar('actor_cost', self.cost )
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('/home/robond/Desktop/summary_actor')

    def operation_learn(self, gradient, state):
        cost_summary, _ = self.sess.run([self.cost_summary, self.train], feed_dict={self.gradient_in:gradient, self.state_in:state})
        return cost_summary

        
class Critic(Model):
    def __init__(self, sess, batch_size, action_dim, state_dim, reward_dim, done_dim):
        super().__init__()

        self.sess = sess
        self.batch_size = batch_size

        self.state_dim = state_dim
        self.state_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.state_dim],
            name='critic_state_in'
            )

        self.action_dim = action_dim
        self.actor_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.action_dim],
            name='critic_action_in'
            )
    
        self.reward_dim = reward_dim
        self.reward_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.reward_dim],
            name='critic_reward_in'
            )

        self.done_dim = done_dim
        self.done_in = tf.placeholder(
            dtype=tf.bool,
            shape=[None, self.done_dim],
            name='critic_done_in'
        )

        self.target_in = tf.placeholder(
            dtype=tf.float32,
            shape=[None, 1],
            name='critic_target_in'
            )

    def __call__(self, OptimizerEn, reuse):

        with tf.variable_scope(name_or_scope=self.name, dtype=tf.float32, reuse=reuse):
            NetHandle = self.CreateNet(OptimizerEn=OptimizerEn)
        return NetHandle


    def CreateNet(self, OptimizerEn=False):
        l0_s = self.state_in
        l0_a = self.actor_in

        l1_s_w = tf.get_variable(
                name='11_s_w',
                shape=[self.state_dim, 60],
                dtype=tf.float32,
                initializer=ortho_init( np.sqrt(2) ),
                trainable=OptimizerEn
                )
        l1_a_w = tf.get_variable(
                name='l1_a_w',
                shape=[self.action_dim, 60],
                dtype=tf.float32,
                initializer=ortho_init( np.sqrt(2) ),
                trainable=OptimizerEn
                )
        l1_b = tf.get_variable(
                name='l1_b',
                shape=[60],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0),
                trainable=OptimizerEn
                )
        l1 = tf.matmul(l0_a, l1_a_w) + tf.matmul(l0_s, l1_s_w) + l1_b
        l1 = tf.nn.elu(l1)

        l2_w = tf.get_variable(
                name='l2_w',
                shape=[60, 40],
                dtype=tf.float32,
                initializer=ortho_init( np.sqrt(2) ),
                trainable=OptimizerEn                
                )
        l2_b = tf.get_variable(
                name='l2_b',
                shape=[40],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0),
                trainable=OptimizerEn
                )
        l2 = tf.matmul(l1, l2_w) + l2_b
        l2 = tf.nn.elu(l2)

        l3_w = tf.get_variable(
                name='l3_w',
                shape=[40, 20],
                dtype=tf.float32,
                initializer=ortho_init( np.sqrt(2) ),
                trainable=OptimizerEn
                ) 
        l3_b = tf.get_variable(
                name='l3_b',
                shape=[20],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0),
                trainable=OptimizerEn
                )
        l3 = tf.matmul(l2, l3_w) + l3_b
        l3 = tf.nn.elu(l3)

        l4 = tf.layers.dense(
            inputs=l3,
            units=1,
            kernel_initializer=tf.random_uniform_initializer(minval=-(3e-3), maxval=(3e-3)),
            trainable=OptimizerEn,
            name='critic_net_out',
        )

        return l4

    def BuildTargetGraph(self, NetOut, GAMMA=0.5):
        self.target = tf.add(
            tf.where(
                self.done_in,
                tf.constant(0, dtype=tf.float32, shape=[self.batch_size, 1]),
                NetOut * GAMMA
            ),
            self.reward_in,
            name='critic_target'
        )

    def BuildCostGraph(self, NetOut):
        self.cost = tf.reduce_mean(tf.square(self.target_in - NetOut), name='critic_cost')
        self.train = tf.train.AdamOptimizer(0.001).minimize(self.cost, name='critic_train')

    def BuildGradientGraph(self, NetOut):
         gradient = tf.gradients(NetOut, self.actor_in)
         self.gradient = tf.reshape(gradient, (self.batch_size, self.action_dim), name='critic_gradient')

    def BuildSummaryGraph(self, NetOut):
        self.cost_summary = tf.summary.scalar('critic_cost', self.cost )
        self.QB_summary = tf.summary.scalar('critic_Q', tf.reduce_mean(NetOut))
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

