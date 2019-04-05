from quad_controller_rl.tasks.base_task import BaseTask
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from gym import spaces
from collections import namedtuple
import numpy as np

class HoverTask(BaseTask):
    def __init__(self):
        self.observation_space = spaces.Box(
            np.array( [-150.0, -150.0,   0.0, -1.0, -1.0, -1.0, -1.0] ),
            np.array( [ 150.0,  150.0, 300.0,  1.0,  1.0,  1.0,  1.0] )
        )

        self.action_space = spaces.Box(
            np.array( [-25.0, -25.0, -25.0, -25.0, -25.0, -25.0] ),
            np.array( [ 25.0,  25.0,  25.0,  25.0,  25.0,  25.0] )
        )

    def reset(self):
        return Pose(position=Point(0.0, 0.0, np.random.normal(10, 0.5)), orientation=Quaternion(0.0, 0.0, 0.0, 0.0)), \
               Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0))

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        State = np.array(
            [pose.position.x, pose.position.y, pose.position.z,
             pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        )

        Reward = 0.0
        Reward += -min( abs(pose.position.x), 20 )
        Reward += -min( abs(pose.position.y), 20 )
        Reward += -min( abs(10 - pose.position.z), 20 )

        Done = False
        if timestamp >= 5:
            Done = True

        action = self.agent.step(State, Reward, Done)

        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)
            return Wrench(force=Vector3(action[0], action[1], action[2]), torque=Vector3(action[3], action[4], action[5])), Done
        else:
            return Wrench(), Done
