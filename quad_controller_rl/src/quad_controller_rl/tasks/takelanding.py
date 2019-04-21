from quad_controller_rl.tasks.base_task import BaseTask
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from gym import spaces
import numpy as np

class LANDING(BaseTask):
    def __init__(self):
        self.observation_space = spaces.Box(
            np.array(
                [0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0]
            ),
            np.array(
                [0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0]
            )
        )

        self.action_space = spaces.Box(
            np.array([-25.0, -25.0, -25.0, -25.0, -25.0, -25.0]),
            np.array([ 25.0,  25.0,  25.0,  25.0,  25.0,  25.0])
        )

        self.star_pose = Point(0.0, 0.0, 10.0)
        self.save_pose = Point(0.0, 0.0, 10.0)
        self.target_pose = Point(0.0, 0.0, 10.0)

    def reset(self):
        self.agent.LANDING()

        return Pose(
            position=Point(0.0,0.0,10.0),
            orientation=Quaternion(0.0, 0.0, 0.0, 0.0)
        ), Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0)
        )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        Reward = 0.0
        Done = False

        self.target_pose = Point( self.star_pose.x, self.star_pose.y, self.star_pose.z * (1-timestamp/10) )

        State = np.array(
            [
                (pose.position.x), (pose.position.y), (pose.position.z),
                (pose.orientation.x), (pose.orientation.y), (pose.orientation.z), (pose.orientation.w),
                (self.target_pose.x), (self.target_pose.y), (self.target_pose.z), timestamp,
                (linear_acceleration.x), (linear_acceleration.y), (linear_acceleration.z)
            ]
        )

        Reward -= min(abs(pose.position.z - self.save_pose.z), 20)    +\
                  min(abs(pose.position.x - self.save_pose.x), 20)*0.1+\
                  min(abs(pose.position.y - self.save_pose.y), 20)*0.1

        self.save_pose = Point( pose.position.x, pose.position.y, pose.position.z * (1-timestamp/10) )

        if pose.position.z <= 0.2:
            Reward += timestamp
            Done = True

        if timestamp >= 11 or abs(pose.position.x) > 50 or abs(pose.position.y) > 50 or abs(pose.position.z) > 50:
            Reward -= 20
            Done = True

        action = self.agent.step(State, Reward, Done)
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)
            return Wrench(
                force=Vector3(action[0], action[1], action[2]),
                torque=Vector3(action[3], action[4], action[5])
            ), Done
        else:
            return Wrench(), Done