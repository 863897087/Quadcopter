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
                 0.0, 0.0, 0.0, 0.0]
            ),
            np.array(
                [0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0]
            )
        )

        self.action_space = spaces.Box(
            np.array([-25.0, -25.0, -25.0, -25.0, -25.0, -25.0]),
            np.array([ 25.0,  25.0,  25.0,  25.0,  25.0,  25.0])
        )

        self.target_pose = Point(0.0, 0.0, 10.0)
        self.down_time = 10

    def reset(self):
        #self.agent.LANDING()

        self.target_pose = Point(0.0, 0.0, np.random.normal(10, 0.5))
        return Pose(
            position=self.target_pose,
            orientation=Quaternion(0.0, 0.0, 0.0, 0.0)
        ), Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0)
        )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        Reward = 0.0
        Done = False

        self.target_pose.z = (self.target_pose.z / self.down_time) * (self.down_time - timestamp)

        State = np.array(
            [
                (pose.position.x), (pose.position.y), (pose.position.z),
                (pose.orientation.x), (pose.orientation.y), (pose.orientation.z), (pose.orientation.w),
                (self.target_pose.x), (self.target_pose.y), (self.target_pose.z), timestamp
            ]
        )

        Reward -= \
            min(abs(pose.position.x - self.target_pose.x), 20) + \
            min(abs(pose.position.y - self.target_pose.y), 20) + \
            min(abs(pose.position.z - self.target_pose.z), 20)
        Reward *= 0.1

        if pose.position.z <= 0.1:
            Done = True
            if self.down_time - timestamp <= 0:
                Reward += 20
            else:
                Reward -= 10

        if timestamp >= 15:
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