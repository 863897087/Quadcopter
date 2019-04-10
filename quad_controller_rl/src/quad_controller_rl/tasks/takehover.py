from quad_controller_rl.tasks.base_task import BaseTask
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from gym import spaces
import numpy as np

class HOVER(BaseTask):
    def __init__(self):
        self.observation_space = spaces.Box(
            np.array(
                [-150.0, -150.0,   0.0,
                 -150.0, -150.0, -10.0]
            ),
            np.array(
                [ 150.0,  150.0, 300.0,
                  150.0,  150.0, 290.0]
            )
        )

        self.action_space = spaces.Box(
            np.array( [-25.0, -25.0, -25.0, -25.0, -25.0, -25.0] ),
            np.array( [ 25.0,  25.0,  25.0,  25.0,  25.0,  25.0] )
        )

        self.target_pose = Point(0.0, 0.0, 10.0)

    def reset(self):
        return Pose(position=Point(0.0, 0.0, np.random.normal(10, 0.5)), orientation=Quaternion(0.0, 0.0, 0.0, 0.0)), \
               Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0))

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        Reward = 0.0
        Done = False
        State = np.array(
            [
                (pose.position.x),                      (pose.position.y),                      (pose.position.z),
                (pose.position.x - self.target_pose.x), (pose.position.y - self.target_pose.y), (pose.position.z - self.target_pose.z)
            ]
        )

        Reward -= \
            abs(pose.position.x - self.target_pose.x) + \
            abs(pose.position.y - self.target_pose.y) + \
            abs(pose.position.z - self.target_pose.z)

        if timestamp >= 5:
            Done = True

        action = self.agent.step(State, Reward, Done)
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)
            return Wrench(force=Vector3(action[0], action[1], action[2]), torque=Vector3(action[3], action[4], action[5])), Done
        else:
            return Wrench(), Done
