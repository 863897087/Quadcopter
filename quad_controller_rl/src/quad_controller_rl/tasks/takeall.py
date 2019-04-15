from quad_controller_rl.tasks.base_task import BaseTask
from quad_controller_rl.tasks.takeoff import Takeoff
from quad_controller_rl.tasks.takehover import HOVER
from quad_controller_rl.tasks.takelanding import LANDING
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from gym import spaces
import numpy as np

class ALL(BaseTask):
    def __init__(self):
        self.Takeoff_En = True
        self.Takeoff_Number = Takeoff()

        self.HOVER_En = False
        self.HOVER_Number = HOVER()

        self.LANDING_En = False
        self.LANDING_Number = LANDING()

        self.action_space = spaces.Box(
            np.array([-25.0, -25.0, -25.0, -25.0, -25.0, -25.0]),
            np.array([25.0, 25.0, 25.0, 25.0, 25.0, 25.0])
        )

    def reset(self):

        if self.Takeoff_En is True:
            print("reset takeoff start")
            self.agent.Takeoff()
            self.Takeoff_Number.set_agent(self.agent)
            return self.Takeoff_Number.reset()

        if self.HOVER_En is True:
            print("reset hover start")
            self.agent.HOVER()
            self.HOVER_Number.set_agent(self.agent)
            return self.HOVER_Number.reset()

        if self.LANDING_En is True:
            print("reset landing start")
            self.agent.LANDING()
            self.LANDING_Number.set_agent(self.agent)
            return self.LANDING_Number.reset()

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):

        if self.Takeoff_En is True:

            Wrench_number, self.HOVER_En = self.Takeoff_Number.update(timestamp, pose, angular_velocity, linear_acceleration)
            if self.HOVER_En is True:
                self.Takeoff_En = False
                self.LANDING_En = False

            print("update start HOVER {}".format(self.HOVER_En))
            return Wrench_number, self.HOVER_En

        if self.HOVER_En is True:

            Wrench_number, self.LANDING_En = self.HOVER_Number.update(timestamp, pose, angular_velocity, linear_acceleration)
            if self.LANDING_En is True:
                self.Takeoff_En = False
                self.HOVER_En = False

            print("update start LANDING {}".format(self.LANDING_En))
            return Wrench_number, self.LANDING_En

        if self.LANDING_En is True:

            Wrench_number, self.Takeoff_En = self.LANDING_Number.update(timestamp, pose, angular_velocity, linear_acceleration)
            if self.Takeoff_En is True:
                self.HOVER_En = False
                self.LANDING_En = False

            print("update start TAKEOFF {}".format(self.Takeoff_En))
            return Wrench_number, self.Takeoff_En



