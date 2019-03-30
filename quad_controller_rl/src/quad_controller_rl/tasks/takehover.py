from quad_controller_rl.tasks.base_task import BaseTask
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench

class HoverTask(BaseTask):
    
    def __init__(self):
        pass

    def reset(self):
        return Pose(), Twist()

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        Done = False
        return Wrench(), Done
