"""Generic base class for reinforcement learning agents."""

class BaseAgent:
    """Generic base class for reinforcement reinforcement agents."""

    def __init__(self, task):
        """Initialize policy and other agent parameters.

        Should be able to access the following (OpenAI Gym spaces):
            task.observation_space  # i.e. state space
            task.action_space
        """
        pass

    def step(self, state, reward, done):
        """Process state, reward, done flag, and return an action.

        Params
        ======
        - state: current state vector as NumPy array, compatible with task's state space
        - reward: last reward received
        - done: whether this episode is complete

        Returns
        =======
        - action: desired action vector as NumPy array, compatible with task's action space
        """
        raise NotImplementedError("{} must override step()".format(self.__class__.__name__))


    def LANDING(self):
        print("BaseAgent LANDING")

    def HOVER(self):
        print("BaseAgent HOVER")

    def Takeoff(self):
        print("BaseAgent Takeoff")

    def SetTakeoff(self):
        print("BaseAgent SetTakeoff")

    def SetTakehover(self):
        print("BaseAgent SetTakehover")

    def SetTakelanding(self):
        print("BaseAgent SetTakelanding")
