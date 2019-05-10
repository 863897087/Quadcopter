# **四轴飞行器控制器**

四轴飞行器学会飞行！

使用深度强化学习智能体，控制四轴飞行器执行飞行任务。

## 安装

本项目使用ROS(机器人操作系统)作为智能体和模拟器之间的主要沟通机制。安装ROS虚拟机。

## ROS虚拟机

压缩的VM镜像：[RoboVM_V2.1.0.zip](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DQN/RoboVM_V2.1.0.zip)

镜像系统登陆身份

​	用户名：`robond`

​	密码：`robo-nd`

安装虚拟机：[VMWare](http://www.vmware.com/)

配置虚拟机，至少需要2个处理器和4GB RAM内存。

## ROS镜像运行

在系统中打开终端`terminator`，如果系统提示`Do you want to source ROS`，选择`y`。

## 项目目录

`catkin_ws`的目录是 [catkin workspace](http://wiki.ros.org/catkin/workspaces)，使用它来管理和进行所有基于`ROS`的项目 。

```terminator
catkin_ws/
	src/
		RL-Quadcopter/
	build/
	devel/
```

`RL-Quadcopter`的目录是`git`保存目录。其中`requirement.txt`是环境需要的库，使用`pip3`安装。其中`quad_controller_rl`是`ROS`工程文件夹。

```terminator
RL-Quadcopter/
	quad_controller_rl/
	README.md
	requirements.txt
	.git
	.gitignore
```

`quad_controller_rl`是`ROS`工程文件夹。`launch`目录中保存启动脚本。`src`目录中保存智能体项目。

```terminator
quad_controller_rl/
	.idea/
	launch/
	notebooks/
	out/
	scripts/
	sim/
	src/
		quad_controller_rl
	srv/
	CMakeLists.txt
	package.xml
	setup.py
```

`quad_controller_rl`目录是任务和智能体的保存目录。

```terminator
quad_controller_rl/
	.idea/
	agents/
	tasks/
	__pycache__/
	__init__.py
	util.py
```

`agents`目录是智能体目录。`base_agent`智能体基类。`__init__`子类列表。`policy_gradients`训练模型。`policy_search`实例模型。`modle_agent`加载训练好的模型和参数。

```terminator
agents/
	__pycache__/
	base_agent.py
	__init__.py
	policy_search.py
	policy_gradients.py
	modle_agent.py
```

`tasks`目录是任务目录。`takeoff`起飞任务。`takehover`悬停任务。`takelanding`降落任务。`takeall`执行所有任务。

```terminator
tasks/
	__pycache__/
	base_task.py
	__init__.py
	takeoff.py
	takehover.py
	takelanding.py
	takeall.py
```

## 环境搭建

安装`pip3`:

```terminator
$ sudo apt-get update
$ sudo apt-get -y install python3-pip
```

安装项目所需的`Python`包：

```terminator
$ pip3 install -r requirements.txt
$ pip3 install TensorFlow=1.10.0
```

## 模拟器

为你的主机 OS 在[这里](https://github.com/udacity/RoboND-Controls-Lab/releases)下载优达学城四轴飞行器模拟器，它的昵称为 **DroneSim**。

要想打开模拟器，你只需运行下载的可执行文件即可。你也许需要在运行部分介绍的 `roslaunch` 步骤_之后_打开模拟器，便于将它连接至正在运行的 ROS master。

*请注意：如果你使用的是虚拟机（VM），你无法在 VM 内运行模拟器。你需要在**主机操作系统**中下载并运行模拟器，再将它连接至虚拟机（见下文）*

### 将模拟器连接至 VM

如果你在虚拟机内运行 ROS，你需要通过几个步骤来保证它能与主机系统中的模拟器连接。如果你没有使用虚拟机，可以忽略这些步骤。

#### 在 VM 中允许网络连接

- **VMWare**：使用默认设定即可。为了验证，你可以在运行虚拟机的情况下，打开虚拟机的菜单 > 网络适配器。NAT 一栏应被勾选。

#### 为主机和虚拟机获取 IP 地址

在主机终端中运行 `ifconfig`。这将显示所有可用的网络接口，包括物理接口和虚拟接口。其中应有名为 `vmnet` 或 `vboxnet` 的接口。请记下该接口的 IP 地址（`inet` 或 `inet addr`），比如 `192.168.56.1`，这是你的**主机 IP 地址**。

请在虚拟机内重复该步骤。在这里接口名称也许有所不同，但 IP 地址的前缀相同。请记下完整的 IP 地址，比如 `192.168.56.101`，这是你的**虚拟机 IP 地址**。

#### 编辑模拟器设定

在模拟器的 `_Data` 或 `/Contents` 文件夹内（在 Mac中请右键点击 app > 显示包目录），编辑 `ros_settings.txt`：

- 将 `vm-ip` 设置为 **虚拟机 IP 地址**，并将 `vm-override` 设置为 `true`。
- 将`host-ip` 设置为 **主机 IP 地址**，并将 `host-override` 设置为 `true`。

主机和/或虚拟机的 IP 地址可以在重启时改变。如果你遇到任何连接问题，请确保实际的 IP 地址与 `ros_settings.txt` 中的一致。

## 项目运行

在虚拟机中使用[ROS镜像运行](#ROS镜像运行)。之后执行以下代码，了解`~/`的详细信息，将`catkin_ws`文件夹存到虚拟机中`~/`文件夹下。

```terminator
cd ~/
pwd
```

将`lastoff`文件夹、`lasthover`文件夹、`lastlanding`文件存在`~/.ros/`文件。注意`.ros`是隐藏文件夹

```terminator
cd ~/.ros/
```

建立`ROS`节点

```terminator
cd ~/catkin_ws/
catkin_make
```

启动命令行`tab`补全功能和其它实用的`ROS`应用

```terminator
source devel/setup.bash
```

![ROS节点](.\picture\catkin_make.png)

打开运行目录

```terminator
cd ~/catkin_ws/src/RL-Quadcopter/quad_controller_rl/launch
```

首先运行默认`task`和`agent`，看`ROS`系统和模拟器之间的链接是否正常。默认的`task`文件是`takeoff.py`。默认的`agent`文件是`policy_search.py`。`ros`等待模拟器的链接、运行。

```terminator
roslaunch quad_controller_rl rl_controller.launch
```

![normal test](.\picture\run_test.png)

打开模拟器运行成功

![Runlog](.\picture\connect_test.png)

运行`takeoff.py`任务和`policy_gradients.py`智能体。训练起飞模型。存储训练的模型固定路径是`~/.ros/modle`。

```terminator
roslaunch quad_controller_rl rl_controller.launch \
agent:=DDPG task:=Takeoff
```

![takeoff](.\picture\takeoff_run.png)

打开模拟器运行成功

![takeoff](.\picture\takeoff_log.png)

运行`takeoff.py`任务和`modle_agent.py`智能体。加载训练完成的起飞模型。加载模型的固定路径是`~/.ros/lastoff`。

```terminator
roslaunch quad_controller_rl rl_controller.launch \
agent:=MODLE task:=Takeoff
```

![takeoff_modle](.\picture\takeoff_modle.png)

打开模拟器运行成功

![takeoff_modle](.\picture\takeoff_modle_log.png)

运行`takehover.py`任务和`policy_gradients.py`智能体。训练悬停模型。存储训练的模型固定路径是`~/.ros/modle`。

```terminator
roslaunch quad_controller_rl rl_controller.launch \
agent:=DDPG task:=HOVER
```

![takehover](.\picture\takehover_modle.png)

打开模拟器运行成功

![takehover](.\picture\takehover_log.png)

运行`takehover.py`任务和`modle_agent.py`智能体。加载训练完成的悬停模型。加载模型的固定路径是`~/.ros/lasthover`。

```terminator
roslaunch quad_controller_rl rl_controller.launch \
agent:=MODLE task:=HOVER
```

![takehover_modle](.\picture\takehover_run.png)

打开模拟器运行成功

![takehover_modle](.\picture\takehover_modle_log.png)

运行`takelanding.py`任务和`policy_gradients.py`智能体。训练降落模型。存储训练的模型固定路径是`~/.ros/modle`。

```terminator
roslaunch quad_controller_rl rl_controller.launch \
agent:=DDPG task:=LANDING
```

![takelanding](.\picture\takelanding.png)

打开模拟器运行成功

![takelanding](.\picture\takelanding_log.png)

运行`takelanding.py`任务和`modle_agent.py`智能体。加载训练完成的降落模型。加载模型的固定路径是`~/.ros/lastlanding`。

```terminator
roslaunch quad_controller_rl rl_controller.launch \
agent:=MODLE task:=LANDING
```

![takelanding_modle](.\picture\takelanding_modle.png)

打开模拟器运行成功

![takelanding_modle](.\picture\takelanding_modle_log.png)

运行`takeall.py`任务和`modle_agent.py`智能体。加载训练完成的所有模型。加载模型的固定路径是`~/.ros/lastoff`、`~/.ros/lasthover`、`~/.ros/lastlanding`。

```terminator
roslaunch quad_controller_rl rl_controller.launch \
agent:=MODLE task:=ALL
```

![takeall](.\picture\takeall_modle.png)

打开模拟器运行成功

![takeall](.\picture\takeall_modle_log.png)

执行模型训练的过程中可以观察`critic_Q`值、`critic_cost`值、`critic_gradient`值和`actor_cost`值。

```terminator
cd ~/Desktop/
tensorboard --logdir summary_critic/
tensorboard --logdir summary_actor/
```

![tensorboard](.\picture\tensorboard_log.png)

