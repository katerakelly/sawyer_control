# Sawyer Control
## Authors:
Originally written by Murtaza Dalal and Shikhar Bahl

## Description
Sawyer Control is a repository that enables RL algorithms to control Rethink Sawyer robots via an OpenAI Gym Style interface. It is both a ROS package and a set of Sawyer (Gym) Envs combined in one. The ROS portion of the repo handles the actual control of the robot and is executed in Python 2. The environments are all in Python 3, and communicate to the robot via the ROS interface. Currently, this repo is capable of utilizing both the state information of the robot as well as visual input from a Microsoft Kinect sensor.

## Setup Instructions:
1. Make sure ros kinetic is installed and make sure to add source /opt/ros/kinetic/setup.bash to your bashrc
2. Install intera interface from the rethink website and set up intera.sh with the correct ip and hostname of your robot
3. Git clone the following in ~/catkin_ws/src/:
* Urdfdom: https://github.com/ros/urdfdom.git
* urdf_parser_py: https://github.com/ros/urdf_parser_py
* pykdl utils: https://github.com/gt-ros-pkg/hrl-kdl
4. switch to the indigo-devel branch on urdf_parser_py and hrl_kdl
5. run `git clone https://github.com/mdalal2020/sawyer_control.git` in ~/catkin_ws/src/
6. run `catkin_make`
7. Make sure you are on system python
8. run `pip install -r system_python_requirements.txt`
9. install anaconda 2 (Do not install anaconda 3!) and type no when it asks you prepend the anaconda path to the bashrc
10. manually add in the anaconda path to the bashrc (see example bashrc below)
11. run `conda create -n <env_name> python=3.5 anaconda`
12. source activate your python 3 conda environment
13. run `pip install -r python3_requirements.txt`
14. install kinect2 bridge: https://github.com/code-iai/iai_kinect2/tree/master/kinect2_bridge

Example Bashrc:
```
source /opt/ros/kinetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export PATH="$PATH:$HOME/anaconda2/bin"
export PYTHONPATH=$PYTHONPATH:/opt/ros/kinetic/lib/python2.7/dist-packages/:
```
Useful aliases:
```
alias saw="cd ~/catkin_ws/; ./intera.sh; cd ~/"
alias enable="rosrun intera_interface enable_robot.py -e"
alias disable="rosrun intera_interface enable_robot.py -d"
alias reset="rosrun intera_interface enable_robot.py -r"
alias stop="rosrun intera_interface enable_robot.py -S"
alias status="rosrun intera_interface enable_robot.py -s"
alias exp_nodes="roslaunch ~/catkin_ws/src/sawyer_control/exp_nodes.launch"
alias kinect="roslaunch kinect2_bridge kinect2_bridge.launch"
```

## Environments
All environments inherit from `SawyerEnvBase` which contains all of the core functionality that is central to using the Sawyer Robot including different control modes, robot state observations, and safety boxes. Note, unless you have a good reason not to, you should ALWAYS have the safety box enabled, ie set `use_safety_box=True`. This environment also provides functionality for changing various settings (see configs section) as well as having a fixed goal (as opposed to the default functionality, which is multi-goal).

There are two main control modes for the Sawyer robot, torque and position, each of which has differing functionality and settings. In terms of the actual controller, all of the actions are executed using intera's in built torque/joint position controller, and the environment merely provides an abstraction around this. For the torque control mode, changing around the `torque_action_scale` will be quite important (generally larger values around 5 or so are better) and the control frequency is 20Hz. The position control mode in the environment does end-effector position control, an option that intera does not provide. As a result, given a desired end-effector position, the environment uses inverse kinematics to compute the corresponding joint angles, and commands the intera impedance controller to move to those angles. The position controller can go as fast you like (by setting the `max_speed` option in `SawyerEnvBase`). Setting higher max speeds will result in a loss of accuracy, so I would recommend testing each speed on a variety of different reaching tasks to see what level of accuracy works for you. In general, I've found anything between .1-.4 works pretty well if your action magnitudes are up to 1cm. You can set this option using `position_action_scale` in `SawyerEnvBase`.

Currently the repository holds two main environments, `SawyerReachXYZEnv` and `SawyerPushXYEnv`. For `SawyerReachXYZEnv` the goal is to reach a target end-effector position. For `SawyerPushXYEnv` the goal is to push an object to a desired position. Since the environment is in the real world, for which we don't have access to the state information of the object to push, you must wrap this environment with `ImageEnv`. See the usage section for how to to this.

## Usage:

### Basic workflow:

1. Make sure robot is enabled

2. open a new terminal and run the following commands:
```
saw
exp_nodes
```
3. (Optional) open another terminal/tab and run the following commands:
```
saw
kinect
```
3. Now open another terminal/tab and run your algorithm on the sawyer. Note you must run `saw` in any new tab that accesses environments/scripts in `sawyer_control`.

### Using Environments

To import the reaching or pushing environment:

```
from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
env = SawyerReachXYZEnv(...)
env.step(action=...)
```

To use the `ImageEnv` wrapper (which is compatible with any SawyerEnv):
```
from sawyer_control.envs.sawyer_pushing import SawyerPushXYEnv
from sawyer_control.core import ImageEnv
env = SawyerPushXYEnv(...)
image_env = ImageEnv(env, ...)
image = env.get_flat_image()
```

### Configs:
All the important/hardcoded settings for robot/env details are stored in the config files. Please do not change the ros_config or base_config files. If you wish to modify settings, make a new configuration file and have it import all the standard configs and modify the rest, see `austri_config.py` for an example. Then add the file to the config dictionary in `config.py` and simply pass in the name to the env to obtain the desired settings.

KATE 12/30: I ignored this (this is what git branches are for!) and did edit these configs.

### Common Problems:
Q: Can't communicate with master node?

A: Check that you ran `saw` in the terminal before running your experiment script

Q: The robot doesn't move!

A: Double check that you ran `exp_nodes` before running the experiment

Q: I ran `exp_nodes` and the robot still doesn't move!

A: Run `status` and check if `Ready=False`. If so the robot is in homing mode (Try pressing the grey button on the robot and moving the arm around, alternatively you might just have to re-start until this error goes away. Unfortunately, I don't have a consistent fix for this).

Q: I changed something in this repo and it doesn't seem to have an effect.

A: Did you change something that touches ROS? e.g. a config file, a message etc...then you need to run `catkin_make` in `~/ros_ws`!!

Q: The arm is just moving upwards all the time in torque control mode.

A: You probably need to up the `torque_action_scale`, I recommend something like 4/5, but it depends on how safe you want the environment and how constrained your space is. The problem occurs because you are applying too small torques, so the solution is to apply larger torques.


## Features:
* Torque and Position Control Modes
* End-Effector Reaching Environment
* Pushing Environment
* Vision Wrapper
* Gym-Style Interface

## Understanding the Code (KATE 12/30)
Make sure you are on the branch `mslac`.

The way the current controller works is that the policy outputs delta *end-effector* positions, which are converted to target joint angles via IK.
The controller takes these joint angles as waypoints and fits a spline to them.
At a fixed rate, the policy provides a new waypoint, and the trajectory is updated.
The control commands are sent to the robot as joint velocities.

Here are the relevant files:
* `base_config.py` and `ros_config.py` - a bunch of config settings like safety box and reset configuration. I edited these to have defaults that work for vestri. NOTE: the robot needs to be moved backward on the pedestal and so these will need to be changed! Bug Glen about the bolts for the Sawyer. It scares me that $45k is relying on a couple of clamps.
* `sawyer_peg.py` - this is the env that should be wrapped by an mslac env. there is a lot of optional stuff in here to do full 6dof control on the end-effector, but for right now, it's just 3-dof (the orientation is fixed to always point straight down). In the past, I've had some trouble getting 6-dof to work. The current observation given to the policy is the endpoint pose (pos + quat) as well as the endpoint velocity. The current action space is simply end-effector position.
* `sawyer_env_base.py` - the base environment. Has a lot (too much) stuff in it, but the most important methods are `request_observation()` which gets the observations from the ROS message, and `request_angle_action()` which calls ROS to execute the action.
* `angle_action_server.py` - this is where the `execute_action()` function is defined. It calls the controller with the action for each joint.
* `velocity_controller.py` - this is where the controller is written. Note that `control_rate` is the rate at which the underlying Sawyer controller is called, and that `update_rate` is the rate of the policy. It is implemented so that control is fixed frequency (currently 20Hz) but reset can take as long as needed. There is a safety feature that if forces are too high, it will stop.

To work out bugs with this code I was using a little test script in `~/test_sawyer/test.py`.

Existing issues:
* I think the safety box is slightly messed up, there are some action sequences that generate very weird IK (e.g. try giving it all positive 1s as actions).
* The controller "finishes" its trajectory at the very end of the episode - I guess the policy stops sending stuff but the ROS process is still running. This seems like a problem because the robot moves quite a lot at the end of the trajectory to fully finish the last action. Maybe it doesn't matter for the learning since it's the last timestep, but just physically I think it's going to be a problem.
* The robot needs to be moved back and bolted down. We need to get a wooden plank or bin to put in front of the robot.
* The control is only 3-DoF right now - probably ideally we want full 6DoF?
