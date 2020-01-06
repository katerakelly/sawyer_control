import sys
import numpy as np
import time

from sawyer_control.src.sawyer_control.envs.mslac_reacher import MslacReacherEnv
from sawyer_control.src.sawyer_control.envs.mslac_peg import MslacPegInsertionEnv

##################################################################################
##################################################################################
##################################################################################

"""
Instrucs for running:

1) turn on robot
2) terminal A:
	saw
	enable
	exp_nodes
3) terminal B:
	saw
	conda activate sawyer_control
	cd src/sawyer_control/src/sawyer_control/envs/test_sawyer
		<this should load a new pythonpath that allows cv etc to be found>
	python test.py


Possible robot errors:

-If ready=False, robot is in homing mode. 
You need to open the e-stop and press the grey button on the robot's ee,
and move the arm a little bit (manually). 
Also consider restarting if things aren't working...

-If you already release the estop but it says estop_button: 3,
you just need to do `reset`. Then, after enable, 
you should see estep_button:0 and ready=True.


Relevant code files:
	envs
		mslac_reacher.py
		mslac_peg.py
		sawyer_env_base.py
	ros nodes
		scripts/angle_action_server.py
		pd_controllers/velocity_controller.py
		exp_nodes.launch

"""

##################################################################################
##################################################################################
##################################################################################


##########################################
# create env (does a reset inside there)
##########################################
env = MslacPegInsertionEnv()


##########################################
# test it by moving it yourself (no scaling/etc.)
# this just sends desired joint angle positions directly to robot
##########################################
# import IPython
# IPython.embed()
# test_angles = env.reset_joint_positions.copy()
# test_angles[0]-=0.1
# env._act(test_angles)


##########################################
# test by sending commands in the new -1 to 1 action range
# here, 0 is the reset position, -1 is low and 1 is high for each joint
##########################################
# import IPython
# IPython.embed()
# env.step(np.array([0,0,0,0,0,0,0]))
# env.step(np.array([0.1,0,0,0,0,0,0]))
# env.step(np.array([-0.1,0,0,0,0,0,0]))


##########################################
# test random commands
##########################################

# start_time = time.time()
# num_steps = 100
# change_freq = 5
# for i in range(num_steps):
# 	if i%change_freq==0:
# 		ac = np.random.uniform(env.action_lows, env.action_highs)
# 	print("trying ac: ", ac)
# 	env.step(ac)

# print("\n\nTime taken: ", time.time()-start_time)
# print("Should have taken: ", num_steps*env.timestep)

##########################################
# test random SMOOTH commands
##########################################

start_time = time.time()
num_steps = 100
change_freq = 10
ac = np.zeros((7,))
for i in range(num_steps):
	if i%change_freq==0:
		vel = np.random.uniform(env.action_lows/3, env.action_highs/1.5)
		sign = 1
		if np.random.randint(2)>0:
			sign = -1
	ac = ac + sign*vel
	print("trying ac: ", ac)
	env.step(ac)

print("\n\nTime taken: ", time.time()-start_time)
print("Should have taken: ", num_steps*env.timestep)



##################################################################################
##################################################################################
##################################################################################

##########################################
# UNDERSTANDING HOW THE ROBOT MOVES 
##########################################

"""
0: rotate body left
	0.4 straight ahead
	limits: 0 to the right, 0.3

1: lift shoulder down
	0 is horizontal
	limits: -0.9 up, -0.7 down

2: rotate shoulder toward down toward side of arm
	limits: -0.8 rotated upward toward the other/missing hand, -0.1 rotated down toward side (ee down)

3: lift forearm down
	limits: 1.6 right-angle with shoulder, 2.1 down and 45-deg angle with shoulder

4: rotate forearm clockwise
	0 is buttons pointing forward
	limits: 0.6 hand sweep to L, to 2.7 hand sweep to R

5: rotate hand clockwise (toward robot's left)
	0 is in line with forearm
	limits: -0.7 point to robot's right, to 0.7 pointing to robot's left

6: rotate ee clockwise (black thing aligned with back of hand)
	limits: -1.5 rotate L and black thing goes toward robot R, 1.5 rotate R and black thing goes toward robot L

#limits right now:
lows: 0, -0.9, -0.8, 1.6, 0.6, -0.7, -1.5
highs: 0.3, -0.7, -0.1, 2.1, 2.7, 0.7, 1.5

#testing makes these limits seem fine. 
a) It can't hit the bottom of the cage
	1 shoulder all the way down (-0.7)
	2 shoulder rotated all the way down toward side of body (-0.1)
	3 forearm all the way down (2.1)
	5 hand/ee pointed down to lowsest point in its arc(0)
	4 swept in both directions while here
b) It can't hit the left of cage
	0 rotate body all the way left (0.3)
	2 shoulder rotated all the way down toward side of body (-0.1)
	3 lift forearm up (1.6)
c) I don't think it can hit the right of cage
	0 rotate body all the way right (0)
	2 shoulder rotated all the way up toward R side (-0.8)
	1 shoulder all the way down (-0.7) 
	1 shoulder all the way up (-0.9)
d) It cannot reach upward at all
e) It cannot rotate backward




############

Example topic outputs:

rostopic echo -n 1 /robot/state
  ready: True
  enabled: True
  stopped: False
  error: False
  lowVoltage: False
  estop_button: 0
  estop_source: 0

rostopic echo -n 1 /robot/joint_states
  name: 
  - head_pan
  - right_j0
  - right_j1
  - right_j2
  - right_j3
  - right_j4
  - right_j5
  - right_j6
  - torso_t0
  position: [0.5597392578125, -0.5550078125, -1.070587890625, -0.1909248046875, 2.3332509765625, 0.3215478515625, 0.3741708984375, -1.0871748046875, 0.0]
  velocity: [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, 0.0]
  effort: [0.052, -0.424, -15.2, -1.672, -5.584, 0.304, 0.168, 0.052, 0.0]


rostopic echo -n 1 /robot/limb/right/endpoint_state
	pose: 
	  position: 
	    x: 0.589071476483
	    y: -0.233236801619
	    z: 0.47302967698
	  orientation: 
	    x: 0.277159573158
	    y: 0.833932483173
	    z: -0.298555951573
	    w: 0.372294947986
	twist: 
	  linear: 
	    x: -0.000947022574253
	    y: -0.00121524077737
	    z: -0.000249181715562
	  angular: 
	    x: 0.00200475881992
	    y: -0.00200965631648
	    z: -0.000410425661771
	wrench: 
	  force: 
	    x: -2.51468313214
	    y: 3.44215142101
	    z: 2.52177747361
	  torque: 
	    x: 0.534232319021
	    y: 0.511620720856
	    z: -0.142879202892
	valid: True
"""