import sys
import numpy as np
import time

from sawyer_control.envs.mslac_peg import SawyerPegInsertionEnv

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
"""

##################################################################################
##################################################################################
##################################################################################


##########################################
# create env (does a reset inside there)
##########################################
env = SawyerPegInsertionEnv()


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
# change_freq = 10
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

"""