from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box
import gym
import os

from pyquaternion import Quaternion
from sawyer_control.coordinates import quat_2_euler, euler_2_rot, euler_2_quat
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable


class MslacPegInsertionEnv(SawyerEnvBase):

    '''
    Inserting a peg into a box (which is at a fixed location)
    '''

    def __init__(self, *args, **kwargs):

        # init
        Serializable.quick_init(self, locals())
        SawyerEnvBase.__init__(self, **kwargs)

        # params
        self.timestep = 0.1 # this env will be running at 1/x Hz
        self.im_width = 32
        self.im_height = 32
        self.num_joint_dof = 7

        # note: this env is currently written only for this mode
        assert self.action_mode=='joint_position'

        # limits
        joint_vel_limit = 0.07 #deg/sec
        self.limits_joint_vel = np.array([joint_vel_limit]*self.num_joint_dof)
        self.limits_lows_joint_pos = np.array([0, -0.9, -0.8, 1.6, 0.6, -0.7, -1.5])
        self.limits_highs_joint_pos = np.array([0.3, -0.7, -0.1, 2.1, 2.7, 0.7, 1.5])
        self.limits_lows_ee_pos = -1*np.ones(3)
        self.limits_highs_ee_pos = np.ones(3)
        self.limits_lows_ee_angles = -1*np.ones(4)
        self.limits_highs_ee_angles = np.ones(4)

        # set obs/ac spaces
        self._set_action_space()
        self._set_observation_space()

        # ranges
        self.action_range = self.action_highs-self.action_lows
        self.joint_range = (self.limits_highs_joint_pos - self.limits_lows_joint_pos)  

        # reset position (note: this is the "0" of our new (-1,1) action range)
        self.reset_joint_positions = self.limits_lows_joint_pos + self.joint_range/2.0
        self.reset_duration = 6.0 # seconds to allow for reset

        # goal pose is when the peg is correctly inserted
        # TODO just made this goal up
        self.goal_ee_position = np.array([0.45, 0.1, 0.23])
        self.goal_ee_orientation = np.array([1, 0, 0, 0])
        self.goal_ee = np.concatenate((self.goal_ee_position, self.goal_ee_orientation))

        # reset robot to initialize
        self.reset()

    ####################################
    ####################################

    def _set_observation_space(self):
        ''' [14] : observation is [7] joint angles + [3] ee pos + [4] ee angles '''
        self.obs_lows = np.concatenate((self.limits_lows_joint_pos, self.limits_lows_ee_pos, self.limits_lows_ee_angles))
        self.obs_highs = np.concatenate((self.limits_highs_joint_pos, self.limits_highs_ee_pos, self.limits_highs_ee_angles))
        self.observation_space = Box(self.obs_lows, self.obs_highs, dtype=np.float32)

    def _set_action_space(self):
        ''' [7] : actions are desired positions for each joint '''
        self.action_lows = -1 * np.ones(self.num_joint_dof)
        self.action_highs = np.ones(self.num_joint_dof)
        self.action_space = Box(self.action_lows, self.action_highs, dtype=np.float32)

    ####################################
    ####################################

    def _get_image(self):
        # TODO haven't integrated this function with the mslac codebase yet
        return self.get_image(self.im_width, self.im_height)

    def _get_obs(self):
        ''' [7] joint angles + [7] ee pose '''
        angles = self._get_joint_angles()
        ee_pose = self._get_endeffector_pose()
        return np.concatenate([angles, ee_pose])

    ####################################
    ####################################

    def compute_rewards(self, obs, action=None):
        # TODO
        return 0 ##-np.linalg.norm(obs[:3] - self.goal_pose[:3])

    def step(self, action):

        # convert from given (-1,1) to joint positions (low,high)
        action = np.clip(action, -1, 1)
        action_scaled = (((action - self.action_lows) * self.joint_range) / self.action_range) + self.limits_lows_joint_pos

        # enforce joint velocity limits on this scaled action
        feasible_scaled_action = self.make_feasible(action_scaled)

        # TODO safety box?

        # take a step
        self._act(feasible_scaled_action, self.timestep)

        # get updated observation
        obs = self._get_obs()

        # reward/etc. after taking the step
        reward = self.compute_rewards(obs)
        done = False
        info = self._get_info()
        return obs, reward, done, info

    def make_feasible(self, desired_positions):
        # get current positions
        curr_positions = self._get_joint_angles()

        # compare the implied vel to the max vel allowed
        max_vel = self.limits_joint_vel*self.timestep
        implied_vel = np.abs(desired_positions-curr_positions)

        # limit the vel 
        actual_vel = np.min([implied_vel, max_vel], axis=0)

        # find the actual position, based on this vel
        sign = np.sign(desired_positions-curr_positions)
        actual_difference = sign*actual_vel
        feasible_positions = curr_positions+actual_difference
        return feasible_positions

    def reset(self):
        # reset the arm to these positions
        self._act(self.reset_joint_positions, self.reset_duration, reset=True)

        # return the observation
        return self._get_obs()