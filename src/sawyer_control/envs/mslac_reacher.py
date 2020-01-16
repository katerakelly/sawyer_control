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

import cv2
import copy
import rospy
from sawyer_control.srv import image

class MslacReacherEnv(SawyerEnvBase):

    '''
    Reach to a fixed ee goal position
    '''

    def __init__(self, *args, **kwargs):

        # init
        Serializable.quick_init(self, locals())
        SawyerEnvBase.__init__(self, **kwargs)

        # params
        self.timestep = 0.25 # this env will be running at 1/x Hz
        self.num_joint_dof = 7

        # note: this env is currently written only for this mode
        assert self.action_mode in ['joint_position', 'joint_delta_position']

        # vel limits
        joint_vel_lim = 0.07 #deg/sec
        self.limits_lows_joint_vel = -np.array([joint_vel_lim]*self.num_joint_dof)
        self.limits_highs_joint_vel = np.array([joint_vel_lim]*self.num_joint_dof)

        # position limits (tight)
        # self.limits_lows_joint_pos = np.array([0, -0.9, -0.8, 1.6, 0.6, -0.7, -1.5])
        # self.limits_highs_joint_pos = np.array([0.3, -0.7, -0.1, 2.1, 2.7, 0.7, 1.5])

        # position limits (tight) with peg/camera attached
        # self.limits_lows_joint_pos = np.array([0, -0.9, -0.8, 1.6, 0.6, -0.7, -1.5])
        # self.limits_highs_joint_pos = np.array([0.3, -0.8, -0.3, 2.1, 2.7, 0.7, 1.5])

        # position limits (loose, need safety box!)
        self.limits_lows_joint_pos = np.array([0, -1, -1, 1.3, 0.6, -0.7, -1.5])
        self.limits_highs_joint_pos = np.array([0.4, -0.6, 0, 2.5, 2.7, 0.7, 1.5])

        # safety box (calculated for our current single-task peg setup)
        self.safety_box_ee_low = np.array([0.36,-0.26,0.196])
        self.safety_box_ee_high = np.array([0.8,0.26,0.5])

        # ee limits
        self.limits_lows_ee_pos = -1*np.ones(3)
        self.limits_highs_ee_pos = np.ones(3)
        self.limits_lows_ee_angles = -1*np.ones(4)
        self.limits_highs_ee_angles = np.ones(4)

        # set obs/ac spaces
        self._set_action_space()
        self._set_observation_space()

        # ranges
        self.action_range = self.action_highs-self.action_lows
        self.joint_pos_range = (self.limits_highs_joint_pos - self.limits_lows_joint_pos)
        self.joint_vel_range = (self.limits_highs_joint_vel - self.limits_lows_joint_vel)

        # reset position (note: this is the "0" of our new (-1,1) action range)
        self.reset_joint_positions = self.limits_lows_joint_pos + self.joint_pos_range/2.0
        self.reset_duration = 4.0 # seconds to allow for reset

        # goal ee pose
        """
        x: 0.417594278477
        y: -0.224770123311
        z: 0.314778205152
        """
        self.goal_ee_position = np.array([0.418, -0.225, 0.315])

        # reset robot to initialize
        self.reset()

        # Added by Tony

        self.sparse_reward = False  # TODO Tony: hardcoded. Also not expecting to run sparse reward on robot

    ####################################
    ####################################

    # added by Tony
    def override_action_mode(self, mode):
        assert mode in ['joint_position', 'joint_delta_position']
        self.action_mode = mode
        print("ACTION MODE: ", self.action_mode)

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

    def _get_image(self, width, height, double_camera):
        if double_camera:
            image =  self.get_double_image(output_width=width, output_height=height)
        else:
            image = self.get_image(width=width, height=height)
        return image

    def get_double_image(self, output_width=84, output_height=84): # override base_env, for double camera
        double_image = self.request_double_image()


        input_w = 480*2 # input width
        input_h = 640 # input height

        if (double_image is None):
            raise Exception('Unable to get double image(s) from image server')
        double_image = np.array(double_image).reshape((input_w, input_h, 3))
        double_image = double_image[:, :int(input_w/2), :]  # crop such that width:high = 2:1: make height: 640->480

        processed_w = input_w
        processed_h = input_w/2

        double_image = copy.deepcopy(double_image)
        double_image = cv2.resize(double_image, (0, 0), fx=output_width/processed_w, fy=output_height/processed_h, interpolation=cv2.INTER_AREA)
        double_image = np.asarray(double_image).reshape((output_width, output_height, 3))[:, :, ::-1]
        return double_image # np.expand_dims(combined_image, axis=0)

    def request_double_image(self):
        rospy.wait_for_service('images_webcam_double')
        try:
            request = rospy.ServiceProxy('images_webcam_double', image, persistent=True)
            obs = request()
            return (
                    obs.image
            )
        except rospy.ServiceException as e:
            print(e)

    def _get_obs(self):
        ''' [7] joint angles + [7] ee pose (3 position, 4 angle quaternion)'''
        angles = self._get_joint_angles()
        ee_pose = self._get_endeffector_pose()
        return np.concatenate([angles, ee_pose])

    ####################################
    ####################################

    def compute_rewards(self, obs, action=None):
        ee_xyz = obs[self.num_joint_dof:self.num_joint_dof+3]
        goal_xyz = self.goal_ee_position

        # distance between the points
        score = np.linalg.norm(ee_xyz - goal_xyz)

        #print("in hardware getting reward/score ", score, "TODO: make sure number is reasonable")

        dist = 5*score

        assert self.sparse_reward is False # for now

        # # Sparse reward setting
        # if self.sparse_reward:
        #     dist = min(dist, self.truncation_dist) # if dist too large: return the reward at truncate_dist

        # use GPS cost function: log + quadratic encourages precision near insertion
        reward = -(dist ** 2 + math.log10(dist ** 2 + 1e-5))

        # if self.sparse_reward:
        #     # offset the whole reward such that when dist>truncation_dist, the reward will be exactly 0
        #     reward = reward - (-(self.truncation_dist ** 2 + math.log10(self.truncation_dist ** 2 + 1e-5)))

        return reward

    def step(self, action):

        if self.action_mode=='joint_position':

            # clip incoming action
            desired_joint_positions = np.clip(action, -1, 1)

            # convert from given (-1,1) to joint pos limits (low,high)
            desired_joint_positions_scaled = (((desired_joint_positions - self.action_lows) * self.joint_pos_range) / self.action_range) + self.limits_lows_joint_pos

        elif self.action_mode=='joint_delta_position':

            # clip the incoming vel
            delta_joint_pos = np.clip(action, -1, 1)

            # convert from given (-1,1) to joint vel limits (low,high) # tony: notice vel_range is actually delta range
            delta_joint_pos_scaled = (((delta_joint_pos - self.action_lows) * self.joint_vel_range) / self.action_range) + self.limits_lows_joint_vel
            
            # turn the delta into an action position
            curr_pos = self._get_joint_angles() 
            desired_joint_positions_scaled = curr_pos + delta_joint_pos_scaled
            

        # enforce joint velocity limits on this scaled action
        feasible_scaled_action = self.make_feasible(desired_joint_positions_scaled)

        # take a step
        self._act(feasible_scaled_action, self.timestep)

        # get updated observation
        obs = self._get_obs()

        # reward/etc. after taking the step
        reward = self.compute_rewards(obs)
        done = False
        info = None #self._get_info()
        return obs, reward, done, info

    def make_feasible(self, desired_positions):
        # get current positions
        curr_positions = self._get_joint_angles()

        # compare the implied vel to the max vel allowed
        max_vel = self.limits_highs_joint_vel*self.timestep #[7]
        implied_vel = np.abs(desired_positions-curr_positions) #[7]

        # limit the vel 
        actual_vel = np.min([implied_vel, max_vel], axis=0)

        # find the actual position, based on this vel
        sign = np.sign(desired_positions-curr_positions)
        actual_difference = sign*actual_vel
        feasible_positions = curr_positions+actual_difference

        # find the actual position, taking safety box into account
        feasible_positions = self.consider_safety_box(curr_positions, feasible_positions)

        return feasible_positions

    def consider_safety_box(self, curr_joint_positions, desired_joint_positions):
        # find desired_ee_position
        transformation_matrix = self._get_ee_fk(desired_joint_positions)
        desired_ee_position = np.array([transformation_matrix[3], transformation_matrix[7], transformation_matrix[11]])

        # if that's outside safety box, don't move the robot
        if np.any(desired_ee_position>self.safety_box_ee_high) or np.any(desired_ee_position<self.safety_box_ee_low):
            return curr_joint_positions
        else:
            return desired_joint_positions

    def reset(self):

        # move upward to make sure not stuck
        self._move_ee_upward()

        # move to reset position
        self._act(self.reset_joint_positions, self.reset_duration, reset=True)

        # return the observation
        return self._get_obs()

    def _move_ee_upward(self):
        curr_ee_pose = self._get_endeffector_pose()
        target_position = curr_ee_pose[:3] + np.array([0, 0, 0.05])
        target_quat = curr_ee_pose[3:]
        target_ee_pose = np.concatenate([target_position, target_quat])

        angles = self.request_ik_angles(target_ee_pose, self._get_joint_angles())
        self._act(angles, self.reset_duration, reset=True)