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
from sawyer_control.envs.rotation_utils import quat_magnitude, quat_difference

import cv2
import copy
import rospy
from sawyer_control.srv import image

# for auto reset
import time
import robel

class MslacPegInsertionAutoEnv(SawyerEnvBase):

    '''
    Inserting a peg into a box (which is at a fixed location)

    GOAL RIGHT NOW: 
    x: 0.459949938269
    y: 0.0171827931642
    z: 0.195978444238
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
        self.limits_lows_joint_pos = np.array([-0.2,  -1, -1, 1.5, -0.5, 0,  0.3])
        self.limits_highs_joint_pos = np.array([0.6,   0, 0.2, 2.5, 1.3, 1.3, 2])


        # Added by Tony: reward limits
        self.reward_low = np.array([-10]) # 10 is not precise, just as a big bounding box
        self.reward_high = np.array([10])


        # safety box (calculated for our current single-task peg setup)
        self.safety_box_ee_low = np.array([0.36,-0.29,0.18])
        self.safety_box_ee_high = np.array([0.8,0.32,0.5])

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
        self.joint_pos_box_range = (self.limits_highs_joint_pos - self.limits_lows_joint_pos)
        self.joint_vel_range = (self.limits_highs_joint_vel - self.limits_lows_joint_vel)

        # reset position (note: this is the "0" of our new (-1,1) action range)
        #self.reset_joint_positions = np.array([ 0.38670802, -0.76212013, -0.91199511,  1.92950678,  1.03574216, 0.83555567,  1.1440537 ]) # too close
        #self.reset_joint_positions = np.array([0.20293261, -0.62833202, -0.88585448, 1.55062211, 0.92892969, 0.58760154, 1.14426076]) # too far
        #self.reset_joint_positions = np.array([0.22460157, -0.42119628, -1.40798533, 1.33706546, 1.1406436, 1.51298046, 1.1185683])
        #self.reset_joint_positions = np.array([ 0.41621679, -0.52845901, -1.00537789,  1.67203224,  1.10663569, 1.02080274,  1.14446783]) # for auto reset
        #self.reset_joint_positions = np.array([0.39000782, -1.06390524, -0.97850686, 1.73639643, 0.88786036, 0.91707617, 1.04374897]) # for higher starting pose, outside global cam view
        self.reset_joint_positions = np.array([0.35275489, -0.93977147, -0.88832617, 1.5501895, 0.64358497, 1.07647467, 1.61443853])


        self.reset_duration = 3.0 # seconds to allow for reset

        # Auto Reset
        self.reset_env = gym.make('Meld-v0', device_path='/dev/ttyUSB0')
        # Reset task After arm is lifted
        self.reset_env.reset()
        time.sleep(1)

        # Generate goals
        self.get_task_info()
        self.get_goal_poses()


        # reset robot to initialize
        self.rand_choose_task()
        self.reset()

        # Added by Tony
        self.sparse_reward = False  # TODO Tony: hardcoded. Also not expecting to run sparse reward on robot

    ####################################
    ####################################

    def set_task_for_env(self, task_idx):
        task_info = self.all_task_info[task_idx]
        box_angles = task_info[:3]
        which_box = task_info[3]

        goal_pose = self.all_goal_poses[task_idx]

        # set
        self.change_box_pos(box_angles)
        time.sleep(1)
        self.curr_peg_goal = goal_pose

        print("=====Set new goal, goal id:", task_idx, "box #", which_box, "=====")

    def rand_choose_task(self):
        i = np.random.randint(0, 30)
        self.set_task_for_env(i)
        print("by random sample")


    def get_task_info(self):
        """Generate all task_info (length = 4 for each task)"""
        num_tasks = 30
        np.random.seed(0)

        all_task_info = []
        for i in range(num_tasks):
            box_angles = np.random.random(3) * 3 * np.pi # 3 scalars from 0 to 3 pi
            goal_box = np.array([np.random.randint(0, 3)]) # 0 or 1 or 2
            task_info = np.concatenate([box_angles, goal_box], axis=0)
            all_task_info.append(task_info)
            print(task_info)

        self.all_task_info = all_task_info
        return all_task_info

    def go_over_goals(self):
        """Just for visualization"""
        for task_info in self.all_task_info:
            box_angles = task_info[:3]
            goal_box = task_info[3]
            self.change_box_pos(box_angles)
            print("box id:", goal_box)
            time.sleep(1.5)

    def ee_pos_helper(self):
        ee_pos = self._get_endeffector_pose()
        s = repr(list(ee_pos))
        print(s)
        return ee_pos

    def get_goal_poses(self):
        ENTERED = True
        if ENTERED:
            all_task_info = [[5.17244542, 6.74050098, 5.68091098, 1.],
                     [5.87694939, 3.62271224, 2.8041976, 0.],
                     [9.08230755, 3.61385118, 7.46183269, 2.],
                     [5.35369386, 8.7235428, 0.66949908, 1.],
                     [6.10887597, 3.47059475, 9.02097485, 0.],
                     [9.22326058, 7.53189202, 4.34934052, 2.],
                     [6.79180276, 5.48540731, 5.06462337, 0.],
                     [4.91830456, 3.90809671, 2.4933779, 2.],
                     [4.2991156, 5.35736375, 0.1770897, 1.],
                     [1.41065239, 2.09532972, 3.64257283, 0.],
                     [6.42600493, 3.38828214, 4.11892913, 1.],
                     [0.93569526, 9.14023513, 6.15569981, 2.],
                     [3.37550465, 7.0750502, 5.72866909, 0.],
                     [5.37397796, 4.13372188, 9.31520397, 0.],
                     [6.15240379, 5.98528888, 9.38047743, 0.],
                     [2.38721711, 4.39487549, 2.30365693, 1.],
                     [6.35939082, 2.98955599, 7.33573334, 0.],
                     [6.24416861, 0.12790965, 5.87018615, 0.],
                     [0.90570616, 9.20291365, 4.41693352, 1.],
                     [0.52509862, 4.25207542, 0.18837931, 2.],
                     [2.66539283, 1.1328259, 2.79105561, 2.],
                     [6.49047874, 8.29828976, 8.65416539, 2.],
                     [5.32677957, 8.15339955, 4.79691944, 2.],
                     [0.88536846, 5.42816784, 8.75841032, 0.],
                     [6.29019464, 1.24216559, 6.75122485, 0.],
                     [2.49502302, 3.74937226, 5.2102197, 0.],
                     [7.81257572, 0.04425382, 6.38827036, 2.],
                     [6.92904042, 9.06841339, 2.34444314, 0.],
                     [8.44496285, 6.02168928, 8.40270261, 1.],
                     [4.21405741, 7.9772138, 6.59243686, 0.]] # a 30 x 4 array
            all_goal_poses = [
                [0.44644695520401, 0.013092352077364922, 0.20978321135044098, 0.25502267479896545, 0.9666233658790588,
                 0.016556834802031517, 0.018125327304005623],
                [0.6412569284439087, 0.23274792730808258, 0.20978321135044098, 0.1367698311805725, 0.9897645115852356,
                 0.03883516415953636, 0.012331472709774971],
                [0.63961261510849, -0.12795326113700867, 0.20978321135044098, 0.12423500418663025, 0.9920806884765625,
                 0.01513577252626419, 0.010603347793221474],
                [0.45047497749328613, -0.0009425116586498916, 0.20978321135044098, 0.2057570219039917,
                 0.9785575270652771, 0.0068109952844679356, -0.0065399399027228355],
                [0.6122738122940063, 0.20151856541633606, 0.20978321135044098, 0.020996209233999252, 0.9996973872184753,
                 0.01262995321303606, -0.002195255598053336],
                [0.6408815383911133, -0.12781472504138947, 0.20978321135044098, 0.16768698394298553, 0.9857485294342041,
                 0.013238205574452877, -0.002387358108535409],
                [0.6307583451271057, 0.2225394994020462, 0.20978321135044098, 0.0808701142668724, 0.9963876008987427,
                 0.023703712970018387, 0.010485051199793816],
                [0.6258251070976257, -0.11207244545221329, 0.20978321135044098, 0.23300981521606445, 0.9723166823387146,
                 0.014756248332560062, 0.009433332830667496],
                [0.4509566128253937, 0.021648498252034187, 0.20978321135044098, 0.2762584686279297, 0.9606983661651611,
                 0.021934395655989647, -0.016088159754872322],
                [0.6390601396560669, 0.22676266729831696, 0.20978321135044098, 0.10365007817745209, 0.9932063221931458,
                 0.052747342735528946, 0.003949806094169617],
                [0.4530617594718933, 0.035969823598861694, 0.20978321135044098, 0.29189279675483704, 0.9558082818984985,
                 0.02327265404164791, -0.026220208033919334],
                [0.611844539642334, -0.0931808277964592, 0.20978321135044098, 0.2776620090007782, 0.9603682160377502,
                 0.013458811677992344, 0.020384078845381737],
                [0.6296904683113098, 0.21718545258045197, 0.20978321135044098, 0.0955396443605423, 0.9945368766784668,
                 0.03624923899769783, -0.021320506930351257],
                [0.611138105392456, 0.19876177608966827, 0.20978321135044098, 0.02556633949279785, 0.9994871020317078,
                 0.019084395840764046, -0.002776087960228324],
                [0.612235426902771, 0.19851148128509521, 0.20978321135044098, 0.00945278350263834, 0.9997549653053284,
                 0.017505362629890442, -0.009707285091280937],
                [0.4509289264678955, 0.02739662677049637, 0.20978321135044098, 0.24869869649410248, 0.9682722091674805,
                 0.02311902493238449, -0.007960180751979351],
                [0.6196180582046509, 0.2068660408258438, 0.20978321135044098, 0.0606541633605957, 0.9966773390769958,
                 0.054042235016822815, 0.005893514025956392],
                [0.6277748346328735, 0.21559619903564453, 0.20978321135044098, 0.07181492447853088, 0.9965139031410217,
                 0.04183139279484749, -0.0072663091123104095],
                [0.449075847864151, -0.002864704467356205, 0.20978321135044098, 0.18432830274105072, 0.9828640222549438,
                 -0.000985872931778431, 0.0006370576447807252],
                [0.6116945147514343, -0.09273890405893326, 0.20978321135044098, 0.2891126275062561, 0.9566419124603271,
                 0.02501743659377098, 0.024985501542687416],
                [0.6163972616195679, -0.10207419842481613, 0.20978321135044098, 0.24938009679317474, 0.9678286910057068,
                 0.01989815942943096, 0.02685648575425148],
                [0.6299532651901245, -0.1168404221534729, 0.20978321135044098, 0.19011132419109344, 0.9815333485603333,
                 0.015797365456819534, 0.01415492594242096],
                [0.6261252760887146, -0.11390472203493118, 0.20978321135044098, 0.22248995304107666, 0.9746600985527039,
                 0.02017763815820217, 0.011347559280693531],
                [0.615086019039154, 0.20329628884792328, 0.20978321135044098, 0.050864238291978836, 0.9985918402671814,
                 0.009881934151053429, -0.011382103897631168],
                [0.6252308487892151, 0.2138160914182663, 0.20978321135044098, 0.06529776006937027, 0.997562050819397,
                 0.019170423969626427, -0.015447268262505531],
                [0.631434440612793, 0.22239504754543304, 0.20978321135044098, 0.10639733821153641, 0.9941754341125488,
                 0.01663600467145443, -0.004256087355315685],
                [0.637357234954834, -0.12090951204299927, 0.20978321135044098, 0.1671018749475479, 0.9859217405319214,
                 -0.005868774838745594, -0.0009356314549222589],
                [0.6482738256454468, 0.23656082153320312, 0.20978321135044098, 0.18524488806724548, 0.9823354482650757,
                 0.020378826186060905, -0.01691425032913685],
                [0.44886514544487, 0.017473217099905014, 0.20978321135044098, 0.24751891195774078, 0.9688166975975037,
                 0.011315778829157352, -0.000727725331671536],
                [0.6228170394897461, 0.21204060316085815, 0.20978321135044098, 0.10294251143932343, 0.9936566352844238,
                 0.04524282366037369, 0.001566296094097197]] # a 30 x 7 array
            all_task_info = np.array(all_task_info)
            all_goal_poses = np.array(all_goal_poses)
            assert all_task_info.shape == (30, 4)
            assert all_goal_poses.shape == (30, 7)
            assert (all_task_info - self.all_task_info < 0.01).all()

            self.all_task_info = all_task_info
            self.all_goal_poses = all_goal_poses

            return

        else:
            goal_poses = []
            print("\nEntering goal poses helper\n")
            count = 0
            for task_info in self.all_task_info:
                box_angles = task_info[:3]
                goal_box = task_info[3]
                self.change_box_pos(box_angles)
                time.sleep(1)
                print("===========")
                print("Goal box number:", goal_box)

                good = 'n'
                while not good == '':
                    print("hint: call self.ee_pos_helper()\n")
                    import IPython
                    IPython.embed()

                    input_str = input("Enter goal pos for task" + str(count))
                    try:
                        pose = np.array(eval(input_str))
                        assert len(pose) == 7
                        print("entered successful: ", pose)
                    except:
                        print("syntax error")
                    good = input("confirm?")

                goal_poses.append(pose)
                count += 1
                self._move_ee_upward()

            for p in goal_poses:
                print(list(p))

            print("\n\nrecording finished, please restart the program")
            assert False


    def change_box_pos(self, curr_servo_pos):
        max_angle = 3*np.pi
        assert len(curr_servo_pos) == 3
        assert (curr_servo_pos[0] < max_angle) and (curr_servo_pos[1] < max_angle) and (curr_servo_pos[2] < max_angle)
        self.reset_env.step(curr_servo_pos)
        time.sleep(1)



    # added by Tony
    def override_action_mode(self, mode):
        assert mode in ['joint_position', 'joint_delta_position']
        self.action_mode = mode
        print("ACTION MODE: ", self.action_mode)

    def _set_observation_space(self):
        ''' [14] : observation is [7] joint angles + [3] ee pos + [4] ee angles '''
        self.obs_lows = np.concatenate((self.limits_lows_joint_pos, self.limits_lows_ee_pos, self.limits_lows_ee_angles, self.reward_low))
        self.obs_highs = np.concatenate((self.limits_highs_joint_pos, self.limits_highs_ee_pos, self.limits_highs_ee_angles, self.reward_high))
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
        crop_width = (input_h - (input_w/2))/2
        double_image = double_image[:, int(crop_width):int(input_h-crop_width), :]  # crop such that width:high = 2:1: make height: 640->480

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
        ''' [7] joint angles + [7] ee pose '''
        angles = self._get_joint_angles()
        ee_pose = self._get_endeffector_pose()
        return np.concatenate([angles, ee_pose])

    ####################################
    ####################################

    def quat_score(self, q1, q2):
        assert len(q1) == len(q2) == 4
        return quat_magnitude(quat_difference(q1, q2))


    def compute_rewards(self, obs, action=None):
        # want indices 7-13
        ee_pose = obs[self.num_joint_dof:self.num_joint_dof+7] # same as env._get_endeffector_pose(): gives 3 + 4
        goal_pose = self.curr_peg_goal

        # distance between the points
        score_dist = np.linalg.norm(ee_pose[:3] - goal_pose[:3])
        # score_angle = quat_magnitude(quat_difference(ee_pose[3:], goal_pose[3:]))
        score_angle = self.quat_score(ee_pose[3:], goal_pose[3:])

        alpha = 1.0
        beta = 0.1
        score = alpha*score_dist + beta*score_angle

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

        # print(reward)
        return reward

    def step(self, action):

        if self.action_mode=='joint_position':

            # clip incoming action
            desired_joint_positions = np.clip(action, -1, 1)

            # convert from given (-1,1) to joint pos limits (low,high)
            desired_joint_positions_scaled = (((desired_joint_positions - self.action_lows) * self.joint_pos_box_range) / self.action_range) + self.limits_lows_joint_pos

        elif self.action_mode=='joint_delta_position':

            # clip the incoming vel
            delta_joint_pos = np.clip(action, -1, 1)

            # convert from given (-1,1) to joint vel limits (low,high) # tony: notice vel_range is actually delta range
            delta_joint_pos_box_scaled = (((delta_joint_pos - self.action_lows) * self.joint_vel_range) / self.action_range) + self.limits_lows_joint_vel

            # turn the delta into an action position
            curr_pos = self._get_joint_angles()
            desired_joint_positions_scaled = curr_pos + delta_joint_pos_box_scaled


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

        obs = np.concatenate((obs, np.array([reward]))) # NOTICE

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
            print("sb", end='')
            return curr_joint_positions
        else:
            return desired_joint_positions

    def reset(self):

        # move upward to make sure not stuck
        self._move_ee_upward()

        # self.reset_goal()

        # move to reset position
        self._act(self.reset_joint_positions, self.reset_duration*1.5, reset=True)

        # return the observation
        ob = self._get_obs()
        ob = np.concatenate((ob, np.array([0])))

        new_ee_pos = self.ee_pos_helper()
        new_z = new_ee_pos[2]
        if new_z > 0.22: # successful
            return ob
        else:
            print("WARNING: reset unsuccessful, will try again in 1 sec")
            time.sleep(1)
            return self.reset()

    def _move_ee_upward(self):
        curr_ee_pose = self._get_endeffector_pose()
        target_position = curr_ee_pose[:3] + np.array([0, 0, 0.05])
        target_quat = curr_ee_pose[3:]
        target_ee_pose = np.concatenate([target_position, target_quat])

        angles = self.request_ik_angles(target_ee_pose, self._get_joint_angles())
        self._act(angles, self.reset_duration*2/3, reset=True)