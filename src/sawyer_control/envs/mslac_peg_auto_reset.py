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
from sawyer_control.envs.rotation_utils import quat_magnitude, quat_difference, quat_conjugate, quat_mul

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
        # self.limits_lows_joint_pos = np.array([-0.6,  -1, -1, 1, -0.5, 0,  0.3])
        # self.limits_highs_joint_pos = np.array([0.6,   0, 0.2, 2.5, 1.3, 1.3, 2])

        self.limits_lows_joint_pos = np.array([-1,  -1, -1, 1, 0.7, 0, -0.2])
        self.limits_highs_joint_pos = np.array([0.3, 0, 0.4, 2., 1.6, 1.7, 2.6])


        # Added by Tony: reward limits
        self.reward_low = np.array([-10]) # 10 is not precise, just as a big bounding box
        self.reward_high = np.array([10])


        # safety box (calculated for our current single-task peg setup)
        self.safety_box_ee_low = np.array([0.36,-0.29,0.2])
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
        #self.reset_joint_positions = np.array([0.35275489, -0.93977147, -0.88832617, 1.5501895, 0.64358497, 1.07647467, 1.61443853])

        # self.reset_joint_positions = np.array([0.17672266, -0.71105176, -0.97821778, 1.41540623, 0.88952541, 1.12744629, 1.39725101])

        self.reset_joint_positions = np.array([-0.0360458984375, -0.7513271484375, -0.810923828125, 1.2634794921875, 1.0465224609375, 1.1741064453125, 1.2810009765625])


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
        # print("GOAL POSE OF EE: ", goal_pose)

    def rand_choose_task(self):
        i = np.random.randint(0, 30)
        self.set_task_for_env(i)
        print("by random sample")


    def get_task_info(self):
        """Generate all task_info (length = 4 for each task)"""
        num_tasks = 30
        np.random.seed(0)
        all_task_info = np.array([[5.17244542, 6.74050098, 5.68091098, 0.],
                                  [5.13540301, 3.99285242, 6.0874086, 1.],
                                  [4.1241623, 8.40476252, 9.08230755, 2.],
                                  [3.61385118, 7.46183269, 4.98471718, 0.],
                                  [5.35369386, 8.7235428, 0.66949908, 1.],
                                  [0.8211743, 0.19055391, 7.84725717, 2.],
                                  [7.3339546, 8.19967132, 9.22326058, 0.],
                                  [7.53189202, 4.34934052, 7.35631418, 1.],
                                  [1.1147102, 6.03111354, 1.3510729, 2.],
                                  [8.90329479, 4.91830456, 3.90809671, 0.],
                                  [2.4933779, 7.29698061, 4.2991156, 1.],
                                  [5.35736375, 0.1770897, 5.82107742, 2.],
                                  [5.76886628, 5.81446594, 8.89461609, 0.],
                                  [6.42600493, 3.38828214, 4.11892913, 1.],
                                  [6.57501912, 0.5676117, 6.28412824, 2.],
                                  [6.32061301, 1.98280892, 1.21510173, 0.],
                                  [2.97284217, 3.42789326, 5.37397796, 1.],
                                  [4.13372188, 9.31520397, 0.96174968, 2.],
                                  [1.96861705, 1.52030639, 6.15540095, 0.],
                                  [2.38721711, 4.39487549, 2.30365693, 1.],
                                  [1.49825303, 1.0402612, 6.18576065, 2.],
                                  [1.30234363, 1.85274511, 3.47515286, 0.],
                                  [7.7376789, 0.91515796, 7.8974447, 1.],
                                  [0.90570616, 9.20291365, 4.41693352, 2.],
                                  [9.20575638, 5.70053472, 6.96739509, 0.],
                                  [0.36933624, 2.66539283, 1.1328259, 1.],
                                  [2.79105561, 1.11898239, 2.99692086, 2.],
                                  [3.90433674, 0.60457591, 6.52639597, 0.],
                                  [5.3400929, 2.50123703, 4.93149672, 1.],
                                  [0.88536846, 5.42816784, 8.75841032, 2.], ])

        # all_task_info = []
        # for i in range(num_tasks):
        #     box_angles = np.random.random(3) * 3 * np.pi # 3 scalars from 0 to 3 pi
        #     goal_box = np.array([np.random.randint(0, 3)]) # 0 or 1 or 2
        #     task_info = np.concatenate([box_angles, goal_box], axis=0)
        #     all_task_info.append(task_info)
        #     print(task_info)

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
        ENTERED = True ###True
        if ENTERED:
            all_task_info = [[5.17244542, 6.74050098, 5.68091098, 0.],
                    [5.13540301, 3.99285242, 6.0874086, 1.],
                    [4.1241623, 8.40476252, 9.08230755, 2.],
                    [3.61385118, 7.46183269, 4.98471718, 0.],
                    [5.35369386, 8.7235428, 0.66949908, 1.],
                    [0.8211743, 0.19055391, 7.84725717, 2.],
                    [7.3339546, 8.19967132, 9.22326058, 0.],
                    [7.53189202, 4.34934052, 7.35631418, 1.],
                    [1.1147102, 6.03111354, 1.3510729, 2.],
                    [8.90329479, 4.91830456, 3.90809671, 0.],
                    [2.4933779, 7.29698061, 4.2991156, 1.],
                    [5.35736375, 0.1770897, 5.82107742, 2.],
                    [5.76886628, 5.81446594, 8.89461609, 0.],
                    [6.42600493, 3.38828214, 4.11892913, 1.],
                    [6.57501912, 0.5676117, 6.28412824, 2.],
                    [6.32061301, 1.98280892, 1.21510173, 0.],
                    [2.97284217, 3.42789326, 5.37397796, 1.],
                    [4.13372188, 9.31520397, 0.96174968, 2.],
                    [1.96861705, 1.52030639, 6.15540095, 0.],
                    [2.38721711, 4.39487549, 2.30365693, 1.],
                    [1.49825303, 1.0402612, 6.18576065, 2.],
                    [1.30234363, 1.85274511, 3.47515286, 0.],
                    [7.7376789, 0.91515796, 7.8974447, 1.],
                    [0.90570616, 9.20291365, 4.41693352, 2.],
                    [9.20575638, 5.70053472, 6.96739509, 0.],
                    [0.36933624, 2.66539283, 1.1328259, 1.],
                    [2.79105561, 1.11898239, 2.99692086, 2.],
                    [3.90433674, 0.60457591, 6.52639597, 0.],
                    [5.3400929, 2.50123703, 4.93149672, 1.],
                    [0.88536846, 5.42816784, 8.75841032, 2.], ] # a 30 x 4 array
            all_goal_poses = [[0.6498202681541443, 0.09903092682361603, 0.2154312, 0.20422907173633575, 0.9785065054893494, 0.026300782337784767, -0.01112747099250555],
                    [0.751207709312439, -0.030662626028060913, 0.2154312, 0.028746457770466805, 0.9991078972816467, 0.02062484249472618, -0.02305903285741806],
                    [0.6068632006645203, -0.15877331793308258, 0.2154312, 0.28160005807876587, 0.9593900442123413, 0.016430461779236794, -0.001477352692745626],
                    [0.6536340713500977, 0.09357554465532303, 0.2154312, 0.2243463695049286, 0.9742645621299744, 0.02002793736755848, 0.008727528154850006],
                    [0.7466577291488647, -0.009104977361857891, 0.2154312, -0.04341282695531845, 0.9988676309585571, 0.01943352073431015, -0.001073271967470646],
                    [0.6043043732643127, -0.16320018470287323, 0.2154312, 0.2962016463279724, 0.9550295472145081, 0.009983026422560215, 0.009139946661889553],
                    [0.6412408351898193, 0.10601378232240677, 0.2154312, 0.1688154637813568, 0.9848292469978333, 0.039989572018384933, -0.0036775164771825075],
                    [0.7487481236457825, -0.029543761163949966, 0.2154312, 0.005664797965437174, 0.9996691942214966, 0.023318184539675713, -0.009258490055799484],
                    [0.5864234566688538, -0.18474116921424866, 0.2154312, 0.3980006277561188, 0.9173701405525208, -0.004463544115424156, 0.0027543026953935623],
                    [0.6347795128822327, 0.11529255658388138, 0.2154312, 0.1393696814775467, 0.9895467162132263, 0.029551686719059944, -0.022362396121025085],
                    [0.7477385997772217, -0.0161913875490427, 0.2154312, -0.04649864137172699, 0.998680830001831, 0.020373743027448654, -0.00770576624199748],
                    [0.6002618074417114, -0.17305612564086914, 0.2154312, 0.3209840953350067, 0.946865975856781, 0.018020641058683395, -0.009450388140976429],
                    [0.6468135118484497, 0.10599967837333679, 0.2154312, 0.1947287619113922, 0.9808465242385864, 0.004538155626505613, 0.0004439897311385721],
                    [0.7469571232795715, -0.030136071145534515, 0.2154312, 0.05626113712787628, 0.9983897805213928, -0.0012255299370735884, 0.007141051348298788],
                    [0.6003249883651733, -0.17078657448291779, 0.2154312, 0.30141860246658325, 0.9532735347747803, 0.020342104136943817, 0.0016280176350846887],
                    [0.6429545879364014, 0.10692509263753891, 0.2154312, 0.19006361067295074, 0.9815251231193542, 0.021934229880571365, 0.0017595576355233788],
                    [0.7481274008750916, -0.033671773970127106, 0.2154312, 0.027959292754530907, 0.9993607997894287, 0.02109481766819954, -0.007162897381931543],
                    [0.5842893719673157, -0.18723012506961823, 0.2154312, 0.4069995880126953, 0.9133808612823486, 0.007478300482034683, 0.005547537934035063],
                    [0.6603473424911499, 0.08503494411706924, 0.2154312, 0.25233200192451477, 0.9670886993408203, 0.03138773515820503, -0.00909921620041132],
                    [0.7482395768165588, -0.029708849266171455, 0.2154312, 0.011963780038058758, 0.9995869398117065, 0.0254589281976223, -0.00588678102940321],
                    [0.6006653904914856, -0.16833952069282532, 0.2154312, 0.3479650020599365, 0.9375009536743164, 0.0015795642975717783, 0.003135999199002981],
                    [0.6641683578491211, 0.08553310483694077, 0.2154312, 0.29652783274650574, 0.9545632600784302, 0.007781678345054388, -0.028629997745156288],
                    [0.746563196182251, -0.04451780766248703, 0.2154312, 0.08020345121622086, 0.9965546727180481, 0.021078532561659813, -0.0013732153456658125],
                    [0.5950831174850464, -0.17460352182388306, 0.2154312, 0.348741739988327, 0.9371175765991211, 0.012251224368810654, 0.006308915093541145],
                    [0.6318866014480591, 0.11816545575857162, 0.2154312, 0.14830738306045532, 0.9888176321983337, 0.015330422669649124, -0.0031037162989377975],
                    [0.7480047345161438, -0.035276804119348526, 0.2154312, 0.032082632184028625, 0.9994488954544067, 0.007546599954366684, 0.0039558433927595615],
                    [0.5902729630470276, -0.18066254258155823, 0.2154312, 0.36308911442756653, 0.931625485420227, 0.010797218419611454, 0.011120866984128952],
                    [0.6528969407081604, 0.09660182893276215, 0.2154312, 0.22509269416332245, 0.9740715622901917, 0.01969837211072445, -0.011397111229598522],
                    [0.7485886812210083, -0.03667362779378891, 0.2154312, 0.05067003145813942, 0.9986448884010315, 0.011783520691096783, -0.0014414428733289242],
                    [0.6055131554603577, -0.1606883704662323, 0.2154312, 0.2731553018093109, 0.9618500471115112, 0.013697723858058453, 0.006561298854649067]] # a 30 x 7 array
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
            print("sb")
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
            self._move_ee_diagonal()
            return self.reset()

    def _move_ee_upward(self):
        curr_ee_pose = self._get_endeffector_pose()
        target_position = curr_ee_pose[:3] + np.array([0, 0, 0.05])
        target_quat = curr_ee_pose[3:]
        target_ee_pose = np.concatenate([target_position, target_quat])

        angles = self.request_ik_angles(target_ee_pose, self._get_joint_angles())
        self._act(angles, self.reset_duration*2/3, reset=True)


    def _move_ee_diagonal(self):
        curr_ee_pose = self._get_endeffector_pose()
        quat = curr_ee_pose[3:]
        unit_vec = np.array([0, 0, 0, 1,])
        quat_inv = quat_conjugate(quat)

        v_transformed = quat_mul(quat_mul(quat, unit_vec), quat_inv)

        v_transformed = v_transformed[1:]
        assert len(v_transformed) == 3

        if v_transformed[2] < 0:
            v_transformed[2] = v_transformed[2] * -1

        target_position = curr_ee_pose[:3] + v_transformed * 0.05 # TODO or -0.05???
        target_quat = curr_ee_pose[3:]
        target_ee_pose = np.concatenate([target_position, target_quat])
        angles = self.request_ik_angles(target_ee_pose, self._get_joint_angles())
        self._act(angles, self.reset_duration*1/3, reset=True)

