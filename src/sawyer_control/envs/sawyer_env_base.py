import numpy as np
import sys
import rospy
import gym

from gym.spaces import Box
from sawyer_control.pd_controllers.joint_angle_pd_controller import AnglePDController
from sawyer_control.core.serializable import Serializable
from sawyer_control.core.multitask_env import MultitaskEnv
from sawyer_control.configs.config import config_dict as config


from sawyer_control.srv import observation
from sawyer_control.srv import getRobotPoseAndJacobian
from sawyer_control.srv import ik
from sawyer_control.srv import angle_action
from sawyer_control.srv import image
from sawyer_control.msg import actions
import abc
import cv2
import copy
import collections

class SawyerEnvBase(gym.Env, Serializable, MultitaskEnv, metaclass=abc.ABCMeta):
    def __init__(
            self,
            action_mode='torque',
            use_safety_box=True,
            torque_action_scale=1,
            position_action_scale=0.1,
            config_name = 'base_config',
            fix_goal=False,
            max_speed = 0.05, #.07
            reset_free=False,
            img_start_col=350, #can range from  0-999
            img_start_row=200, #can range from  0-999
            img_col_delta=300, #can range from  0-999
            img_row_delta=600, #can range from  0-999
            height_2d=None,
            constant_hz=False,
            **kwargs  # to get rid of unused kwargs
    ):
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(self)
        self.config = config[config_name]
        self.init_rospy(self.config.UPDATE_HZ)
        self.action_mode = action_mode
        self.max_speed = max_speed

        self.use_safety_box = use_safety_box
        self.AnglePDController = AnglePDController(config=self.config)

        self._set_action_space()
        self._set_observation_space()
        self.get_latest_pose_jacobian_dict()

        self.torque_action_scale = torque_action_scale
        self.position_action_scale = position_action_scale
        self.in_reset = True
        self._state_goal = None
        self.fix_goal = fix_goal

        self.pos_control_reset_position = self.config.POSITION_RESET_POS
        self.reset_free = reset_free

        self.img_start_col = img_start_col
        self.img_start_row = img_start_row
        self.img_col_delta = img_col_delta
        self.img_row_delta = img_row_delta

        self.height_2d = height_2d

        self.torque_window_size = 2
        # Use a deque if the window size becomes too large?
        self.torque_window = [np.asarray([0.0 for _ in range(7)]) for _ in range(self.torque_window_size)]

        self.constant_hz = constant_hz
        print("CONSTANT HZ? " + str(self.constant_hz))

    def _act(self, action):
        if self.action_mode == 'position':
            self._position_act(action * self.position_action_scale)
        else:
            self._torque_act(action*self.torque_action_scale)
        return

    def _position_act(self, action):
        print("ACTION: " + str(action))
        endeffector_pos = self._get_endeffector_position()
        target_ee_pos = (endeffector_pos + action)
        if self.height_2d:
            target_ee_pos[2] = self.height_2d
        target_ee_pos = np.clip(target_ee_pos, self.config.POSITION_SAFETY_BOX_LOWS, self.config.POSITION_SAFETY_BOX_HIGHS)
        target_ee_pos = np.concatenate((target_ee_pos, [self.config.POSITION_CONTROL_EE_ORIENTATION.x, self.config.POSITION_CONTROL_EE_ORIENTATION.y, self.config.POSITION_CONTROL_EE_ORIENTATION.z, self.config.POSITION_CONTROL_EE_ORIENTATION.w]))
        angles = self.request_ik_angles(target_ee_pos, self._get_joint_angles())
        if angles:
            self.send_angle_action(angles, target_ee_pos)
        else:
            print("No IK solution\n target: {} | current: {}".format(target_ee_pos[:3], ee_pos))

    def _get_pd_torques(self, error, p_coeff=150, d_coeff=5):
        """Proportional-Derivative controller used to move joint angles to a given position."""
        joint_velocities = self._get_joint_velocities()
        torques = -p_coeff*error - d_coeff*joint_velocities
        # print("P: " + str(-p_coeff*error) + " | D: " + str(-d_coeff*joint_velocities))
        # PID command to hold still: SENDING ACTION: [ 2.47852817e-01  3.85983616e-01  3.00000000e+00  4.79142368e-01
        #  -4.29153442e-06  4.64792192e-01  4.91886228e-01]


        # Find how much they have exceeded max torques, rescale torques
        # Make it so it is double max possible torque (so it can counteract completely in the other direction).
        fraction_max_torque = torques / self.config.MAX_TORQUES
        # print(np.max(fraction_max_torque))
        torques /= (np.max(fraction_max_torque) / 2.0)

        return torques

    def _torque_act(self, action):
        # print("INIT ACTION: " + str(action))
        # hold_position_mask = np.array(action == 0, dtype=np.float32)  # 1 if nonzero torque is commanded
        # pid_action = self._get_pd_torques(self._get_joint_angles() - self.prev_torque_angle_pos)
        # pid_action = self._get_pd_torques(self._get_joint_angles() - self.config.RESET_ANGLES)
        # action += hold_position_mask * pid_action
        curr_joint_angles, curr_velocities, ee_pose, ee_vel = self.request_observation()

        if self.use_safety_box:
            if self.in_reset:
                safety_box = self.config.RESET_SAFETY_BOX
            else:
                safety_box = self.config.TORQUE_SAFETY_BOX
            ee_safety_box = self.config.TORQUE_EE_SAFETY_BOX
            # ee_info = np.asarray(self.pose_jacobian_dict['_hand'][0])
            ee_info = ee_pose[:3]
            # print(ee_info)
            use_ee_pd_controller = False
            if use_ee_pd_controller:
                if not ee_safety_box.contains(ee_info):
                    desired_ee = np.max([np.min([ee_info, ee_safety_box.high], axis=0), ee_safety_box.low], axis=0)
                    full_ee_state = np.concatenate((desired_ee, ee_pose[3:]))
                    # safe_joint_angles = self.request_ik_angles(full_ee_state, curr_joint_angles)
                    safe_joint_angles = self.request_ik_angles(full_ee_state, self.config.RESET_ANGLES)
                    if not safe_joint_angles:
                        print("No IK solution. Moving to default position.")
                        safe_joint_angles = self.config.RESET_ANGLES
                    safety_adjustment_torques = self._get_pd_torques(curr_joint_angles - safe_joint_angles)
                    safety_adjustment_torques[-1] = 0 #we don't need to move the wrist
                    # print(safety_adjustment_torques)
                    action = safety_adjustment_torques  # add the adjustment torques in

            self.get_latest_pose_jacobian_dict()
            pose_jacobian_dict_of_joints_not_in_box = self.get_pose_jacobian_dict_of_joints_not_in_box(safety_box)
            if len(pose_jacobian_dict_of_joints_not_in_box) > 0:
                # print(pose_jacobian_dict_of_joints_not_in_box)
                forces_dict = self._get_adjustment_forces_per_joint_dict(pose_jacobian_dict_of_joints_not_in_box, safety_box)
                torques = np.zeros(7)
                for joint in forces_dict:
                    jacobian = pose_jacobian_dict_of_joints_not_in_box[joint][1]
                    force = forces_dict[joint]
                    torques = torques + np.dot(jacobian.T, force).T
                torques[-1] = 0 #we don't need to move the wrist
                action = torques
        exceeds_max_velocity = np.array(np.abs(curr_velocities) > self.config.MAX_VELOCITIES, dtype=np.int32)
        same_direction = np.array(curr_velocities * action > 0, dtype=np.int32)
        if np.sum(exceeds_max_velocity * same_direction) > 0:
            print("MASK: " + str(exceeds_max_velocity * same_direction))
        action *= (1 - exceeds_max_velocity * same_direction)

        ## Don't do smoothing - too hard to learn
        # self.torque_window.append(action)
        # self.torque_window.pop(0)
        action += self.config.GRAVITY_COMP_ADJUSTMENT
        # action = (self.torque_window[0]*0.25 + self.torque_window[1] * 0.75)


        # Limit wrist motion
        if action[-1] > 0 and curr_joint_angles[-1] > self.config.WRIST_ANGLE_HIGH:
            action[-1] = 0
        if action[-1] < 0 and curr_joint_angles[-1] < self.config.WRIST_ANGLE_LOW:
            action[-1] = 0

        if self.in_reset:
            action = np.clip(action, self.config.RESET_TORQUE_LOW, self.config.RESET_TORQUE_HIGH)
        else:
            action = np.clip(np.asarray(action), self.config.JOINT_TORQUE_LOW, self.config.JOINT_TORQUE_HIGH)
        # print("FINAL ACTION: " + str(action))
        self.send_action(action)
        self.rate.sleep()

    def _wrap_angles(self, angles):
        return angles % (2*np.pi)

    def _get_joint_angles(self):
        angles, _, _, _ = self.request_observation()
        return angles

    def _get_joint_velocities(self):
        _, velocities, _, _ = self.request_observation()
        return velocities

    def _get_endeffector_pose(self):
        # full 7-dof pose of ee
        _, _, endpoint_pose, _ = self.request_observation()
        return endpoint_pose

    def _get_endeffector_position(self):
        # 3-D position of ee
        _, _, endpoint_pose, _ = self.request_observation()
        return endpoint_pose[:3]

    def _get_endeffector_fullvel(self):
        # full 7-dof position + rotational velocities
        _, _, _, endpoint_vel = self.request_observation()
        return endpoint_vel

    def _get_endeffector_posvel(self):
        # x-y-z velocities of ee
        _, _, _, endpoint_vel = self.request_observation()
        return endpoint_vel[:3]

    def compute_angle_difference(self, angles1, angles2):
        deltas = np.abs(angles1 - angles2)
        differences = np.minimum(2 * np.pi - deltas, deltas)
        return differences

    def step(self, action):
        self._act(action)
        observation = self._get_obs()
        reward = self.compute_rewards(action, self.convert_ob_to_goal(observation), self._state_goal)
        info = self._get_info()
        done = False
        ee_pos = observation[-7:-4]
        # print("R:" + str(reward) + " | O: " + str(ee_pos) + " | G: " + str(self._state_goal))
        return observation, reward, done, info

    def _get_obs(self):
        ''' concatenate all parts of the obs into one vector '''
        angles, velocities, endpoint_pose, endpoint_velocity = self.request_observation()
        obs = np.hstack((
            self._wrap_angles(angles),
            velocities,
            endpoint_pose,
            endpoint_velocity,
        ))
        return obs

    def _get_info(self):
        return dict()

    def _safe_move_to_neutral(self):
        for _ in range(self.config.RESET_LENGTH):
            cur_pos, cur_vel, _, _ = self.request_observation()
            torques = self.AnglePDController._compute_pd_forces(cur_pos, cur_vel)
            self._torque_act(torques)
            if self._check_reset_complete():
                break

    def _check_reset_complete(self):
        close_to_desired_reset_pos = self._check_reset_angles_within_threshold()
        velocities = self._get_joint_velocities()
        velocities = np.abs(np.array(velocities))
        VELOCITY_THRESHOLD = .002 * np.ones(7)
        no_velocity = (velocities < VELOCITY_THRESHOLD).all()
        return close_to_desired_reset_pos and no_velocity

    def _check_reset_angles_within_threshold(self):
        desired_neutral = self.AnglePDController._des_angles
        desired_neutral = np.array([desired_neutral[joint] for joint in self.config.JOINT_NAMES])
        actual_neutral = (self._get_joint_angles())
        errors = self.compute_angle_difference(desired_neutral, actual_neutral)
        is_within_threshold = (errors < self.config.RESET_ERROR_THRESHOLD).all()
        return is_within_threshold

    def _reset_robot(self):
        if not self.reset_free:
            # if self.action_mode == "position":
            #     for i in range(5):
            #         self._position_act(self.pos_control_reset_position - self._get_endeffector_pose())
            #     # while np.linalg.norm(self.pos_control_reset_position - self._get_endeffector_pose()) > 0.05:
            #     #     for i in range(5):
            #     #         self._position_act(self.pos_control_reset_position - self._get_endeffector_pose())
            # else:
            self.in_reset = True
            print("STARTING RESET")
            while np.linalg.norm(self._get_joint_angles() - self.config.RESET_ANGLES) > 0.05:
                for i in range(3):
                    self.request_angle_action(self.config.RESET_ANGLES, self.pos_control_reset_position, reset=True)
                self.request_angle_action(self.config.RESET_ANGLES, self.pos_control_reset_position, reset=True)
            print("FINISHED RESET")
            self.in_reset = False

    def move_to_pos(self, target_pos):
        if self.action_mode == "position":
            for i in range(5):
                self._position_act(target_pos - self._get_endeffector_position())
            while np.linalg.norm(target_pos - self._get_endeffector_position()) > 0.05:
                for i in range(5):
                    self._position_act(target_pos - self._get_endeffector_position())
        else:
            raise RuntimeError("We cannot move to position in torque mode")

    def reset(self):
        self._reset_robot()
        self._state_goal = self.sample_goal()
        return self._get_obs()

    def get_latest_pose_jacobian_dict(self):
        self.pose_jacobian_dict = self._get_robot_pose_jacobian_client()
        return self.pose_jacobian_dict

    def _get_robot_pose_jacobian_client(self):
        rospy.wait_for_service('get_robot_pose_jacobian')
        try:
            get_robot_pose_jacobian = rospy.ServiceProxy('get_robot_pose_jacobian', getRobotPoseAndJacobian,
                                                         persistent=True)
            resp = get_robot_pose_jacobian('right')
            pose_jac_dict = self._unpack_pose_jacobian_dict(resp.poses, resp.jacobians)
            return pose_jac_dict
        except rospy.ServiceException as e:
            print(e)

    def _unpack_pose_jacobian_dict(self, poses, jacobians):
        pose_jacobian_dict = {}
        pose_counter = 0
        jac_counter = 0
        poses = np.array(poses)
        jacobians = np.array(jacobians)
        for link in self.config.LINK_NAMES:
            pose = poses[pose_counter:pose_counter + 3]
            jacobian = []
            for i in range(jac_counter, jac_counter+21, 7):
                jacobian.append(jacobians[i:i+7])
            jacobian = np.array(jacobian)
            pose_counter += 3
            jac_counter += 21
            pose_jacobian_dict[link] = [pose, jacobian]
        return pose_jacobian_dict

    def _get_positions_from_pose_jacobian_dict(self):
        poses = []
        for joint in self.pose_jacobian_dict.keys():
            poses.append(self.pose_jacobian_dict[joint][0])
        return np.array(poses)

    def get_pose_jacobian_dict_of_joints_not_in_box(self, safety_box):
        joint_dict = self.pose_jacobian_dict.copy()
        keys_to_remove = []
        for joint in joint_dict.keys():
            if self._pose_in_box(joint_dict[joint][0], safety_box):
                keys_to_remove.append(joint)
        for key in keys_to_remove:
            del joint_dict[key]
        return joint_dict

    def _pose_in_box(self, pose, safety_box):
        within_box = safety_box.contains(pose)
        return within_box

    def _get_adjustment_forces_per_joint_dict(self, joint_dict, safety_box):
        forces_dict = {}
        for joint in joint_dict:
            force = self._get_adjustment_force_from_pose(joint_dict[joint][0], safety_box)
            forces_dict[joint] = force
        return forces_dict

    def _get_adjustment_force_from_pose(self, pose, safety_box):
        x, y, z = 0, 0, 0

        curr_x = pose[0]
        curr_y = pose[1]
        curr_z = pose[2]

        if curr_x > safety_box.high[0]:
            x = -1 * np.exp(np.abs(curr_x - safety_box.high[0]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE
        elif curr_x < safety_box.low[0]:
            x = np.exp(np.abs(curr_x - safety_box.low[0]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE

        if curr_y > safety_box.high[1]:
            y = -1 * np.exp(np.abs(curr_y - safety_box.high[1]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE
        elif curr_y < safety_box.low[1]:
            y = np.exp(np.abs(curr_y - safety_box.low[1]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE

        if curr_z > safety_box.high[2]:
            z = -1 * np.exp(np.abs(curr_z - safety_box.high[2]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE
        elif curr_z < safety_box.low[2]:
            z = np.exp(np.abs(curr_z - safety_box.high[2]) * self.config.SAFETY_FORCE_TEMPERATURE) * self.config.SAFETY_FORCE_MAGNITUDE
        return np.array([x, y, z])

    def _compute_joint_distance_outside_box(self, pose, safety_box):
        curr_x = pose[0]
        curr_y = pose[1]
        curr_z = pose[2]
        if(self._pose_in_box(pose, safety_box)):
            x, y, z = 0, 0, 0
        else:
            x, y, z = 0, 0, 0
            if curr_x > safety_box.high[0]:
                x = np.abs(curr_x - safety_box.high[0])
            elif curr_x < safety_box.low[0]:
                x = np.abs(curr_x - safety_box.low[0])
            if curr_y > safety_box.high[1]:
                y = np.abs(curr_y - safety_box.high[1])
            elif curr_y < safety_box.low[1]:
                y = np.abs(curr_y - safety_box.low[1])
            if curr_z > safety_box.high[2]:
                z = np.abs(curr_z - safety_box.high[2])
            elif curr_z < safety_box.low[2]:
                z = np.abs(curr_z - safety_box.low[2])
        return np.linalg.norm([x, y, z])

    def _set_action_space(self):
        if self.action_mode == 'position':
            self.action_space = Box(
                self.config.POSITION_CONTROL_LOW,
                self.config.POSITION_CONTROL_HIGH,
                dtype=np.float32,
            )
        else:
            self.action_space = Box(
                self.config.JOINT_TORQUE_LOW,
                self.config.JOINT_TORQUE_HIGH,
                dtype=np.float32,
            )

    def _set_observation_space(self):
        if self.action_mode == 'torque':
            lows = np.hstack((
                self.config.JOINT_VALUE_LOW['position'],
                self.config.JOINT_VALUE_LOW['velocity'],
                self.config.END_EFFECTOR_VALUE_LOW['position'],
                self.config.END_EFFECTOR_VALUE_LOW['angle'],
            ))
            highs = np.hstack((
                self.config.JOINT_VALUE_HIGH['position'],
                self.config.JOINT_VALUE_HIGH['velocity'],
                self.config.END_EFFECTOR_VALUE_HIGH['position'],
                self.config.END_EFFECTOR_VALUE_HIGH['angle'],
            ))
            self.observation_space = Box(
                lows,
                highs,
                dtype=np.float32,
            )
        else:
            print('Not setting observation space')
            pass

    """
    ROS Functions
    """

    def init_rospy(self, update_hz):
        rospy.init_node('sawyer_env', anonymous=True)
        self.action_publisher = rospy.Publisher('actions_publisher', actions, queue_size=10)
        self.rate = rospy.Rate(update_hz)

    def send_action(self, action):
        self.action_publisher.publish(action)

    def send_angle_action(self, action, target):
        self.request_angle_action(action, target)

    def request_image(self):
        rospy.wait_for_service('images')
        try:
            request = rospy.ServiceProxy('images', image, persistent=True)
            obs = request()
            return (
                    obs.image
            )
        except rospy.ServiceException as e:
            print(e)

    def crop_image(self, img):
        endcol = self.img_start_col + self.img_col_delta
        endrow = self.img_start_row + self.img_row_delta
        img = copy.deepcopy(img[self.img_start_row:endrow, self.img_start_col:endcol])
        return img

    def get_image(self, width=84, height=84):
        image = self.request_image()
        if image is None:
            raise Exception('Unable to get image from image server')
        image = np.array(image).reshape(1000, 1000, 3)
        image = copy.deepcopy(image)
        image = cv2.resize(image, (0, 0), fx=width/1000, fy=height/1000, interpolation=cv2.INTER_AREA)
        image = np.asarray(image).reshape(width, height, 3)
        return image

    def request_observation(self):
        rospy.wait_for_service('observations')
        try:
            request = rospy.ServiceProxy('observations', observation, persistent=True)
            obs = request()
            return (
                    np.array(obs.angles),
                    np.array(obs.velocities),
                    np.array(obs.endpoint_pose),
                    np.array(obs.endpoint_velocity)
            )
        except rospy.ServiceException as e:
            print(e)


    def request_angle_action_constant_rate(self, angles, desired_pose, clip_joints=True, reset=False):
        curr_pose = self._get_endeffector_pose()

        if clip_joints:
            # clip the joint angles to avoid contortions
            curr_joint_angles = self._get_joint_angles()
            max_joint_speed = np.ones(len(curr_joint_angles)) * .2
            max_time = 1.0
            old_angles = angles
            angle_diff = angles - curr_joint_angles
            angle_diff = np.clip(angle_diff, -1 * max_joint_speed * max_time, max_joint_speed * max_time)
            angles = curr_joint_angles + angle_diff

            # if np.any(old_angles - angles):
            #     print('clipped joint angles')
            #     print(old_angles, '\n', angles)

        # control the speed by computing an appropriate action duration
        # print('\n REQUESTING ACTION')
        #print('old joints', self._get_joint_angles())
        #print('new joints', angles)
        pos_dist = np.linalg.norm(curr_pose[:3] - desired_pose[:3])
        pos_duration = (pos_dist / self.max_speed) * 1
        # compute a rough distance between quaternions
        quat_dist = 1 - np.abs(np.inner(curr_pose[3:], desired_pose[3:]))
        # the max position distance is about .07 (due to action scaling), so we scale by 0.07 to make them about the same
        quat_duration = (quat_dist / self.max_speed) * .07
        # this may not work because the quat distance can be zero while the joint angle distance is large
        # compute the distance between the current wrist joint angle and the one output by the IK
        # wrist ranges through 3pi, -4.7 to 4.7 so the max this can be is 9.4
        # if we want the outer point of the wrist cuff to travel at no more than max_speed, and the wrist radius is about 5cm, then the max duration for a full 3pi should be about 9.4 seconds, + 20% for safety is 11.28 seconds
        # this calculation comes out to a multiplier of .06 but .12 looks safer in practice, probably because this doesn't take into account that all the joints moving together can make the ee move much faster
        angles_dist = np.abs(angles - self._get_joint_angles())
        angles_dist = max(angles_dist)
        angles_duration = (angles_dist / self.max_speed) * .2
        duration = max(pos_duration, quat_duration, angles_duration)
        if duration > 7:
            print('wanted to do crazy trajectory')
            #raise(Exception)
        rospy.wait_for_service('angle_action')
        try:
            execute_action = rospy.ServiceProxy('angle_action', angle_action, persistent=True)
            execute_action(angles=angles, duration=duration, constant_hz=(not reset))
            return None
        except rospy.ServiceException as e:
            pass

    def request_angle_action(self, angles, pos, reset=False):
        # NOTE: not used for velocity controller, may no longer work!
        if self.constant_hz and not reset:
            self.request_angle_action_constant_rate(angles, pos)
        if reset:
            dist = np.linalg.norm(self._get_endeffector_position() - pos[:3])
            duration = dist/self.max_speed
            rospy.wait_for_service('angle_action')
            try:
                execute_action = rospy.ServiceProxy('angle_action', angle_action, persistent=True)
                execute_action(angles, duration, -1, False)
                return None
            except rospy.ServiceException as e:
                pass
        else:
            dist = np.linalg.norm(self._get_endeffector_position() - pos[:3])
            duration = dist/self.max_speed
            rospy.wait_for_service('angle_action')
            try:
                execute_action = rospy.ServiceProxy('angle_action', angle_action, persistent=True)
                execute_action(angles, duration, 1.0/20, False)
                return None
            except rospy.ServiceException as e:
                pass

    def request_ik_angles(self, ee_pos, seed_angles):
        rospy.wait_for_service('ik')
        try:
            get_joint_angles = rospy.ServiceProxy('ik', ik, persistent=True)
            resp = get_joint_angles(ee_pos, seed_angles)

            return (
                resp.joint_angles
            )
        except rospy.ServiceException as e:
            print(e)

    """
    Multitask functions
    """

    @property
    def goal_dim(self):
        raise NotImplementedError()

    def get_goal(self):
        return self._state_goal

    def set_goal(self, goal):
        self._state_goal = goal

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self._state_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.goal_space.low,
                self.goal_space.high,
                size=(batch_size, self.goal_space.low.size),
            )
        return goals

    """
    Image Env Functions
    """

    def get_env_state(self):
        return self._get_joint_angles(), self._get_endeffector_pose()

    def set_env_state(self, env_state):
        angles, ee_pos = env_state
        for _ in range(3):
            self.send_angle_action(angles, ee_pos)

    def initialize_camera(self, init_fctn):
        pass
