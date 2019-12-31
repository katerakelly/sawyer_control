import numpy as np
from gym.spaces import Box
from pyquaternion import Quaternion
from sawyer_control.coordinates import quat_2_euler, euler_2_rot, euler_2_quat
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable


class SawyerPegEnv(SawyerEnvBase):
    '''
    sawyer peg insertion task
    '''
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())
        SawyerEnvBase.__init__(self, **kwargs)

        # reset pose for start of each episode
        self.pos_control_reset_position = self.config.POSITION_RESET_POS
        pc= self.config.POSITION_CONTROL_EE_ORIENTATION
        self.reset_orientation = np.array([pc.x, pc.y, pc.z, pc.w])
        self.reset_pose = np.concatenate((self.pos_control_reset_position, self.reset_orientation))

        # set ee position and angle safety box
        # TODO have not verified these angles for vestri robot
        self.ee_angle_lows = np.array([-0.05, -0.05, -np.pi * .75])
        self.ee_angle_highs = np.array([0.05, 0.05, np.pi * .75])
        self.ee_pos_lows = self.config.POSITION_SAFETY_BOX_LOWS
        self.ee_pos_highs = self.config.POSITION_SAFETY_BOX_HIGHS
        self.position_action_scale = .05 # 5cm

        self._set_action_space()
        self._set_observation_space()

        # goal pose is when the peg is correctly inserted
        self.goal_pos = self.pos_control_reset_position + np.array([.1, .1, .1]) #TODO just made this up
        self.goal_orientation = self.reset_orientation # TODO change for orientation control
        self.goal_pose = np.concatenate((self.goal_pos, self.goal_orientation))

        # reset robot to initialize
        self.reset()

    def _set_observation_space(self):
        # obs is full 7-dof endpoint pose
        # NOTE add endpoint velocity?
        lows = np.concatenate((self.ee_pos_lows, self.ee_angle_lows))
        highs = np.concatenate((self.ee_pos_highs, self.ee_angle_highs))
        self.observation_space = Box(lows, highs)

    def _set_action_space(self):
        # TODO change for orientation control
        self.action_space = Box(-1 * np.ones(3), np.ones(3), dtype=np.float32)

    def _get_obs(self):
        # obs is full current ee pose: position + orientation (length 7)
        # NOTE: obs also include ee velocity! (length 6)
        _, _, ee_pose, ee_vel = self.request_observation()
        return np.concatenate([ee_pose, ee_vel])

    def _pose_from_obs(self, obs):
        return obs[:7]

    def _vel_from_obs(self, obs):
        return obs[7:]

    def _move_by(self, action):
        # action consists of (x, y, z) position and rotation about the z axis
        # determine new desired pose
        # full pose is (x, y, z) + rot quaternion
        ee_full_pose = self._pose_from_obs(self._get_obs())

        # first deal with the position part
        pos_act = action[:3] * self.position_action_scale
        ee_pos = ee_full_pose[:3]
        target_ee_pos = pos_act + ee_pos
        old_ee_pos = np.copy(target_ee_pos)
        target_ee_pos = np.clip(target_ee_pos, self.ee_pos_lows, self.ee_pos_highs)
        if np.any(old_ee_pos - target_ee_pos):
            print('safety box violated, position clipped')

        # next deal with the orientation
        # scale down the rotation to within +-15 degrees
        # TODO: this only considers rotation about z-axis for now (4DoF)
        '''
        rot_act = action[-1] * 0.262
        rot_act = np.array([0, 0, rot_act])
        # get current euler angles from the pose
        curr_eulers = np.array(quat_2_euler(ee_full_pose[3:]))
        # compute new euler angles by adding the action
        new_eulers = curr_eulers + rot_act
        # handle angle wrap
        if new_eulers[-1] < -np.pi:
            new_eulers[-1] = new_eulers[-1] + 2 * np.pi
        elif new_eulers[-1] > np.pi:
            new_eulers[-1] = new_eulers[-1] - 2 * np.pi
        # clip angles to keep them safe
        if new_eulers[-1] < 0:
            # TODO: this is a hack to prevent double solutions
            new_eulers[-1] = -np.pi
            #new_eulers[-1] = min(new_eulers[-1], -np.pi * .75)
        elif new_eulers[-1] > 0:
            new_eulers[-1] = max(new_eulers[-1], np.pi * .75)
        # compute the desired quaternion
        quat_new = euler_2_quat(*new_eulers)
        '''
        # TODO no rotation for now
        quat_new = self.goal_pose[3:]
        target_pose = np.concatenate([target_ee_pos, quat_new])
        self._move_to(target_pose)

    def _move_to(self, target_ee_pose, reset=False):
        # combine position and orientation and send to IK
        angles = self.request_ik_angles(target_ee_pose, self._get_joint_angles())
        self.request_angle_action(angles, target_ee_pose, reset=reset)

    def compute_rewards(self, action, pose):
        return -np.linalg.norm(pose[:3] - self.goal_pose[:3]), []

    # TODO use this reward function to penalize the full pose of the end-effector
    # it also uses the peaked GPS reward function that seems to work better for tasks
    # that require precision
    def _compute_rewards_pose(self, action, pose):
        # penalize the full pose
        # compute rotation matrix between goal and curent frame
        goal_quat = Quaternion(self.goal_pose[3:])
        curr_quat = Quaternion(pose[3:])
        rotation = (curr_quat * goal_quat.inverse).rotation_matrix

        # determine translation vector
        translation = (pose[:3] - self.goal_pose[:3])[..., None]

        # construct transformation matrix in homogenous coordinates
        transformation = np.concatenate([rotation, translation], axis=1)
        assert transformation.shape == (3, 4)
        # pick three points roughly on the peg in the goal ee frame
        # (include all three directions in order to penalize the full pose)
        x = np.array([-.01, 0, -.01, 1])
        y = np.array([0, -.01, -.01, 1])
        z = np.array([0, 0, -.05, 1])

        # transform each point into the current ee frame
        x_t = transformation.dot(x)
        y_t = transformation.dot(y)
        z_t = transformation.dot(z)

        # NOTE debugging save these for inspection
        points = np.array([x, y, z, x_t, y_t, z_t])

        # stack points into one vector and compute distance
        dist = np.linalg.norm(np.concatenate([x[:-1], y[:-1], z[:-1]]) - np.concatenate([x_t, y_t, z_t]))
        # use cm as base unit
        dist *= 100

        # GPS paper cost function
        gps_reward = -(1.0 * np.square(dist) + 1.0 * np.log(np.square(dist) + 1e-5))
        return gps_reward, points

    def reset(self):
        print('resetting...')
        print('from:', self._get_obs()[:3])
        print('to:', self.reset_pose[:3])
        # return to fixed ee position and orientation
        self._move_to(self.reset_pose, reset=True)
        print('reset complete!')
        return self._pose_from_obs(self._get_obs())

    def step(self, action):
        # for now action is 3 DOF
        self._move_by(action)
        obs = self._get_obs()
        pose = self._pose_from_obs(obs)
        # reward based on position only for now
        reward, points = self.compute_rewards(action, pose)
        print('reward', reward)
        info = self._get_info()
        done = False
        return pose, reward, done, info



