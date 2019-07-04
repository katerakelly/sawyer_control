''' helper methods for coordinates '''
import numpy as np
from pyquaternion import Quaternion


def quat_2_euler(quat):
    # calculates and returns: yaw, pitch, roll from given quaternion
    if not isinstance(quat, Quaternion):
        quat = Quaternion(quat)
    yaw, pitch, roll = quat.yaw_pitch_roll
    #return yaw + np.pi, pitch, roll
    return yaw, pitch, roll

def euler_2_rot(yaw=0.0, pitch=0.0, roll=np.pi):
    yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],[np.sin(yaw), np.cos(yaw), 0.0], [0, 0, 1.0]])
    pitch_matrix = np.array([[np.cos(pitch), 0., np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0, np.cos(pitch)]])
    roll_matrix = np.array([[1.0, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
    return rot_mat

def euler_2_quat(yaw=np.pi/2, pitch=0.0, roll=np.pi):
    rot_mat = euler_2_rot(yaw, pitch, roll)
    return Quaternion(matrix=rot_mat).elements
