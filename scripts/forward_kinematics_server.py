#!/usr/bin/env python

from sawyer_control.srv import *
import rospy
import numpy as np
from urdf_parser_py.urdf import URDF
import intera_interface as ii
from pykdl_utils.kdl_kinematics import KDLKinematics



def handle_ee_fk(ee_pose_msg):
    joint_positions = ee_pose_msg.joint_positions
    pose = kin.forward(joint_positions, 'right_hand')
    pose = np.squeeze(np.asarray(pose))
    # pose = np.zeros((4,4))
    return eePoseResponse(pose.flatten().tolist())


def get_ee_fk_server():
    rospy.init_node('get_ee_fk_server')
    robot = URDF.from_parameter_server(key='robot_description')

    global kin
    global arm

    kin = KDLKinematics(robot, 'base', 'right_hand')

    arm = ii.Limb('right')

    s = rospy.Service('get_ee_fk', eePose, handle_ee_fk)
    rospy.spin()


if __name__ == "__main__":
    get_ee_fk_server()
