#!/usr/bin/env python
from sawyer_control.srv import *
import rospy
import intera_interface as ii
from sawyer_control.pd_controllers.inverse_kinematics import get_pose_stamped, get_joint_angles
from sawyer_control.configs import ros_config
from geometry_msgs.msg import Quaternion

def compute_joint_angle(req):
    ee_pos = req.ee_pos
    if req.seed_angles:
        seed_angles = req.seed_angles
    else:
        seed_angles = ros_config.RESET_ANGLES
    seed_angles = dict(zip(joint_names, seed_angles))
    pose = get_pose_stamped(ee_pos[0], ee_pos[1], ee_pos[2], Quaternion(x=ee_pos[3], y=ee_pos[4], z=ee_pos[5], w=ee_pos[6]))
    print("IK POSE: " + str(pose))
    print("BASE ANGLES: " + str(seed_angles))
    ik_angles = get_joint_angles(pose, seed_angles, True, False)
    ik_angles = [ik_angles[joint] for joint in joint_names]
    return ikResponse(ik_angles)

def inverse_kinematics_server():
    rospy.init_node('ik_server', anonymous=True)
    global arm
    global joint_names
    arm = ii.Limb('right')
    joint_names = arm.joint_names()
    s = rospy.Service('ik', ik, compute_joint_angle)
    rospy.spin()

if __name__ == "__main__":
    inverse_kinematics_server()
