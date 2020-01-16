#!/usr/bin/env python
from sawyer_control.srv import *
import rospy
import intera_interface as ii
from sawyer_control.pd_controllers.inverse_kinematics import get_pose_stamped, get_joint_angles
from sawyer_control.configs import ros_config


from geometry_msgs.msg import Quaternion

POSITION_CONTROL_EE_ORIENTATION=Quaternion(
    x=1.0, y=-.5, z=0.0, w=0.0
)

def compute_joint_angle(req):
    ee_pose = req.ee_pos

    Q = Quaternion(x=ee_pose[3], y=ee_pose[4], z=ee_pose[5], w=ee_pose[6]) #ros_config.POSITION_CONTROL_EE_ORIENTATION
    pose = get_pose_stamped(ee_pose[0], ee_pose[1], ee_pose[2], Q)

    reset_angles = req.curr_joint_angles #ros_config.RESET_ANGLES
    reset_angles = dict(zip(joint_names, reset_angles))

    ik_angles = get_joint_angles(pose, reset_angles, True, False)
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
