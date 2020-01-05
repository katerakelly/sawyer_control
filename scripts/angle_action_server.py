#!/usr/bin/env python
import intera_interface as ii
from sawyer_control.pd_controllers.velocity_controller import VelocityController
from sawyer_control.srv import *
import rospy

def execute_action(action_msg):
    # unpack the message
    action = action_msg.angles
    duration = action_msg.duration
    reset = action_msg.reset
    rate = action_msg.rate

    # assign actions to joints and execute motion
    joint_names = arm.joint_names()
    joint_to_values = dict(zip(joint_names, action))
    if joint_to_values:
        ja = [joint_to_values[name] for name in arm.joint_names()]
        controller.update_plan([ja], duration=duration, reset=reset, action_update_rate=rate)
        return angle_actionResponse(True)
    return angle_actionResponse(False)

def angle_action_server():
    rospy.init_node('angle_action_server', anonymous=True)
    global arm
    global controller
    arm = ii.Limb('right')
    arm.set_joint_position_speed(0.1)
    controller = VelocityController(control_rate=1000)
    s = rospy.Service('angle_action', angle_action, execute_action)
    rospy.spin()

if __name__ == '__main__':
    angle_action_server()
