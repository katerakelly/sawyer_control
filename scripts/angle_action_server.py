#!/usr/bin/env python
import intera_interface as ii
from sawyer_control.srv import angle_action, angle_actionResponse
import rospy

import rospy
import numpy as np
from intera_core_msgs.msg import JointCommand
import intera_interface
from intera_interface import CHECK_VERSION
from scipy.interpolate import CubicSpline

from sawyer_control.srv import observation

# constants for robot control
max_vel_mag = np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316])
max_accel_mag = np.array([3.5, 2.5, 5, 5, 5, 5, 5])

class CSpline:
    def __init__(self, points, duration=1., bc_type='clamped'):
        n_points = points.shape[0]
        self._duration = duration
        self._cs = CubicSpline(np.linspace(0, duration, n_points), points, bc_type=bc_type)

    def get(self, t):
        t = np.array(min(t, self._duration))

        return self._cs(t), self._cs(t, nu=1), self._cs(t, nu=2)

class PositionController():
    def __init__(self, control_rate):
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        print("Robot enabled...")
        self.limb = intera_interface.Limb("right")
        self.head = intera_interface.Head()
        self.head.set_pan(angle=0, speed=1.0, timeout=10.0, active_cancellation=True)
        print("Done initializing controller.")
        self._cmd_publisher = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, queue_size=100)
        # self.action_subscriber = rospy.Subscriber('/robot/limb/right/joint_command', JointCommand, queue_size=100)
        self.control_rate = rospy.Rate(control_rate)

        self.action_update_rate = rospy.Rate(10)
        self.move_calls_between_update = 0

        self.jointnames = self.limb.joint_names()
        self.prev_joint = np.array([self.limb.joint_angle(j) for j in self.jointnames])
        self.update_plan([self.prev_joint, ])

        self.force_threshold = 20 # Force in Newtons
        self.force_check_interval = 1
        self.last_safe_position = self.limb.joint_angles()

        self.move_timer = rospy.Timer(rospy.Duration(1.0 / control_rate), self.move)

    def update_plan(self, waypoints, duration=1.5):
        """
        Updates the current plan with waypoints
        :param waypoints: List of arrays containing waypoint joint angles
        :param duration: trajectory duration
        """

        if not waypoints:
            return

        rospy.logerr(self.move_calls_between_update)
        self.move_calls_between_update = 0

        self.prev_joint = np.array([self.limb.joint_angle(j) for j in self.jointnames])
        self.waypoints = np.array([self.prev_joint] + waypoints)

        self.spline = CSpline(self.waypoints, duration)

        self.start_time = rospy.get_time()  # in seconds
        # finish_time = start_time + duration  # in seconds

        # while time < finish_time:
        self.action_update_rate.sleep()

    def move(self, timer_event):
        """Move according to the current plan"""

        self.move_calls_between_update += 1

        forces = self.limb.endpoint_effort()["force"]
        forces = np.array([forces.x, forces.y, forces.z])

        if max(abs(forces)) > self.force_threshold:
            # abandon the plan and stay here, the force is too high
            rospy.logerr(forces)
            rospy.logerr("Forces too high - going back one step")
            self.limb.move_to_joint_positions(self.last_safe_position)
            current_joints = np.array([self.limb.joint_angle(j) for j in self.jointnames])
            self.update_plan([current_joints, ])
        else:
            self.last_safe_position = self.limb.joint_angles()

        time = rospy.get_time()
        dt = time - self.start_time
        # rospy.logerr(str(dt))
        pos, velocity, acceleration = self.spline.get(dt)
        command = JointCommand()
        command.mode = JointCommand.VELOCITY_MODE # important difference to previous controller
        command.names = self.jointnames
        command.position = pos
        command.velocity = np.clip(velocity, -max_vel_mag, max_vel_mag)
        command.acceleration = np.clip(acceleration, -max_accel_mag, max_accel_mag)
        self._cmd_publisher.publish(command)
        self.control_rate.sleep()

def execute_action(action_msg):
    action = action_msg.angles
    joint_names = arm.joint_names()
    joint_to_values = dict(zip(joint_names, action))
    duration = action_msg.duration
    if joint_to_values:
        ja = [joint_to_values[name] for name in arm.joint_names()]
        controller.update_plan([ja], duration=duration)
        return angle_actionResponse(True)
    return angle_actionResponse(False)

def angle_action_server():
    rospy.init_node('angle_action_server', anonymous=True)
    global arm
    global controller
    arm = ii.Limb('right')
    arm.set_joint_position_speed(0.1)
    controller = PositionController(control_rate=1000)
    s = rospy.Service('angle_action', angle_action, execute_action)
    rospy.spin()

if __name__ == '__main__':
    angle_action_server()
