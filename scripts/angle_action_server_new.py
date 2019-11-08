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

class VelocityController():
    def __init__(self, control_rate, update_rate=20):
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        print("Robot enabled...")
        self.limb = intera_interface.Limb("right")
        self.head = intera_interface.Head()
        self.head.set_pan(angle=0, speed=1.0, timeout=10.0, active_cancellation=True)
        print("Done initializing controller.")
        self._cmd_publisher = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, queue_size=100)
        # self.action_subscriber = rospy.Subscriber('/robot/limb/right/joint_command', JointCommand, queue_size=100)
        self.control_rate = rospy.Rate(control_rate)

        # get a new action this many times per second (this is the frequency of the policy)
        self.action_update_rate = rospy.Rate(update_rate)
        self.move_calls_between_update = 0

        self.jointnames = self.limb.joint_names()
        self.prev_joint = np.array([self.limb.joint_angle(j) for j in self.jointnames])
        self.update_plan([self.prev_joint, ])

        self.force_threshold = 20 # Force in Newtons
        self.force_check_interval = 1
        self.last_safe_position = self.limb.joint_angles()

        # will call self.move() control_rate times per second
        self.move_timer = rospy.Timer(rospy.Duration(1.0 / control_rate), self.move)

    def update_plan(self, waypoints, duration=1.5, constant_hz=True):
        """
        Updates the current plan with waypoints
        :param waypoints: List of arrays containing waypoint joint angles
        :param duration: trajectory duration
        """

        if not waypoints:
            return

        self.move_calls_between_update = 0

        self.prev_joint = np.array([self.limb.joint_angle(j) for j in self.jointnames])
        self.waypoints = np.array([self.prev_joint] + waypoints)

        # plot a course to the goal position given a desired duration (usually computed from safety bounds)
        self.spline = CSpline(self.waypoints, duration)


        self.start_time = rospy.get_time()
        # policy controls the system at a fixed frequency
        if constant_hz:
            # waits for just 1/control_rate seconds
            self.action_update_rate.sleep()

        # during reset, take the whole alloted time to complete the move
        else:
            finish_time = self.start_time + duration
            while rospy.get_time() < finish_time:
                print("UPDATED PLAN")
                self.action_update_rate.sleep()

    def move(self, timer_event):
        """ Move according to the current plan, will be called by ROS 1/control_rate times per second """

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
    constant_hz = action_msg.constant_hz
    duration = action_msg.duration
    joint_names = arm.joint_names()
    joint_to_values = dict(zip(joint_names, action))
    if joint_to_values:
        ja = [joint_to_values[name] for name in arm.joint_names()]
        controller.update_plan([ja], duration=duration, constant_hz=constant_hz)
        return angle_actionResponse(True)
    return angle_actionResponse(False)

def angle_action_server():
    rospy.init_node('angle_action_server', anonymous=True)
    global arm
    global controller
    arm = ii.Limb('right')
    arm.set_joint_position_speed(0.1)
    controller = VelocityController(control_rate=100, update_rate=20)
    s = rospy.Service('angle_action', angle_action, execute_action)
    rospy.spin()

if __name__ == '__main__':
    angle_action_server()
