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
# max_accel_mag = np.array([3.5, 2.5, 5, 5, 5, 5, 5])

class CSpline:
    def __init__(self, points, duration=1., bc_type='natural'):
        n_points = points.shape[0]
        self._duration = duration
        self._cs = CubicSpline(np.linspace(0, duration, n_points), points, bc_type=bc_type)

    def get(self, t):
        t = np.array(min(t, self._duration))
        return self._cs(t), self._cs(t, nu=1), self._cs(t, nu=2)

    def get_vel(self, t):
        t_arr = np.array(t)
        pos = self._cs(t_arr)
        vel = self._cs(t_arr, nu=1)
        accel = self._cs(t_arr, nu=2)
        if t>self._duration:
            # I have completed executing my plan, so I should stop
            # note that t gets reset to 0 once a new waypoint comes in
            # so this "vel=0" only happens if no new command comes in and we're already at the waypoint
            vel = 0*vel
        return vel

class VelocityController():
    '''
    policy still outputs end-effector positions

    this controller fits a spline to these waypoints
    and commands the robot with velocities
    # TODO: does this make any difference??

    :param: control_rate - frequency controller sends commands to robot
    :param: update_rate - frequency policy sends new waypoint and trajectory is updated
    '''
    def __init__(self, control_rate, update_rate=20):
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        print("Robot enabled...")
        self.limb = intera_interface.Limb("right")
        self.head = intera_interface.Head()
        self.head.set_pan(angle=0, speed=1.0, timeout=10.0, active_cancellation=True)
        print("Done initializing controller.")
        self._cmd_publisher = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, queue_size=100)
        self.control_rate = rospy.Rate(control_rate)

        # get a new action this many times per second (this is the frequency of the policy)
        self.action_update_rate = rospy.Rate(update_rate)
        self.rate_is_correct = False
        self.move_calls_between_update = 0

        self.jointnames = self.limb.joint_names()
        self.prev_joint = np.array([self.limb.joint_angle(j) for j in self.jointnames])
        self.update_plan([self.prev_joint, ])

        self.force_threshold = 20 # Force in Newtons
        self.force_check_interval = 1
        self.last_safe_position = self.limb.joint_angles()

        # will call self.move() control_rate times per second
        self.move_timer = rospy.Timer(rospy.Duration(1.0 / control_rate), self.move)

    def update_plan(self, waypoints, duration=1.5, reset=False, action_update_rate=None):

        """ 
        This function is called in the case of a new waypoint being sent.
        Updates the current plan (self.spline) by creating a new one, from curr to goal
        :param waypoints: List of arrays containing waypoint joint angles (or just a single goal waypoint)
        :param duration: trajectory duration (how long it should take to achieve this waypoint)
        """

        ############################################################
        #### set update rate, to control plan making (i.e. control sending)
        ############################################################

        if not self.rate_is_correct and action_update_rate is not None:
            self.action_update_rate = rospy.Rate(action_update_rate)
            self.rate_is_correct = True

        if not waypoints:
            return

        ############################################################
        #### make plan
        ############################################################
        # given (curr,goal) where time is duration
        # create spline (curr,a,b,c,d,...,goal)
            # total time curr-->goal is `duration`
            # time betw each step of spline is `dt` in the move function below
        
        # Each SLOW call to this `update_plan` creates a new splines from curr to goal
        # Each FAST call to `move` progresses it forward by 1 entry of the spline
        ############################################################

        self.move_calls_between_update = 0

        # make a spline between prev_joint and waypoints
        self.prev_joint = np.array([self.limb.joint_angle(j) for j in self.jointnames])
        self.waypoints = np.array([self.prev_joint] + waypoints)
        self.spline = CSpline(self.waypoints, duration)

        ############################################################
        #### execute that command for fixed amount of time (by doing sleep)
        #### or if doing reset, do it as long as needed
        ############################################################

        self.start_time = rospy.get_time()
        # policy controls the system at a fixed frequency
        if not reset:
            # waits for just 1/control_rate seconds
            self.action_update_rate.sleep()
            # if action_update_rate is not None:
                # print("Running commands at fixed rate: ", action_update_rate, " Hz, ", 1/action_update_rate, " sec")

        # during reset, take the whole alloted time to complete the move
        else:
            # print("Running a command for ", duration, " seconds.")
            finish_time = self.start_time + duration
            while rospy.get_time() < finish_time:
                self.action_update_rate.sleep()

    def move(self, timer_event):
        """ 
        This function is called all the time, regardless of new waypoints being sent or not.
        Called FAST, by ROS 1/control_rate times per second .
        Move according to the current plan (self.spline)
        """

        self.move_calls_between_update += 1

        ########################################
        ### check forces
        ########################################

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

        ########################################
        ### get the right point from spline
        # which point = dt = {0, ..., duration}
        # note that start_time only gets reset when receiving a new command
        # so if not new command comes in, then dt = {0, ..., duration, ....}
        ########################################

        time = rospy.get_time()
        dt = time - self.start_time
        velocity = self.spline.get_vel(dt)

        ############################
        ### send velocity command
        ############################

        command = JointCommand()
        command.mode = JointCommand.VELOCITY_MODE
        command.names = self.jointnames
        command.velocity = np.clip(velocity, -max_vel_mag, max_vel_mag)
        self._cmd_publisher.publish(command)
        self.control_rate.sleep()
