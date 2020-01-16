#!/usr/bin/env python
from sawyer_control.srv import *
import rospy
from sensor_msgs.msg import Image as Image_msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
import copy
import thread
import numpy as np


class Latest_observation(object):
    def __init__(self):
        # color image:
        self.img_cv2_ee = None
        self.img_cv2_overall = None
        # self.img_cropped = None
        self.img_msg_ee = None
        self.img_msg_overall = None

class WebcamRecorder(object):
    def __init__(self):

        rospy.Subscriber("/overall_webcam/image_raw", Image_msg, self.store_latest_overall_image)
        rospy.Subscriber("/ee_webcam/image_raw", Image_msg, self.store_latest_ee_image)
        self.ltob = Latest_observation()
        self.bridge = CvBridge()

        def spin_thread():
            rospy.spin()

        thread.start_new(spin_thread, ())

    def store_latest_overall_image(self, data):
        self.incoming_width_overall = data.width
        self.incoming_height_overall = data.height
        self.ltob.img_msg_overall = data
        self.ltob.img_cv2_overall = self.bridge.imgmsg_to_cv2(data, "bgr8")  # (480, 640)
    def store_latest_ee_image(self, data):
        self.incoming_width_ee = data.width
        self.incoming_height_ee = data.height
        self.ltob.img_msg_ee = data
        self.ltob.img_cv2_ee = self.bridge.imgmsg_to_cv2(data, "bgr8")  # (480, 640)

def get_image_from_webcam(unused):
    img_ee = recorder.ltob.img_cv2_ee
    img_overall = recorder.ltob.img_cv2_overall

    img_ee = np.array(img_ee)
    img_overall = np.array(img_overall)

    img = np.concatenate([img_overall, img_ee], axis=0)
    image = img.flatten().tolist()
    return imageResponse(image)

def image_server():
    s = rospy.Service('images_webcam_double', image, get_image_from_webcam)
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node('double_webcam_image_server', anonymous=True)
    recorder = WebcamRecorder()
    image_server()

