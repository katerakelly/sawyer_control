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
        self.img_cv2 = None
        self.img_cropped = None
        self.img_msg = None

class WebcamRecorder(object):
    def __init__(self):

        rospy.Subscriber("/overall_webcam/image_raw", Image_msg, self.store_latest_image)
        self.ltob = Latest_observation()
        self.bridge = CvBridge()

        def spin_thread():
            rospy.spin()

        thread.start_new(spin_thread, ())

    def store_latest_image(self, data):
        self.incoming_width = data.width
        self.incoming_height = data.height
        self.ltob.img_msg = data
        self.ltob.img_cv2 = self.bridge.imgmsg_to_cv2(data, "bgr8")  # (480, 640)

def get_image_from_webcam(unused):
    img = recorder.ltob.img_cv2
    img = np.array(img)
    image = img.flatten().tolist()
    return imageResponse(image)

def image_server():
    s = rospy.Service('images_webcam_overall', image, get_image_from_webcam)
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node('overall_webcam_image_server', anonymous=True)
    recorder = WebcamRecorder()
    image_server()

