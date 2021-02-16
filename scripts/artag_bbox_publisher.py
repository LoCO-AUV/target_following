#!/usr/bin/python

# This code is a part of the LoCO AUV project.
# Copyright (C) The Regents of the University of Minnesota

# Maintainer: Junaed Sattar <junaed@umn.edu> and the Interactive Robotics and Vision Laboratory

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from ar_recog.msg import Tags
from cv_bridge import CvBridge, CvBridgeError
from threading import Lock
import numpy as np
from target_following.msg import TargetObservation, TargetObservations


'''
class for tracking a AR-Tag using the ar_recog package
kill the basic ar_recog node before launching the ar_recog_pid_test node (tag_detect.launch)
it publishes the target(tag) bounding box
-- Xahid 03/26/18
'''
class ARTagBBox(object):
    def __init__(self):
        rospy.init_node('artag_bbox_publisher')

        self.bbox = None
        self.curr_img = None
        self.bridge = CvBridge()
        self.publish_image = True
        self.mutex = Lock()

        self.im_pub  = rospy.Publisher("/artag_bbox/image_raw", Image, queue_size=3)
        self.bbox_pub  = rospy.Publisher("/target/observation", TargetObservations, queue_size=3)
        
        self.im_sub  = rospy.Subscriber("/aqua/ar/image", Image, self.image_callback)
        self.ar_sub = rospy.Subscriber("/aqua/tags", Tags, self.artag_callback)
        

        
    def artag_callback(self, msg):
        self.mutex.acquire()
        
        if self.curr_img is None:
            self.mutex.release()
            return
        
        obs = TargetObservation()
        obs.header.stamp = rospy.Time.now()

        mobs = TargetObservations()
        mobs.header.stamp = obs.header.stamp
        
        #print ('Got tags', len(msg.tags))
        if len(msg.tags) == 0:
	    self.cur_id = None
            self.bbox = None
            obs.target_visible = False
            
        else:
            tag = msg.tags[0]
	    self.cur_id = tag.id
            a = (tag.cwCorners[0], tag.cwCorners[1])  
            b = (tag.cwCorners[2], tag.cwCorners[3])  
            c = (tag.cwCorners[4], tag.cwCorners[5])  
            d = (tag.cwCorners[6], tag.cwCorners[7])  
            
            top_left_x = min(a[0], b[0], c[0], d[0])
            top_left_y = min(a[1], b[1], c[1], d[1])

            bottom_right_x = max(a[0], b[0], c[0], d[0])
            bottom_right_y = max(a[1], b[1], c[1], d[1])
            
            width  = bottom_right_x - top_left_x -1.0
            height = bottom_right_y - top_left_y -1.0

            rows, cols, channels = self.curr_img.shape

            self.bbox = np.zeros(4)
            self.bbox[0] = top_left_x 
            self.bbox[1] = top_left_y 
            self.bbox[2] = width 
            self.bbox[3] = height

            
            obs.target_visible = True
            obs.top_left_x = self.bbox[0]
            obs.top_left_y = self.bbox[1]
            obs.width = self.bbox[2]
            obs.height = self.bbox[3]
            obs.image_width = cols
            obs.image_height = rows
            obs.class_prob = 1.0
            obs.class_name = 'aqua'

            mobs.observations.append(obs)

            
        self.mutex.release()
        self.bbox_pub.publish(mobs)

        
    def image_callback(self, msg):
        
        self.mutex.acquire()
        try:
            self.curr_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            
            if self.bbox is None:
		print ('here')
		self.im_pub.publish(self.bridge.cv2_to_imgmsg(self.curr_img, "rgb8"))
            else:
		print (self.cur_id, tuple(self.bbox.astype('int32')))
                top_left_x, top_left_y, w, h = tuple(self.bbox.astype('int32'))
                bottom_right = (top_left_x + w, top_left_y + h)
                if self.publish_image:
                    cv2.rectangle(self.curr_img, (top_left_x, top_left_y), bottom_right, (0, 0, 255), 2)
                    self.im_pub.publish(self.bridge.cv2_to_imgmsg(self.curr_img, "rgb8"))
            
        except CvBridgeError as e:
            print(e)

        self.mutex.release()
        

# for testing
if __name__ == "__main__":
    artbb = ARTagBBox()
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        rate.sleep()
