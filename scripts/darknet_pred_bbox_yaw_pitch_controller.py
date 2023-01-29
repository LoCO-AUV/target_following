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

import rospy
from loco_pilot.msg import Command
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from diver_prediction.msg import TrajectoryPrediction, FramePredictions, DiverPrediction

from math import pi, sqrt, exp, log, tanh, cos, sin
from threading import Lock
import numpy as np

from pid import PID

"""
 Accept bbox as input (msg type TargetObservation)
 Issues (speed, yaw, pitch) commands for following it
 PID controller smooths the control outputs

  Launch the bbox_yaw_pitch_controller.launch file
  Tuned PID params are in data/bbox_yaw_pitch_ctr_pid_params.yaml

	 -- Xahid 03.27.2018
"""

class PredictionFilter(object):
    def __init__(self, std_dev_thresh):
        self.pred_std_dev_thresh = std_dev_thresh
        self.prediction = []

    def update_prediction(self, msg):
        last5_predictions = msg.predictions[-5:]
        last5_boxes = []
        for frame_prediction in last5_predictions:
            diver_data = frame_prediction.diver_predictions[0]
            std_devs = np.array([diver_data.sx_min, diver_data.sy_min, diver_data.sx_max, diver_data.sy_max])
            if not np.any(std_devs > self.pred_std_dev_thresh):
                last5_boxes.append([diver_data.xmin, diver_data.ymin, diver_data.xmax, diver_data.ymax])
        mean_box = np.mean((last5_boxes), axis=0)
        box_w = mean_box[3] - mean_box[1]
        box_h = mean_box[2] - mean_box[0]
        self.prediction = np.array([mean_box[0], mean_box[1], box_w, box_h])
    def get_prediction(self):
        return self.prediction

# Provides helper functions for BBox object
class BBoxFilter(object):
    def __init__(self, img_cols, img_rows, bbox_init_width, bbox_init_height):
        img_center_x = img_cols/2
        img_center_y = img_rows/2

        top_left_x = img_center_x - bbox_init_width/2.0
        top_left_y = img_center_y - bbox_init_height/2.0

        self.img_rows = img_rows
        self.img_cols = img_cols

        # state vector = [ normalized top_left_x,
        #                  normalized top_left_y,   ]
        #                  normalized bbox width, 
        #                  normalized bbox height, 
        #                  normalized pixel velocity of top_left_x,
        #                  normalized pixel velocity of top left y,
        #                  normalized pixel velocity of bbox width,
        #                  normalized pixel velocity of bbox height,
        #]    
        self.state = np.array([top_left_x / float(self.img_cols),
                               top_left_y / float(self.img_rows),
                               bbox_init_width / float(self.img_cols),
                               bbox_init_height / float(self.img_rows),
                               0., 0., 0., 0.])
        
        self.last_measurement = None
        
    def is_initialized(self):
        return self.last_measurement is not None

    def still_tracking(self, thr):
        assert (self.is_initialized())
        return (rospy.Time.now() - self.last_stamp).to_sec() < thr

    def get_bbox(self):
        return self.state[0:4] * np.array([self.img_cols, self.img_rows, self.img_cols, self.img_rows])
    
    def update_estimate(self, measurement):
        #assert measurement.target_visible
        
        self.last_measurement = measurement
        self.last_stamp = rospy.Time.now()

        m_width = self.last_measurement.xmax - self.last_measurement.xmin
        m_height = self.last_measurement.ymax - self.last_measurement.ymin

        z = np.array([self.last_measurement.xmin / self.img_cols,
                      self.last_measurement.ymin / self.img_rows,
                      m_width / self.img_cols,
                      m_height / self.img_rows]) 

        if (not self.is_initialized()):
            self.state[0:4] = z
            return
        
        self.state[0:4] = 0.*self.state[0:4] + 1.*z  

# The controller        
class BBoxReactiveController(object):
    def __init__(self):
        rospy.init_node('pred_bbox_reactive_controller')

        self.vx_pid = PID(kp=3, ki=0, deriv_prediction_dt=0.3, max_deriv_noise_gain=3)
        self.yaw_pid = PID(kp=3, ki=0, deriv_prediction_dt=0.3, max_deriv_noise_gain=3)
        self.pitch_pid = PID(kp=3, ki=0, deriv_prediction_dt=0.3, max_deriv_noise_gain=3)
        self.params_map = {}
        self.set_pid_params()

        self.current_state = None
        self.bbox_filter = None
        self.pred_filter = PredictionFilter(rospy.get_param('~pred_std_dev_thresh'))

        self.current_state_mutex = Lock()        
        self.current_observation_mutex = Lock()
        self.current_pid_mutex = Lock()
        self.pred_mutex = Lock()
        
        self.rate = 20 

        print ("Waiting for /darknet_ros/bounding_boxes to come up")
        rospy.wait_for_message('/darknet_ros/bounding_boxes', BoundingBoxes)
        print ("/darknet_ros/bounding_boxes has come up")

        image_msg = rospy.wait_for_message('/loco_cams/right/image_raw', Image)
        self.image_w = image_msg.width
        self.image_h = image_msg.height
        rospy.loginfo('Aquired base image dimmensions %d %d', self.image_w, self.image_h)


        self.observation_sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.observation_callback, queue_size=3)
        self.pred_sub = rospy.Subscriber("/diver_predictions", TrajectoryPrediction, self.pred_callback, queue_size=3)
        self.rpy_pub = rospy.Publisher('/loco/command', Command, queue_size=3)
        self.cmd_msg = Command()
	
    def pred_callback(self, msg):
        self.pred_mutex.acquire()

        if self.pred_filter is None:
            self.pred_filter = PredictionFilter(rospy.get_param('~pred_std_dev_thresh'))
        self.pred_filter.update_prediction(msg)

        self.pred_mutex.release()

    def observation_callback(self, msg):
        self.current_observation_mutex.acquire()
        self.current_observation = None

        if msg.bounding_boxes:
            relevant_boxes = [box for box in msg.bounding_boxes]
            
            if relevant_boxes:
                assert (len(relevant_boxes) == 1)
                #assert (relevant_obs[0].target_visible)
                #if relevant_obs[0].target_visible:
                self.current_observation = relevant_boxes[0]

        if self.bbox_filter is None and self.current_observation:
            self.bbox_filter = BBoxFilter(self.image_w, self.image_h,self.image_w/2.0, self.image_h/2.0)

        if self.current_observation:
	    #print ('Got a measurement, updating now')
            self.bbox_filter.update_estimate(self.current_observation)

        self.current_observation_mutex.release()
            
    def set_pid_params(self):
        self.params_map['flat_vel_kp'] = rospy.get_param('~flat_vel_kp')
        self.params_map['flat_vel_ki'] = rospy.get_param('~flat_vel_ki')
        self.params_map['flat_vel_deriv_prediction_dt'] = rospy.get_param('~flat_vel_deriv_prediction_dt')
        self.params_map['flat_vel_max_deriv_noise_gain'] = rospy.get_param('~flat_vel_max_deriv_noise_gain')
	    #print (self.params_map['flat_vel_kp'], self.params_map['flat_vel_ki'], self.params_map['flat_vel_deriv_prediction_dt'])

        self.params_map['flat_yaw_kp'] = rospy.get_param('~flat_yaw_kp')
        self.params_map['flat_yaw_ki'] = rospy.get_param('~flat_yaw_ki')
        self.params_map['flat_yaw_deriv_prediction_dt'] = rospy.get_param('~flat_yaw_deriv_prediction_dt')
        self.params_map['flat_yaw_max_deriv_noise_gain'] = rospy.get_param('~flat_yaw_max_deriv_noise_gain')
	    #print (self.params_map['flat_yaw_kp'], self.params_map['flat_yaw_ki'], self.params_map['flat_yaw_deriv_prediction_dt'])
	

        self.params_map['flat_pitch_kp'] = rospy.get_param('~flat_pitch_kp')
        self.params_map['flat_pitch_ki'] = rospy.get_param('~flat_pitch_ki')
        self.params_map['flat_pitch_deriv_prediction_dt'] = rospy.get_param('~flat_pitch_deriv_prediction_dt')
        self.params_map['flat_pitch_max_deriv_noise_gain'] = rospy.get_param('~flat_pitch_max_deriv_noise_gain')

        self.params_map['magnify_speed'] = rospy.get_param('~magnify_speed')
        self.params_map['deadzone_abs_vel_error'] = rospy.get_param('~deadzone_abs_vel_error')
        self.params_map['deadzone_abs_yaw_error'] = rospy.get_param('~deadzone_abs_yaw_error')
        self.params_map['deadzone_abs_pitch_error'] = rospy.get_param('~deadzone_abs_pitch_error')
        self.params_map['target_bbox_image_ratio'] = rospy.get_param('~target_bbox_image_ratio')
        self.params_map['sec_before_giving_up'] = rospy.get_param('~sec_before_giving_up')

        self.params_map['pred_dev_thresh'] = rospy.get_param('~prediction_deviation_thresh')
        self.params_map['pred_std_dev_thresh'] = rospy.get_param('~pred_std_dev_thresh')

        self.vx_pid.set_params(self.params_map['flat_vel_kp'], self.params_map['flat_vel_ki'], 
			       self.params_map['flat_vel_deriv_prediction_dt'], self.params_map['flat_vel_max_deriv_noise_gain'])
        self.yaw_pid.set_params(self.params_map['flat_yaw_kp'], self.params_map['flat_yaw_ki'], 
			       self.params_map['flat_yaw_deriv_prediction_dt'], self.params_map['flat_yaw_max_deriv_noise_gain'])
        self.pitch_pid.set_params(self.params_map['flat_pitch_kp'], self.params_map['flat_pitch_ki'], 
			       self.params_map['flat_pitch_deriv_prediction_dt'], self.params_map['flat_pitch_max_deriv_noise_gain'])
            
    def get_centroid(self, box):
        top_left_x, top_left_y, bbox_width, bbox_height = tuple(box)
        bbox_center_x = top_left_x + bbox_width/2.0
        bbox_center_y = top_left_y + bbox_height/2.0
        return np.array([bbox_center_x, bbox_center_y])

    def get_deviation(self, box1, box2):
        centroid1 = self.get_centroid(box1)
        centroid2 = self.get_centroid(box2)
        return np.linalg.norm(centroid1-centroid2)
    
    def compute_errors_from_estimate(self): 
        bbox = self.bbox_filter.get_bbox()
        pred = self.pred_filter.get_prediction()
        if len(pred) > 0:
            deviation = self.get_deviation(bbox, pred)
            if deviation > self.params_map['pred_dev_thresh']:
                rospy.loginfo('Control from prediction')
                return self.compute_error(pred)
            else:
                rospy.loginfo('Control from detection')
                return self.compute_error(bbox)
        else:
            rospy.loginfo('Control from detection')
            return self.compute_error(bbox)

    def compute_error(self, bbox):
        top_left_x, top_left_y, bbox_width, bbox_height = tuple(bbox)
        
        bbox_center_x = top_left_x + bbox_width/2.0
        bbox_center_y = top_left_y + bbox_height/2.0

        image_center_x = self.bbox_filter.img_cols/2.0
        image_center_y = self.bbox_filter.img_rows/2.0
        
        error_cols = (bbox_center_x - image_center_x) / float(self.bbox_filter.img_cols)  # [-1,1]
        error_rows = (bbox_center_y - image_center_y) / float(self.bbox_filter.img_rows)  # [-1,1]

        bbox_area = bbox_width * bbox_height
        image_area = self.bbox_filter.img_cols * self.bbox_filter.img_rows

        error_bbox_size = self.params_map['target_bbox_image_ratio']*(1.0 - bbox_area/float(image_area)) 
        error_bbox_size = max(0.0, error_bbox_size) # [0, target_bbox_im_ratio] \propoto distance
        
        error_forward = error_bbox_size
        return (error_forward, error_cols, error_rows)

    def _clip(self, value, min_value, max_value):
        if value < min_value:
            return min_value
        elif value > max_value:
            return max_value
        else:
            return value

    def _acquire_all_mutexes(self):
        self.current_observation_mutex.acquire()
        self.current_pid_mutex.acquire()
        self.pred_mutex.acquire()

    def _release_all_mutexes(self):
        self.current_pid_mutex.release()
        self.current_observation_mutex.release()
        self.pred_mutex.release()
        
    def compute_control(self):
        self._acquire_all_mutexes()
        
        now = rospy.Time.now()
        bbox_filter_is_active = (self.bbox_filter is not None and self.bbox_filter.is_initialized() 
				     and self.bbox_filter.still_tracking(self.params_map['sec_before_giving_up']))

        if bbox_filter_is_active:
            ss, yy, pp, rr, hh = 0, 0, 0, 0, 0
            error_forward, error_yaw, error_pitch = self.compute_errors_from_estimate()
            #print (error_forward, error_yaw,  error_pitch)
        
            self.vx_pid.update(error_forward, now.to_sec())
            self.yaw_pid.update(error_yaw, now.to_sec())
            self.pitch_pid.update(error_pitch, now.to_sec())

            if self.vx_pid.is_initialized(): # forward pseudospeed
                ss = self._clip(self.vx_pid.control-self.params_map['target_bbox_image_ratio'], 0, 1)  
                if ss <= self.params_map['deadzone_abs_vel_error']:
                    ss = 0.0 
                else: 
                    ss = self._clip(self.params_map['magnify_speed']*ss, 0, 1)  

            if self.yaw_pid.is_initialized(): # yaw pseudospeed
                yy = self._clip(self.yaw_pid.control, -1, 1)
                if abs(yy) <= self.params_map['deadzone_abs_yaw_error']:
                    yy = 0.0           

            if self.pitch_pid.is_initialized(): # pitch pseudospeed         
                pp = self._clip(self.pitch_pid.control, -1, 1) 
                if abs(pp) <= self.params_map['deadzone_abs_pitch_error']:
                    pp = 0.0

            print ('V, yaw, pitch : ', (ss, yy,  pp) )
            self.set_vyprh_cmd(ss, yy, pp, rr, hh)

        else:
            print ('Target out of sight or statioanry')
            self.set_vyprh_cmd(0, 0, 0, 0, 0)

        self._release_all_mutexes()        
        return 

	

    def set_vyprh_cmd(self, ss, yy, pp, rr, hh):
        self.cmd_msg.throttle = ss+0.2
        self.cmd_msg.yaw = yy
        self.cmd_msg.pitch = pp
        #self.cmd_msg.roll = rr
        #self.cmd_msg.heave = hh
	
    def publish_control(self):
        #print ('publishing ', self.cmd_msg)
        self.rpy_pub.publish(self.cmd_msg)
        
if __name__ == "__main__":
    bbrc = BBoxReactiveController()
    
    rate = rospy.Rate(bbrc.rate)
    while not rospy.is_shutdown():
        rospy.loginfo("Target Follower Computing Control")
        bbrc.compute_control()
        rospy.loginfo("Target Follower Publishing Control")
        bbrc.publish_control()
        rate.sleep()
