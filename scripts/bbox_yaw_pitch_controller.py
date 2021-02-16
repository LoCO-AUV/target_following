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
from mini_pilot.msg import Command
from std_msgs.msg import Float32
from target_following.msg import TargetObservation, TargetObservations

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
        return (rospy.Time.now() - self.last_measurement.header.stamp).to_sec() < thr
    
    def get_bbox(self):
        return self.state[0:4] * np.array([self.img_cols, self.img_rows, self.img_cols, self.img_rows])
    
    def update_estimate(self, measurement):
        assert measurement.target_visible
        
        self.last_measurement = measurement
        self.last_measurement.header.stamp = rospy.Time.now()
            
        z = np.array([self.last_measurement.top_left_x / self.img_cols,
                      self.last_measurement.top_left_y / self.img_rows,
                      self.last_measurement.width / self.img_cols,
                      self.last_measurement.height / self.img_rows]) 

        if (not self.is_initialized()):
            self.state[0:4] = z
            return
        
        self.state[0:4] = 0.*self.state[0:4] + 1.*z 
        
        

# The controller        
class BBoxReactiveController(object):
    def __init__(self):
        rospy.init_node('bbox_reactive_controller')

        self.vx_pid = PID(kp=3, ki=0, deriv_prediction_dt=0.3, max_deriv_noise_gain=3)
        self.yaw_pid = PID(kp=3, ki=0, deriv_prediction_dt=0.3, max_deriv_noise_gain=3)
	self.pitch_pid = PID(kp=3, ki=0, deriv_prediction_dt=0.3, max_deriv_noise_gain=3)
	self.params_map = {}
	self.set_pid_params()

        self.current_state = None
	self.bbox_filter = None

        self.current_state_mutex = Lock()        
        self.current_observation_mutex = Lock()
        self.current_pid_mutex = Lock()
        
        self.rate = 20 

        print ("Waiting for /target/observation to come up")
        rospy.wait_for_message('/target/observation', TargetObservations)
        print ("/target/observation has come up")
        
        self.observation_sub = rospy.Subscriber("/target/observation", TargetObservations, self.observation_callback, queue_size=3)
	self.rpy_pub = rospy.Publisher('/loco/command', Command, queue_size=3)
	self.cmd_msg = Command()
	
        
    def observation_callback(self, msg):
        self.current_observation_mutex.acquire()
        self.current_observation = None

        if msg.observations:
            relevant_obs = [obs for obs in msg.observations]
            
            if relevant_obs:
                assert (len(relevant_obs) == 1)
                #assert (relevant_obs[0].target_visible)
                if relevant_obs[0].target_visible:
                    self.current_observation = relevant_obs[0]

        if self.bbox_filter is None and self.current_observation:
            self.bbox_filter = BBoxFilter(self.current_observation.image_width, self.current_observation.image_height,
                                          self.current_observation.image_width/2.0, self.current_observation.image_height/2.0)

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

        self.vx_pid.set_params(self.params_map['flat_vel_kp'], self.params_map['flat_vel_ki'], 
			       self.params_map['flat_vel_deriv_prediction_dt'], self.params_map['flat_vel_max_deriv_noise_gain'])
        self.yaw_pid.set_params(self.params_map['flat_yaw_kp'], self.params_map['flat_yaw_ki'], 
			       self.params_map['flat_yaw_deriv_prediction_dt'], self.params_map['flat_yaw_max_deriv_noise_gain'])
        self.pitch_pid.set_params(self.params_map['flat_pitch_kp'], self.params_map['flat_pitch_ki'], 
			       self.params_map['flat_pitch_deriv_prediction_dt'], self.params_map['flat_pitch_max_deriv_noise_gain'])
            
   
            
    def compute_errors_from_estimate(self): 
        bbox = self.bbox_filter.get_bbox()
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

    def _release_all_mutexes(self):
        self.current_pid_mutex.release()
        self.current_observation_mutex.release()


        
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
        bbrc.compute_control()
        bbrc.publish_control()
        rate.sleep()
