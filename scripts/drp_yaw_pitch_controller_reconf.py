#!/usr/bin/python3
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
from target_following.msg import TargetObservation, TargetObservations
from adroc.msg import DiverRelativePosition

from math import pi, sqrt, exp, log, tanh, cos, sin
from threading import Lock
import numpy as np

from pid import PID

from dynamic_reconfigure.server import Server
from target_following.cfg import DRPControllerParamsConfig
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

"""
 Accepts DRP as input (msg type DiverRelativePosition)
 Issues (speed, yaw, pitch) commands for following it
 PID controller smooths the control outputs

"""       
# The controller        
class DRPReactiveController(object):
    def __init__(self):
        rospy.init_node('drp_reactive_controller')
        self.params_map = {}

        self.vx_pid = PID(kp=3, ki=0, deriv_prediction_dt=0.3, max_deriv_noise_gain=3)
        self.yaw_pid = PID(kp=3, ki=0, deriv_prediction_dt=0.3, max_deriv_noise_gain=3)
        self.pitch_pid = PID(kp=3, ki=0, deriv_prediction_dt=0.3, max_deriv_noise_gain=3)
        self.controller_active = False

        self.current_state = None
        self.current_observation = None
        self.observation_ts = None

        self.current_state_mutex = Lock()        
        self.current_observation_mutex = Lock()
        self.current_pid_mutex = Lock()
        
        self.rate = 20 

        print ("Waiting for /drp/drp_target to come up")
        msg = rospy.wait_for_message('/drp/drp_target', DiverRelativePosition)
        self.image_w = msg.image_w
        self.image_h = msg.image_h
        print ("/drp/drp_target has come up")

        self.controller_params_cfg = None
        self.dynamic_reconfigure_srv = Server(DRPControllerParamsConfig, self.dynamic_reconfigure_callback)
        self.set_pid_params()
        
        self.observation_sub = rospy.Subscriber("/drp/drp_target", DiverRelativePosition, self.observation_callback, queue_size=3)
        self.rpy_pub = rospy.Publisher('/loco/command', Command, queue_size=3)
        self.cmd_msg = Command()

        rospy.Service('drp_reactive_controller/start', Trigger, self.start_service_handler)
        rospy.Service('drp_reactive_controller/stop', Trigger, self.stop_service_handler)

    def start_service_handler(self, request):
        self.controller_active = True
        t = TriggerResponse()
        t.success=True
        t.message="DRP Controller started"
        return t

    def stop_service_handler(self, request):
        self.controller_active = False
        t = TriggerResponse()
        t.success=True
        t.message="DRP Controller stopped"
        return t

    def dynamic_reconfigure_callback(self, config, level):
        self.controller_params_cfg = config
        self.set_pid_params()
        return self.controller_params_cfg
        
    def observation_callback(self, msg):
        self.current_observation_mutex.acquire()
        self.current_observation = None

        self.current_observation = [msg.target_x, msg.target_y, msg.pseudo_distance]
        self.observation_ts = rospy.Time.now().to_sec()

        self.current_observation_mutex.release()
            
    def set_pid_params(self):
        self.params_map['flat_vel_kp'] = self.controller_params_cfg.flat_vel_kp
        self.params_map['flat_vel_ki'] = self.controller_params_cfg.flat_vel_ki
        self.params_map['flat_vel_deriv_prediction_dt'] = self.controller_params_cfg.flat_vel_deriv_prediction_dt
        self.params_map['flat_vel_max_deriv_noise_gain'] = self.controller_params_cfg.flat_vel_max_deriv_noise_gain
    #print (self.params_map['flat_vel_kp'], self.params_map['flat_vel_ki'], self.params_map['flat_vel_deriv_prediction_dt'])

        self.params_map['flat_yaw_kp'] = self.controller_params_cfg.flat_yaw_kp
        self.params_map['flat_yaw_ki'] = self.controller_params_cfg.flat_yaw_ki
        self.params_map['flat_yaw_deriv_prediction_dt'] = self.controller_params_cfg.flat_yaw_deriv_prediction_dt
        self.params_map['flat_yaw_max_deriv_noise_gain'] = self.controller_params_cfg.flat_yaw_max_deriv_noise_gain
    #print (self.params_map['flat_yaw_kp'], self.params_map['flat_yaw_ki'], self.params_map['flat_yaw_deriv_prediction_dt'])
    

        self.params_map['flat_pitch_kp'] = self.controller_params_cfg.flat_pitch_kp
        self.params_map['flat_pitch_ki'] = self.controller_params_cfg.flat_pitch_ki
        self.params_map['flat_pitch_deriv_prediction_dt'] = self.controller_params_cfg.flat_pitch_deriv_prediction_dt
        self.params_map['flat_pitch_max_deriv_noise_gain'] = self.controller_params_cfg.flat_pitch_max_deriv_noise_gain

        self.params_map['magnify_speed'] = self.controller_params_cfg.magnify_speed
        self.params_map['deadzone_abs_vel_error'] = self.controller_params_cfg.deadzone_abs_vel_error
        self.params_map['deadzone_abs_yaw_error'] = self.controller_params_cfg.deadzone_abs_yaw_error
        self.params_map['deadzone_abs_pitch_error'] = self.controller_params_cfg.deadzone_abs_pitch_error
        self.params_map['sec_before_giving_up'] = self.controller_params_cfg.sec_before_giving_up

        self.vx_pid.set_params(self.params_map['flat_vel_kp'], self.params_map['flat_vel_ki'], 
                                self.params_map['flat_vel_deriv_prediction_dt'], self.params_map['flat_vel_max_deriv_noise_gain'])
        self.yaw_pid.set_params(self.params_map['flat_yaw_kp'], self.params_map['flat_yaw_ki'], 
                                self.params_map['flat_yaw_deriv_prediction_dt'], self.params_map['flat_yaw_max_deriv_noise_gain'])
        self.pitch_pid.set_params(self.params_map['flat_pitch_kp'], self.params_map['flat_pitch_ki'], 
                                self.params_map['flat_pitch_deriv_prediction_dt'], self.params_map['flat_pitch_max_deriv_noise_gain'])
            
   
            
    def compute_errors_from_estimate(self): 
        tx, ty, pd = self.current_observation

        #Our image target point is centered horizontally, and 1/3 of the way down vertically.
        image_setpoint_x = self.image_w/2.0 
        image_setpoint_y = self.image_h/2.0 #########

        #Since PD is 0 if very far away and 1.0 if at ideal position, error should decrease as we get closer.        
        error_forward = 1.0 - pd
        error_x = (tx - image_setpoint_x)/ float(self.image_w) #Pixel difference between target point and DRP point, normalized by image size.
        error_y = (ty - image_setpoint_y)/ float(self.image_h)
        
        return (error_forward, error_x, error_y)


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
        
        now = rospy.Time.now().to_sec()
        target_active = (self.current_observation is not None and ( now - self.observation_ts  < (self.params_map['sec_before_giving_up'])))

        if target_active:
            ss, yy, pp, rr, hh = 0, 0, 0, 0, 0
            error_forward, error_yaw, error_pitch = self.compute_errors_from_estimate()
            print ("error forward: ", error_forward, "error_yaw: ", error_yaw,  "error_pitch: ", error_pitch)
      
            self.vx_pid.update(error_forward, now)
            self.yaw_pid.update(error_yaw, now)
            self.pitch_pid.update(error_pitch, now)
 
            if self.vx_pid.is_initialized(): # forward pseudospeed
                ss = self._clip(self.vx_pid.control, -1, 1)  
                #if ss <= self.params_map['deadzone_abs_vel_error']:
                    #pass
                    #ss = 0.0 
                # else: #####
                ss = self._clip(self.params_map['magnify_speed']*ss, -1, 1)   #######

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
            print ('Target out of sight.')
            self.set_vyprh_cmd(0, 0, 0, 0, 0)

        self._release_all_mutexes()        
        return 

    def set_vyprh_cmd(self, ss, yy, pp, rr, hh):
        self.cmd_msg.throttle = ss+0 # 0.2
        self.cmd_msg.yaw = yy
        self.cmd_msg.pitch = pp
        #self.cmd_msg.roll = rr
        #self.cmd_msg.heave = hh
    


    def publish_control(self):
        #print ('publishing ', self.cmd_msg)
        if self.controller_active:
            self.rpy_pub.publish(self.cmd_msg)


        
if __name__ == "__main__":
    drprc = DRPReactiveController()
    
    rate = rospy.Rate(drprc.rate)
    while not rospy.is_shutdown(): 
        drprc.compute_control()
        drprc.publish_control()
        rate.sleep()
