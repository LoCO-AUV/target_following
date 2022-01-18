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

import numpy as np
from math import pi

class PID(object):
    def __init__(self, kp, ki, deriv_prediction_dt, max_deriv_noise_gain, max_window_size=3):
        self.kp = kp
        self.kd = kp * deriv_prediction_dt
        self.ki = ki

        assert (max_window_size >= 3)
        
        self.timestamps_of_errors = []   # in secs
        self.bounded_window_of_errors = []
        self.bounded_window_of_error_derivs = []
        
        self.max_window_size = max_window_size
        self.max_deriv_noise_gain = max_deriv_noise_gain
        self.deriv_prediction_dt = deriv_prediction_dt
        self.integral_of_errors = 0
        self.control = 0

        self.alpha = None    # weight [0,1] on the previous estimate of derivative. 0 ignore previous estimate
        
        
    def erase_history(self):
        self.timestamps_of_errors = []  
        self.bounded_window_of_errors = []
        self.bounded_window_of_error_derivs = []
        self.integral_of_errors = 0
        self.control = 0
        
        
    def set_params(self, kp, ki, deriv_prediction_dt, max_deriv_noise_gain, alpha=None):
        self.kp = kp
        self.ki = ki
        self.kd = kp * deriv_prediction_dt
        self.max_deriv_noise_gain = max_deriv_noise_gain
        self.deriv_prediction_dt = deriv_prediction_dt
        self.alpha = alpha
        
    def is_initialized(self):
        W = len(self.bounded_window_of_errors)
        return (W >= self.max_window_size)

    
    def compute_error_derivative(self):
        assert (len(self.bounded_window_of_errors) >= 2)
        dt1 = self.timestamps_of_errors[-1] - self.timestamps_of_errors[-2]
        assert (dt1 > 0)
        
        curr_error_diff = self.bounded_window_of_errors[-1] - self.bounded_window_of_errors[-2]

        if self.bounded_window_of_error_derivs:
            prev_deriv = self.bounded_window_of_error_derivs[-1]
            alpha, beta = 0, 1

            if self.alpha is None and (self.deriv_prediction_dt > 0 or self.max_deriv_noise_gain > 0):
                # See Astrom's book "Automatically Tuning PID Controllers, page 21"
                alpha = self.deriv_prediction_dt / (self.deriv_prediction_dt + self.max_deriv_noise_gain * dt1)
                beta = self.kp * self.max_deriv_noise_gain * alpha
                
            elif self.alpha:
                # Relative weights are manually set 
                alpha = self.alpha
                beta = (1.0 - self.alpha)/dt1 

            error_derivative = alpha * prev_deriv + beta * curr_error_diff
            return error_derivative

        else:
            return curr_error_diff / dt1
        
        

    def compute_error_integral(self):
        dt1 = self.timestamps_of_errors[-1] - self.timestamps_of_errors[-2]
        assert (dt1 > 0)
        curr_error = self.bounded_window_of_errors[-1]
        return curr_error * dt1

    
    def update(self, error, timestamp):
        self.bounded_window_of_errors.append(error)
        self.timestamps_of_errors.append(timestamp)
        
        W = len(self.bounded_window_of_errors)
        if (W > self.max_window_size):
            self.bounded_window_of_errors = self.bounded_window_of_errors[1:]
            self.timestamps_of_errors = self.timestamps_of_errors[1:]

                
        if W > 1:
            error_derivative = self.compute_error_derivative()
            self.bounded_window_of_error_derivs.append(error_derivative)
            if (len(self.bounded_window_of_error_derivs) > self.max_window_size - 1):
                self.bounded_window_of_error_derivs = self.bounded_window_of_error_derivs[1:]

                    
        if self.is_initialized():
            curr_error = self.bounded_window_of_errors[-1]
            error_derivative = self.bounded_window_of_error_derivs[-1]
            curr_error_integral = self.compute_error_integral()
        
            self.integral_of_errors += curr_error_integral
            self.control = self.kp*curr_error + self.kd*error_derivative + self.ki*self.integral_of_errors 


    

'''
Uncomment the following lines and use pid_dyn_reconf.launch file for pid tuning
using dynamic reconfig
	 -- Xahid 03.26.2018
'''

        
"""
from dynamic_reconfigure.server import Server
from target_following.cfg import PIDControllerParamsConfig

import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import rospy

## kp = 3.9, ki = 3.6, deriv= 0.14, max_Deriv_gain = 1.6, alpha = 0.5
class PIDTestNode(object):
    def __init__(self):
        rospy.init_node('pid_testing_node')
        plt.ion()
            
        self.f = 0.3    # hz
        self.w = 2. * pi * self.f
        self.samples = 1000
        self.t = np.linspace(0,10., num=self.samples)
        self.A = 1.
        self.dt = self.t[1] - self.t[0]

        self.kp = 3.
        self.ki = 0.
        self.deriv_prediction_dt = 0.1
        self.max_deriv_noise_gain = 5.
        self.pid = PID(self.kp, self.ki, self.deriv_prediction_dt, self.max_deriv_noise_gain)
        
        self.states_plot = None
        self.setpoints_plot = None

        self.fig = None 
        
        self.controller_params_cfg = None
        self.dynamic_reconfigure_srv = Server(PIDControllerParamsConfig, self.dynamic_reconfigure_callback)
        
    def update_plot(self):
        self.pid.erase_history()
        
        states = np.zeros(self.samples)
        controls = np.zeros(self.samples-1)
        setpoints = self.A * np.cos(self.w * self.t)
        errors = np.zeros(self.samples)
    
        for i in xrange(self.samples-1):
            errors[i] = setpoints[i] - states[i]
            self.pid.update(errors[i], self.t[i])

            if i >= 2:
                assert (self.pid.is_initialized())
                states[i+1] = states[i] + self.pid.control * self.dt
                controls[i] = self.pid.control
            else:
                assert not (self.pid.is_initialized())

        if self.states_plot is None:
            self.fig = plt.figure()
            self.states_plot, = plt.plot(self.t, states)
            self.setpoints_plot, = plt.plot(self.t, setpoints)
            plt.legend(['State', 'Setpoint'])
        
            print "Started showing"
            plt.show()
            print "Finished showing"

        self.states_plot.set_data(self.t, states)
        self.setpoints_plot.set_data(self.t, setpoints)
        #plt.draw()
        #plt.pause(0.01)
        
    def dynamic_reconfigure_callback(self, config, level):
        self.controller_params_cfg = config

        if config.manual_deriv_filter_weights:
            self.pid.set_params(config.kp, config.ki, config.deriv_prediction_dt, config.max_deriv_noise_gain, alpha=config.alpha)
        
        else:
            self.pid.set_params(config.kp, config.ki, config.deriv_prediction_dt, config.max_deriv_noise_gain)
        
        self.update_plot()
        return self.controller_params_cfg


    def run(self):
        self.update_plot()
        
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            plt.draw()
            plt.pause(0.01)
            rate.sleep()


# for testing        
if __name__ == "__main__":
    pid_test_node= PIDTestNode()
    pid_test_node.run()

  """ 
