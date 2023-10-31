#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:21:46 2020

@author: kris
"""

import rosbag
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
import math

sync_step = 10;

deadrek_x=np.array([])
deadrek_y=np.array([])
deadrek_t=np.array([])




print('reading deadreck bag')
bag = rosbag.Bag('/home/kris/Desktop/auv_deadreck.bag')
odom_topic = 'odometry/filtered'
for topic, msg, t in bag.read_messages(topics=[odom_topic]):
    if topic == odom_topic:
        deadrek_x = np.append(deadrek_x, msg.pose.pose.position.x)
        deadrek_y = np.append(deadrek_y, msg.pose.pose.position.y)
        deadrek_t = np.append(deadrek_t, msg.header.stamp.to_sec())
        
groundtruth_x=np.array([])
groundtruth_y=np.array([])
groundtruth_t=np.array([])
print('reading ground truth bag')
bag2 = rosbag.Bag('/data/pos_datasets/reprocessed/dutch_harbor/bag/auv_renav_trimmed.bag')
topics = bag2.get_type_and_topic_info()[1].keys()
odom_topic = '/nav/processed/odometry'

for topic, msg, t in bag2.read_messages(topics=[odom_topic]):
    if topic == odom_topic:
        if msg.header.stamp.to_sec() >= deadrek_t[0]:
            groundtruth_x = np.append(groundtruth_x, msg.pose.pose.position.x)
            groundtruth_y = np.append(groundtruth_y, msg.pose.pose.position.y)
            groundtruth_t = np.append(groundtruth_t, msg.header.stamp.to_sec())


        
deadrek_x = deadrek_x + (groundtruth_x[sync_step]-deadrek_x[sync_step])       
deadrek_y = deadrek_y + (groundtruth_y[sync_step]-deadrek_y[sync_step])  




t_vals = np.linspace(groundtruth_t[0], groundtruth_t[-1], 100)

deadrek_x_interp = np.interp(t_vals, deadrek_t, deadrek_x)
deadrek_y_interp = np.interp(t_vals, deadrek_t, deadrek_y)
groundtruth_x_interp = np.interp(t_vals, groundtruth_t, groundtruth_x)
groundtruth_y_interp = np.interp(t_vals, groundtruth_t, groundtruth_y)


size = len(t_vals)
error = np.zeros(size)
for i in range(0,size):
    error[i] = np.sqrt((deadrek_x_interp[i] -groundtruth_x_interp[i])**2 + (deadrek_y_interp[i] - groundtruth_y_interp[i])**2) #deadrek_y[i] - groundtruth_y[i] #deadrek_y[i] - groundtruth_y[i]#,2) pow(deadrek_x[i] - groundtruth_x[i],2) + pow(

#error = np.sqrt(error)
#error =  math.sqrt(deadrek_x**2 + deadrek_y**2) - math.sqrt(groundtruth_x**2 + groundtruth_y**2)

print(groundtruth_t[0],deadrek_t[0])

fig, ax = plt.subplots()

t_vals = (t_vals - t_vals[0])/60.0

ax.plot(t_vals,error, label='magnitude error')

legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
plt.title("Ground Truth vs Odometry Only Comparison")
plt.show()