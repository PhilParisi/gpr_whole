#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:21:46 2020

@author: kris
"""
"""
Created on Wed Dec 23 18:44:04 2020

@author: kris
"""

import yaml
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import rosbag
from nav_msgs.msg import Odometry


def addPlot(axs,data,datatype,step_number,y_lbl):
    for item in datatype:
        key = item[0]
        lbl = item[1]
        time=np.array(data['series']['per_step_series']['float_data']['elapsed_time(minutes)'])
        
        try:
            value=np.array(data['series']['per_step_series']['float_data'][key])
        except KeyError:
            value=np.array(data['series']['per_step_series']['int_data'][key])
        
        time = time[:step_number]
        value = value[:step_number]
        axs.plot(time, value,label = lbl,linewidth=3)
    #    trans_color=(color[0],color[1],color[2],.2)
        
        try:
            var=np.array(data['series']['per_step_series']['variance'][key])
            var = var[:step_number]
            axs.fill_between(time, value-np.sqrt(var), value+np.sqrt(var), alpha=0.2)
        except KeyError:
            pass
    axs.set(xlabel='', ylabel=y_lbl)
    axs.legend(loc='upper right', shadow=False, fontsize='small')
    

show_history = False
#set_number = 17
set_number = -1


# Read YAML file
print("loading")
with open("/mnt/aux_data2/Dropbox/Ubuntu/Data/bpslam_runs/dissertation1.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)

if set_number < 0:
    set_number = data_loaded['metrics']['particle_metrics']['last_resamping_step']

fig1, ax = plt.subplots()
print("plotting")
colormap = cm.get_cmap('Reds', 256)
step_number=0;
best_particle=0;
best_particle_weight=0;
for key in data_loaded['metrics']['particle_metrics']['particles']:
    if show_history and data_loaded['metrics']['particle_metrics']['particles'][key]['resamping_step']  <  set_number:
        sol_x=np.array(data_loaded['metrics']['particle_metrics']['particles'][key]['x'])
        sol_y=np.array(data_loaded['metrics']['particle_metrics']['particles'][key]['y'])    
        color = (.2,.2,.2,.2)
        ax.plot(sol_x, sol_y, c=color,linewidth=1)
    if data_loaded['metrics']['particle_metrics']['particles'][key]['resamping_step']  ==  set_number:
        sol_x=np.array(data_loaded['metrics']['particle_metrics']['particles'][key]['x'])
        sol_y=np.array(data_loaded['metrics']['particle_metrics']['particles'][key]['y'])    
        color = colormap(data_loaded['metrics']['particle_metrics']['particles'][key]['weight'])
        step_number = data_loaded['metrics']['particle_metrics']['particles'][key]['step_number']
        ax.plot(sol_x, sol_y, c=color,linewidth=2)
        if(data_loaded['metrics']['particle_metrics']['particles'][key]['weight']>best_particle_weight):
            best_particle_weight=data_loaded['metrics']['particle_metrics']['particles'][key]['weight'];
            best_particle=key;
    
odom_x=np.array(data_loaded['series']['per_step_series']['float_data']['odom_x'])
odom_y=np.array(data_loaded['series']['per_step_series']['float_data']['odom_y'])

odom_x = odom_x[:step_number]
odom_y = odom_y[:step_number]
    
ax.plot(odom_x, odom_y, 'b-', label='Ground Truth Trajectory', linewidth=2)

legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
plt.title("Ground Truth vs Odometry Only Comparison")

## make subplots

fig2, timeseries = plt.subplots(4,1)

addPlot(timeseries[0],data_loaded,[('magnitude_error','Magnitude Error')],step_number,'Error (M)')
addPlot(timeseries[1],data_loaded,[('num_particles', 'Number Particles')],step_number,'Particles')
addPlot(timeseries[2],data_loaded,[('max_cholesky_size', 'Largest Cholesky Factor Size')],step_number,'Chol Size (GByte)')
addPlot(timeseries[3],data_loaded,[('cumulative_time','Cumulative Compute Time'),('latest_ping_time', 'Real Time')],step_number,'Time (s)')
fig2.text(0.5, 0.04, 'time since start (min)', ha='center', va='center')


fig3, error_plot = plt.subplots()

best_particle_x = np.array(data_loaded['metrics']['particle_metrics']['particles'][best_particle]['x'])
best_particle_y = np.array(data_loaded['metrics']['particle_metrics']['particles'][best_particle]['y'])
best_particle_x = best_particle_x[:step_number]
best_particle_y = best_particle_y[:step_number]

error_x = np.subtract(best_particle_x,odom_x)
error_y = best_particle_y-odom_y

net_error = np.sqrt(error_x**2,error_y**2)
time=np.array(data_loaded['series']['per_step_series']['float_data']['elapsed_time(minutes)'])
time = time[:step_number]
#error_plot.plot(best_particle_x, best_particle_y, linewidth=2)
#error_plot.plot(odom_x, odom_y, linewidth=2)











sync_step = 10;

deadrek_x=np.array([])
deadrek_y=np.array([])
deadrek_t=np.array([])




print('reading deadreck bag')
bag = rosbag.Bag('/data/pos_datasets/reprocessed/wiggles_bank/auv_deadreck_wiggles.bag')
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
bag2 = rosbag.Bag('/data/pos_datasets/reprocessed/wiggles_bank/wiggles_bank_trim.bag')
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







addPlot(error_plot,data_loaded,[('magnitude_error','Magnitude Error')],step_number,'Error (M)')
error_plot.plot(time, net_error,label='Best Particle Error', linewidth=2)
error_plot.plot(t_vals, error,label='Odom only error', linewidth=2)
error_plot.legend(loc='upper right', shadow=False, fontsize='small')
fig3.text(0.5, 0.04, 'time since start (min)', ha='center', va='center')
plt.show()