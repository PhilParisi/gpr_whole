#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:16:14 2019

@author: kris
"""
import yaml
import time
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

from yaml import CLoader as Loader, CDumper as Dumper

def main():
    dir = "/home/kris/Dropbox/Ubuntu/Data/tests/variable_block_size"
    experiments = listdir(dir)
    experiments.sort()
    fig, ax = plt.subplots()
    for experiment in experiments:
        
        with open(dir+'/'+experiment+'/'+"analytics.yml") as file:
            analytics = yaml.full_load(file)
            print analytics["total_time"]
            
            cum_time = np.zeros(len(analytics["tile_time"]))
            cum_time[0]=analytics["tile_time"][0]
            for i in range(1,len(analytics["tile_time"])):
                cum_time[i]=analytics["tile_time"][i]+cum_time[i-1]
            
            ax.plot(cum_time, label=experiment)

    legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
    plt.title('Tile compute time for various block sizes')
    plt.ylabel('Compute Time (s)')
    plt.xlabel('tile number')
    plt.show()




    mem = np.zeros(len(experiments))
    block_size = np.zeros(len(experiments))
    i=0
    for experiment in experiments:
        
        with open(dir+'/'+experiment+'/'+"analytics.yml") as file:
            analytics = yaml.full_load(file)
            with open(dir+'/'+experiment+'/'+"config.yml") as cfgfile: 
                config = yaml.full_load(cfgfile)
                maximum =  max(analytics["tile_nnz"])
                maxpos = analytics["tile_nnz"].index(maximum)
                maximum = maximum * config["block/size"]**2 * 4   
                print("max: ", experiment, maximum , "index: ", maxpos)
                mem[i] = maximum*1e-6
                block_size[i] = config["block/size"]
        i=i+1

    x=np.arange(len(experiments))
    fig, ax = plt.subplots()
    plt.scatter(block_size, mem)
    
    plt.title('Cholesky Factor Memory Usage')
    plt.ylabel('memory usage (MB)')
    plt.xlabel('block size')
    plt.show()
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
