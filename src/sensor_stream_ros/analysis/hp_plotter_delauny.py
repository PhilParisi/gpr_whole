#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:16:14 2019

@author: kris
"""
import yaml
import time
from os import listdir
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib.colors import LogNorm
import math

from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner


from yaml import CLoader as Loader, CDumper as Dumper


def Remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 


def main():
    print "running"
    with open("/home/kris/Dropbox/Ubuntu/Data/tests/hp_tests/sigma_lenght_slide/hyperparam_analytics.yml") as file:
        analytics = yaml.full_load(file)
        #print analytics["hyperparams"]
        x = np.zeros(len(analytics["hyperparams"]))
        y = np.zeros(len(analytics["sigma2e"]))
        lml = np.zeros(len(analytics["lml"]))
        for i in range(len(x)):
            x[i]=np.log10(analytics["hyperparams"][i][0])
            y[i]=np.log10(analytics["sigma2e"][i])
            lml[i]=analytics["lml"][i]
            if math.isnan(lml[i]):
                lml[i]=0;
            if lml[i] < 0:
                lml[i] = 0
            lml[i]=lml[i]
        
        triang = Triangulation(x, y)
        subdiv = 3
        refiner = UniformTriRefiner(triang)
        tri_refi, z_test_refi = refiner.refine_field(lml, subdiv=subdiv)

#        plt.figure()
#        plt.gca().set_aspect('equal')
#        plt.tripcolor(tri_refi, z_test_refi, shading='flat')
#        plt.colorbar()
#        plt.title('tripcolor of Delaunay triangulation, flat shading')
       # 
        
        
        levels = np.arange(1000, 35000., 1000)
        cmap = cm.get_cmap(name='Blues', lut=None)
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_title('Hyperparameter Optimization')

        # 1) plot of the refined (computed) data contours:
        ax.tricontour(tri_refi, z_test_refi, levels=levels, cmap=cmap,
                       linewidths=[2.0, 0.5, 0.5, 0.5, 0.5])
#        ax.tripcolor(tri_refi, z_test_refi, shading='flat')
        plt.show()
        
        
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
