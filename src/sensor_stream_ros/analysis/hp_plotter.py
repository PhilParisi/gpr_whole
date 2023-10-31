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
        
        hp = Remove(analytics["hyperparams"])
        sigma2e=Remove(analytics["sigma2e"])

        x = np.zeros((len(hp),len(sigma2e)))
        y = np.zeros((len(sigma2e),len(hp)))
        lml = np.zeros((len(hp),len(sigma2e)))
        for i in range(len(sigma2e)):
            for j in range (len(hp)):
                x[i,j]= np.log(sigma2e[i])
                y[i,j]= np.log(hp[j][0])  
                lml[i,j] = np.log(analytics["lml"][len(hp)*j+i])

                
            
        z_max, z_min = lml.max(), lml.max()
        
        

        fig, ax = plt.subplots()
        c = ax.pcolormesh(x,y,lml,shading='gouraud')
#        ax.set_title('pcolor')
#        fig.colorbar(c, ax=ax)
        plt.show()
        
#        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#        surf = ax.plot_surface(x,y,lml, cmap=cm.jet,
#                       linewidth=0, antialiased=True)
#        
#                # Customize the z axis.
#        ax.set_zlim(10, 11)
#        
##        ax.zaxis.set_major_locator(LinearLocator(10))
#        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#        
#        # Add a color bar which maps values to colors.
#        fig.colorbar(surf, shrink=0.5, aspect=5)
#        plt.show()
#    print "running"
#    with open("/home/kris/Data/SensorStream/hp_test/sigma_lenght_slide/hyperparam_analytics.yml") as file:
#        analytics = yaml.full_load(file)
#        #print analytics["hyperparams"]
#        x = np.zeros(len(analytics["hyperparams"]))
#        y = np.zeros(len(analytics["sigma2e"]))
#        lml = np.zeros(len(analytics["lml"]))
#        for i in range(len(x)):
#            x[i]=np.log(analytics["hyperparams"][i][0])
#            y[i]=np.log(analytics["sigma2e"][i])
#            lml[i]=np.log(analytics["lml"][i])
##            if lml[i] < 10:
##                lml[i] = 10
#
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(x, y, lml, c='r', marker='o')
#        plt.show()
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
