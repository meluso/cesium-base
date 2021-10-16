# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:08:45 2021

@author: John Meluso
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from model_agent import Bounds, Objective
import numpy as np
import fig_settings as fs


#%% Run start
if __name__ == '__main__':   

    fs.set_fonts() 


    # Define Function Inputs
    fn = "ackley"
    x_min = -32.768,
    x_max = 32.768
    neighbors = [1]
    ndivs = 500
    
    # Create objective function from model
    obj = Objective(fn,neighbors,Bounds(x_min,x_max),True)
    
    # Create meshgrid
    x_range = np.linspace(x_min,x_max,ndivs)
    y_range = np.linspace(x_min,x_max,ndivs)
    x_mesh, y_mesh = np.meshgrid(x_range,y_range)
    z_mesh = np.round(np.zeros((ndivs,ndivs)),decimals=0)
    
    # Create z values
    for x_ind in np.arange(len(x_range)):
        for y_ind in np.arange(len(y_range)):
            z_mesh[x_ind,y_ind] = obj(x_range[x_ind],[y_range[y_ind]])
    
    # Create figure
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25,azim=-35)
    ls = LightSource(270, 45)
    rgb = ls.shade(z_mesh, cmap=plt.cm.viridis,
                    vert_exag=0.1,
                    )
    surf = ax.plot_surface(x_mesh, y_mesh, z_mesh,
                            facecolors=rgb,
                            linewidth=0,
                            antialiased=False,
                            shade=False
                            )
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1,x_2)$')
    plt.savefig('../figures/fn_' + fn + '.png', bbox_inches='tight')
    plt.show()
    
    # # Create figure
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # surf = ax.contourf(x_mesh,y_mesh,z_mesh,cmap=plt.cm.viridis)
    # #plt.savefig('../figures/' + fn + '.pdf', format='pdf')
    # plt.show()