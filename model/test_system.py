# -*- coding: utf-8 -*-
"""
@author: John Meluso
@date: 2018-10-21
@name: test_system.py

-------------------------------------------------------------------------------
Description:

This file tests the system class.

The file takes in the amount of nodes that are present in the system and the
objective function to show how many connections occur between the nodes through
a network diagram.

The plot that follows shows the amount of nodes that have the same amount of
connections.

-------------------------------------------------------------------------------
Change Log:

Date:       Author:    Description:
2018-10-10  jmeluso    Initial version.
2019-07-09  jmeluso    Updated to new inputs System(n,obj,edg,tri,con,div,itr).
2019-11-04  jmeluso    Updated to new inputs System(n,obj,edg,tri,con,tmp,crt).

-------------------------------------------------------------------------------
"""

import sys
import data_manager as dm
import get_params as gp
import run_model as rm
import test_plot as tp
import numpy as np
from numpy.random import default_rng

# Create the random number generator
rng = default_rng()


def test_system(run_mode,obj='absolute-sum',plot=False):

    # Check for random
    if run_mode == 'random':
        params_all = gp.get_params()
        case = rng.integers(len(params_all))
        params_run = params_all[case]
        
    else:
        
        # Map inputs
        net_mode = run_mode[0]
        agt_mode = run_mode[1]
    
        # Set network mode
        if net_mode == 'holme-kim':
            net_opts = {
                    'n': 64,
                    'edg': 2,
                    'tri': 0.5,
                    }
        else:
            print('Not a valid network mode.')
            
        # Set agent mode
        if agt_mode == 'estimate':
            agt_opts = {
                    'obj': obj,
                    'norm': True,
                    'p': 0.5,
                    'tmp': 100,
                    'itr': 1,
                    'crt': 2.62,
                    }
        else:  # agt_mode == 'default'
            agt_mode = 'default'
            agt_opts = {
                    'obj': obj,
                    'norm': True,
                    'tmp': 100,
                    'itr': 1,
                    'crt': 2.62,
                    }
    
        # Construct parameters
        params_run = {
            'ind': 999999,
            'run': 999999,
            'net': net_mode,
            'agt': agt_mode,
            'con': 0.001,
            'cyc': 100,
            'net_opts': net_opts,
            'agt_opts': agt_opts
            }
    
    # Run test
    summary, history, system = rm.run_model(params_run)
    dm.save_test(summary, history, system)
    if not(sys.platform.startswith('linux')) and plot:
        tp.plot_test()
    return summary[-3]


if __name__ == '__main__':

    # Specify run mode
    run_modes = [
        # 'random',
        #('holme-kim','default'),
        # ('holme-kim','estimate')
        ]
    
    # Specify test functions
    test_fns = [
        'absolute-sum',
        'sphere',
        'ackley',
        'levy'
        ]
    
    # Num runs
    test_runs = 20
    
    # Create duration array
    duration = np.zeros((len(run_modes),len(test_fns),test_runs))
    
    # Run num tests for each function
    for ii, mode in enumerate(run_modes):
        for jj, fn in enumerate(test_fns):
            for kk in range(test_runs):
                duration[ii,jj,kk] += test_system(mode,fn)
    
    # Average results
    dur_means = np.mean(duration,axis=2)
