# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:02:17 2020

@author: John Meluso
"""

import datetime as dt
import model_system as sy
from numpy.random import default_rng

# Create the random number generator
rng = default_rng()


def run_model(params):
    """Function which runs one instance of the system model for a given set of
    input parameters and saves the resulting data."""

    # Print input parameters
    print('Run parameters: ' + str(params))

    # Start timer
    t_start = dt.datetime.now()

    # Generate a system
    system = sy.System(
        params['net'],
        params['agt'],
        params['con'],
        params['cyc'],
        params['net_opts'],
        params['agt_opts']
        )

    # Run the system
    results = system.run()

    # Stop timer
    t_stop = dt.datetime.now()
    t_elapsed = t_stop - t_start
    print('Run Time Elapsed: ' + str(t_elapsed) + '\n')

    # Build data to return
    summary = build_summary(params,t_elapsed,results)
    history = results.perf_system

    # Return results
    return summary, history, system


def build_summary(params,elapsed,results):
    '''Builds a summary list to return the results of the simulation for the
    specified input parameters.'''
    
    summary = []
    
    # Append system properties
    summary.append(params['ind'])
    summary.append(params['run'])
    summary.append(params['net'])
    summary.append(params['agt'])
    summary.append(params['con'])
    summary.append(params['cyc'])
    
    # Append network properties
    for par in return_network(params['net'],params['net_opts']):
        summary.append(par)
        
    # Append agent properties
    for par in return_agent(params['agt'],params['agt_opts']):
        summary.append(par)
        
    # Append test time elapsed
    summary.append(elapsed.total_seconds())
        
    # Append test returns
    summary.append(results.design_cycles)
    summary.append(results.perf_system[-1])
    
    

    return summary


def return_network(net_type,net_opts):
    '''Return network properties corresponding to the specified type of
    network.'''
    
    if net_type == 'holme-kim':
        
        properties = [
            net_opts['n'],
            net_opts['edg'],
            net_opts['tri']
            ]
	
	else: 
		print('Not a valid network type.')
	
    return properties
        

def return_agent(agt_type,agt_opts):
    '''Return agent properties corresponding to the specified type of agent.'''
    
    if agt_type == 'estimate-definitions':
        properties = [
            agt_opts['obj'],
            agt_opts['norm'],
            agt_opts['p'],
            agt_opts['tmp'],
            agt_opts['itr'],
            agt_opts['crt']
            ]
    else:  # agt_type == 'default'
        properties = [
            agt_opts['obj'],
            agt_opts['norm'],
            agt_opts['tmp'],
            agt_opts['itr'],
            agt_opts['crt']
            ]
    return properties
    


    
    
    
    
    