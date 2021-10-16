# -*- coding: utf-8 -*-
"""
@author: John Meluso
@date: 2018-10-18
@name: test_agent.py

-------------------------------------------------------------------------------
Description:

This file tests the agent class.

The file creantes an agent using its location, the amount of neighbors that it
has, and the objective funtion that is being used. The file prints out results
for a specified (iterations) number of estimates.

-------------------------------------------------------------------------------
Change Log:

Date:       Author:    Description:
2018-10-10  jmeluso    Initial version.
2019-10-28  jmeluso    Updated to allow for a specified number of iterations
                       and added clearer comments.

-------------------------------------------------------------------------------
"""

import model_agent as ag
from numpy.random import random_sample as rand
import matplotlib.pylab as plt
import doe_lhs as doe

# Specify test parameters
obj_fns = ["absolute-sum","ackley","griewank","langermann","levy","rosenbrock",
           "schwefel","sphere","styblinski-tang"]
obj_num=len(obj_fns)
iterations=10

for obj_ in obj_fns:

    # Create an instance of an agent
    index=2
    a1 = ag.Agent(loc=index,
                  nbr=[0,3],
                  obj=obj_,
                  tmp=10,
                  itr=1,
                  mthd="future",
                  prob=0.5,
                  crt=2.62)
    print(("Location = " + str(a1.location)))
    print(("Neighbors = " + str(a1.neighbors)))
    print(("Min Bound = " + str(a1.obj_bounds.xmin)))
    print(("Max Bound = " + str(a1.obj_bounds.xmax)))
    print(("Estimate type = " + str(a1.est_type)))
    print(("Objective function type = " + a1.fn))

    if a1.mthd == "future":
    	# Create a history vector for a set of 4 agents
    	sys_vect = [ag.Obj_Eval() for i in range(4)]
    	hypercube = doe.lhs(4,101)

    	# Get bounds from agent
    	min_val = a1.obj_bounds.xmin
    	max_val = a1.obj_bounds.xmax

    	# Scale all of the agents' samples to that range
    	for i in range(4):
    	    for h in range(101):
    	        hypercube[h][i] = min_val + (max_val - min_val)*hypercube[h][i]

    	# Generate history
    	for h in range(0,101):
    	    lhs_vect = hypercube[h][:]
    	    sys_vect[2] = a1.rand_hist_init(lhs_vect)
    	    est_vect = [ag.Obj_Eval(rand(),rand()),
    	                ag.Obj_Eval(rand(),rand()),
    	                sys_vect[2],
    	                ag.Obj_Eval(rand(),rand())]
    	    a1.save_history(est_vect)

    	print(("History = " + str(len(a1.history)) + " points"))

    # Create output vectors for the main agent
    x_vals = []
    fx_vals = []

    # Set initial estimates for the agent
    a1.initialize_estimates()

    # Create initialization vector for system
    init_vect = [(a1.obj_bounds.xmax-a1.obj_bounds.xmin)*rand()+a1.obj_bounds.xmin,
                 (a1.obj_bounds.xmax-a1.obj_bounds.xmin)*rand()+a1.obj_bounds.xmin,
                 (a1.obj_bounds.xmax-a1.obj_bounds.xmin)*rand()+a1.obj_bounds.xmin,
                 (a1.obj_bounds.xmax-a1.obj_bounds.xmin)*rand()+a1.obj_bounds.xmin]
    init_vect[index] = a1.curr_est.x

    # Populate f(x) values based on initial values
    a1.initialize_evaluations(init_vect)

    # Save the initial estimate to the archival vectors
    x_vals.append(a1.curr_est.x)
    fx_vals.append(a1.curr_est.fx)

    print(("Agent Performance (t=0): f(" + str(a1.curr_est.x) + ") = " \
           + str(a1.curr_est.fx)))

    # Create an estimate
    for ind in range(0,iterations):

        # Populate the system vector with random data
        sys_vect = [ag.Obj_Eval((a1.obj_bounds.xmax-a1.obj_bounds.xmin)*\
                                rand()+a1.obj_bounds.xmin,0),
                    ag.Obj_Eval((a1.obj_bounds.xmax-a1.obj_bounds.xmin)*\
                                rand()+a1.obj_bounds.xmin,0),
                    ag.Obj_Eval((a1.obj_bounds.xmax-a1.obj_bounds.xmin)*\
                                rand()+a1.obj_bounds.xmin,0),
                    ag.Obj_Eval((a1.obj_bounds.xmax-a1.obj_bounds.xmin)*\
                                rand()+a1.obj_bounds.xmin,0)]

        # Fill the estimate into the appropriate system vector slots (at index)
        sys_vect[index].x = a1.curr_est.x
        sys_vect[index].fx = a1.curr_est.fx

        # Generate a new estiamte from the system vector
        received_est = a1.generate_estimate(sys_vect)

        # Save the updated estimate to the archival vectors
        x_vals.append(a1.curr_est.x)
        fx_vals.append(a1.curr_est.fx)

        # Print the updated estimate
        print(("Agent Performance (t=" + str(ind+1) + "): f(" + str(a1.curr_est.x) \
               + ") = " + str(a1.curr_est.fx)))

    # Plot x values vs f(x) values
    plt.scatter(x_vals,fx_vals)
    plt.title(obj_)
    plt.xlabel("x value")
    plt.ylabel("f(x) value")
    plt.show()

    # Plot historical trend
    plt.plot(fx_vals)
    plt.title(obj_)
    plt.xlabel("Iteration")
    plt.ylabel("f(x) value")
    plt.show()
