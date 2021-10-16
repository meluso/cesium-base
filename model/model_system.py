# -*- coding: utf-8 -*-
"""
@author: John Meluso
@date: 2018-10-10
@name: model_system.py

-------------------------------------------------------------------------------
Description:

This file contains a model a complex engineered system development process. It
creates the system network, the agents assigned to each node in the network,
and the process of designing the system.

The file takes in a certain amount of nodes to create a system along with an
objective function to find the amount of design cycles that is required for the
system to converge.

Parameters:

    n = [2,3,...,100,...,1000,...,inf)
        An integer value greater than or equal to 2 which specifies the number
        of agents in the system. The default value is 1000.
    obj = (string)
        A string input which specifies the objective function for all agents
        in the system to use to evaluate the quality of a design. The input
        must be one of the following terms, specified with quotes:
            "sphere"          - uses the sphere function as the objective,
                                which is also the default setting
            "ackley"          - uses the Ackley function as the objective
            "rosenbrock"      - uses the Rosenbrock function as the objective
            "styblinski-tang" - uses the Styblinski-Tang function as objective
    edg = [1,2,...,n]
        The number of random edges created for each new node added to the
        network during system generation.
    tri = [0,1]
        Probability of adding a triangle after adding a random edge.
    con = (0,inf)
        The threshold for system convergence. The simulation terminates when
        the system objective evaluation is less than this value away from the
        previous three objective evaluations and the number of evaluations is
        greater than three.
    cyc = (1,inf)
        The maximum number of design cycles the system will undergo before
        stopping.
    tmp = (0.01, 50000]
        A number which determines the initial temp of the dual annealing
        algorithm. The domain options are set by the algorithm.
    itr = [1,2,...,inf)
        The number of iterations that the dual annealing algorithm will run per
        execution. The default value is 1 to increase the difficulty of
        converging, but the value may be increased by integer values.
	mthd = {"","future"}
		Estimate method used by agents. If the value is set to anything other
		than "future" (e.g. ""), the agent will use only current values of
		their designs. If the method is "future", it allows for agents to use
		either current estimates or future predictions with a probability
		specified by parameter p.
	p = [0,1]
		The probability of a node using a specified estimate method. If the
	    method is "", then all agents use current estimates only. If the method
		is "future", then the probability p is used.
    crt = (0,3]
        The cooling rate of the algorithm, which sets how quickly the
        probability distribution of sampling further-off points contracts.
        The domain options are set by the algorithm.

-------------------------------------------------------------------------------
Change Log:
Date:       Author:    Description:
2018-10-10  jmeluso    Initial version.
2019-05-02  jmeluso    Corrected misallocation of probability for communication
                       type to network triangle formation. Now triangles are
                       set to form with a 50% probability statically.
2019-06-19  jmeluso    Removed historical estimate code.
2019-07-09  jmeluso    Added div and itr parameters inhereted from model_agent,
                       plus edg, tri, and con parameters for Monte Carlo
                       testing. Corrected termination evaluation to never
                       complete within the first 3 design cycles to allow for
                       design evolution by removing the else statement. Now,
                       objective evaluations are compared against the previous
                       three objective evaluations and must be within the
                       convergence threshold for all three before terminating.
2019-10-30  jmeluso    Updated to reflect change from basin-hopping to dual
                       annealing in model_agent.
2019-11-04  jmeluso    Differentiated between simulated annealing cooling rate
                       (visit parameter) and initial temperature.
2020-09-24  jmeluso    Reinstituted multiple estimate types with associated
                       parameters from original miscommunication model as
					   method of type "future".
2020-10-06  jmeluso    Added if statement to run from gutter or by direct call.
2021-03-31  jmeluso    Grouped agent and network parameters.

-------------------------------------------------------------------------------
"""

# Import python packages
from networkx.generators.random_graphs \
    import powerlaw_cluster_graph as gen_holme_kim
import numpy as np
import model_agent as ag
import model_network as mn
import doe_lhs as doe

class System(object):
    '''Defines a class system which contains a specified number of agents that
    are engineers designing various components. Also includes methods for
    advancing the system from an initial to a final converged design.'''

    def __init__(self, 
                  net_type = 'holme-kim',
                  agt_type = 'default',
                  con = 0.01,
                  cyc = 100,
                  net_opts = {
                      'n': 1000,
                      'edg': 2,
                      'tri': 0.5,
                      },
                  agt_opts = {
                      'obj': 'sphere',
                      'norm': True,
                      'tmp': 100,
                      'itr': 5,
                      'crt': 2.62,
                      }
                 ):
        '''Initializes an instance of the system model.'''

         # System Development Process Properties
        self.cyc = cyc  # Max number of design cycles
        self.conv_lim = con  # System convergence limit

        # Generate the network using generate_network
        self.system = self.generate_network(
            net_type,net_opts,
            agt_type,agt_opts
            )
        
        # Network Properties
        self.n = len(self.system)  # The number of agents in the network

        # Vector (old and new) of the agents' returned values
        self.vect = [ag.Obj_Eval() for ii in range(self.n)]
        self.vect_new = [ag.Obj_Eval() for ii in range(self.n)]
        self.archive = []

		# Generate the system history if needed
        if agt_type == 'estimate-definitions':
            self.generate_history()
        # else agt_type == 'default'
            # don't generate history

        for ii in range(self.n):

            # Initialize the agent's estimate
            self.system[ii].initialize_estimates()

            # Return the estimates to the system
            self.vect_new[ii] = self.system[ii].get_estimate()

        # Create a vector of just estimates (x's) for evaluation initialization
        est_vect = [self.vect_new[ii].x for ii in range(self.n)]

        for ii in range(self.n):

            # Populate the initial function evaluations of the agents
            self.system[ii].initialize_evaluations(est_vect)

            # Return the estimates to the system
            self.vect_new[ii] = self.system[ii].get_estimate()

        # Transfer new values to system as current system design
        self.vect = self.vect_new
        self.archive.append([self.vect[ii].x.copy() for ii in range(self.n)])


    def __repr__(self):
        '''Returns a representation of the agent'''
        return self.__class__.__name__
    
    
    def generate_network(self,net_type,net_opts,agt_type,agt_opts):
        '''Creates a system of the specified type with the specified network
        properties. In each case, the network is generated in model_graph
        and returned to here for agent assignment.'''
        
        # Create the specified network by calling model_graph
        if net_type == 'holme-kim':
            
            self.graph = mn.gen_holme_kim(net_opts)
            
        else: # net_type == 'holme-kim', the default
            
            print('Not a valid network type.')
            
        # Create an empty system
        system = []
        
        # Attach an agent of class model_agent to each node
        for ii in self.graph:

            # Get a list of all the neighbors of node i
            nbrs = np.sort([j for j in self.graph.adj[ii]])

            # Create agent in system with specified inputs for its neighbors,
            # probability of objective function, dual annealing cooling rate,
            # and number of iterations.
            system.append(ag.Agent(
                ii,
                nbrs,
                agt_type,
                agt_opts
                ))

        # Return the generated network of agents
        return system


    def generate_history(self):
        '''Creates a historical profile for all of the agents through Latin
        Hypercube sampling all of the agents a specified number of times.'''
        
        self.samples = 101  # The number of hypercube sampling partitions
        self.hypercube = doe.lhs(self.n,self.samples) # Create the hypercube sample

        # Scale the hypercube samples to the agents' bounds
        for ii in range(self.n):

            # Get bounds from agent
            min_val = self.system[ii].obj_bounds.xmin
            max_val = self.system[ii].obj_bounds.xmax

            # Scale all of the agents' samples to that range
            for h in range(self.samples):
                self.hypercube[h][ii] = min_val + (max_val - min_val)* \
                                       self.hypercube[h][ii]

        # Cycle through all of the hypercube sample vectors
        for h in range(self.samples):

            # Give the agents initial points to run one optimization on
            for ii in range(self.n):
                self.vect_new[ii] = \
                    self.system[ii].rand_hist_init(self.hypercube[h])

            # Give agents the optimized vector to evaluate and save
            for a in self.system:
                a.save_history(self.vect_new)


    def design_cycle(self):
        '''Perform a single design cycle with all of the agents.'''

        # For each agent
        for ii in range(self.n):

            # Perform one design cycle with current system vector
            self.vect_new[ii] = self.system[ii].generate_estimate(self.vect)

        # Record all system design values returned by agents
        self.vect = self.vect_new
        self.archive.append([self.vect[ii].x.copy() for ii in range(self.n)])


    def run(self):
        '''Designs the system. Assumes the system has already initialized
        each agent.'''

        # Initialize design cycle counter and convergence flag
        dc = 0
        cv = 0

        # Create performance vector and sum
        perf_vect = [self.vect[ii].fx for ii in range(self.n)]
        perf_sys = []
        perf_sys.append(sum(perf_vect)/self.n)

        # Perform design cycles until converged
        while (cv == 0) and (dc < self.cyc):

            # Increment design cycle counter
            dc = dc + 1

            # Perform a design cycle by calling design_cycle
            self.design_cycle()

            # Evaluate the system's performance
            perf_vect = [self.vect[ii].fx for ii in range(self.n)]
            perf_sys.append(sum(perf_vect)/self.n)

            # Check convergence conditions
            if dc > 3:

                # Check current evaluation against current-3, -2, and -1
                if (abs(perf_sys[-1 ] - perf_sys[-1 - 3]) \
                    < self.conv_lim) & \
                   (abs(perf_sys[-1] - perf_sys[-1 - 2]) \
                    < self.conv_lim) & \
                   (abs(perf_sys[-1] - perf_sys[-1 - 1]) \
                    < self.conv_lim):
                    cv = 1 # Set the convergence flag to terminate



        # Collect all of the information from this system to return after the
        # run of the simulation. It returns the following information: the
        # number of design cycles required for convergence, the system
        # performance at the end of the design cycles, the mean degree of the
        # network, the final performance of each agent, and the degree of each
        # agent.

        k_agents = [d for n, d in self.graph.degree()] # Get the agents' degrees
        k_mean = np.mean(k_agents)
        perf_ag = [[ii.x,ii.fx] for ii in self.vect]

        # Build the results vector
        results = Results(dc, perf_sys, k_mean, perf_ag, k_agents)

        # Return the results
        return results


class Results(object):
    '''An object which returns a specified set of properties from the system
    after completion of a simulation run.'''

    def __init__(self, des_cyc, perf_system, k_mean, perf_agents, k_agents):
        '''Creates an instances of the results object to return the outputs of
        the simulation to the monte carlo.'''

        self.design_cycles = des_cyc
        self.perf_system = perf_system
        self.k_mean = k_mean
        self.perf_agents = np.array(perf_agents)
        self.k_agents = k_agents

