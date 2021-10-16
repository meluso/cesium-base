# -*- coding: utf-8 -*-
"""
@author: John Meluso
@date: 2020-10-06
@name: plot_test.py

-------------------------------------------------------------------------------
Description:

This file plots results saved by directly running model_system.py.

-------------------------------------------------------------------------------
Change Log:

Date:       Author:    Description:
2020-10-06  jmeluso    Initial version.

-------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import networkx as nx
import datetime as dt
import data_manager as dm


def plot_test():
    """Plot data from a test instead of a nominal run."""
    filename = dm.get_test_loc()
    plot_data(filename)

def plot_data(filename):

    # Start timer
    t_start = dt.datetime.now()

    # Load restults from test
    summary, history, system = dm.load_data(filename,True)

    # Save figures? True or False
    save_figs = False

    # Plot the system
    options = {
            'node_color': 'red',
            'node_size': 15,
            'width': 1
            }

    # Create new figure for network graph
    fig_network = plt.figure()

    # Build network graph in figure
    nx.draw_kamada_kawai(system.graph, **options)
    if save_figs:
        plt.savefig("../tests/network_graph.tif", format='tif', dpi=250)
        plt.savefig("../tests/network_graph.eps", format='eps', dpi=250)
    plt.show()

    #degree histogram
    hist = [float(x)/(system.n) for x in nx.degree_histogram(system.graph)]

    # Create a linear plot of the degree distribution
    #plt.plot(hist, color='b',marker=".",ls="none")
    #plt.show()

    # Create a logarithmic plot of the degree distribution
    fig_log = plt.figure()

    # Build logarithmic plot of the degree distribution
    plt.loglog(hist, color='b',marker=".",ls="none")
    plt.ylabel("Fraction of Nodes")
    plt.xlabel("Artifact Degree")
    if save_figs:
        plt.savefig("../tests/degree_distribution.tif", format='tif', dpi=250)
        plt.savefig("../tests/degree_distribution.eps", format='eps', dpi=250)
    plt.show()

    # Create figure for system performance
    fig_system = plt.figure()

    # Plot the system performance
    plt.plot(history)
    plt.xlabel("Design Cycle")
    plt.ylabel("System Performance")
    #plt.semilogy(results.perf_system)
    plt.show()

    # Stop timer
    t_stop = dt.datetime.now()
    print('Test Plot Time Elapsed: ' + str(t_stop - t_start))


if __name__ == '__main__':
    plot_test()
