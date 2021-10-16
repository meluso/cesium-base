# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:59:17 2020

@author: John Meluso
"""

import sys
import pickle
import csv


def get_test_loc():
    """Sets test variables for test saving and loading."""

    # Set location based on platform
    if sys.platform.startswith('linux'):
        loc = "~/cesium/data/test/"
    else:
        loc = "../data/test/"
    return loc + "test"


def save_test(summary, history, system):
    """Saves data from a test run."""

    # Get test variables
    dir_prefix = get_test_loc()

    # Write run summary and history to location
    save_data(dir_prefix, summary, history, system)


def load_test():
    """Loads data from a test run."""

    # Get test variables
    dir_prefix = get_test_loc()

    # Read run summary and history to variables
    return load_data(dir_prefix, True)


def save_data(dir_prefix, summary, history, system=[]):
    """Saves data from a run of the model."""

    # Write run summary to location
    with open(dir_prefix + "_summary.csv","w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(summary)

    # Write run history to location
    with open(dir_prefix + "_history.csv","w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(history)

    if not(system == []): # Write system to location
        pickle.dump(system, open(dir_prefix + "_system.pickle","wb"))


def load_data(dir_prefix, get_sys=False):
    """Loads data from a run of the model."""

    # Read run summary to variable
    with open(dir_prefix + "_summary.csv","r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        summary = next(reader)

    # Read run history to variable
    with open(dir_prefix + "_history.csv","r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        line = next(reader)
        history = [float(xx) for xx in line]

    if get_sys:

        # Return the system object
        system = pickle.load(open(dir_prefix + "_system.pickle","rb"))
        return summary, history, system

    else:

        # Just return the summary and history
        return summary, history

