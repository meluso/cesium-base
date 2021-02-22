# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:00:03 2021

@author: John Meluso
"""

import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
import data_import as di
import matplotlib.pyplot as plt

def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# Import data from file
df = di.import_execset()
params = di.import_params()

# Independent variables to try
# - Nodes (x_num_nodes) (4 options)
# - Objective Functions (x_objective_fn) (4 options)
# - Triangle Formation Probability (x_prob_triangle) (11 options)
# - System Convergence Threshold (x_conv_threshold) (7 options)
# - Future Estimate Probability (x_est_prob) (11 options)

# Create array for each input parameter.
# nod = [50, 100, 500, 1000]
# obj = ["absolute-sum","sphere","levy","ackley"]
# edg = [2]
# tri = np.round(np.arange(0,1.1,0.1),decimals=1)
# con = np.array([0.01,0.05,0.1,0.5,1,5,10])
# cyc = [100]
# tmp = [0.1]
# itr = [1]
# mth = ["future"]
# prb = np.round(np.arange(0,1.1,0.1),decimals=1)
# crt = [2.62]

# Dependent variables to try
# - Number of Cycles to Convergence or Timeout (y_num_cycles)
# - Ending System Performance (y_sys_perf)

#%% Slicing ####################

# Create masks for base set
bm_nod = (df['x_num_nodes'] == 1000)
bm_tri = (df['x_prob_triangle'] == 0.1)
bm_con = (df['x_conv_threshold'] == 1)
bm_prb = (df['x_est_prob'] == 0)

# Create frames for objective functions
by_abs = df[df['x_objective_fn'] == 'absolute-sum']
by_sph = df[df['x_objective_fn'] == 'sphere']
by_ack = df[df['x_objective_fn'] == 'ackley']
by_lvy = df[df['x_objective_fn'] == 'levy']

#%% Means and standard deviations #####################

by_abs_nodes = by_abs.groupby('x_num_nodes')[['y_num_cycles','y_sys_perf']]
groups = [name for name,unused_df in by_abs_nodes]
abs_nodes_mean = by_abs_nodes.mean()
abs_nodes_std = by_abs_nodes.std()

by_sph_nodes = by_sph.groupby('x_num_nodes')[['y_num_cycles','y_sys_perf']]
sph_nodes_mean = by_sph_nodes.mean()
sph_nodes_std = by_sph_nodes.std()

by_ack_nodes = by_ack.groupby('x_num_nodes')[['y_num_cycles','y_sys_perf']]
ack_nodes_mean = by_ack_nodes.mean()
ack_nodes_std = by_ack_nodes.std()

by_lvy_nodes = by_lvy.groupby('x_num_nodes')[['y_num_cycles','y_sys_perf']]
lvy_nodes_mean = by_lvy_nodes.mean()
lvy_nodes_std = by_lvy_nodes.std()

#%% Figures for objective/nodes means and stdevs ################

fig, axs = plt.subplots(2,2,figsize=(7.5,6))
fig.suptitle('(Graph #1) Mean/StDev by Objective Fn & Node Count')

axs[0,0].set_xlabel('y_sys_perf')
axs[0,0].set_ylabel('y_num_cycles')
axs[0,0].set_title('Absolute-Sum')
for ii in np.arange(len(groups)):
    axs[0,0].errorbar(x=abs_nodes_mean['y_sys_perf'].iloc[ii],
                      y=abs_nodes_mean['y_num_cycles'].iloc[ii],
                      xerr=abs_nodes_std['y_sys_perf'].iloc[ii],
                      yerr=abs_nodes_std['y_sys_perf'].iloc[ii],
                      ls='',lw=1,marker='o',ms=5,
                      label='n='+str(groups[ii]))

axs[0,1].set_xlabel('y_sys_perf')
axs[0,1].set_ylabel('y_num_cycles')
axs[0,1].set_title('Sphere')
for ii in np.arange(len(groups)):
    axs[0,1].errorbar(x=sph_nodes_mean['y_sys_perf'].iloc[ii],
                          y=sph_nodes_mean['y_num_cycles'].iloc[ii],
                          xerr=sph_nodes_std['y_sys_perf'].iloc[ii],
                          yerr=sph_nodes_std['y_sys_perf'].iloc[ii],
                          ls='',lw=1,marker='o',ms=5,
                          label='n='+str(groups[ii]))

axs[1,0].set_xlabel('y_sys_perf')
axs[1,0].set_ylabel('y_num_cycles')
axs[1,0].set_title('Ackley')
for ii in np.arange(len(groups)):
    axs[1,0].errorbar(x=ack_nodes_mean['y_sys_perf'].iloc[ii],
                          y=ack_nodes_mean['y_num_cycles'].iloc[ii],
                          xerr=ack_nodes_std['y_sys_perf'].iloc[ii],
                          yerr=ack_nodes_std['y_sys_perf'].iloc[ii],
                          ls='',lw=1,marker='o',ms=5,
                          label='n='+str(groups[ii]))

axs[1,1].set_xlabel('y_sys_perf')
axs[1,1].set_ylabel('y_num_cycles')
axs[1,1].set_title('Levy')
for ii in np.arange(len(groups)):
    axs[1,1].errorbar(x=lvy_nodes_mean['y_sys_perf'].iloc[ii],
                          y=lvy_nodes_mean['y_num_cycles'].iloc[ii],
                          xerr=lvy_nodes_std['y_sys_perf'].iloc[ii],
                          yerr=lvy_nodes_std['y_sys_perf'].iloc[ii],
                          ls='',lw=1,marker='o',ms=5,
                          label='n='+str(groups[ii]))

handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.025),
               borderaxespad=0.,ncol=5)

plt.tight_layout(rect=(0,0.07,1,1))
plt.show()


#%% Boxplots example ###################

# Construct figure
fig, axs = plt.subplots(4,2,figsize=(7.5,10))

# Absolute-Sum Function
by_abs.boxplot(
    ax=axs[0,0],
    column=['y_sys_perf'],
    by=['x_num_nodes'])
by_abs.boxplot(
    ax=axs[0,1],
    column=['y_num_cycles'],
    by=['x_num_nodes'])

# Sphere Function
by_sph.boxplot(
    ax=axs[1,0],
    column=['y_sys_perf'],
    by=['x_num_nodes'])
by_sph.boxplot(
    ax=axs[1,1],
    column=['y_num_cycles'],
    by=['x_num_nodes'])

# Ackley Function
by_ack.boxplot(
    ax=axs[2,0],
    column=['y_sys_perf'],
    by=['x_num_nodes'])
by_ack.boxplot(
    ax=axs[2,1],
    column=['y_num_cycles'],
    by=['x_num_nodes'])

# Levy Function
by_lvy.boxplot(
    ax=axs[3,0],
    column=['y_sys_perf'],
    by=['x_num_nodes'])
by_lvy.boxplot(
    ax=axs[3,1],
    column=['y_num_cycles'],
    by=['x_num_nodes'])

plt.tight_layout()
plt.show()


#%% Boxplots example ###################
df_est = df[bm_nod & bm_tri & bm_con]
df_est_abs = df_est[df_est['x_objective_fn'] == 'absolute-sum']

# Construct figure
fig = plt.figure()
ax = fig.gca()
df_est_abs.boxplot(
    ax=ax,
    column=['y_sys_perf'],
    by=['x_est_prob'])
ax.set_ylim(0,10)
plt.show()


#%% Boxplots example ###################
df_est = df[bm_nod & bm_tri & bm_con]
dfe_abs = df_est[df_est['x_objective_fn'] == 'absolute-sum']
dfe_sph = df_est[df_est['x_objective_fn'] == 'sphere']
dfe_lvy = df_est[df_est['x_objective_fn'] == 'levy']
dfe_ack = df_est[df_est['x_objective_fn'] == 'ackley']

# Construct figure
fig = plt.figure(figsize=(16,16))
fig = plt.suptitle('Estimate Method Prob')

# scatter plots for each
ax1 = plt.subplot(2,2,1)
dfe_abs.boxplot(ax=ax1,column=['y_sys_perf'],by=['x_est_prob'])
ax1.set_title('Absolute Sum')

# scatter plots for each
ax2 = plt.subplot(2,2,2)
dfe_sph.boxplot(ax=ax2,column=['y_sys_perf'],by=['x_est_prob'])
ax2.set_title('Sphere')

# scatter plots for each
ax3 = plt.subplot(2,2,3)
dfe_lvy.boxplot(ax=ax3,column=['y_sys_perf'],by=['x_est_prob'])
ax3.set_title('Levy')

# scatter plots for each
ax4 = plt.subplot(2,2,4)
dfe_lvy.boxplot(ax=ax4,column=['y_sys_perf'],by=['x_est_prob'])
ax4.set_title('Ackley')

# Show all
plt.tight_layout()
plt.show()


#%% Get all current estimates for each objective function ###################
df_base = df[bm_nod & bm_tri & bm_con & bm_prb]
dfb_abs = df_base[df_base['x_objective_fn'] == 'absolute-sum']
dfb_sph = df_base[df_base['x_objective_fn'] == 'sphere']
dfb_lvy = df_base[df_base['x_objective_fn'] == 'levy']
dfb_ack = df_base[df_base['x_objective_fn'] == 'ackley']

fig1 = plt.figure()
fig1 = plt.suptitle('No Variation')

# scatter plots for each
ax1 = plt.subplot(2,2,1)
plt.scatter(dfb_abs.y_num_cycles, dfb_abs.y_sys_perf)
ax1.set_xlim((0,105))
ax1.set_title('Absolute Sum')

ax2 = plt.subplot(2,2,2)
plt.scatter(dfb_sph.y_num_cycles, dfb_sph.y_sys_perf)
ax2.set_xlim((0,105))
ax2.set_title('Sphere')

ax3 = plt.subplot(2,2,3)
plt.scatter(dfb_lvy.y_num_cycles, dfb_lvy.y_sys_perf)
ax3.set_xlim((0,105))
ax3.set_title('Levy')

ax4 = plt.subplot(2,2,4)
plt.scatter(dfb_ack.y_num_cycles, dfb_ack.y_sys_perf)
ax4.set_xlim((0,105))
ax4.set_title('Ackley')


#%% Get all current estimates for each objective function ###################
df_est = df[bm_nod & bm_tri & bm_con]
dfe_abs = df_est[df_est['x_objective_fn'] == 'absolute-sum']
dfe_sph = df_est[df_est['x_objective_fn'] == 'sphere']
dfe_lvy = df_est[df_est['x_objective_fn'] == 'levy']
dfe_ack = df_est[df_est['x_objective_fn'] == 'ackley']

fig2 = plt.figure()
fig2 = plt.suptitle('Estimate Method Prob')

# scatter plots for each
ax1 = plt.subplot(2,2,1)
plt.scatter(dfe_abs.x_est_prob, dfe_abs.y_sys_perf)
ax1.set_xlim((0,1))
ax1.set_title('Absolute Sum')

ax2 = plt.subplot(2,2,2)
plt.scatter(dfe_sph.x_est_prob, dfe_sph.y_sys_perf)
ax2.set_xlim((0,1))
ax2.set_title('Sphere')

ax3 = plt.subplot(2,2,3)
plt.scatter(dfe_lvy.x_est_prob, dfe_lvy.y_sys_perf)
ax3.set_xlim((0,1))
ax2.set_title('Levy')

ax4 = plt.subplot(2,2,4)
plt.scatter(dfe_ack.x_est_prob, dfe_ack.y_sys_perf)
ax4.set_xlim((0,1))


#%% Get all current estimates for each objective function ###################
df_con = df[bm_nod & bm_tri & bm_prb]
dfc_abs = df_con[df_con['x_objective_fn'] == 'absolute-sum']
dfc_sph = df_con[df_con['x_objective_fn'] == 'sphere']
dfc_lvy = df_con[df_con['x_objective_fn'] == 'levy']
dfc_ack = df_con[df_con['x_objective_fn'] == 'ackley']

fig3 = plt.figure()
fig3 = plt.suptitle('Convergence Limit Variation')

# scatter plots for each
ax1 = plt.subplot(2,2,1)
plt.scatter(dfc_abs.x_conv_threshold, dfc_abs.y_sys_perf)
ax1.set_xlim((0,10))
ax1.set_title('Absolute Sum')

ax2 = plt.subplot(2,2,2)
plt.scatter(dfc_sph.x_conv_threshold, dfc_sph.y_sys_perf)
ax2.set_xlim((0,10))
ax2.set_title('Sphere')

ax3 = plt.subplot(2,2,3)
plt.scatter(dfc_lvy.x_conv_threshold, dfc_lvy.y_sys_perf)
ax3.set_xlim((0,10))
ax2.set_title('Levy')

ax4 = plt.subplot(2,2,4)
plt.scatter(dfc_ack.x_conv_threshold, dfc_ack.y_sys_perf)
ax4.set_xlim((0,10))


#%% Get all current estimates for each objective function ###################
df_tri = df[bm_nod & bm_con & bm_prb]
dft_abs = df_tri[df_tri['x_objective_fn'] == 'absolute-sum']
dft_sph = df_tri[df_tri['x_objective_fn'] == 'sphere']
dft_lvy = df_tri[df_tri['x_objective_fn'] == 'levy']
dft_ack = df_tri[df_tri['x_objective_fn'] == 'ackley']

fig4 = plt.figure()
fig4 = plt.suptitle('Triangle Probability Variation')

# scatter plots for each
ax1 = plt.subplot(2,2,1)
plt.scatter(dft_abs.x_prob_triangle, dft_abs.y_sys_perf)
ax1.set_xlim((0,1))
ax1.set_title('Absolute Sum')

ax2 = plt.subplot(2,2,2)
plt.scatter(dft_sph.x_prob_triangle, dft_sph.y_sys_perf)
ax2.set_xlim((0,1))
ax2.set_title('Sphere')

ax3 = plt.subplot(2,2,3)
plt.scatter(dft_lvy.x_prob_triangle, dft_lvy.y_sys_perf)
ax3.set_xlim((0,1))
ax2.set_title('Levy')

ax4 = plt.subplot(2,2,4)
plt.scatter(dft_ack.x_prob_triangle, dft_ack.y_sys_perf)
ax4.set_xlim((0,1))


#%% Get all current estimates for each objective function ###################
df_nod = df[bm_tri & bm_con & bm_prb]
dfn_abs = df_nod[df_nod['x_objective_fn'] == 'absolute-sum']
dfn_sph = df_nod[df_nod['x_objective_fn'] == 'sphere']
dfn_lvy = df_nod[df_nod['x_objective_fn'] == 'levy']
dfn_ack = df_nod[df_nod['x_objective_fn'] == 'ackley']

fig5 = plt.figure()
fig5 = plt.suptitle('Node Number Variation')

# scatter plots for each
ax1 = plt.subplot(2,2,1)
plt.scatter(dfn_abs.x_num_nodes, dfn_abs.y_sys_perf)
ax1.set_xlim((0,1000))
ax1.set_title('Absolute Sum')

ax2 = plt.subplot(2,2,2)
plt.scatter(dfn_sph.x_num_nodes, dfn_sph.y_sys_perf)
ax2.set_xlim((0,1000))
ax2.set_title('Sphere')

ax3 = plt.subplot(2,2,3)
plt.scatter(dfn_lvy.x_num_nodes, dfn_lvy.y_sys_perf)
ax3.set_xlim((0,1000))
ax2.set_title('Levy')

ax4 = plt.subplot(2,2,4)
plt.scatter(dfn_ack.x_num_nodes, dfn_ack.y_sys_perf)
ax4.set_xlim((0,1000))


#%% Get average for each parameter combination

# Get number of unique parameter combinations
count = df.groupby('index_case')['index_run'].count()
means = df.groupby('index_case')['y_sys_perf'].mean()
stdev = df.groupby('index_case')['y_sys_perf'].std()

# 10158-10163
leftovers = [params[10158],params[10158],params[10158],params[10158],params[10158],params[10158],
             params[10159],params[10159],params[10159],params[10159],params[10159],params[10159],params[10159],params[10159],
             params[10160],params[10160],params[10160],params[10160],params[10160],params[10160],params[10160],params[10160],
             params[10161],params[10161],params[10161],params[10161],params[10161],params[10161],params[10161],params[10161],
             params[10162],params[10162],params[10162],params[10162],params[10162],params[10162],params[10162],params[10162],
             params[10163],params[10163],params[10163],params[10163],params[10163],params[10163],params[10163],params[10163]]
leftovers_new = []
missed_list = [4, 26, 60, 62, 79, 80, 81, 90]
# 10158:        60, 62, 79, 80, 81, 90
# 10159: 4, 26, 60, 62, 79, 80, 81, 90
# 10160: 4, 26, 60, 62, 79, 80, 81, 90
# 10161: 4, 26, 60, 62, 79, 80, 81, 90
# 10162: 4, 26, 60, 62, 79, 80, 81, 90
# 10163: 4, 26, 60, 62, 79, 80, 81, 90
ind = 1
for ii in np.arange(len(leftovers)):
    ind += 1
    leftovers_new.append(leftovers[ii].copy())
    leftovers_new[ii]['run'] = missed_list[np.mod(ind,len(missed_list))]
















