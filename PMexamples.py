#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:53:20 2019

@author: iaji
"""

import matplotlib.pyplot as plt
import numpy as np


plt.bar(np.arange(3), VELO_arr_df[91])
ax = plt.gca()
velType = np.arange(3)
ax.tick_params(axis = 'both', labelsize = 24)
ax.set_ylim([-20, 30])
ax.set_xticks(velType)
ax.set_xticklabels(('stationary', 'translational', 'rotational'), fontsize = 36)
ax.set_ylabel('velocities [mm/s and rad/s]', fontsize = 36)

np.min([VELO_arr_df[91], VELO_arr_df[608], VELO_arr_df[728], VELO_arr_df[877], VELO_arr_df[1394]])
