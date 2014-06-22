#!/usr/bin/env python
# encoding: utf-8
"""
utils_math.py

Created by Loic Matthey on 2013-09-08.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import numpy as np
import scipy.interpolate as spint


####################### INTERPOLATION FUNCTIONS #############################

def interpolate_data_2d(all_points, data, param1_space_int=None, param2_space_int=None, interpolation_numpoints=200, interpolation_method='linear', mask_when_nearest=True, mask_x_condition=None, mask_y_condition=None):

    # Construct the interpolation
    if param1_space_int is None:
        param1_space_int = np.linspace(all_points[:, 0].min(), all_points[:, 0].max(), interpolation_numpoints)
    if param2_space_int is None:
        param2_space_int = np.linspace(all_points[:, 1].min(), all_points[:, 1].max(), interpolation_numpoints)

    data_interpol = spint.griddata(all_points, data, (param1_space_int[None, :], param2_space_int[:, None]), method=interpolation_method)

    if interpolation_method == 'nearest' and mask_when_nearest:
        # Let's mask the points outside of the convex hull

        # The linear interpolation will have nan's on points outside of the convex hull of the all_points
        data_interpol_lin = spint.griddata(all_points, data, (param1_space_int[None, :], param2_space_int[:, None]), method='linear')

        # Mask
        data_interpol[np.isnan(data_interpol_lin)] = np.nan

    # Mask it based on some conditions
    if not mask_x_condition is None:
        data_interpol[mask_x_condition(param1_space_int), :] = 0.0
    if not mask_y_condition is None:
        data_interpol[:, mask_y_condition(param2_space_int)] = 0.0

    return data_interpol


