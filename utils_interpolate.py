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

def interpolate_data_2d(all_points, data, param1_space_int=None, param2_space_int=None, interpolation_numpoints=200, interpolation_method='linear', mask_when_nearest=True, mask_x_condition=None, mask_y_condition=None, mask_smaller_than=None, mask_greater_than=None):

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

    if mask_smaller_than is not None:
        data_interpol = np.ma.masked_less(data_interpol, mask_smaller_than)
    if mask_greater_than is not None:
        data_interpol = np.ma.masked_greater(data_interpol, mask_greater_than)
    return data_interpol


def gridify(parameters_flat, bins=100, parameter_linspace=None):
    '''
        Given a flat array of parameter values, will generate a new indexing that collapses over a finite number of intervals over the parameter space

        Useful to convert random samples to a grid-like representation.

        Still need to use this indexing later, for each to avg stuff or select them.
    '''

    if parameter_linspace is None:
        # Split up the parameter space according to the bins we want
        params_unique = np.unique(parameters_flat)
        parameter_linspace = np.linspace(params_unique.min(), params_unique.max(), bins + 1)
    else:
        bins = parameter_linspace.size - 1

    result_index = np.empty((bins, parameters_flat.shape[0]), dtype=bool)

    for x_i, x in enumerate(parameter_linspace[:-1]):
        result_index[x_i] = (parameters_flat > x) & (parameters_flat <= parameter_linspace[x_i + 1])

    return result_index, parameter_linspace







