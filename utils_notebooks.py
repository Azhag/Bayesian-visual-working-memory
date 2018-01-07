#!/usr/bin/env python
# encoding: utf-8
"""
utils_plot.py

Created by Loic Matthey on 2013-09-08.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import numpy as np

import scipy.stats as spst

import pandas as pd

import utils_math
import utils_directional_stats

def avg_lastaxis(array_name, array):
    return [(array_name, utils_math.nanmean(array, axis=-1))]

def avg_twice_lastaxis(array_name, array):
    return [(array_name, utils_math.nanmean(utils_math.nanmean(array, axis=-1), axis=-1))]
def std_lastaxis(array_name, array):
    return [('std_' + array_name, utils_math.nanstd(array, axis=-1))]


def process_precision(array_name, array):
    outputs = avg_lastaxis(array_name, array)
    outputs.extend(std_lastaxis(array_name, array))
    outputs.extend(avg_lastaxis(array_name + "_stddev", (2./array)**0.5))
    return outputs

def process_fi(array_name, array):
    outputs = avg_twice_lastaxis(array_name, array)
    outputs.extend(avg_twice_lastaxis(array_name + "_stddev", (2./array)**0.5))
    return outputs

def process_marginal_fi(array_name, array):
    # Marginal FI/Inv FI have (mean, std), just keep mean
    outputs_all = avg_lastaxis(array_name, array)

    if array_name.find('inv') > -1:
        outputs_all.extend(avg_lastaxis(array_name + "_stddev", (2.*array)**0.5))
    else:
        outputs_all.extend(avg_lastaxis(array_name + "_stddev", (2./array)**0.5))

    outputs = [(o[0], o[1][:, 0]) for o in outputs_all]
    return outputs

def process_em_fits(array_name, array):
    emfits_all = utils_math.nanmean(array, axis=-1)

    outputs = [(array_name + "_" + colname, emfits_all[:, col_i])
        for col_i, colname in enumerate(['kappa',
                                         'target',
                                         'nontargets',
                                         'random',
                                         'LL',
                                         'bic'])]

    outputs.append((array_name + '_fidelity', 1./utils_directional_stats.kappa_to_stddev(emfits_all[:, 0])**2.))
    outputs.append((array_name + '_stddev', utils_directional_stats.kappa_to_stddev(emfits_all[:, 0])))

    outputs.extend(std_lastaxis(array_name + '_kappa', array[:, 0]))
    outputs.extend(std_lastaxis(array_name + '_fidelity', 1./utils_directional_stats.kappa_to_stddev(array[:, 0])**2.))
    outputs.extend(std_lastaxis(array_name + '_stddev', utils_directional_stats.kappa_to_stddev(array[:, 0])))

    return outputs


def construct_pandas_dataframe(data_pbs, pandas_columns_with_processing, num_repetitions):
    parameter_names_sorted = data_pbs.dataset_infos['parameters']
    filter_data = None
    result_parameters_flat = None

    pandas_column_data = []

    for result_array_name, result_processing in pandas_columns_with_processing:
        # Extract data
        res_array = np.array(data_pbs.dict_arrays[result_array_name]['results_flat']).squeeze()

        # Filter completed only
        if filter_data is None:
            repeats_completed = data_pbs.dict_arrays[result_array_name]['repeats_completed']
            filter_data = repeats_completed == (num_repetitions - 1)
        res_array = res_array[filter_data[:res_array.shape[0]]]

        # Keep parameters
        if result_parameters_flat is None:
            result_parameters_flat = np.array(data_pbs.dict_arrays[result_array_name]['parameters_flat'])
            result_parameters_flat = result_parameters_flat[filter_data]

        # Transform into list of columns for Pandas
        pandas_column_data.extend(result_processing['process'](result_processing['name'], res_array))

    # Add all parameters to Pandas columns
    for param_i, param_name in enumerate(parameter_names_sorted):
        pandas_column_data.append((param_name, result_parameters_flat[:, param_i]))

    df_out = pd.DataFrame.from_items(pandas_column_data)

    # Remove NaN
    df_out = df_out.dropna()

    return df_out


def remove_outliers(df, n_stddev=5):
    outliers = np.sum(np.abs(spst.zscore(df)) < n_stddev, axis=-1)
    return df[outliers >= outliers.max()]

def df_add_quantize_parameters(df, parameters, nQuantiles):
    param_qbins = dict()
    param_qbins_middle = dict()

    for param_name in parameters:
        param_factored, param_qbins[param_name] = pd.qcut(df[param_name], nQuantiles, retbins=True, labels=False)
        param_qbins_middle[param_name] = (
            (param_qbins[param_name][:-1] + param_qbins[param_name][1:])/2.
        ).astype(df[param_name].dtype)
        df.loc[:, (param_name + "_qi")] = param_factored

    return df, param_qbins, param_qbins_middle


def filter_dataframe(df, parameters_values):
    filter_mask = None
    for key, value in parameters_values.iteritems():
        new_filter = (df[key] == value)
        if filter_mask is None:
            filter_mask = new_filter
        else:
            filter_mask = filter_mask & new_filter

    if filter_mask is None:
        return df
    else:
        return df[filter_mask]


def filter_quantized_param(df, target_parameters, param_qbins):
    quantized_parameters_targets = dict()

    for key, value in target_parameters.iteritems():
        target_qi = (np.digitize(value, param_qbins[key], right=False).item() - 1)
        quantized_parameters_targets[key + "_qi"] = target_qi

    return filter_dataframe(df, quantized_parameters_targets)
