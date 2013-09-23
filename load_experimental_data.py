'''
     Load experimental data in python, because Matlab sucks ass.
'''

import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# import matplotlib.patches as plt_patches
# import matplotlib.gridspec as plt_grid
import os
# import bottleneck as bn
import em_circularmixture
import pandas as pd

from utils import *


def convert_wrap(dataset, keys_to_convert = ['item_angle', 'probe_angle', 'response', 'error', 'err'], max_angle=np.pi/2.):
    '''
        Takes a dataset and a list of keys. Each data associated with these keys will be converted to radian,
            and wrapped in a [-max_angle, max_angle] interval
    '''
    for key in keys_to_convert:
        if key in dataset:
            dataset[key] = wrap_angles(np.deg2rad(dataset[key]), bound = max_angle)


def preprocess_simultaneous(dataset, parameters):
    '''
        For simultaneous datasets, there is no 'probe' key, the first item in 'item_angle' is the target one.
        The 'error' key is missing and called 'err', which we will correct as well.
    '''

    # Extract parameters
    # params_lists: List( (param_name, default_value) )
    params_list = [('convert_radians', True), ('correct_probe', True)]
    for curr_param in params_list:
        if curr_param[0] in parameters:
            # Set to provided value
            exec(curr_param[0] +" = parameters['" + curr_param[0] + "']")
        else:
            # Default value
            exec(curr_param[0] +" = " + str(curr_param[1]))

    # Rename the error field
    dataset['error'] = dataset['err']
    del dataset['err']

    # Assign probe field correctly
    dataset['probe'] = np.zeros(dataset['error'].shape, dtype= int)

    # Convert everything to radians, spanning a -np.pi/2:np.pi
    if convert_radians:   #pylint: disable=E0602
        convert_wrap(dataset)

    # Compute additional errors, between the response and all items
    compute_all_errors(dataset)

    # Create arrays per subject
    create_subject_arrays(dataset)

    # Fit the mixture model, and save the responsibilities per datapoint.
    # em_circularmixture.fit()




def preprocess_sequential(dataset, parameters):
    '''
        For sequential datasets, need to convert to radians and correct the probe indexing.
    '''
    # Extract parameters
    # params_lists: List( (param_name, default_value) )
    params_list = [('convert_radians', True), ('correct_probe', True)]
    for curr_param in params_list:
        if curr_param[0] in parameters:
            # Set to provided value
            exec(curr_param[0] + " = parameters['" + curr_param[0] + "']")
        else:
            # Default value
            exec(curr_param[0] + " = " + str(curr_param[1]))


    # Convert everything to radians, spanning a -np.pi/2:np.pi
    if convert_radians:  #pylint: disable=E0602
        convert_wrap(dataset)

    # Correct the probe field, Matlab format for indices...
    if correct_probe and 'probe' in dataset:  #pylint: disable=E0602
        dataset['probe'] = dataset['probe'].astype(int)
        dataset['probe'] -= 1

    # Compute additional errors, between the response and all items
    compute_all_errors(dataset)

    # Create arrays per subject
    create_subject_arrays(dataset)

    # Fit the mixture model, and save the responsibilities per datapoint.
    # em_circularmixture.fit()



def preprocess_doublerecall(dataset, parameters):
    '''
        This is the dataset where both colour and orientation can be recalled.
        - There are two groups of subjects, either with 6 or 3 items shown (no intermediates...). Stored in 'n_items'
        - Subjects recalled either colour or orientation, per blocks. Stored in 'cond'
        - Subject report their confidence, which is cool.

        Things to change:
        - 'item_location' really contains 'item_angle'...
        - item_location and probe_location should be wrapped back into -pi:pi.
        - Should compute the errors.
    '''

    # Make some aliases
    dataset['item_angle'] = dataset['item_location']
    dataset['probe_angle'] = dataset['probe_location']
    dataset['response'] = np.empty(dataset['probe_angle'].size)
    dataset['target'] = np.empty(dataset['probe_angle'].size)
    dataset['probe'] = np.zeros(dataset['probe_angle'].shape, dtype= int)

    # Get shortcuts for colour and orientation trials
    dataset['colour_trials'] = (dataset['cond'] == 1.).flatten()
    dataset['angle_trials'] = (dataset['cond'] == 2.).flatten()
    dataset['3_items_trials'] = (dataset['n_items'] == 3.0).flatten()
    dataset['6_items_trials'] = (dataset['n_items'] == 6.0).flatten()

    # Wrap everything around
    dataset['item_angle'] = wrap_angles(dataset['item_angle'], np.pi)
    dataset['probe_angle'] = wrap_angles(dataset['probe_angle'], np.pi)
    dataset['item_colour'] = wrap_angles(dataset['item_colour'], np.pi)
    dataset['probe_colour'] = wrap_angles(dataset['probe_colour'], np.pi)

    # Remove wrong trials
    reject_ids = (dataset['reject'] == 1.0).flatten()
    for key in dataset:
        if type(dataset[key]) == np.ndarray and dataset[key].shape[0] == reject_ids.size and key in ('probe_colour', 'probe_angle', 'item_angle', 'item_colour'):
            dataset[key][reject_ids] = np.nan

    # Compute the errors
    dataset['errors_angle_all'] = wrap_angles(dataset['item_angle'] - dataset['probe_angle'], np.pi/2.)
    dataset['errors_colour_all'] = wrap_angles(dataset['item_colour'] - dataset['probe_colour'], np.pi/2.)
    dataset['error_angle'] = dataset['errors_angle_all'][:, 0]
    dataset['error_colour'] = dataset['errors_colour_all'][:, 0]
    dataset['error'] = np.where(~np.isnan(dataset['error_angle']), dataset['error_angle'], dataset['error_colour'])


    # Fit the mixture model
    dataset['em_fits'] = dict(kappa=np.empty(dataset['probe_angle'].size), mixt_target=np.empty(dataset['probe_angle'].size), mixt_nontarget=np.empty(dataset['probe_angle'].size), mixt_random=np.empty(dataset['probe_angle'].size), resp_target=np.empty(dataset['probe_angle'].size), resp_nontarget=np.empty(dataset['probe_angle'].size), resp_random=np.empty(dataset['probe_angle'].size), train_LL=np.empty(dataset['probe_angle'].size), test_LL=np.empty(dataset['probe_angle'].size))
    for key in dataset['em_fits']:
        dataset['em_fits'][key].fill(np.nan)

    # Angles trials
    ids_angle = (dataset['cond'] ==  2.0).flatten()
    for n_items in np.unique(dataset['n_items']):
        ids_n_items = (dataset['n_items'] == n_items).flatten()
        ids_filtered = ids_angle & ids_n_items

        dataset['target'][ids_filtered] = dataset['item_angle'][ids_filtered, 0]
        dataset['response'][ids_filtered] = dataset['probe_angle'][ids_filtered, 0]

        # params_fit = em_circularmixture.fit(dataset['probe_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:])
        cross_valid_outputs = em_circularmixture.cross_validation_kfold(dataset['probe_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:], K=10, shuffle=True)
        params_fit = cross_valid_outputs['best_fit']
        resp = em_circularmixture.compute_responsibilities(dataset['probe_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:], params_fit)

        dataset['em_fits']['kappa'][ids_filtered] = params_fit['kappa']
        dataset['em_fits']['mixt_target'][ids_filtered] = params_fit['mixt_target']
        dataset['em_fits']['mixt_nontarget'][ids_filtered] = params_fit['mixt_nontargets']
        dataset['em_fits']['mixt_random'][ids_filtered] = params_fit['mixt_random']
        dataset['em_fits']['resp_target'][ids_filtered] = resp['target']
        dataset['em_fits']['resp_nontarget'][ids_filtered] = np.sum(resp['nontargets'], axis=1)
        dataset['em_fits']['resp_random'][ids_filtered] = resp['random']
        dataset['em_fits']['train_LL'][ids_filtered] = params_fit['train_LL']
        dataset['em_fits']['test_LL'][ids_filtered] = cross_valid_outputs['best_test_LL']

    # Colour trials
    ids_colour = (dataset['cond'] ==  1.0).flatten()
    for n_items in np.unique(dataset['n_items']):
        ids_n_items = (dataset['n_items'] == n_items).flatten()
        ids_filtered = ids_colour & ids_n_items

        dataset['target'][ids_filtered] = dataset['item_colour'][ids_filtered, 0]
        dataset['response'][ids_filtered] = dataset['probe_colour'][ids_filtered, 0]

        # params_fit = em_circularmixture.fit(dataset['probe_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 1:])
        cross_valid_outputs = em_circularmixture.cross_validation_kfold(dataset['probe_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 1:], K=10, shuffle=True)
        params_fit = cross_valid_outputs['best_fit']
        resp = em_circularmixture.compute_responsibilities(dataset['probe_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 1:], params_fit)

        dataset['em_fits']['kappa'][ids_filtered] = params_fit['kappa']
        dataset['em_fits']['mixt_target'][ids_filtered] = params_fit['mixt_target']
        dataset['em_fits']['mixt_nontarget'][ids_filtered] = params_fit['mixt_nontargets']
        dataset['em_fits']['mixt_random'][ids_filtered] = params_fit['mixt_random']
        dataset['em_fits']['resp_target'][ids_filtered] = resp['target']
        dataset['em_fits']['resp_nontarget'][ids_filtered] = np.sum(resp['nontargets'], axis=1)
        dataset['em_fits']['resp_random'][ids_filtered] = resp['random']
        dataset['em_fits']['train_LL'][ids_filtered] = params_fit['train_LL']
        dataset['em_fits']['test_LL'][ids_filtered] = cross_valid_outputs['best_test_LL']

    # Try with Pandas for some advanced plotting
    dataset_filtered = dict((k, dataset[k].flatten()) for k in ('n_items', 'trial', 'subject', 'reject', 'rating', 'probe_colour', 'probe_angle', 'cond', 'error', 'error_angle', 'error_colour', 'response', 'target'))
    dataset_filtered.update(dataset['em_fits'])
    dataset['panda'] = pd.DataFrame(dataset_filtered)



def load_dataset(filename = '', specific_preprocess = lambda x, p: x, parameters={}):
    '''
        Load datasets.
        Supports
            - Simultaneous and sequential Gorgoraptis_2011 data: Exp1 and Exp2.
            - Dual recall data DualRecall_Bays.
    '''

    # Load everything
    dataset = sio.loadmat(filename, mat_dtype=True)

    # Specific operations, for different types of datasets
    specific_preprocess(dataset, parameters)


    return dataset


def load_multiple_datasets(datasets_descr = []):
    '''
        Takes a list of datasets descriptors, and loads them up.
    '''

    datasets = []
    for datasets_descr in datasets_descr:
        datasets.append( load_dataset(filename=datasets_descr['filename'], specific_preprocess=datasets_descr['preprocess'], parameters=datasets_descr['parameters']))

    return datasets


def reconstruct_colours_exp1(datasets=('Data/ad.mat', 'Data/gb.mat', 'Data/kf.mat', 'Data/md.mat', 'Data/sf.mat', 'Data/sw.mat', 'Data/wd.mat', 'Data/zb.mat')):
    '''
        The colour is missing from the simultaneous experiment dataset
        Reconstruct it.
    '''

    all_colours = []
    all_preangles = []
    all_targets = []
    for dataset_fn in datasets:
        print dataset_fn
        curr_data = sio.loadmat(dataset_fn, mat_dtype=True)

        all_colours.append(curr_data['probe_colour'])
        all_preangles.append(wrap_angles(np.deg2rad(curr_data['probe_pre_angle']), bound=np.pi/2.))
        all_targets.append(wrap_angles(np.deg2rad(curr_data['item_angle'][:, 0]), bound=np.pi/2.))

    print "Data loaded"

    all_colours = np.array(all_colours)
    all_targets = np.array(all_targets)
    all_preangles = np.array(all_preangles)

    # Ordering in original data
    order_subjects = [0, 4, 1, 5, 2, 6, 3, 7]
    all_colours = all_colours[order_subjects]
    all_targets = all_targets[order_subjects]
    all_preangles = all_preangles[order_subjects]

    # Flatten everything
    all_colours_ = []
    all_targets_ = []
    all_preangles_ = []
    for curr_colour in all_colours:
        all_colours_.extend(np.squeeze(curr_colour).tolist())
    print "Colours flat"
    for curr_target in all_targets:
        all_targets_.extend(np.squeeze(curr_target).tolist())
    print "Targets flat"
    for curr_preangle in all_preangles:
        all_preangles_.extend(np.squeeze(curr_preangle).tolist())
    print "Preangles flat"


    return (np.array(all_colours_), np.array(all_targets_), np.array(all_preangles_))



def compute_all_errors(dataset = {}):
    '''
        Will compute the error between the response and all possible items
    '''

    # Get the difference between angles
    # Should also wrap it around
    dataset['errors_all'] = wrap_angles(dataset['item_angle'] - dataset['response'], bound=np.pi/2.)

    # Sanity check, verify that errors computed are the same as precomputed ones.
    assert all(np.abs(dataset['errors_all'][np.arange(dataset['probe'].size), dataset['probe'][:, 0]] - dataset['error'][:, 0]) < 10**-6), "Errors computed are different than errors given in the data"


def create_subject_arrays(dataset = {}):
    '''
        Create arrays with errors per subject and per num_target
        also create an array with the precision per subject and num_target directly
    '''

    unique_subjects = np.unique(dataset['subject'])
    unique_n_items = np.unique(dataset['n_items'])

    errors_subject_nitems = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)
    precision_subject_nitems = np.zeros((unique_subjects.size, unique_n_items.size))
    precision_subject_nitems_theo = np.zeros((unique_subjects.size, unique_n_items.size))
    precision_subject_nitems_nochance = np.zeros((unique_subjects.size, unique_n_items.size))
    precision_subject_nitems_raw = np.zeros((unique_subjects.size, unique_n_items.size))

    for n_items_i, n_items in enumerate(unique_n_items):
        for subject_i, subject in enumerate(unique_subjects):
            # Get the responses and errors
            errors_subject_nitems[subject_i, n_items_i] = dataset['error'][(dataset['subject']==subject) & (dataset['n_items'] == n_items)]

            # Compute the precision
            precision_subject_nitems[subject_i, n_items_i] = compute_precision(errors_subject_nitems[subject_i, n_items_i], remove_chance_level=True, correct_orientation=True, use_wrong_precision=True)
            precision_subject_nitems_theo[subject_i, n_items_i] = compute_precision(errors_subject_nitems[subject_i, n_items_i], remove_chance_level=False, correct_orientation=True, use_wrong_precision=False)
            precision_subject_nitems_nochance[subject_i, n_items_i] = compute_precision(errors_subject_nitems[subject_i, n_items_i], remove_chance_level=True, correct_orientation=True, use_wrong_precision=False)
            precision_subject_nitems_raw[subject_i, n_items_i] = compute_precision(errors_subject_nitems[subject_i, n_items_i], remove_chance_level=False)

    dataset['errors_subject_nitems'] = errors_subject_nitems
    dataset['precision_subject_nitems_bays'] = precision_subject_nitems
    dataset['precision_subject_nitems_theo'] = precision_subject_nitems_theo
    dataset['precision_subject_nitems_theo_nochance'] = precision_subject_nitems_nochance
    dataset['precision_subject_nitems_bays_chance'] = precision_subject_nitems_raw


def check_bias_all(dataset):
    '''
        Get an histogram of the errors between the response and all non targets
            If biased towards 0-values, should indicate misbinding errors.

        (if you do this with respect to all targets, it's retarded and should always be biased)
    '''

    # Remove all the probes, by setting them to nan
    all_errors = dataset['errors_all'].copy()
    all_errors[np.arange(dataset['probe'].size), dataset['probe'][:, 0]] = np.nan

    # Now filter all nans
    all_errors = all_errors[~np.isnan(all_errors)]

    # Some plots
    plt.figure()
    plt.hist(all_errors, bins=20)


def check_bias_bestnontarget(dataset):
    '''
        Get an histogram of errors between response and best nontarget.
        Should be more biased towards 0 than the overall average
    '''

    # all_errors = dataset['errors_all']
    all_errors = dataset['errors_all'][(dataset['n_items'] > 1)[:, 0]]  # filter trials with only no nontargets
    probe_indices = dataset['probe'][(dataset['n_items'] > 1)[:, 0], 0]

    ## Remove all the probes, by setting them to nan
    # all_errors[np.arange(probe_indices.size), probe_indices] = np.nan

    # Keep only the best non target for each trial
    # Use Bottleneck
    # min_indices = bn.nanargmin(np.abs(all_errors), axis=1)
    # all_errors = all_errors[np.arange(probe_indices.size), min_indices]

    ## More efficient, use masked arrays
    masked_errors = np.ma.masked_invalid(all_errors)
    # mask the probed item
    masked_errors[np.arange(probe_indices.size), probe_indices] = np.ma.masked

    # Get the best non targets
    all_errors = all_errors[np.arange(probe_indices.size), np.ma.argmin(np.abs(masked_errors), axis=1)]

    ## Superslow
    # all_errors = np.array([all_errors[i, np.nanargmin(np.abs(all_errors[i]))] for i in xrange(all_errors.shape[0]) if not np.isnan(np.nanargmin(np.abs(all_errors[i])))])

    # Some plots
    plt.figure()
    plt.hist(all_errors, bins=20)


def compute_precision(errors, remove_chance_level=True, correct_orientation=True, use_wrong_precision=True):
    '''
        Compute the precision (1./circ_std**2). Remove the chance level if desired.
    '''

    if correct_orientation:
        # Correct for the fact that bars are modelled in [0, pi] and not [0, 2pi]
        errors = errors.copy()*2.0

    # avg_error = np.mean(np.abs(errors), axis=0)

    # Angle population vector
    error_mean_vector = np.mean(np.exp(1j*errors), axis=0)

    # Population mean
    # error_mean_error = np.angle(error_mean_vector)

    # Circular standard deviation estimate
    error_std_dev_error = np.sqrt(-2.*np.log(np.abs(error_mean_vector)))

    # Precision
    if use_wrong_precision:
        precision = 1./error_std_dev_error
    else:
        precision = 1./error_std_dev_error**2.

    if remove_chance_level:
        # Expected precision under uniform distribution
        x = np.logspace(-2, 2, 100)

        precision_uniform = np.trapz(errors.size/(np.sqrt(x)*np.exp(x+errors.size*np.exp(-x))), x)

        # Remove the chance level
        precision -= precision_uniform

    if correct_orientation:
        # The obtained precision is for half angles, correct it
        precision *= 2.

    return precision


def check_oblique_effect(data, nb_bins=100):
    '''
        Humans are more precise for vertical and horizontal bars than diagonal orientations.

        Check if present.
    '''

    # Construct the list of (target angles, errors), see if there is some structure in that
    errors_per_angle = np.array(zip(data['item_angle'][np.arange(data['probe'].size), data['probe'][:, 0]], data['error'][:, 0]))

    plt.figure()
    plt.plot(errors_per_angle[:, 0], errors_per_angle[:, 1], 'x')

    plt.figure()
    plt.plot(errors_per_angle[:, 0], np.abs(errors_per_angle[:, 1]), 'x')

    discrete_x = np.linspace(-np.pi/2., np.pi/2., nb_bins)
    avg_error = np.zeros(discrete_x.shape)

    for x_i in np.arange(discrete_x.size):
        if x_i < discrete_x.size - 1:
            # Check what data comes in the current interval x[x_i, x_i+1]
            avg_error[x_i] = np.mean(errors_per_angle[np.logical_and(errors_per_angle[:, 0] > discrete_x[x_i], errors_per_angle[:, 0] < discrete_x[x_i+1]), 1])

    plt.figure()
    plt.plot(discrete_x, avg_error)

    plt.figure()
    plt.plot(discrete_x, avg_error**2.)

    plt.figure()
    plt.plot(discrete_x, np.abs(avg_error))


def plots_doublerecall(dataset):
    '''
        Create plots for the double recall dataset
    '''

    to_plot = {'resp_vs_targ':True, 'error_boxplot':True, 'resp_rating':True, 'em_fits':True, 'loglik':True, 'resp_distrib':True, 'resp_conds':True}

    dataset_pd = dataset['panda']

    dataset_pd['error_abs'] = dataset_pd.error.abs()

    # Show distributions of responses wrt target angles/colour
    if to_plot['resp_vs_targ']:

        # Plot scatter and marginals for the orientation trials
        scatter_marginals(dropnan(dataset['item_angle'][dataset['angle_trials'] & dataset['3_items_trials'], 0]), dropnan(dataset['probe_angle'][dataset['angle_trials'] & dataset['3_items_trials']]), xlabel ='Target angle', ylabel='Response angle', title='Angle trials, 3 items', figsize=(9, 9), factor_axis=1.1, bins=61)
        scatter_marginals(dropnan(dataset['item_angle'][dataset['angle_trials'] & dataset['6_items_trials'], 0]), dropnan(dataset['probe_angle'][dataset['angle_trials'] & dataset['6_items_trials']]), xlabel ='Target angle', ylabel='Response angle', title='Angle trials, 6 items', figsize=(9, 9), factor_axis=1.1, bins=61)

        # Plot scatter and marginals for the colour trials
        scatter_marginals(dropnan(dataset['item_colour'][dataset['colour_trials']& dataset['3_items_trials'], 0]), dropnan(dataset['probe_colour'][dataset['colour_trials'] & dataset['3_items_trials']]), xlabel ='Target colour', ylabel='Response colour', title='Colour trials, 3 items', figsize=(9, 9), factor_axis=1.1, bins=61, show_colours=True)
        scatter_marginals(dropnan(dataset['item_colour'][dataset['colour_trials'] & dataset['6_items_trials'], 0]), dropnan(dataset['probe_colour'][dataset['colour_trials'] & dataset['6_items_trials']]), xlabel ='Target colour', ylabel='Response colour', title='Colour trials, 6 items', figsize=(9, 9), factor_axis=1.1, bins=61, show_colours=True)



    # dataset_pd[ids_filtered][ids_targets_responses].boxplot('error_angle_abs', by='rating')
    # dataset_pd[ids_filtered][ids_nontargets_responses].boxplot('error_angle_abs', by='rating')
    if to_plot['error_boxplot']:
        dataset_pd.boxplot(column=['error_abs'], by=['cond', 'n_items', 'rating'])

    # for i in dataset_pd.subject.unique():
    #     dataset_pd[dataset_pd.subject == i].boxplot(column=['error_angle'], by=['n_items', 'rating'])

    # Show distribution responsibility as a function of rating
    if to_plot['resp_rating']:
        # dataset_grouped_nona_rating = dataset_pd[dataset_pd.n_items == 3.0].dropna(subset=['error']).groupby(['rating'])
        dataset_grouped_nona_rating = dataset_pd[dataset_pd.n_items == 6.0][dataset_pd.cond == 1.].dropna(subset=['error']).groupby(['rating'])
        f, axes = plt.subplots(dataset_pd.rating.nunique(), 3)
        i = 0
        bins = np.linspace(0., 1.0, 31)
        for name, group in dataset_grouped_nona_rating:
            print name

            # Compute histograms and normalize per rating condition
            counts_target, bins_edges = np.histogram(group.resp_target, bins=bins)
            counts_nontarget, bins_edges = np.histogram(group.resp_nontarget, bins=bins)
            counts_random, bins_edges = np.histogram(group.resp_random, bins=bins)
            dedges = np.diff(bins_edges)[0]

            sum_counts = float(np.sum(counts_target) + np.sum(counts_nontarget) + np.sum(counts_random))
            counts_target = counts_target/sum_counts
            counts_nontarget = counts_nontarget/sum_counts
            counts_random = counts_random/sum_counts

            # Print Responsibility target density estimation
            # group.resp_target.plot(kind='kde', ax=axes[i, 0])
            axes[i, 0].bar(bins_edges[:-1], counts_target, dedges, color='b')
            axes[i, 0].set_xlim((0.0, 1.0))
            axes[i, 0].set_ylim((0.0, 0.35))
            axes[i, 0].text(0.5, 0.8, "T " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 0].transAxes)

            # Print Responsibility nontarget density estimation
            # group.resp_nontarget.plot(kind='kde', ax=axes[i, 1])
            axes[i, 1].bar(bins_edges[:-1], counts_nontarget, dedges, color='g')
            axes[i, 1].set_xlim((0.0, 1.0))
            axes[i, 1].set_ylim((0.0, 0.35))
            axes[i, 1].text(0.5, 0.8, "NT " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 1].transAxes)

            # Print Responsibility random density estimation
            # group.resp_random.plot(kind='kde', ax=axes[i, 1])
            axes[i, 2].bar(bins_edges[:-1], counts_random, dedges, color='r')
            axes[i, 2].set_xlim((0.0, 1.0))
            axes[i, 2].set_ylim((0.0, 0.35))
            axes[i, 2].text(0.5, 0.8, "R " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 2].transAxes)

            i += 1

        plt.suptitle("Colour trials")

        dataset_grouped_nona_rating = dataset_pd[dataset_pd.n_items == 6.0][dataset_pd.cond == 2.].dropna(subset=['error']).groupby(['rating'])
        f, axes = plt.subplots(dataset_pd.rating.nunique(), 3)
        i = 0
        bins = np.linspace(0., 1.0, 31)
        for name, group in dataset_grouped_nona_rating:
            print name

            # Compute histograms and normalize per rating condition
            counts_target, bins_edges = np.histogram(group.resp_target, bins=bins)
            counts_nontarget, bins_edges = np.histogram(group.resp_nontarget, bins=bins)
            counts_random, bins_edges = np.histogram(group.resp_random, bins=bins)
            dedges = np.diff(bins_edges)[0]

            sum_counts = float(np.sum(counts_target) + np.sum(counts_nontarget) + np.sum(counts_random))
            counts_target = counts_target/sum_counts
            counts_nontarget = counts_nontarget/sum_counts
            counts_random = counts_random/sum_counts

            # Print Responsibility target density estimation
            # group.resp_target.plot(kind='kde', ax=axes[i, 0])
            axes[i, 0].bar(bins_edges[:-1], counts_target, dedges, color='b')
            axes[i, 0].set_xlim((0.0, 1.0))
            axes[i, 0].set_ylim((0.0, 0.35))
            axes[i, 0].text(0.5, 0.8, "T " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 0].transAxes)

            # Print Responsibility nontarget density estimation
            # group.resp_nontarget.plot(kind='kde', ax=axes[i, 1])
            axes[i, 1].bar(bins_edges[:-1], counts_nontarget, dedges, color='g')
            axes[i, 1].set_xlim((0.0, 1.0))
            axes[i, 1].set_ylim((0.0, 0.35))
            axes[i, 1].text(0.5, 0.8, "NT " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 1].transAxes)

            # Print Responsibility random density estimation
            # group.resp_random.plot(kind='kde', ax=axes[i, 1])
            axes[i, 2].bar(bins_edges[:-1], counts_random, dedges, color='r')
            axes[i, 2].set_xlim((0.0, 1.0))
            axes[i, 2].set_ylim((0.0, 0.35))
            axes[i, 2].text(0.5, 0.8, "R " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 2].transAxes)

            i += 1

        plt.suptitle("Angle trials")


    # Add condition names
    dataset_pd['cond_name'] = np.array(['Colour', 'Angle'])[np.array(dataset_pd['cond']-1, dtype=int)]

    dataset_grouped_nona_conditems = dataset_pd.dropna(subset=['error']).groupby(['cond_name', 'n_items'])
    dataset_grouped_nona_conditems_mean = dataset_grouped_nona_conditems.mean()[['mixt_target', 'mixt_nontarget', 'mixt_random', 'kappa', 'train_LL', 'test_LL']]

    # Show inferred mixture proportions and kappa
    if to_plot['em_fits']:
        ax = dataset_grouped_nona_conditems_mean[['mixt_target', 'mixt_nontarget', 'mixt_random', 'kappa']].plot(secondary_y='kappa', kind='bar')
        ax.set_ylabel('Mixture proportions')
        ax.right_ax.set_ylabel('Kappa')

    # Show loglihood of fit
    if to_plot['loglik']:
        f, ax = plt.subplots(1, 1)
        dataset_grouped_nona_conditems_mean[['train_LL', 'test_LL']].plot(kind='bar', ax=ax, secondary_y='test_LL')

    # Show boxplot of responsibilities
    if to_plot['resp_distrib']:
        dataset_grouped_nona_conditems.boxplot(column=['resp_target', 'resp_nontarget', 'resp_random'])

    # Show distributions of responsibilities
    if to_plot['resp_conds']:
        f, axes = plt.subplots(dataset_pd.cond_name.nunique()*dataset_pd.n_items.nunique(), 3)
        i = 0
        bins = np.linspace(0., 1.0, 31)
        for name, group in dataset_grouped_nona_conditems:
            print name

            # Print Responsibility target density estimation
            # group.resp_target.plot(kind='kde', ax=axes[i, 0])
            group.resp_target.hist(ax=axes[i, 0], color='b', bins=bins)
            axes[i, 0].text(0.5, 0.85, "T " + ' '.join([str(x) for x in name]), fontweight='bold', horizontalalignment='center', transform = axes[i, 0].transAxes)
            axes[i, 0].set_xlim((0.0, 1.0))

            # Print Responsibility nontarget density estimation
            # group.resp_nontarget.plot(kind='kde', ax=axes[i, 1])
            group.resp_nontarget.hist(ax=axes[i, 1], color='g', bins=bins)
            axes[i, 1].text(0.5, 0.85, "NT " + ' '.join([str(x) for x in name]), fontweight='bold', horizontalalignment='center', transform = axes[i, 1].transAxes)
            axes[i, 1].set_xlim((0.0, 1.0))

            # Print Responsibility random density estimation
            # group.resp_random.plot(kind='kde', ax=axes[i, 1])
            group.resp_random.hist(ax=axes[i, 2], color='r', bins=bins)
            axes[i, 2].text(0.5, 0.85, "R " + ' '.join([str(x) for x in name]), fontweight='bold', horizontalalignment='center', transform = axes[i, 2].transAxes)
            axes[i, 2].set_xlim((0.0, 1.0))

            i += 1

    # Extract some parameters
    fitted_parameters = dataset_grouped_nona_conditems_mean.iloc[0].loc[['kappa', 'mixt_target', 'mixt_nontarget', 'mixt_random']]
    print fitted_parameters



if __name__ == '__main__':
    ## Load data
    data_dir = '/Users/loicmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/experimental_data/'

    print sys.argv

    if False or (len(sys.argv) > 1 and sys.argv[1]):
    # keys:
    # 'probe', 'delayed', 'item_colour', 'probe_colour', 'item_angle', 'error', 'probe_angle', 'n_items', 'response', 'subject']
        (data_sequen, data_simult, data_dualrecall) = load_multiple_datasets([dict(filename=os.path.join(data_dir, 'Gorgoraptis_2011', 'Exp1.mat'), preprocess=preprocess_sequential, parameters={}), dict(filename=os.path.join(data_dir, 'Gorgoraptis_2011', 'Exp2.mat'), preprocess=preprocess_simultaneous, parameters={}), dict(filename=os.path.join(data_dir, 'DualRecall_Bays', 'rate_data.mat'), preprocess=preprocess_doublerecall, parameters={})])


    # Check for bias towards 0 for the error between response and all items
    # check_bias_all(data_simult)

    # Check for bias for the best non-probe
    # check_bias_bestnontarget(data_simult)

    # check_bias_all(data_sequen)
    # check_bias_bestnontarget(data_sequen)

    # print data_simult['precision_subject_nitems_bays']
    # print data_simult['precision_subject_nitems_theo']

    # prec_exp = np.mean(data_simult['precision_subject_nitems_bays'], axis=0)
    # prec_theo = np.mean(data_simult['precision_subject_nitems_theo'], axis=0)
    # fi_fromexp = prec_exp**2./4.
    # fi_fromtheo = prec_theo**2./4.
    # print "Precision experim", prec_exp
    # print "FI from exp", fi_fromexp
    # print "Precision no chance level removed", prec_theo
    # print "FI no chance", fi_fromtheo

    # check_oblique_effect(data_simult, nb_bins=50)

    # np.save('processed_experimental_230613.npy', dict(data_simult=data_simult, data_sequen=data_sequen))

    plots_doublerecall(data_dualrecall)



    plt.show()





