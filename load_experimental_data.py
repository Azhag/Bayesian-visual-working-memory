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
import os.path
import cPickle as pickle
# import bottleneck as bn
import em_circularmixture
import em_circularmixture_allitems_uniquekappa
import pandas as pd

import dataio as DataIO

import utils


def convert_wrap(dataset, keys_to_convert = ['item_angle', 'probe_angle', 'response', 'error', 'err'], multiply_factor=2., max_angle=np.pi):
    '''
        Takes a dataset and a list of keys. Each data associated with these keys will be converted to radian,
            and wrapped in a [-max_angle, max_angle] interval
    '''
    for key in keys_to_convert:
        if key in dataset:
            dataset[key] = utils.wrap_angles(np.deg2rad(multiply_factor*dataset[key]), bound = max_angle)


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
    if 'err' in dataset:
        dataset['error'] = dataset['err']
        del dataset['err']

    # Assign probe field correctly
    dataset['probe'] = np.zeros(dataset['error'].shape, dtype= int)


    # Convert everything to radians, spanning a -np.pi:np.pi
    if convert_radians: #pylint: disable=E0602
        convert_wrap(dataset, multiply_factor=2, max_angle=np.pi)

    # Make some aliases
    dataset['n_items'] = dataset['n_items'].astype(int)
    dataset['subject'] = dataset['subject'].astype(int)

    dataset['n_items_size'] = np.unique(dataset['n_items']).size
    dataset['subject_size'] = np.unique(dataset['subject']).size

    # Compute additional errors, between the response and all items
    compute_all_errors(dataset)

    # Create arrays per subject
    create_subject_arrays(dataset)

    # Reconstruct the colour information_
    if 'item_colour' not in dataset:
        reconstruct_colours_exp1(dataset, datadir = parameters.get('datadir', ''))

    # Fit the mixture model, and save the responsibilities per datapoint.
    if parameters['fit_mixture_model']:
        if 'mixture_model_cache' in parameters:
            mixture_model_cache_filename = os.path.join(parameters.get('datadir', ''), parameters['mixture_model_cache'])
            fit_mixture_model(dataset, caching_save_filename=mixture_model_cache_filename)
        else:
            fit_mixture_model(dataset)

    ## Save item in a nice format for the model fit
    dataset['data_to_fit'] = {}
    dataset['data_to_fit']['n_items'] = np.unique(dataset['n_items'])
    for n_items in dataset['data_to_fit']['n_items']:
        ids_n_items = (dataset['n_items'] == n_items).flatten()

        if n_items not in dataset['data_to_fit']:
            dataset['data_to_fit'][n_items] = {}
            dataset['data_to_fit'][n_items]['N'] = np.sum(ids_n_items)
            dataset['data_to_fit'][n_items]['probe'] = np.unique(dataset['probe'][ids_n_items])
            dataset['data_to_fit'][n_items]['item_features'] = np.empty((dataset['data_to_fit'][n_items]['N'], n_items, 2))
            dataset['data_to_fit'][n_items]['response'] = np.empty((dataset['data_to_fit'][n_items]['N'], 1))

        dataset['data_to_fit'][n_items]['item_features'][..., 0] = dataset['item_angle'][ids_n_items, :n_items]
        dataset['data_to_fit'][n_items]['item_features'][..., 1] = dataset['item_colour'][ids_n_items, :n_items]
        dataset['data_to_fit'][n_items]['response'] = dataset['response'][ids_n_items].flatten()

    # Perform Vtest for circular uniformity
    dataset['vtest_nitems'] = np.empty(dataset['n_items_size'])*np.nan
    for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
        if n_items > 1:
            dataset['vtest_nitems'][n_items_i] = utils.V_test(utils.dropnan(dataset['errors_nontarget_nitems'][n_items_i]).flatten())['pvalue']

    # Do per subject and nitems, get average histogram
    compute_average_histograms(dataset)



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



def preprocess_dualrecall(dataset, parameters):
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
    dataset['response'] = np.empty((dataset['probe_angle'].size, 1))
    dataset['target'] = np.empty(dataset['probe_angle'].size)
    dataset['probe'] = np.zeros(dataset['probe_angle'].shape, dtype= int)

    dataset['n_items'] = dataset['n_items'].astype(int)
    dataset['cond'] = dataset['cond'].astype(int)

    dataset['n_items_size'] = np.unique(dataset['n_items']).size
    dataset['subject_size'] = np.unique(dataset['subject']).size

    # Get shortcuts for colour and orientation trials
    dataset['colour_trials'] = (dataset['cond'] == 1).flatten()
    dataset['angle_trials'] = (dataset['cond'] == 2).flatten()
    dataset['3_items_trials'] = (dataset['n_items'] == 3).flatten()
    dataset['6_items_trials'] = (dataset['n_items'] == 6).flatten()

    # Wrap everything around
    multiply_factor = 2.
    dataset['item_angle'] = utils.wrap_angles(multiply_factor*dataset['item_angle'], np.pi)
    dataset['probe_angle'] = utils.wrap_angles(multiply_factor*dataset['probe_angle'], np.pi)
    dataset['item_colour'] = utils.wrap_angles(multiply_factor*dataset['item_colour'], np.pi)
    dataset['probe_colour'] = utils.wrap_angles(multiply_factor*dataset['probe_colour'], np.pi)

    # Remove wrong trials
    reject_ids = (dataset['reject'] == 1.0).flatten()
    for key in dataset:
        if type(dataset[key]) == np.ndarray and dataset[key].shape[0] == reject_ids.size and key in ('probe_colour', 'probe_angle', 'item_angle', 'item_colour'):
            dataset[key][reject_ids] = np.nan

    # Compute the errors
    dataset['errors_angle_all'] = utils.wrap_angles(dataset['item_angle'] - dataset['probe_angle'], np.pi)
    dataset['errors_colour_all'] = utils.wrap_angles(dataset['item_colour'] - dataset['probe_colour'], np.pi)
    dataset['error_angle'] = dataset['errors_angle_all'][:, 0]
    dataset['error_colour'] = dataset['errors_colour_all'][:, 0]
    dataset['error'] = np.where(~np.isnan(dataset['error_angle']), dataset['error_angle'], dataset['error_colour'])

    dataset['errors_nitems'] = np.empty(dataset['n_items_size'], dtype=np.object)
    dataset['errors_all_nitems'] = np.empty(dataset['n_items_size'], dtype=np.object)

    for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
        ids_filtered = dataset['angle_trials'] & (dataset['n_items'] == n_items).flatten()

        dataset['errors_nitems'][n_items_i] = dataset['error_angle'][ids_filtered]
        dataset['errors_all_nitems'][n_items_i
        ] = dataset['errors_angle_all'][ids_filtered]


    ### Fit the mixture model
    if parameters['fit_mixture_model']:

        dataset['em_fits'] = dict(kappa=np.empty(dataset['probe_angle'].size), mixt_target=np.empty(dataset['probe_angle'].size), mixt_nontarget=np.empty(dataset['probe_angle'].size), mixt_random=np.empty(dataset['probe_angle'].size), resp_target=np.empty(dataset['probe_angle'].size), resp_nontarget=np.empty(dataset['probe_angle'].size), resp_random=np.empty(dataset['probe_angle'].size), train_LL=np.empty(dataset['probe_angle'].size), test_LL=np.empty(dataset['probe_angle'].size))
        for key in dataset['em_fits']:
            dataset['em_fits'][key].fill(np.nan)

        # Angles trials
        for n_items in np.unique(dataset['n_items']):
            ids_n_items = (dataset['n_items'] == n_items).flatten()
            ids_filtered = dataset['angle_trials'] & ids_n_items

            dataset['target'][ids_filtered] = dataset['item_angle'][ids_filtered, 0]
            dataset['response'][ids_filtered] = dataset['probe_angle'][ids_filtered]

            # params_fit = em_circularmixture.fit(dataset['probe_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:])
            print dataset['probe_angle'][ids_filtered, 0].shape, dataset['item_angle'][ids_filtered, 0].shape, dataset['item_angle'][ids_filtered, 1:].shape

            cross_valid_outputs = em_circularmixture.cross_validation_kfold(dataset['probe_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:], K=10, shuffle=True, debug=False)
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
        for n_items in np.unique(dataset['n_items']):
            ids_n_items = (dataset['n_items'] == n_items).flatten()
            ids_filtered = dataset['colour_trials'] & ids_n_items

            dataset['target'][ids_filtered] = dataset['item_colour'][ids_filtered, 0]
            dataset['response'][ids_filtered] = dataset['probe_colour'][ids_filtered]

            # params_fit = em_circularmixture.fit(dataset['probe_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 1:])
            cross_valid_outputs = em_circularmixture.cross_validation_kfold(dataset['probe_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 1:], K=10, shuffle=True, debug=False)
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

    ## Save item in a nice format for the model fit
    dataset['data_to_fit'] = {}
    dataset['data_to_fit']['n_items'] = np.unique(dataset['n_items'])
    for n_items in dataset['data_to_fit']['n_items']:
        ids_n_items = (dataset['n_items'] == n_items).flatten()
        ids_filtered = dataset['angle_trials'] & ids_n_items

        if n_items not in dataset['data_to_fit']:
            dataset['data_to_fit'][n_items] = {}
            dataset['data_to_fit'][n_items]['N'] = np.sum(ids_filtered)
            dataset['data_to_fit'][n_items]['probe'] = np.unique(dataset['probe'][ids_filtered])
            dataset['data_to_fit'][n_items]['item_features'] = np.empty((dataset['data_to_fit'][n_items]['N'], n_items, 2))
            dataset['data_to_fit'][n_items]['response'] = np.empty((dataset['data_to_fit'][n_items]['N'], 1))

        dataset['data_to_fit'][n_items]['item_features'][..., 0] = dataset['item_angle'][ids_filtered, :n_items]
        dataset['data_to_fit'][n_items]['item_features'][..., 1] = dataset['item_colour'][ids_filtered, :n_items]
        dataset['data_to_fit'][n_items]['response'] = dataset['probe_angle'][ids_filtered].flatten()

    # Try with Pandas for some advanced plotting
    dataset_filtered = dict((k, dataset[k].flatten()) for k in ('n_items', 'trial', 'subject', 'reject', 'rating', 'probe_colour', 'probe_angle', 'cond', 'error', 'error_angle', 'error_colour', 'response', 'target'))

    if parameters['fit_mixture_model']:
        dataset_filtered .update(dataset['em_fits'])

    dataset['panda'] = pd.DataFrame(dataset_filtered)


def preprocess_bays09(dataset, parameters):
    '''
        The Bays2009 dataset is completely different...
        Some preprocessing is already done, so just do the plots we care about
    '''

    # Extract parameters
    # params_lists: List( (param_name, default_value) )
    params_list = []
    for curr_param in params_list:
        if curr_param[0] in parameters:
            # Set to provided value
            exec(curr_param[0] +" = parameters['" + curr_param[0] + "']")
        else:
            # Default value
            exec(curr_param[0] +" = " + str(curr_param[1]))

    # Make some aliases
    dataset['n_items'] = dataset['N'].astype(int)
    dataset['n_items_size'] = np.unique(dataset['n_items']).size
    dataset['subject'] = dataset['subject'].astype(int)
    dataset['subject_size'] = np.unique(dataset['subject']).size
    dataset['error'] = dataset['E']
    dataset['response'] = dataset['Y']
    dataset['item_angle'] = dataset['X']
    dataset['item_colour'] = dataset['A'] - np.pi
    dataset['probe'] = np.zeros(dataset['error'].shape, dtype= int)
    dataset['errors_nitems'] = np.empty(dataset['n_items_size'], dtype=np.object)
    dataset['errors_nontarget_nitems'] = np.empty(dataset['n_items_size'], dtype=np.object)
    dataset['errors_subject_nitems'] = np.empty((dataset['subject_size'], dataset['n_items_size']), dtype=np.object)
    dataset['errors_nontarget_subject_nitems'] = np.empty((dataset['subject_size'], dataset['n_items_size']), dtype=np.object)
    dataset['vtest_nitems'] = np.empty(dataset['n_items_size'])*np.nan
    dataset['precision_subject_nitems_bays'] = np.nan*np.empty((dataset['subject_size'], dataset['n_items_size']))
    dataset['precision_subject_nitems_theo'] = np.nan*np.empty((dataset['subject_size'], dataset['n_items_size']))
    dataset['precision_subject_nitems_theo_nochance'] = np.nan*np.empty((dataset['subject_size'], dataset['n_items_size']))
    dataset['precision_subject_nitems_bays_notreatment'] = np.nan*np.empty((dataset['subject_size'], dataset['n_items_size']))



    # Fit mixture model
    if parameters.get('fit_mixture_model', False):
        if 'mixture_model_cache' in parameters:
            mixture_model_cache_filename = os.path.join(parameters.get('datadir', ''), parameters['mixture_model_cache'])
            fit_mixture_model(dataset, caching_save_filename=mixture_model_cache_filename)
        else:
            fit_mixture_model(dataset)

    # Compute errors and Vtests
    for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
        for subject_i, subject in enumerate(np.unique(dataset['subject'])):
            # Data per subject
            ids_filtered = (dataset['subject']==subject).flatten() & (dataset['n_items'] == n_items).flatten()

            dataset['errors_subject_nitems'][subject_i, n_items_i] = dataset['error'][ids_filtered, 0]
            dataset['errors_nontarget_subject_nitems'][subject_i, n_items_i] = dataset['error'][ids_filtered, 1:n_items]

            # Precisions
            # Compute the precision
            dataset['precision_subject_nitems_bays'][subject_i, n_items_i] = compute_precision(dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=True, correct_orientation=False, use_wrong_precision=True)
            dataset['precision_subject_nitems_theo'][subject_i, n_items_i] = compute_precision(dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=False, correct_orientation=False, use_wrong_precision=False)
            dataset['precision_subject_nitems_theo_nochance'][subject_i, n_items_i] = compute_precision(dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=True, correct_orientation=False, use_wrong_precision=False)
            dataset['precision_subject_nitems_bays_notreatment'][subject_i, n_items_i] = compute_precision(dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=False, correct_orientation=False, use_wrong_precision=True)

        # Data collapsed accross subjects
        ids_filtered = (dataset['n_items'] == n_items).flatten()

        dataset['errors_nitems'][n_items_i] = dataset['error'][ids_filtered, 0]
        dataset['errors_nontarget_nitems'][n_items_i] = dataset['error'][ids_filtered, 1:n_items]

        if n_items > 1:
            dataset['vtest_nitems'][n_items_i] = utils.V_test(utils.dropnan(dataset['errors_nontarget_nitems'][n_items_i]).flatten())['pvalue']

    # Save item in a nice format for the model fit
    dataset['data_to_fit'] = {}
    dataset['data_to_fit']['n_items'] = np.unique(dataset['n_items'])
    for n_items in dataset['data_to_fit']['n_items']:
        ids_n_items = (dataset['n_items'] == n_items).flatten()

        if n_items not in dataset['data_to_fit']:
            dataset['data_to_fit'][n_items] = {}
            dataset['data_to_fit'][n_items]['N'] = np.sum(ids_n_items)
            dataset['data_to_fit'][n_items]['probe'] = np.unique(dataset['probe'][ids_n_items])
            dataset['data_to_fit'][n_items]['item_features'] = np.empty((dataset['data_to_fit'][n_items]['N'], n_items, 2))
            dataset['data_to_fit'][n_items]['response'] = np.empty((dataset['data_to_fit'][n_items]['N'], 1))

        dataset['data_to_fit'][n_items]['item_features'][..., 0] = dataset['item_angle'][ids_n_items, :n_items]
        dataset['data_to_fit'][n_items]['item_features'][..., 1] = dataset['item_colour'][ids_n_items, :n_items]
        dataset['data_to_fit'][n_items]['response'] = dataset['response'][ids_n_items].flatten()


    # Perform Bootstrap analysis if required
    if parameters.get('should_compute_bootstrap', False):
        try:
            bootstrap_cache_filename = os.path.join(parameters.get('datadir', ''), parameters.get('bootstrap_cache', None))
        except AttributeError:
            # Raised if bootstrap_cache is None
            bootstrap_cache_filename = None

        compute_bootstrap(dataset, caching_save_filename=bootstrap_cache_filename)

    # Do per subject and nitems, get average histogram
    compute_average_histograms(dataset)


######

def load_dataset(filename='', preprocess=lambda x, p: x, parameters={}, name=''):
    '''
        Load datasets.
        Supports
            - Simultaneous and sequential Gorgoraptis_2011 data: Exp1 and Exp2.
            - Dual recall data DualRecall_Bays.
    '''
    # Add datadir
    filename = os.path.join(parameters.get('datadir', ''), filename)

    # Load everything
    dataset = sio.loadmat(filename, mat_dtype=True)

    # Set its name
    dataset['name'] = name

    # Specific operations, for different types of datasets
    preprocess(dataset, parameters)


    return dataset


def load_multiple_datasets(datasets_descr = []):
    '''
        Takes a list of datasets descriptors, and loads them up.
    '''

    datasets = []
    for datasets_descr in datasets_descr:
        datasets.append( load_dataset(filename=datasets_descr['filename'], preprocess=datasets_descr['preprocess'], parameters=datasets_descr['parameters'], name=datasets_descr['name']))

    return datasets


def reconstruct_colours_exp1(dataset, datadir='', datasets=('Data/ad.mat', 'Data/gb.mat', 'Data/kf.mat', 'Data/md.mat', 'Data/sf.mat', 'Data/sw.mat', 'Data/wd.mat', 'Data/zb.mat')):
    '''
        The colour is missing from the simultaneous experiment dataset
        Reconstruct it.
    '''

    all_colours = []
    all_preangles = []
    all_targets = []
    for dataset_fn in datasets:
        dataset_fn = os.path.join(datadir, dataset_fn)
        print dataset_fn
        curr_data = sio.loadmat(dataset_fn, mat_dtype=True)

        all_colours.append(curr_data['item_colour'])
        all_preangles.append(utils.wrap_angles(curr_data['probe_pre_angle'], bound=np.pi))
        all_targets.append(utils.wrap_angles(np.deg2rad(curr_data['item_angle'][:, 0]), bound=np.pi))

    print "Data loaded"

    all_colours = np.array(all_colours)
    all_targets = np.array(all_targets)
    all_preangles = np.array(all_preangles)

    # Ordering in original data
    order_subjects = [0, 2, 4, 6, 1, 3, 5, 7]

    all_colours = all_colours[order_subjects]
    all_targets = all_targets[order_subjects]
    all_preangles = all_preangles[order_subjects]

    # Convert colour ids into angles.
    # Assume uniform coverage over circle...
    nb_colours = np.unique(all_colours[0][:, 0]).size

    # Make it so 0 is np.nan, and then 1...8 are possible colour angles
    colours_possible = np.r_[np.nan, np.linspace(-np.pi, np.pi, nb_colours, endpoint=False)]

    size_colour_arr = np.sum([col.shape[0] for col in all_colours])
    item_colour = np.empty((size_colour_arr, all_colours[0].shape[1]))
    start_i = 0
    for i in xrange(all_colours.size):
        # Get the indices. 0 will be np.nan, 1 .. nb_colours will work directly.
        colours_indices = np.ma.masked_invalid(all_colours[i]).filled(fill_value = 0.0).astype(int)

        # Get the colours!
        # Indexing is annoying, as we have different shapes for different subjects
        item_colour[start_i:start_i+colours_indices.shape[0]] = colours_possible[colours_indices]

        start_i += colours_indices.shape[0]

    item_preangle_arr = np.empty((0, all_preangles[0].shape[1]))
    for arr in all_preangles:
        item_preangle_arr = np.r_[item_preangle_arr, arr]

    dataset['item_colour'] = item_colour
    dataset['item_preangle'] = item_preangle_arr
    dataset['all_targets'] = all_targets




def compute_all_errors(dataset = {}):
    '''
        Will compute the error between the response and all possible items
    '''

    # Get the difference between angles
    # Should also wrap it around
    dataset['errors_all'] = utils.wrap_angles(dataset['item_angle'] - dataset['response'], bound=np.pi)

    # Sanity check, verify that errors computed are the same as precomputed ones.
    # assert all(np.abs(dataset['errors_all'][np.arange(dataset['probe'].size), dataset['probe'][:, 0]] - dataset['error'][:, 0]) < 10**-6), "Errors computed are different than errors given in the data"


def create_subject_arrays(dataset = {}, double_precision=True   ):
    '''
        Create arrays with errors per subject and per num_target
        also create an array with the precision per subject and num_target directly
    '''

    unique_subjects = np.unique(dataset['subject'])
    unique_n_items = np.unique(dataset['n_items'])

    dataset['errors_subject_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)
    dataset['errors_all_subject_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)
    dataset['errors_nontarget_subject_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)
    precision_subject_nitems = np.nan*np.empty((unique_subjects.size, unique_n_items.size))
    precision_subject_nitems_theo = np.nan*np.empty((unique_subjects.size, unique_n_items.size))
    precision_subject_nitems_nochance = np.nan*np.empty((unique_subjects.size, unique_n_items.size))
    precision_subject_nitems_raw = np.nan*np.empty((unique_subjects.size, unique_n_items.size))

    dataset['response_subject_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)
    dataset['item_angle_subject_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)

    for n_items_i, n_items in enumerate(unique_n_items):
        for subject_i, subject in enumerate(unique_subjects):
            ids_filtered = ((dataset['subject']==subject) & (dataset['n_items'] == n_items)).flatten()

            # Get the errors
            dataset['errors_subject_nitems'][subject_i, n_items_i] = dataset['errors_all'][ids_filtered, 0]
            dataset['errors_all_subject_nitems'][subject_i, n_items_i] = dataset['errors_all'][ids_filtered]
            dataset['errors_nontarget_subject_nitems'][subject_i, n_items_i] = dataset['errors_all'][ids_filtered, 1:]

            # Get the responses and correct item angles
            dataset['response_subject_nitems'][subject_i, n_items_i] = dataset['response'][ids_filtered]
            dataset['item_angle_subject_nitems'][subject_i, n_items_i] = dataset['item_angle'][ids_filtered]

            # Compute the precision
            precision_subject_nitems[subject_i, n_items_i] = compute_precision(dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=True, correct_orientation=True, use_wrong_precision=True)
            precision_subject_nitems_theo[subject_i, n_items_i] = compute_precision(dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=False, correct_orientation=True, use_wrong_precision=False)
            precision_subject_nitems_nochance[subject_i, n_items_i] = compute_precision(dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=True, correct_orientation=False, use_wrong_precision=False)
            precision_subject_nitems_raw[subject_i, n_items_i] = compute_precision(dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=False, correct_orientation=True, use_wrong_precision=True)

    # if double_precision:
    #     precision_subject_nitems *= 2.
    #     precision_subject_nitems_theo *= 2.
    #     # precision_subject_nitems_nochance *= 2.
    #     precision_subject_nitems_raw *= 2.

    dataset['precision_subject_nitems_bays'] = precision_subject_nitems
    dataset['precision_subject_nitems_theo'] = precision_subject_nitems_theo
    dataset['precision_subject_nitems_theo_nochance'] = precision_subject_nitems_nochance
    dataset['precision_subject_nitems_bays_notreatment'] = precision_subject_nitems_raw

    dataset['errors_nitems'] = np.array([utils.flatten_list(dataset['errors_subject_nitems'][:, n_item_i]) for n_item_i in xrange(unique_n_items.size)])
    dataset['errors_all_nitems'] = np.array([utils.flatten_list(dataset['errors_all_subject_nitems'][:, n_item_i]) for n_item_i in xrange(unique_n_items.size)])
    dataset['errors_nontarget_nitems'] = dataset['errors_all_nitems'][:, :, 1:]
    dataset['precision_nitems_bays'] = np.mean(precision_subject_nitems, axis=0)
    dataset['precision_nitems_theo'] = np.mean(precision_subject_nitems_theo, axis=0)
    dataset['precision_nitems_theo_nochance'] = np.mean(precision_subject_nitems_nochance, axis=0)
    dataset['precision_nitems_bays_notreatment'] = np.mean(precision_subject_nitems_raw, axis=0)


def compute_precision(errors, remove_chance_level=True, correct_orientation=True, use_wrong_precision=True):
    '''
        Compute the precision (1./circ_std**2). Remove the chance level if desired.
    '''

    # if correct_orientation:
    #     # Correct for the fact that bars are modelled in [0, pi] and not [0, 2pi]
    #     errors = errors.copy()*2.0

    # Circular standard deviation estimate
    error_std_dev_error = utils.angle_circular_std_dev(errors)

    # Precision
    if use_wrong_precision:
        precision = 1./error_std_dev_error
    else:
        precision = 1./error_std_dev_error**2.

    if remove_chance_level:
        # Remove the chance level
        precision -= utils.compute_precision_chance(errors.size)

    if correct_orientation:
        # The obtained precision is for half angles, correct it
        precision *= 2.

    return precision


def fit_mixture_model(dataset, caching_save_filename=None):
    '''
        Fit the mixture model onto classical responses/item_angle values

        If caching_save_filename is not None:
        - Will try to open the file provided and use 'em_fits', 'em_fits_subjects_nitems' and 'em_fits_nitems' instead of computing them.
        - If file does not exist, compute and save it.
    '''

    should_fit_model = True
    save_caching_file = False

    if caching_save_filename is not None:

        if os.path.exists(caching_save_filename):
            # Got file, open it and try to use its contents
            try:
                with open(caching_save_filename, 'r') as file_in:
                    # Load and assign values
                    cached_data = pickle.load(file_in)
                    dataset.update(cached_data)
                    should_fit_model = False
                    print "reloaded mixture model from cache", caching_save_filename

            except IOError:
                print "Error while loading ", caching_save_filename, "falling back to computing the EM fits"
        else:
            # No file, create it after everything is computed
            save_caching_file = True


    if should_fit_model:

        # Initalisize empty arrays
        dataset['em_fits'] = dict(kappa=np.empty(dataset['probe'].size), mixt_target=np.empty(dataset['probe'].size), mixt_nontarget=np.empty(dataset['probe'].size), mixt_random=np.empty(dataset['probe'].size), resp_target=np.empty(dataset['probe'].size), resp_nontarget=np.empty(dataset['probe'].size), resp_random=np.empty(dataset['probe'].size), train_LL=np.empty(dataset['probe'].size), test_LL=np.empty(dataset['probe'].size))
        for key in dataset['em_fits']:
            dataset['em_fits'][key].fill(np.nan)
        dataset['target'] = np.empty(dataset['probe'].size)
        dataset['em_fits_subjects_nitems'] = dict()
        for subject in np.unique(dataset['subject']):
            dataset['em_fits_subjects_nitems'][subject] = dict()

        dataset['em_fits_nitems'] = dict(mean=dict(), std=dict(), values=dict())

        # Compute mixture model fits per n_items and per subject
        for n_items in np.unique(dataset['n_items']):
            for subject in np.unique(dataset['subject']):
                ids_filtered = (dataset['subject']==subject).flatten() & (dataset['n_items'] == n_items).flatten()
                print "Fit mixture model, %d items, subject %d, %d datapoints" % (subject, n_items, np.sum(ids_filtered))

                dataset['target'][ids_filtered] = dataset['item_angle'][ids_filtered, 0]

                # params_fit = em_circularmixture.fit(dataset['response'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:])
                params_fit = em_circularmixture_allitems_uniquekappa.fit(dataset['response'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:])
                params_fit['mixt_nontargets_sum'] = np.sum(params_fit['mixt_nontargets'])
                # print dataset['response'][ids_filtered, 0].shape, dataset['item_angle'][ids_filtered, 0].shape, dataset['item_angle'][ids_filtered, 1:].shape

                # cross_valid_outputs = em_circularmixture.cross_validation_kfold(dataset['response'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:], K=10, shuffle=True, debug=False)
                # params_fit = cross_valid_outputs['best_fit']
                # resp = em_circularmixture.compute_responsibilities(dataset['response'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:], params_fit)
                resp = em_circularmixture_allitems_uniquekappa.compute_responsibilities(dataset['response'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:], params_fit)

                dataset['em_fits']['kappa'][ids_filtered] = params_fit['kappa']
                dataset['em_fits']['mixt_target'][ids_filtered] = params_fit['mixt_target']
                # dataset['em_fits']['mixt_nontarget'][ids_filtered] = params_fit['mixt_nontargets']
                dataset['em_fits']['mixt_nontarget'][ids_filtered] = params_fit['mixt_nontargets_sum']
                dataset['em_fits']['mixt_random'][ids_filtered] = params_fit['mixt_random']
                dataset['em_fits']['resp_target'][ids_filtered] = resp['target']
                dataset['em_fits']['resp_nontarget'][ids_filtered] = np.sum(resp['nontargets'], axis=1)
                dataset['em_fits']['resp_random'][ids_filtered] = resp['random']
                dataset['em_fits']['train_LL'][ids_filtered] = params_fit['train_LL']
                # dataset['em_fits']['test_LL'][ids_filtered] = cross_valid_outputs['best_test_LL']

                dataset['em_fits_subjects_nitems'][subject][n_items] = params_fit


            ## Now compute mean/std em_fits per n_items
            dataset['em_fits_nitems']['mean'][n_items] = dict()
            dataset['em_fits_nitems']['std'][n_items] = dict()
            dataset['em_fits_nitems']['values'][n_items] = dict()

            # Need to extract the values for a subject/nitems pair, for all keys of em_fits. Annoying dictionary indexing needed
            emfits_keys = params_fit.keys()
            for key in emfits_keys:
                values_allsubjects = [dataset['em_fits_subjects_nitems'][subject][n_items][key] for subject in np.unique(dataset['subject'])]

                dataset['em_fits_nitems']['mean'][n_items][key] = np.mean(values_allsubjects)
                dataset['em_fits_nitems']['std'][n_items][key] = np.std(values_allsubjects)
                dataset['em_fits_nitems']['values'][n_items][key] = values_allsubjects


    ## Construct array versions of the em_fits_nitems mixture proportions, for convenience
    if 'em_fits_nitems_arrays' not in dataset:
        dataset['em_fits_nitems_arrays'] = dict()

        # Check if mixt_nontargets is array or not
        if 'mixt_nontargets_sum' in dataset['em_fits_nitems']['mean'].values()[0]:
            dataset['em_fits_nitems_arrays']['mean'] = np.array([[dataset['em_fits_nitems']['mean'][item][em_key] for item in np.unique(dataset['n_items'])] for em_key in ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random']])
            dataset['em_fits_nitems_arrays']['std'] = np.array([[dataset['em_fits_nitems']['std'][item][em_key] for item in np.unique(dataset['n_items'])] for em_key in ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random']])
        else:
            dataset['em_fits_nitems_arrays']['mean'] = np.array([[dataset['em_fits_nitems']['mean'][item][em_key] for item in np.unique(dataset['n_items'])] for em_key in ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random']])
            dataset['em_fits_nitems_arrays']['std'] = np.array([[dataset['em_fits_nitems']['std'][item][em_key] for item in np.unique(dataset['n_items'])] for em_key in ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random']])

    if 'sem' not in dataset['em_fits_nitems_arrays']:
        dataset['em_fits_nitems_arrays']['sem'] = dataset['em_fits_nitems_arrays']['std']/np.sqrt(dataset['subject_size'])

    if save_caching_file:
        try:
            with open(caching_save_filename, 'w') as filecache_out:
                data_em = dict((key, dataset[key]) for key in ['em_fits', 'em_fits_nitems', 'em_fits_subjects_nitems', 'em_fits_nitems_arrays'])

                pickle.dump(data_em, filecache_out, protocol=2)

        except IOError:
            print "Error writing out to caching file ", caching_save_filename


def compute_bootstrap(dataset, caching_save_filename=None, nb_bootstrap_samples=1000):
    '''
        Compute bootstrap estimates per subject/nitems.

        If caching_save_filename is not None:
        - Will try to open the file provided and use 'bootstrap_subject_nitems', 'bootstrap_nitems' and 'bootstrap_nitems_pval' instead of computing them.
        - If file does not exist, compute and save it.
    '''

    should_compute_bootstrap = True
    save_caching_file = False

    if caching_save_filename is not None:

        if os.path.exists(caching_save_filename):
            # Got file, open it and try to use its contents
            try:
                with open(caching_save_filename, 'r') as file_in:
                    # Load and assign values
                    cached_data = pickle.load(file_in)
                    dataset.update(cached_data)
                    should_compute_bootstrap = False

            except IOError:
                print "Error while loading ", caching_save_filename, "falling back to computing the EM fits"
        else:
            # No file, create it after everything is computed
            save_caching_file = True


    if should_compute_bootstrap:

        print "Computing bootstrap..."

        dataset['bootstrap_nitems_pval'] = np.nan*np.empty(dataset['n_items_size'])
        dataset['bootstrap_nitems'] = np.empty(dataset['n_items_size'], dtype=np.object)
        dataset['bootstrap_subject_nitems'] = np.empty((dataset['subject_size'], dataset['n_items_size']), dtype=np.object)
        dataset['bootstrap_subject_nitems_pval'] = np.nan*np.empty((dataset['subject_size'], dataset['n_items_size']))


        for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
            if n_items > 1:
                for subject_i, subject in enumerate(np.unique(dataset['subject'])):
                    print "Nitems %d, subject %d" % (n_items, subject)

                    # Bootstrap per subject and nitems
                    ids_filtered = (dataset['subject']==subject).flatten() & (dataset['n_items'] == n_items).flatten()

                    # Compute bootstrap if required

                    bootstrap = em_circularmixture_allitems_uniquekappa.bootstrap_nontarget_stat(dataset['response'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:n_items], nb_bootstrap_samples=nb_bootstrap_samples)
                    dataset['bootstrap_subject_nitems'][subject_i, n_items_i] = bootstrap
                    dataset['bootstrap_subject_nitems_pval'][subject_i, n_items_i] = bootstrap['p_value']

                    print dataset['bootstrap_subject_nitems_pval'][:, n_items_i]

                print "Nitems %d, all subjects" % (n_items)

                # Data collapsed accross subjects
                ids_filtered = (dataset['n_items'] == n_items).flatten()

                bootstrap = em_circularmixture_allitems_uniquekappa.bootstrap_nontarget_stat(dataset['response'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:n_items], nb_bootstrap_samples=nb_bootstrap_samples)
                dataset['bootstrap_nitems'][n_items_i] = bootstrap
                dataset['bootstrap_nitems_pval'][n_items_i] = bootstrap['p_value']

                print dataset['bootstrap_nitems_pval']

    if save_caching_file:
        try:
            with open(caching_save_filename, 'w') as filecache_out:
                cached_data = dict((key, dataset[key]) for key in ['bootstrap_subject_nitems', 'bootstrap_nitems', 'bootstrap_nitems_pval', 'bootstrap_subject_nitems_pval'])

                pickle.dump(cached_data, filecache_out, protocol=2)

        except IOError:
            print "Error writing out to caching file ", caching_save_filename


def compute_average_histograms(dataset):
    '''
        Do per subject and nitems, get average histogram
    '''

    angle_space = np.linspace(-np.pi, np.pi, 51)

    dataset['hist_cnts_target_subject_nitems'] = np.empty((dataset['subject_size'], dataset['n_items_size'], angle_space.size - 1))*np.nan
    dataset['hist_cnts_nontarget_subject_nitems'] = np.empty((dataset['subject_size'], dataset['n_items_size'], angle_space.size - 1))*np.nan
    dataset['pvalue_nontarget_subject_nitems'] = np.empty((dataset['subject_size'], dataset['n_items_size']))*np.nan

    for subject_i, subject in enumerate(np.unique(dataset['subject'])):
        for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
            dataset['hist_cnts_target_subject_nitems'][subject_i, n_items_i], x, bins = utils.histogram_binspace(utils.dropnan(dataset['errors_subject_nitems'][subject_i, n_items_i]), bins=angle_space, norm='density')

            dataset['hist_cnts_nontarget_subject_nitems'][subject_i, n_items_i], x, bins = utils.histogram_binspace(utils.dropnan(dataset['errors_nontarget_subject_nitems'][subject_i, n_items_i]), bins=angle_space, norm='density')

            if n_items > 1:
                dataset['pvalue_nontarget_subject_nitems'][subject_i, n_items_i] = utils.V_test(utils.dropnan(dataset['errors_nontarget_subject_nitems'][subject_i, n_items_i]).flatten())['pvalue']

    dataset['hist_cnts_target_nitems_stats'] = dict(mean=np.mean(dataset['hist_cnts_target_subject_nitems'], axis=0), std=np.std(dataset['hist_cnts_target_subject_nitems'], axis=0), sem=np.std(dataset['hist_cnts_target_subject_nitems'], axis=0)/np.sqrt(dataset['subject_size']))

    dataset['hist_cnts_nontarget_nitems_stats'] = dict(mean=np.mean(dataset['hist_cnts_nontarget_subject_nitems'], axis=0), std=np.std(dataset['hist_cnts_nontarget_subject_nitems'], axis=0), sem=np.std(dataset['hist_cnts_nontarget_subject_nitems'], axis=0)/np.sqrt(dataset['subject_size']))

######

def plots_check_bias_nontarget(dataset, dataio=None):
    '''
        Get an histogram of the errors between the response and all non targets
            If biased towards 0-values, should indicate misbinding errors.

        (if you do this with respect to all targets, it's retarded and should always be biased)
    '''
    n_items_space = np.unique(dataset['n_items'])
    angle_space = np.linspace(-np.pi, np.pi, 20)

    # Get histograms of errors, per n_item
    for nitems_i in xrange(n_items_space.size):
        utils.hist_samples_density_estimation(dataset['errors_nitems'][nitems_i], bins=angle_space, title='%s N=%d' % (dataset['name'], n_items_space[nitems_i]), dataio=dataio, filename='hist_bias_targets_%ditems_{label}_{unique_id}.pdf' % (n_items_space[nitems_i]))

    # Get histograms of bias to nontargets. Do that by binning the errors to others nontargets of the array.
    utils.plot_hists_bias_nontargets(dataset['errors_all_nitems'][n_items_space>1], bins=20, dataio=dataio, label='allnontargets', remove_first_column=True)

    rayleigh_test = utils.rayleigh_test(dataset['errors_all_nitems'][n_items_space>1].flatten())
    v_test = utils.V_test(dataset['errors_all_nitems'][n_items_space>1].flatten())
    print rayleigh_test
    print v_test



def plots_check_bias_bestnontarget(dataset, dataio=None):
    '''
        Get an histogram of errors between response and best nontarget.
        Should be more biased towards 0 than the overall average
    '''
    n_items_space = np.unique(dataset['n_items'])

    # Compute the errors to the best non target
    errors_nontargets = dataset['errors_all_nitems'][n_items_space>1]
    errors_nontargets = np.array([errors_nontargets_nitem[~np.all(np.isnan(errors_nontargets_nitem), axis=1), :] for errors_nontargets_nitem in errors_nontargets])

    indices_bestnontarget = [np.nanargmin(np.abs(errors_nontargets[n_item_i][..., 1:]), axis=-1) for n_item_i in xrange(errors_nontargets.shape[0])]
    # indices_bestnontarget = np.nanargmin(np.abs(errors_nontargets), axis=2)

    # Index of the argmin of absolute error. Not too bad, easy to index into.
    errors_bestnontargets_nitems = np.array([ errors_nontargets[n_items_i][ xrange(errors_nontargets[n_items_i].shape[0]), indices_bestnontarget[n_items_i] + 1]   for n_items_i in xrange(errors_nontargets.shape[0]) ])

    # Show histograms per n_items, like in Bays2009 figure
    utils.plot_hists_bias_nontargets(errors_bestnontargets_nitems, bins=20, label='bestnontarget', dataio=dataio)



def plots_check_bias_nontarget_randomized(dataset, dataio=None):
    '''
        Plot the histogram of errors to nontargets, after replacing all nontargets by random angles.
        If show similar bias, would be indication of low predictive power of distribution of errors to nontargets.
    '''

    n_items_space = np.unique(dataset['n_items'])

    # Copy item_angles
    new_item_angles = dataset['item_angle'].copy()

    # Will resample multiple times
    errors_nitems_new_dict = dict()
    nb_resampling = 100

    for resampling_i in xrange(nb_resampling):

        # Replace nontargets randomly
        nontarget_indices = np.nonzero(~np.isnan(new_item_angles[:, 1:]))
        new_item_angles[nontarget_indices[0], nontarget_indices[1]+1] = 2*np.pi*np.random.random(nontarget_indices[0].size) - np.pi

        # Compute errors
        new_all_errors = utils.wrap_angles(new_item_angles - dataset['response'], bound=np.pi)

        for n_items in n_items_space:
            ids_filtered = (dataset['n_items'] == n_items).flatten()

            if n_items in errors_nitems_new_dict:
                errors_nitems_new_dict[n_items] = np.r_[errors_nitems_new_dict[n_items], new_all_errors[ids_filtered]]
            else:
                errors_nitems_new_dict[n_items] = new_all_errors[ids_filtered]

    errors_nitems_new = np.array([val for key, val in errors_nitems_new_dict.items()])

    utils.plot_hists_bias_nontargets(errors_nitems_new[n_items_space>1], bins=20, label='allnontarget_randomized_%dresamplings' % nb_resampling, dataio=dataio, remove_first_column=True)

    ### Do same for best non targets
    # TODO Convert this for data_dualrecall
    errors_nontargets = errors_nitems_new[1:, :, 1:]
    indices_bestnontarget = np.nanargmin(np.abs(errors_nontargets), axis=2)

    # Index of the argmin of absolute error. Not too bad, easy to index into.
    errors_bestnontargets_nitems = np.array([ errors_nontargets[n_items_i, xrange(errors_nontargets.shape[1]), indices_bestnontarget[n_items_i]]   for n_items_i in xrange(errors_nontargets.shape[0]) ])

    # Show histograms
    utils.plot_hists_bias_nontargets(errors_bestnontargets_nitems, bins=20, label='bestnontarget_randomized_%dresamplings' % nb_resampling, dataio=dataio)




def plots_check_oblique_effect(data, nb_bins=100):
    '''
        Humans are more precise for vertical and horizontal bars than diagonal orientations.

        Check if present.
    '''

    # Construct the list of (target angles, errors), see if there is some structure in that
    errors_per_angle = np.array(zip(data['item_angle'][np.arange(data['probe'].size), data['probe'][:, 0]], data['error'][:, 0]))

    # response_per_angle = np.array(zip(data['item_angle'][np.arange(data['probe'].size), data['probe'][:, 0]], data['response']))
    # response_per_colour = np.array(zip(data['item_colour'][np.arange(data['probe'].size), data['probe'][:, 0]], data['response']))

    plt.figure()
    plt.plot(errors_per_angle[:, 0], errors_per_angle[:, 1], 'x')

    plt.figure()
    plt.plot(errors_per_angle[:, 0], np.abs(errors_per_angle[:, 1]), 'x')

    discrete_x = np.linspace(-np.pi/2., np.pi/2., nb_bins)
    avg_error = np.zeros(discrete_x.shape)
    std_error = np.zeros(discrete_x.shape)

    for x_i in np.arange(discrete_x.size):
        if x_i < discrete_x.size - 1:
            # Check what data comes in the current interval x[x_i, x_i+1]
            avg_error[x_i] = utils.mean_angles(errors_per_angle[np.logical_and(errors_per_angle[:, 0] > discrete_x[x_i], errors_per_angle[:, 0] < discrete_x[x_i+1]), 1])
            std_error[x_i] = utils.angle_circular_std_dev(errors_per_angle[np.logical_and(errors_per_angle[:, 0] > discrete_x[x_i], errors_per_angle[:, 0] < discrete_x[x_i+1]), 1])

    plt.figure()
    plt.plot(discrete_x, avg_error)

    plt.figure()
    plt.plot(discrete_x, avg_error**2.)

    plt.figure()
    plt.plot(discrete_x, np.abs(avg_error))

    plt.figure()
    plt.plot(errors_per_angle[:, 0], errors_per_angle[:, 1], 'x')
    plt.plot(discrete_x, avg_error, 'ro')


def plots_histograms_errors_targets_nontargets_nitems(dataset, dataio=None):
    '''
        Create subplots showing histograms of errors to targets and nontargets

        Adds Vtest texts on the nontargets
    '''

    angle_space = np.linspace(-np.pi, np.pi, 51)
    bins_center = angle_space[:-1] + np.diff(angle_space)[0]/2

    # Histogram, collapsing across subjects
    f1, axes1 = plt.subplots(ncols=dataset['n_items_size'], figsize=(dataset['n_items_size']*6, 6), sharey=True)
    f2, axes2 = plt.subplots(ncols=dataset['n_items_size']-1, figsize=((dataset['n_items_size']-1)*6, 6), sharey=True)

    for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
        utils.hist_angular_data(dataset['errors_nitems'][n_items_i], bins=angle_space, title='%s N=%d' % (dataset['name'], n_items), norm='density', ax_handle=axes1[n_items_i], pretty_xticks=True)
        axes1[n_items_i].set_ylim([0., 2.0])

        if n_items > 1:
            utils.hist_angular_data(utils.dropnan(dataset['errors_nontarget_nitems'][n_items_i]), bins=angle_space, title='%s N=%d' % (dataset['name'], n_items), norm='density', ax_handle=axes2[n_items_i-1], pretty_xticks=True)

            axes2[n_items_i-1].text(0.02, 0.96, "Vtest pval: %.4f" % (dataset['vtest_nitems'][n_items_i]), transform=axes2[n_items_i-1].transAxes, horizontalalignment='left', fontsize=13)

            axes2[n_items_i - 1].set_ylim([0., 0.3])



    f1.canvas.draw()
    f2.canvas.draw()

    if dataio is not None:
        plt.figure(f1.number)
        # plt.tight_layout()
        dataio.save_current_figure("hist_error_target_all_{label}_{unique_id}.pdf")
        plt.figure(f2.number)
        # plt.tight_layout()
        dataio.save_current_figure("hist_error_nontarget_all_{label}_{unique_id}.pdf")

    # Do per subject and nitems, using average histogram
    f3, axes3 = plt.subplots(ncols=dataset['n_items_size'], figsize=(dataset['n_items_size']*6, 6), sharey=True)
    f4, axes4 = plt.subplots(ncols=dataset['n_items_size']-1, figsize=((dataset['n_items_size']-1)*6, 6), sharey=True)

    for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):

        axes3[n_items_i].bar(bins_center, dataset['hist_cnts_target_nitems_stats']['mean'][n_items_i], width=2.*np.pi/(angle_space.size-1), align='center', yerr=dataset['hist_cnts_target_nitems_stats']['sem'][n_items_i])
        # axes3[n_items_i].set_title('N=%d' % n_items)
        axes3[n_items_i].set_xlim([bins_center[0]-np.pi/(angle_space.size-1), bins_center[-1]+np.pi/(angle_space.size-1)])
        axes3[n_items_i].set_ylim([0., 2.0])
        axes3[n_items_i].set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        axes3[n_items_i].set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=16)

        if n_items > 1:
            axes4[n_items_i-1].bar(bins_center, dataset['hist_cnts_nontarget_nitems_stats']['mean'][n_items_i], width=2.*np.pi/(angle_space.size-1), align='center', yerr=dataset['hist_cnts_nontarget_nitems_stats']['sem'][n_items_i])
            # axes4[n_items_i-1].set_title('N=%d' % n_items)
            axes4[n_items_i-1].set_xlim([bins_center[0]-np.pi/(angle_space.size-1), bins_center[-1]+np.pi/(angle_space.size-1)])

            # axes4[n_items_i-1].text(0.02, 0.96, "Vtest pval: %.4f" % (pvalue_nontarget_subject_nitems_mean[n_items_i]), transform=axes4[n_items_i-1].transAxes, horizontalalignment='left', fontsize=13)
            axes4[n_items_i-1].text(0.02, 0.96, "Vtest pval: %.4f" % (dataset['vtest_nitems'][n_items_i]), transform=axes4[n_items_i-1].transAxes, horizontalalignment='left', fontsize=14)

            # TODO Add bootstrap there instead.

            axes4[n_items_i-1].set_ylim([0., 0.3])
            axes4[n_items_i-1].set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
            axes4[n_items_i-1].set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=16)

        utils.scatter_marginals(utils.dropnan(dataset['data_to_fit'][n_items]['item_features'][:, 0, 0]), utils.dropnan(dataset['data_to_fit'][n_items]['response']), xlabel ='Target angle', ylabel='Response angle', title='%s histogram responses, %d items' % (dataset['name'], n_items), figsize=(9, 9), factor_axis=1.1, bins=61)
        # utils.scatter_marginals(utils.dropnan(dataset['item_angle'][dataset['angle_trials'] & dataset['3_items_trials'], 0]), utils.dropnan(dataset['probe_angle'][dataset['angle_trials'] & dataset['3_items_trials']]), xlabel ='Target angle', ylabel='Response angle', title='%s Angle trials, 3 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61)
        # utils.scatter_marginals(utils.dropnan(dataset['item_angle'][dataset['angle_trials'] & dataset['6_items_trials'], 0]), utils.dropnan(dataset['probe_angle'][dataset['angle_trials'] & dataset['6_items_trials']]), xlabel ='Target angle', ylabel='Response angle', title='%s Angle trials, 6 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61)

    f3.canvas.draw()
    f4.canvas.draw()

    if dataio is not None:
        plt.figure(f3.number)
        # plt.tight_layout()
        dataio.save_current_figure("hist_error_target_persubj_{label}_{unique_id}.pdf")
        plt.figure(f4.number)
        # plt.tight_layout()
        dataio.save_current_figure("hist_error_nontarget_persubj_{label}_{unique_id}.pdf")


def plots_em_mixtures(dataset, dataio=None, use_sem=True):
    '''
        Do plots for the mixture models and kappa
    '''
    T_space_exp = np.unique(dataset['n_items'])

    f, ax = plt.subplots()

    if use_sem:
        errorbars = 'sem'
    else:
        errorbars = 'std'

    # Mixture probabilities
    utils.plot_mean_std_area(T_space_exp, dataset['em_fits_nitems_arrays']['mean'][1], np.ma.masked_invalid(dataset['em_fits_nitems_arrays'][errorbars][1]).filled(0.0), xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Target')
    utils.plot_mean_std_area(T_space_exp, np.ma.masked_invalid(dataset['em_fits_nitems_arrays']['mean'][2]).filled(0.0), np.ma.masked_invalid(dataset['em_fits_nitems_arrays'][errorbars][2]).filled(0.0), xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Nontarget')
    utils.plot_mean_std_area(T_space_exp, dataset['em_fits_nitems_arrays']['mean'][3], np.ma.masked_invalid(dataset['em_fits_nitems_arrays'][errorbars][3]).filled(0.0), xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Random')

    ax.legend(prop={'size':15})

    ax.set_title('Mixture model for EM fit %s' % dataset['name'])
    ax.set_xlim([1.0, T_space_exp.max()])
    ax.set_ylim([0.0, 1.1])
    ax.set_xticks(range(1, T_space_exp.max()+1))
    ax.set_xticklabels(range(1, T_space_exp.max()+1))

    f.canvas.draw()

    if dataio is not None:
        dataio.save_current_figure('emfits_mixtures_{label}_{unique_id}.pdf')

    # Kappa
    f, ax = plt.subplots()

    ax = utils.plot_mean_std_area(T_space_exp,
    dataset['em_fits_nitems_arrays']['mean'][0], np.ma.masked_invalid(dataset['em_fits_nitems_arrays'][errorbars][0]).filled(0.0), linewidth=3, fmt='o-', markersize=8, ylabel='Experimental data', ax_handle=ax)

    ax.legend(prop={'size':15})
    ax.set_title('Kappa for EM fit %s' % dataset['name'])
    ax.set_xlim([0.9, T_space_exp.max()+0.1])
    ax.set_ylim([0.0, np.max(dataset['em_fits_nitems_arrays']['mean'][0])*1.1])
    ax.set_xticks(range(1, T_space_exp.max()+1))
    ax.set_xticklabels(range(1, T_space_exp.max()+1))
    ax.get_figure().canvas.draw()

    if dataio is not None:
        dataio.save_current_figure('emfits_kappa_{label}_{unique_id}.pdf')

def plots_precision(dataset, dataio=None, use_sem=True):
    '''
        Do plots for the mixture models and kappa
    '''
    T_space_exp = np.unique(dataset['n_items'])

    precisions_to_plot = [['precision_subject_nitems_theo', 'Precision Theo'],['precision_subject_nitems_bays_notreatment', 'Precision BaysNoTreat'],['precision_subject_nitems_bays', 'Precision Bays'],['precision_subject_nitems_theo_nochance', 'Precision TheoNoChance']]

    for precision_to_plot, precision_title in precisions_to_plot:
        f, ax = plt.subplots()

        # Compute the errorbars
        precision_mean = np.mean(dataset[precision_to_plot], axis=0)
        precision_errors = np.std(dataset[precision_to_plot], axis=0)
        if use_sem:
            precision_errors /= np.sqrt(dataset['subject_size'])

        # Now show the precision
        utils.plot_mean_std_area(T_space_exp, precision_mean, precision_errors, xlabel='Number of items', label="Precision", ax_handle=ax, linewidth=3, fmt='o-', markersize=5)

        ax.legend(prop={'size':15})

        ax.set_title('%s %s' % (precision_title, dataset['name']))
        ax.set_xlim([1.0, T_space_exp.max()])
        ax.set_ylim([0.0, np.max(precision_mean)+np.max(precision_errors)])
        ax.set_xticks(range(1, T_space_exp.max()+1))
        ax.set_xticklabels(range(1, T_space_exp.max()+1))

        f.canvas.draw()

        if dataio is not None:
            dataio.save_current_figure('%s_{label}_{unique_id}.pdf' % precision_title)


def plots_bays2009(dataset, dataio=None):
    '''

    Some plots for the Bays2009 data
    '''

    plots_histograms_errors_targets_nontargets_nitems(dataset, dataio)

    plots_precision(dataset, dataio)

    plots_em_mixtures(dataset, dataio)


def plots_gorgo11(dataset, dataio=None):
    '''
        Plots for Gorgo11, assuming sequential data
    '''
    plots_histograms_errors_targets_nontargets_nitems(dataset, dataio)

    plots_precision(dataset, dataio)

    plots_em_mixtures(dataset, dataio)


def plots_dualrecall(dataset):
    '''
        Create plots for the double recall dataset
    '''

    to_plot = {'resp_vs_targ':True, 'error_boxplot':True, 'resp_rating':True, 'em_fits':True, 'loglik':True, 'resp_distrib':True, 'resp_conds':True}

    dataset_pd = dataset['panda']

    dataset_pd['error_abs'] = dataset_pd.error.abs()

    # Show distributions of responses wrt target angles/colour
    if to_plot['resp_vs_targ']:

        # Plot scatter and marginals for the orientation trials
        utils.scatter_marginals(utils.dropnan(dataset['item_angle'][dataset['angle_trials'] & dataset['3_items_trials'], 0]), utils.dropnan(dataset['probe_angle'][dataset['angle_trials'] & dataset['3_items_trials']]), xlabel ='Target angle', ylabel='Response angle', title='%s Angle trials, 3 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61)
        utils.scatter_marginals(utils.dropnan(dataset['item_angle'][dataset['angle_trials'] & dataset['6_items_trials'], 0]), utils.dropnan(dataset['probe_angle'][dataset['angle_trials'] & dataset['6_items_trials']]), xlabel ='Target angle', ylabel='Response angle', title='%s Angle trials, 6 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61)

        # Plot scatter and marginals for the colour trials
        utils.scatter_marginals(utils.dropnan(dataset['item_colour'][dataset['colour_trials']& dataset['3_items_trials'], 0]), utils.dropnan(dataset['probe_colour'][dataset['colour_trials'] & dataset['3_items_trials']]), xlabel ='Target colour', ylabel='Response colour', title='%s Colour trials, 3 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61, show_colours=True)
        utils.scatter_marginals(utils.dropnan(dataset['item_colour'][dataset['colour_trials'] & dataset['6_items_trials'], 0]), utils.dropnan(dataset['probe_colour'][dataset['colour_trials'] & dataset['6_items_trials']]), xlabel ='Target colour', ylabel='Response colour', title='%s Colour trials, 6 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61, show_colours=True)


    if 'em_fits' in dataset:

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
            _, axes = plt.subplots(dataset_pd.rating.nunique(), 3)
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

        # Regroup some data
        dataset_grouped_nona_conditems = dataset_pd.dropna(subset=['error']).groupby(['cond_name', 'n_items'])
        dataset_grouped_nona_conditems_mean = dataset_grouped_nona_conditems.mean()[['mixt_target', 'mixt_nontarget', 'mixt_random', 'kappa', 'train_LL', 'test_LL']]

        # Show inferred mixture proportions and kappa
        if to_plot['em_fits']:
            # ax = dataset_grouped_nona_conditems_mean[['mixt_target', 'mixt_nontarget', 'mixt_random', 'kappa']].plot(secondary_y='kappa', kind='bar')
            ax = dataset_grouped_nona_conditems_mean[['mixt_target', 'mixt_nontarget', 'mixt_random']].plot(kind='bar')
            ax.set_ylabel('Mixture proportions')

            ax = dataset_grouped_nona_conditems_mean[['kappa']].plot(kind='bar')
            ax.set_ylabel('Kappa')

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


def plot_bias_close_feature(dataset, dataio=None):
    '''
        Check if there is a bias in the response towards closest item (either closest wrt cued feature, or wrt all features)
    '''
    number_items_considered = 2

    # Error to nontarget
    bias_to_nontarget = np.abs(dataset['errors_nontarget_nitems'][number_items_considered-1][:, :number_items_considered-1].flatten())
    bias_to_target = np.abs(dataset['errors_nitems'][number_items_considered-1].flatten())
    ratio_biases = bias_to_nontarget/ bias_to_target
    response = dataset['data_to_fit'][number_items_considered]['response']

    target = dataset['data_to_fit'][number_items_considered]['item_features'][:, 0]
    nontarget = dataset['data_to_fit'][number_items_considered]['item_features'][:, 1]

    # Distance between probe and closest nontarget, in full feature space
    dist_target_nontarget_torus = utils.dist_torus(target, nontarget)

    # Distance only looking at recalled feature
    dist_target_nontarget_recalled = np.abs(utils.wrap_angles((target[:, 0] - nontarget[:, 0])))

    # Distance only looking at cued feature.
    # Needs more work. They are only a few possible values, so we can group them and get a boxplot for each
    dist_target_nontarget_cue = np.round(np.abs(utils.wrap_angles((target[:, 1] - nontarget[:, 1]))), decimals=8)
    dist_distinct_values = np.unique(dist_target_nontarget_cue)
    bias_to_nontarget_grouped_dist_cue = []
    for dist_value in dist_distinct_values:
        bias_to_nontarget_grouped_dist_cue.append(bias_to_nontarget[dist_target_nontarget_cue == dist_value])

    # Check if the response is closer to the target or nontarget, in relative terms.
    # Need to compute a ratio linking bias_to_target and bias_to_nontarget.
    # Two possibilities: response was between target and nontarget, or response was "behind" the target.
    ratio_response_close_to_nontarget = bias_to_nontarget/dist_target_nontarget_recalled
    indices_filter_other_side = bias_to_nontarget > dist_target_nontarget_recalled
    ratio_response_close_to_nontarget[indices_filter_other_side] = bias_to_nontarget[indices_filter_other_side]/(dist_target_nontarget_recalled[indices_filter_other_side] + bias_to_target[indices_filter_other_side])

    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(dist_target_nontarget_torus, bias_to_nontarget, 'x')
    ax[0, 0].set_xlabel('Distance full feature space')
    ax[0, 0].set_ylabel('Error to nontarget')

    ax[0, 1].boxplot(bias_to_nontarget_grouped_dist_cue, positions=dist_distinct_values)
    ax[0, 1].set_ylabel('Error to nontarget')
    ax[0, 1].set_xlabel('Distance cued feature only')

    # ax[1, 0].plot(dist_target_nontarget_recalled, np.ma.masked_greater(ratio_biases, 100), 'x')
    ax[1, 0].plot(dist_target_nontarget_recalled, bias_to_nontarget, 'x')
    # ax[1, 0].plot(dist_target_nontarget_recalled, np.ma.masked_greater(bias_to_nontarget/dist_target_nontarget_recalled, 30), 'x')
    ax[1, 0].set_xlabel('Distance recalled feature only')
    ax[1, 0].set_ylabel('Error to nontarget')

    ax[1, 1].plot(dist_target_nontarget_recalled, ratio_response_close_to_nontarget, 'x')
    ax[1, 1].set_xlabel('Distance recalled feature only')
    ax[1, 1].set_ylabel('Normalised distance to nontarget')


    f.suptitle('Effect of distance between items on bias of response towards nontarget')

    if dataio:
        f.set_size_inches(16, 16, forward=True)
        dataio.save_current_figure('plot_bias_close_feature_{label}_{unique_id}.pdf')



def load_data_simult(data_dir='../../experimental_data/', fit_mixture_model=False):
    '''
        Convenience function, automatically load the Gorgoraptis_2011 dataset.
    '''

    if data_dir == '../../experimental_data/':
        experim_datadir = os.environ.get('WORKDIR_DROP', os.path.split(utils.__file__)[0])
        data_dir = os.path.normpath(os.path.join(experim_datadir, data_dir))

    data_simult =  load_multiple_datasets([dict(name='gorgo11', filename='Exp2_withcolours.mat', preprocess=preprocess_simultaneous, parameters=dict(fit_mixture_model=fit_mixture_model, datadir=os.path.join(data_dir, 'Gorgoraptis_2011'), mixture_model_cache='em_simult.pickle'))])[0]

    return data_simult


def load_data_bays09(data_dir='../../experimental_data/', fit_mixture_model=False):
    '''
        Convenience function, automatically load the Bays2009 dataset.
    '''

    if data_dir == '../../experimental_data/':
        experim_datadir = os.environ.get('WORKDIR_DROP', os.path.split(utils.__file__)[0])
        data_dir = os.path.normpath(os.path.join(experim_datadir, data_dir))

    (data_bays2009, ) = load_multiple_datasets([dict(name='bays09', filename='colour_data.mat', preprocess=preprocess_bays09, parameters=dict(datadir=os.path.join(data_dir, 'Bays2009'), fit_mixture_model=fit_mixture_model, mixture_model_cache='em_bays_allitems.pickle'))])

    return data_bays2009


def load_data_gorgo11(data_dir='../../experimental_data/', fit_mixture_model=False):
    '''
        Convenience function, automatically load the Gorgo11 simultaneous dataset.
    '''

    return load_data_simult(data_dir, fit_mixture_model)


def load_data_dualrecall(data_dir='../../experimental_data/', fit_mixture_model=False):
    '''
        Convenience function, automatically load the Double recall dataset (unpublished).
    '''

    if data_dir == '../../experimental_data/':
        experim_datadir = os.environ.get('WORKDIR_DROP', os.path.split(utils.__file__)[0])
        data_dir = os.path.normpath(os.path.join(experim_datadir, data_dir))

    (data_dualrecall, ) = load_multiple_datasets([dict(name='dualrecall', filename=os.path.join(data_dir, 'DualRecall_Bays', 'rate_data.mat'), preprocess=preprocess_dualrecall, parameters=dict(fit_mixture_model=fit_mixture_model, mixture_model_cache='em_dualrecall_allitems.pickle'))])


    return data_dualrecall



if __name__ == '__main__':
    ## Load data
    experim_datadir = os.environ.get('WORKDIR_DROP', os.path.split(utils.__file__)[0])
    data_dir = os.path.normpath(os.path.join(experim_datadir, '../../experimental_data/'))
    # data_dir = '/Users/loicmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/experimental_data/'

    # data_dir = os.path.normpath(os.path.join(experim_datadir, '../experimental_data/'))

    print sys.argv

    if True or (len(sys.argv) > 1 and sys.argv[1]):
    # keys:
    # 'probe', 'delayed', 'item_colour', 'probe_colour', 'item_angle', 'error', 'probe_angle', 'n_items', 'response', 'subject']
        # (data_sequen, data_simult, data_dualrecall) = load_multiple_datasets([dict(filename='Exp1.mat', preprocess=preprocess_sequential, parameters=dict(datadir=os.path.join(data_dir, 'Gorgoraptis_2011'))), dict(filename='Exp2_withcolours.mat', preprocess=preprocess_simultaneous, parameters=dict(datadir=os.path.join(data_dir, 'Gorgoraptis_2011'), fit_mixture_model=True)), dict(filename=os.path.join(data_dir, 'DualRecall_Bays', 'rate_data.mat'), preprocess=preprocess_dualrecall, parameters=dict(fit_mixture_model=True))])
        (data_simult,) = load_multiple_datasets([dict(name='Gorgo_simult', filename='Exp2_withcolours.mat', preprocess=preprocess_simultaneous, parameters=dict(datadir=os.path.join(data_dir, 'Gorgoraptis_2011'), fit_mixture_model=True, mixture_model_cache='em_simult.pickle'))])
        # (data_bays2009, ) = load_multiple_datasets([dict(name='Bays2009', filename='colour_data.mat', preprocess=preprocess_bays09, parameters=dict(datadir=os.path.join(data_dir, 'Bays2009'), fit_mixture_model=True, mixture_model_cache='em_bays.pickle', should_compute_bootstrap=True, bootstrap_cache='bootstrap_1000samples.pickle'))])
        # data_dualrecall = load_data_dualrecall(data_dir=data_dir, fit_mixture_model=True)
        data_bays2009 = load_data_bays09(data_dir=data_dir, fit_mixture_model=True)
        data_gorgo11 = load_data_gorgo11(data_dir=data_dir, fit_mixture_model=True)


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

    # plots_check_oblique_effect(data_simult, nb_bins=50)

    # np.save('processed_experimental_230613.npy', dict(data_simult=data_simult, data_sequen=data_sequen))

    # plots_dualrecall(data_dualrecall)

    plt.rcParams['font.size'] = 16
    dataio = None

    # dataio = DataIO.DataIO(label='experiments_bays2009')
    # plots_check_bias_nontarget(data_simult, dataio=dataio)
    # plots_check_bias_bestnontarget(data_simult, dataio=dataio)
    # plots_check_bias_nontarget_randomized(data_simult, dataio=dataio)

    # plots_bays2009(data_bays2009, dataio=dataio)

    # dataio = DataIO.DataIO(label='experiments_gorgo11')
    plots_gorgo11(data_gorgo11, dataio)

    # plots_precision(data_gorgo11, dataio)
    # plots_precision(data_bays2009, dataio)

    # dataio = DataIO.DataIO(label='experiments_bays2009')
    # plot_bias_close_feature(data_bays2009, dataio)

    # dataio = DataIO.DataIO(label='experiments_gorgo11')
    # plot_bias_close_feature(data_gorgo11, dataio)

    plt.show()





