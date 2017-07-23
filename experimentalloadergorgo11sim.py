'''
    Small class system to simplify the process of loading Experimental datasets
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
from experimentalloader import ExperimentalLoader

class ExperimentalLoaderGorgo11Simultaneous(ExperimentalLoader):
    """docstring for ExperimentalLoaderGorgo11Simultaneous"""
    def __init__(self, dataset_description):
        super(self.__class__, self).__init__(dataset_description)


    def preprocess(self, parameters):
        '''
        For simultaneous datasets, there is no 'probe' key, the first item in 'item_angle' is the target one.
        The 'error' key is missing and called 'err', which we will correct as well.
        '''

        # Rename the error field
        if 'err' in self.dataset:
            self.dataset['error'] = self.dataset['err']
            del self.dataset['err']

        # Assign probe field correctly
        self.dataset['probe'] = np.zeros(self.dataset['error'].shape, dtype=int)

        # Convert everything to radians, spanning a -np.pi:np.pi
        if parameters.get('convert_radians', True):
            self.convert_wrap(multiply_factor=2, max_angle=np.pi)

        # Make some aliases
        self.dataset['n_items'] = self.dataset['n_items'].astype(int)
        self.dataset['subject'] = self.dataset['subject'].astype(int)

        self.dataset['n_items_size'] = np.unique(self.dataset['n_items']).size
        self.dataset['subject_size'] = np.unique(self.dataset['subject']).size

        # Compute additional errors, between the response and all items
        self.compute_all_errors()

        # Create arrays per subject
        self.create_subject_arrays()

        # Reconstruct the colour information_
        if 'item_colour' not in self.dataset:
            self.reconstruct_colours_exp1()

        # Fit the mixture model, and save the responsibilities per datapoint.
        if parameters['fit_mixture_model']:
            self.fit_mixture_model_cached(caching_save_filename=parameters.get('mixture_model_cache', None))

        ## Save item in a nice format for the model fit
        self.generate_data_to_fit()

        # Save data in a better format to fit the new collapsed mixture model
        self.generate_data_subject_split()

        # Fit the new Collapsed mixture model
        if parameters.get('fit_mixture_model', False):
            self.fit_collapsed_mixture_model_cached(caching_save_filename=parameters.get('collapsed_mixture_model_cache', None))

        # Perform Bootstrap analysis if required
        if parameters.get('should_compute_bootstrap', False):
            self.compute_bootstrap_cached(
                caching_save_filename=parameters.get('bootstrap_cache', None),
                nb_bootstrap_samples=parameters.get('nb_bootstrap_samples', 1000))

        # Perform Vtest for circular uniformity
        self.compute_vtest()

        # Do per subject and nitems, get average histogram
        self.compute_average_histograms()


    def create_subject_arrays(self, double_precision=True):
        '''
            Create arrays with errors per subject and per num_target
            also create an array with the precision per subject and num_target directly
        '''

        unique_subjects = np.unique(self.dataset['subject'])
        unique_n_items = np.unique(self.dataset['n_items'])

        self.dataset['errors_subject_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_all_subject_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_nontarget_subject_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)
        self.dataset['precision_subject_nitems_bays'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size))
        self.dataset['precision_subject_nitems_theo'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size))
        self.dataset['precision_subject_nitems_theo_nochance'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size))
        self.dataset['precision_subject_nitems_bays_notreatment'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size))

        self.dataset['response_subject_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)
        self.dataset['item_angle_subject_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)

        for n_items_i, n_items in enumerate(unique_n_items):
            for subject_i, subject in enumerate(unique_subjects):
                ids_filtered = ((self.dataset['subject']==subject) & (self.dataset['n_items'] == n_items) & (self.dataset.get('masked', False) == False)).flatten()


                # Get the errors
                self.dataset['errors_subject_nitems'][subject_i, n_items_i] = self.dataset['errors_all'][ids_filtered, 0]
                self.dataset['errors_all_subject_nitems'][subject_i, n_items_i] = self.dataset['errors_all'][ids_filtered]
                self.dataset['errors_nontarget_subject_nitems'][subject_i, n_items_i] = self.dataset['errors_all'][ids_filtered, 1:]

                # Get the responses and correct item angles
                self.dataset['response_subject_nitems'][subject_i, n_items_i] = self.dataset['response'][ids_filtered]
                self.dataset['item_angle_subject_nitems'][subject_i, n_items_i] = self.dataset['item_angle'][ids_filtered]

                # Compute the precision
                self.dataset['precision_subject_nitems_bays'][subject_i, n_items_i] = self.compute_precision(self.dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=True, correct_orientation=True, use_wrong_precision=True)
                self.dataset['precision_subject_nitems_theo'][subject_i, n_items_i] = self.compute_precision(self.dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=False, correct_orientation=True, use_wrong_precision=False)
                self.dataset['precision_subject_nitems_theo_nochance'][subject_i, n_items_i] = self.compute_precision(self.dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=True, correct_orientation=False, use_wrong_precision=False)
                self.dataset['precision_subject_nitems_bays_notreatment'][subject_i, n_items_i] = self.compute_precision(self.dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=False, correct_orientation=True, use_wrong_precision=True)

        # if double_precision:
        #     precision_subject_nitems *= 2.
        #     precision_subject_nitems_theo *= 2.
        #     # self.dataset['precision_subject_nitems_theo_nochance'] *= 2.
        #     self.dataset['precision_subject_nitems_bays_notreatment'] *= 2.


        self.dataset['errors_nitems'] = np.array([utils.flatten_list(self.dataset['errors_subject_nitems'][:, n_item_i]) for n_item_i in xrange(unique_n_items.size)])
        self.dataset['errors_all_nitems'] = np.array([utils.flatten_list(self.dataset['errors_all_subject_nitems'][:, n_item_i]) for n_item_i in xrange(unique_n_items.size)])
        self.dataset['errors_nontarget_nitems'] = self.dataset['errors_all_nitems'][:, :, 1:]
        self.dataset['precision_nitems_bays'] = np.mean(self.dataset['precision_subject_nitems_bays'], axis=0)
        self.dataset['precision_nitems_theo'] = np.mean(self.dataset['precision_subject_nitems_theo'], axis=0)
        self.dataset['precision_nitems_theo_nochance'] = np.mean(self.dataset['precision_subject_nitems_theo_nochance'], axis=0)
        self.dataset['precision_nitems_bays_notreatment'] = np.mean(self.dataset['precision_subject_nitems_bays_notreatment'], axis=0)


    def compute_vtest(self):
        self.dataset['vtest_nitems'] = np.empty(self.dataset['n_items_size'])*np.nan
        for n_items_i, n_items in enumerate(np.unique(self.dataset['n_items'])):
            if n_items > 1:
                self.dataset['vtest_nitems'][n_items_i] = utils.V_test(utils.dropnan(self.dataset['errors_nontarget_nitems'][n_items_i]).flatten())['pvalue']



    def reconstruct_colours_exp1(self, datasets=('Data/ad.mat', 'Data/gb.mat', 'Data/kf.mat', 'Data/md.mat', 'Data/sf.mat', 'Data/sw.mat', 'Data/wd.mat', 'Data/zb.mat')):
        '''
            The colour is missing from the simultaneous experiment dataset
            Reconstruct it.
        '''

        all_colours = []
        all_preangles = []
        all_targets = []
        for dataset_fn in datasets:
            dataset_fn = os.path.join(self.datadir, dataset_fn)
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

        self.dataset['item_colour'] = item_colour
        self.dataset['item_preangle'] = item_preangle_arr
        self.dataset['all_targets'] = all_targets


