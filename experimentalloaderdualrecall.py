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

class ExperimentalLoaderDualRecall(ExperimentalLoader):
    """docstring for ExperimentalLoaderDualRecall"""
    def __init__(self, dataset_description):
        super(ExperimentalLoaderDualRecall, self).__init__(dataset_description)

    def preprocess(self, parameters):
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
        self.dataset['item_angle'] = self.dataset['item_location']
        self.dataset['probe_angle'] = self.dataset['probe_location']
        self.dataset['response'] = np.empty((self.dataset['probe_angle'].size, 1))
        self.dataset['target'] = np.empty(self.dataset['probe_angle'].size)
        self.dataset['probe'] = np.zeros(self.dataset['probe_angle'].shape, dtype= int)

        self.dataset['n_items'] = self.dataset['n_items'].astype(int)
        self.dataset['cond'] = self.dataset['cond'].astype(int)

        self.dataset['n_items_size'] = np.unique(self.dataset['n_items']).size
        self.dataset['subject_size'] = np.unique(self.dataset['subject']).size

        # Get shortcuts for colour and orientation trials
        self.dataset['colour_trials'] = (self.dataset['cond'] == 1).flatten()
        self.dataset['angle_trials'] = (self.dataset['cond'] == 2).flatten()
        self.dataset['3_items_trials'] = (self.dataset['n_items'] == 3).flatten()
        self.dataset['6_items_trials'] = (self.dataset['n_items'] == 6).flatten()

        # Wrap everything around
        multiply_factor = 2.
        self.dataset['item_angle'] = utils.wrap_angles(multiply_factor*self.dataset['item_angle'], np.pi)
        self.dataset['probe_angle'] = utils.wrap_angles(multiply_factor*self.dataset['probe_angle'], np.pi)
        self.dataset['item_colour'] = utils.wrap_angles(multiply_factor*self.dataset['item_colour'], np.pi)
        self.dataset['probe_colour'] = utils.wrap_angles(multiply_factor*self.dataset['probe_colour'], np.pi)

        # Remove wrong trials
        reject_ids = (self.dataset['reject'] == 1.0).flatten()
        for key in self.dataset:
            if type(self.dataset[key]) == np.ndarray and self.dataset[key].shape[0] == reject_ids.size and key in ('probe_colour', 'probe_angle', 'item_angle', 'item_colour'):
                self.dataset[key][reject_ids] = np.nan

        # Compute the errors
        self.dataset['errors_angle_all'] = utils.wrap_angles(self.dataset['item_angle'] - self.dataset['probe_angle'], np.pi)
        self.dataset['errors_colour_all'] = utils.wrap_angles(self.dataset['item_colour'] - self.dataset['probe_colour'], np.pi)
        self.dataset['error_angle'] = self.dataset['errors_angle_all'][:, 0]
        self.dataset['error_colour'] = self.dataset['errors_colour_all'][:, 0]
        self.dataset['error'] = np.where(~np.isnan(self.dataset['error_angle']), self.dataset['error_angle'], self.dataset['error_colour'])

        self.dataset['errors_nitems'] = np.empty(self.dataset['n_items_size'], dtype=np.object)
        self.dataset['errors_all_nitems'] = np.empty(self.dataset['n_items_size'], dtype=np.object)

        for n_items_i, n_items in enumerate(np.unique(self.dataset['n_items'])):
            ids_filtered = self.dataset['angle_trials'] & (self.dataset['n_items'] == n_items).flatten()

            self.dataset['errors_nitems'][n_items_i] = self.dataset['error_angle'][ids_filtered]
            self.dataset['errors_all_nitems'][n_items_i
            ] = self.dataset['errors_angle_all'][ids_filtered]


        ### Fit the mixture model
        if parameters['fit_mixture_model']:

            self.dataset['em_fits'] = dict(kappa=np.empty(self.dataset['probe_angle'].size), mixt_target=np.empty(self.dataset['probe_angle'].size), mixt_nontarget=np.empty(self.dataset['probe_angle'].size), mixt_random=np.empty(self.dataset['probe_angle'].size), resp_target=np.empty(self.dataset['probe_angle'].size), resp_nontarget=np.empty(self.dataset['probe_angle'].size), resp_random=np.empty(self.dataset['probe_angle'].size), train_LL=np.empty(self.dataset['probe_angle'].size), test_LL=np.empty(self.dataset['probe_angle'].size))
            for key in self.dataset['em_fits']:
                self.dataset['em_fits'][key].fill(np.nan)

            # Angles trials
            for n_items in np.unique(self.dataset['n_items']):
                ids_n_items = (self.dataset['n_items'] == n_items).flatten()
                ids_filtered = self.dataset['angle_trials'] & ids_n_items

                self.dataset['target'][ids_filtered] = self.dataset['item_angle'][ids_filtered, 0]
                self.dataset['response'][ids_filtered] = self.dataset['probe_angle'][ids_filtered]

                # params_fit = em_circularmixture.fit(self.dataset['probe_angle'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 1:])
                print self.dataset['probe_angle'][ids_filtered, 0].shape, self.dataset['item_angle'][ids_filtered, 0].shape, self.dataset['item_angle'][ids_filtered, 1:].shape

                cross_valid_outputs = em_circularmixture.cross_validation_kfold(self.dataset['probe_angle'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 1:], K=10, shuffle=True, debug=False)
                params_fit = cross_valid_outputs['best_fit']
                resp = em_circularmixture.compute_responsibilities(self.dataset['probe_angle'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 1:], params_fit)

                self.dataset['em_fits']['kappa'][ids_filtered] = params_fit['kappa']
                self.dataset['em_fits']['mixt_target'][ids_filtered] = params_fit['mixt_target']
                self.dataset['em_fits']['mixt_nontarget'][ids_filtered] = params_fit['mixt_nontargets']
                self.dataset['em_fits']['mixt_random'][ids_filtered] = params_fit['mixt_random']
                self.dataset['em_fits']['resp_target'][ids_filtered] = resp['target']
                self.dataset['em_fits']['resp_nontarget'][ids_filtered] = np.sum(resp['nontargets'], axis=1)
                self.dataset['em_fits']['resp_random'][ids_filtered] = resp['random']
                self.dataset['em_fits']['train_LL'][ids_filtered] = params_fit['train_LL']
                self.dataset['em_fits']['test_LL'][ids_filtered] = cross_valid_outputs['best_test_LL']

            # Colour trials
            for n_items in np.unique(self.dataset['n_items']):
                ids_n_items = (self.dataset['n_items'] == n_items).flatten()
                ids_filtered = self.dataset['colour_trials'] & ids_n_items

                self.dataset['target'][ids_filtered] = self.dataset['item_colour'][ids_filtered, 0]
                self.dataset['response'][ids_filtered] = self.dataset['probe_colour'][ids_filtered]

                # params_fit = em_circularmixture.fit(self.dataset['probe_colour'][ids_filtered, 0], self.dataset['item_colour'][ids_filtered, 0], self.dataset['item_colour'][ids_filtered, 1:])
                cross_valid_outputs = em_circularmixture.cross_validation_kfold(self.dataset['probe_colour'][ids_filtered, 0], self.dataset['item_colour'][ids_filtered, 0], self.dataset['item_colour'][ids_filtered, 1:], K=10, shuffle=True, debug=False)
                params_fit = cross_valid_outputs['best_fit']
                resp = em_circularmixture.compute_responsibilities(self.dataset['probe_colour'][ids_filtered, 0], self.dataset['item_colour'][ids_filtered, 0], self.dataset['item_colour'][ids_filtered, 1:], params_fit)

                self.dataset['em_fits']['kappa'][ids_filtered] = params_fit['kappa']
                self.dataset['em_fits']['mixt_target'][ids_filtered] = params_fit['mixt_target']
                self.dataset['em_fits']['mixt_nontarget'][ids_filtered] = params_fit['mixt_nontargets']
                self.dataset['em_fits']['mixt_random'][ids_filtered] = params_fit['mixt_random']
                self.dataset['em_fits']['resp_target'][ids_filtered] = resp['target']
                self.dataset['em_fits']['resp_nontarget'][ids_filtered] = np.sum(resp['nontargets'], axis=1)
                self.dataset['em_fits']['resp_random'][ids_filtered] = resp['random']
                self.dataset['em_fits']['train_LL'][ids_filtered] = params_fit['train_LL']
                self.dataset['em_fits']['test_LL'][ids_filtered] = cross_valid_outputs['best_test_LL']

        ## Save item in a nice format for the model fit
        self.dataset['data_to_fit'] = {}
        self.dataset['data_to_fit']['n_items'] = np.unique(self.dataset['n_items'])
        for n_items in self.dataset['data_to_fit']['n_items']:
            ids_n_items = (self.dataset['n_items'] == n_items).flatten()
            ids_filtered = self.dataset['angle_trials'] & ids_n_items

            if n_items not in self.dataset['data_to_fit']:
                self.dataset['data_to_fit'][n_items] = {}
                self.dataset['data_to_fit'][n_items]['N'] = np.sum(ids_filtered)
                self.dataset['data_to_fit'][n_items]['probe'] = np.unique(self.dataset['probe'][ids_filtered])
                self.dataset['data_to_fit'][n_items]['item_features'] = np.empty((self.dataset['data_to_fit'][n_items]['N'], n_items, 2))
                self.dataset['data_to_fit'][n_items]['response'] = np.empty((self.dataset['data_to_fit'][n_items]['N'], 1))

            self.dataset['data_to_fit'][n_items]['item_features'][..., 0] = self.dataset['item_angle'][ids_filtered, :n_items]
            self.dataset['data_to_fit'][n_items]['item_features'][..., 1] = self.dataset['item_colour'][ids_filtered, :n_items]
            self.dataset['data_to_fit'][n_items]['response'] = self.dataset['probe_angle'][ids_filtered].flatten()

        # Try with Pandas for some advanced plotting
        dataset_filtered = dict((k, self.dataset[k].flatten()) for k in ('n_items', 'trial', 'subject', 'reject', 'rating', 'probe_colour', 'probe_angle', 'cond', 'error', 'error_angle', 'error_colour', 'response', 'target'))

        if parameters['fit_mixture_model']:
            dataset_filtered.update(self.dataset['em_fits'])

        self.dataset['panda'] = pd.DataFrame(dataset_filtered)

