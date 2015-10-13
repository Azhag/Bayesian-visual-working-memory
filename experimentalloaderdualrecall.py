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
        super(self.__class__, self).__init__(dataset_description)

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
        self.dataset['n_items'] = self.dataset['n_items'].astype(int)
        self.dataset['cond'] = self.dataset['cond'].astype(int)
        self.dataset['subject'] = self.dataset['subject'].astype(int)

        self.dataset['probe'] = np.zeros(self.dataset['probe_angle'].shape[0], dtype=int)

        self.dataset['n_items_space'] = np.unique(self.dataset['n_items'])
        self.dataset['n_items_size'] = self.dataset['n_items_space'].size

        self.dataset['subject_space'] = np.unique(self.dataset['subject'])
        self.dataset['subject_size'] = self.dataset['subject_space'].size

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


        ### Split the data up
        self.generate_data_to_fit()

        ### Fit the mixture model
        if parameters['fit_mixture_model']:
            self.fit_mixture_model_cached(caching_save_filename=parameters.get('mixture_model_cache', None), saved_keys=['em_fits', 'em_fits_angle_nitems_subjects', 'em_fits_angle_nitems', 'em_fits_colour_nitems_subjects', 'em_fits_colour_nitems', 'em_fits_angle_nitems_arrays', 'em_fits_colour_nitems_arrays'])

        # Try with Pandas for some advanced plotting
        dataset_filtered = dict((k, self.dataset[k].flatten()) for k in ('n_items', 'trial', 'subject', 'reject', 'rating', 'probe_colour', 'probe_angle', 'cond', 'error', 'error_angle', 'error_colour', 'response', 'target'))
        if parameters['fit_mixture_model']:
            dataset_filtered.update(self.dataset['em_fits'])

        self.dataset['panda'] = pd.DataFrame(dataset_filtered)


    def fit_mixture_model(self):
        '''
            Special fitting for this dual recall dataset
        '''

        self.dataset['em_fits'] = dict(kappa=np.empty(self.dataset['probe_angle'].size), mixt_target=np.empty(self.dataset['probe_angle'].size), mixt_nontarget=np.empty(self.dataset['probe_angle'].size), mixt_random=np.empty(self.dataset['probe_angle'].size), resp_target=np.empty(self.dataset['probe_angle'].size), resp_nontarget=np.empty(self.dataset['probe_angle'].size), resp_random=np.empty(self.dataset['probe_angle'].size), train_LL=np.empty(self.dataset['probe_angle'].size), test_LL=np.empty(self.dataset['probe_angle'].size))
        for key in self.dataset['em_fits']:
            self.dataset['em_fits'][key].fill(np.nan)

        self.dataset['em_fits_angle_nitems_subjects'] = dict()
        self.dataset['em_fits_angle_nitems'] = dict(mean=dict(), std=dict(), values=dict())
        self.dataset['em_fits_colour_nitems_subjects'] = dict()
        self.dataset['em_fits_colour_nitems'] = dict(mean=dict(), std=dict(), values=dict())

        # This dataset is a bit special with regards to subjects, it's a conditional design:
        # 8 Subjects (1 - 8) only did 6 items, both angle/colour trials
        # 6 Subjects (9 - 14) did 3 items, both angle/colour trials.
        # We have 160 trials per (subject, n_item, condition).

        # Angles trials
        for n_items_i, n_items in enumerate(self.dataset['n_items_space']):
            for subject_i, subject in enumerate(self.dataset['subject_space']):
                ids_filtered = ((self.dataset['subject']==subject) & (self.dataset['n_items'] == n_items) & (self.dataset.get('masked', False) == False)).flatten()

                ids_filtered = self.dataset['angle_trials'] & ids_filtered

                if ids_filtered.sum() > 0:
                    print 'Angle trials, %d items, subject %d, %d datapoints' % (n_items, subject, self.dataset['probe_angle'][ids_filtered, 0].size)

                    # params_fit = em_circularmixture.fit(self.dataset['probe_angle'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 1:])

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

                    self.dataset['em_fits_angle_nitems_subjects'].setdefault(n_items, dict())[subject] = params_fit

            ## Now compute mean/std em_fits per n_items
            self.dataset['em_fits_angle_nitems']['mean'][n_items] = dict()
            self.dataset['em_fits_angle_nitems']['std'][n_items] = dict()
            self.dataset['em_fits_angle_nitems']['values'][n_items] = dict()

            # Need to extract the values for a subject/nitems pair, for all keys of em_fits. Annoying dictionary indexing needed
            emfits_keys = params_fit.keys()
            for key in emfits_keys:
                values_allsubjects = [self.dataset['em_fits_angle_nitems_subjects'][n_items][subject][key] for subject in self.dataset['em_fits_angle_nitems_subjects'][n_items]]

                self.dataset['em_fits_angle_nitems']['mean'][n_items][key] = np.mean(values_allsubjects)
                self.dataset['em_fits_angle_nitems']['std'][n_items][key] = np.std(values_allsubjects)
                self.dataset['em_fits_angle_nitems']['values'][n_items][key] = values_allsubjects


        # Colour trials
        for n_items_i, n_items in enumerate(self.dataset['n_items_space']):
            for subject_i, subject in enumerate(self.dataset['subject_space']):
                ids_filtered = ((self.dataset['subject']==subject) & (self.dataset['n_items'] == n_items) & (self.dataset.get('masked', False) == False)).flatten()

                ids_filtered = self.dataset['colour_trials'] & ids_filtered

                if ids_filtered.sum() > 0:
                    print 'Colour trials, %d items, subject %d, %d datapoints' % (n_items, subject, self.dataset['probe_angle'][ids_filtered, 0].size)

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

                    self.dataset['em_fits_colour_nitems_subjects'].setdefault(n_items, dict())[subject] = params_fit

            ## Now compute mean/std em_fits per n_items
            self.dataset['em_fits_colour_nitems']['mean'][n_items] = dict()
            self.dataset['em_fits_colour_nitems']['std'][n_items] = dict()
            self.dataset['em_fits_colour_nitems']['values'][n_items] = dict()

            # Need to extract the values for a subject/nitems pair, for all keys of em_fits. Annoying dictionary indexing needed
            emfits_keys = params_fit.keys()
            for key in emfits_keys:
                values_allsubjects = [self.dataset['em_fits_colour_nitems_subjects'][n_items][subject][key] for subject in self.dataset['em_fits_colour_nitems_subjects'][n_items]]

                self.dataset['em_fits_colour_nitems']['mean'][n_items][key] = np.mean(values_allsubjects)
                self.dataset['em_fits_colour_nitems']['std'][n_items][key] = np.std(values_allsubjects)
                self.dataset['em_fits_colour_nitems']['values'][n_items][key] = values_allsubjects

        ## Construct array versions of the em_fits_nitems mixture proportions, for convenience
        self.construct_arrays_em_fits()


    def construct_arrays_em_fits(self):
        if 'em_fits_angle_nitems_arrays' not in self.dataset:
            self.dataset['em_fits_angle_nitems_arrays'] = dict()

            self.dataset['em_fits_angle_nitems_arrays']['mean'] = np.array([[self.dataset['em_fits_angle_nitems']['mean'][item][em_key] for item in np.unique(self.dataset['n_items'])] for em_key in ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random']])
            self.dataset['em_fits_angle_nitems_arrays']['std'] = np.array([[self.dataset['em_fits_angle_nitems']['std'][item][em_key] for item in np.unique(self.dataset['n_items'])] for em_key in ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random']])

        if 'sem' not in self.dataset['em_fits_angle_nitems_arrays']:
            self.dataset['em_fits_angle_nitems_arrays']['sem'] = self.dataset['em_fits_angle_nitems_arrays']['std']/np.sqrt(self.dataset['subject_size'])

        if 'em_fits_colour_nitems_arrays' not in self.dataset:
            self.dataset['em_fits_colour_nitems_arrays'] = dict()

            self.dataset['em_fits_colour_nitems_arrays']['mean'] = np.array([[self.dataset['em_fits_colour_nitems']['mean'][item][em_key] for item in np.unique(self.dataset['n_items'])] for em_key in ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random']])
            self.dataset['em_fits_colour_nitems_arrays']['std'] = np.array([[self.dataset['em_fits_colour_nitems']['std'][item][em_key] for item in np.unique(self.dataset['n_items'])] for em_key in ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random']])

        if 'sem' not in self.dataset['em_fits_colour_nitems_arrays']:
            self.dataset['em_fits_colour_nitems_arrays']['sem'] = self.dataset['em_fits_colour_nitems_arrays']['std']/np.sqrt(self.dataset['subject_size'])


    def generate_data_to_fit(self):
        '''
            Split the data up nicely, used in FitExperiment as well
        '''

        self.dataset['response'] = np.nan*np.empty((self.dataset['probe_angle'].size, 1))
        self.dataset['target'] = np.nan*np.empty(self.dataset['probe_angle'].size)
        self.dataset['nontargets'] = np.nan*np.empty((self.dataset['probe_angle'].size, self.dataset['n_items_space'][-1] - 1))
        self.dataset['data_split_nitems_subjects'] = {
            'angle_trials': dict(),
            'colour_trials': dict(),
            'n_items': self.dataset['n_items_space']
        }

        for n_items_i, n_items in enumerate(self.dataset['n_items_space']):
            for subject_i, subject in enumerate(self.dataset['subject_space']):

                ids_filtered = ((self.dataset['subject']==subject) & (self.dataset['n_items'] == n_items) & (self.dataset.get('masked', False) == False)).flatten()

                # Angle trial
                ids_filtered_angle = self.dataset['angle_trials'] & ids_filtered
                if ids_filtered_angle.sum() > 0:
                    self.dataset['target'][ids_filtered_angle] = self.dataset['item_angle'][ids_filtered_angle, 0]
                    self.dataset['nontargets'][ids_filtered_angle] = self.dataset['item_angle'][ids_filtered_angle, 1:]
                    self.dataset['response'][ids_filtered_angle] = self.dataset['probe_angle'][ids_filtered_angle]

                    self.dataset['data_split_nitems_subjects']['angle_trials'].setdefault(n_items, dict())[subject] = dict(
                            target=self.dataset['target'][ids_filtered_angle],
                            nontargets=self.dataset['nontargets'][ids_filtered_angle],
                            response=self.dataset['response'][ids_filtered_angle],
                            item_features=self.dataset['item_angle'][ids_filtered_angle],
                            probe=self.dataset['probe'][ids_filtered_angle],
                            N=np.sum(ids_filtered_angle)
                        )

                # Colour trial
                ids_filtered_colour = self.dataset['colour_trials'] & ids_filtered
                if ids_filtered_colour.sum() > 0:
                    self.dataset['target'][ids_filtered_colour] = self.dataset['item_colour'][ids_filtered_colour, 0]
                    self.dataset['nontargets'][ids_filtered_colour] = self.dataset['item_colour'][ids_filtered_colour, 1:]
                    self.dataset['response'][ids_filtered_colour] = self.dataset['probe_colour'][ids_filtered_colour]

                    self.dataset['data_split_nitems_subjects']['colour_trials'].setdefault(n_items, dict())[subject] = dict(
                            target=self.dataset['target'][ids_filtered_colour],
                            nontargets=self.dataset['nontargets'][ids_filtered_colour],
                            response=self.dataset['response'][ids_filtered_colour],
                            item_features=self.dataset['item_colour'][ids_filtered_colour],
                            probe=self.dataset['probe'][ids_filtered_colour],
                            N=np.sum(ids_filtered_colour)
                        )

            # Also store a version collating subjects across
            self.dataset['data_split_nitems'] = dict(colour_trials=dict(), angle_trials=dict())

            ids_filtered = ((self.dataset['n_items'] == n_items) & (self.dataset.get('masked', False) == False)).flatten()
            ids_filtered_angle = self.dataset['angle_trials'] & ids_filtered
            self.dataset['data_split_nitems']['angle_trials'][n_items] = dict(
                            target=self.dataset['target'][ids_filtered_angle],
                            nontargets=self.dataset['nontargets'][ids_filtered_angle],
                            response=self.dataset['response'][ids_filtered_angle],
                            item_features=self.dataset['item_angle'][ids_filtered_angle],
                            probe=self.dataset['probe'][ids_filtered_angle],
                            N=np.sum(ids_filtered_angle)
                        )
            ids_filtered_colour = self.dataset['colour_trials'] & ids_filtered
            self.dataset['data_split_nitems']['colour_trials'][n_items] = dict(
                            target=self.dataset['target'][ids_filtered_colour],
                            nontargets=self.dataset['nontargets'][ids_filtered_colour],
                            response=self.dataset['response'][ids_filtered_colour],
                            item_features=self.dataset['item_colour'][ids_filtered_colour],
                            probe=self.dataset['probe'][ids_filtered_colour],
                            N=np.sum(ids_filtered_colour)
                        )

















