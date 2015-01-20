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

em_circular_mixture_to_use = em_circularmixture

import dataio as DataIO

import utils
from experimentalloader import ExperimentalLoader

class ExperimentalLoaderGorgo11Sequential(ExperimentalLoader):
    """docstring for ExperimentalLoaderGorgo11Sequential"""
    def __init__(self, dataset_description):
        super(ExperimentalLoaderGorgo11Sequential, self).__init__(dataset_description)


    def preprocess(self, parameters):
        '''
        For sequential datasets, need to convert to radians and correct the probe indexing.
        '''

        # Convert everything to radians, spanning a -np.pi/2:np.pi
        if parameters.get('convert_radians', True):  #pylint: disable=E0602
            self.convert_wrap()

        # Correct the probe field, Matlab format for indices...
        # if parameters.get('correct_probe', True) and 'probe' in self.dataset:  #pylint: disable=E0602
            # self.dataset['probe'] = self.dataset['probe'].astype(int)
            # self.dataset['probe'] -= 1


        self.dataset['n_items'] = self.dataset['n_items'].astype(int)
        self.dataset['subject'] = self.dataset['subject'].astype(int)

        self.dataset['Ntot'] = self.dataset['probe'].size

        self.dataset['n_items_size'] = np.unique(self.dataset['n_items']).size
        self.dataset['subject_size'] = np.unique(self.dataset['subject']).size


        # Will remove delayed trials
        self.dataset['masked'] = self.dataset['delayed'] == 1

        # Compute additional errors, between the response and all items
        self.compute_all_errors()

        # Create arrays per subject
        self.create_subject_arrays()


        # Fit the mixture model, and save the responsibilities per datapoint.
        if parameters.get('fit_mixture_model', False):
            self.fit_mixture_model_cached(caching_save_filename=parameters.get('mixture_model_cache', None), saved_keys=['em_fits', 'em_fits_nitems_mean_arrays', 'em_fits_nitems_trecall', 'em_fits_nitems_trecall_arrays', 'em_fits_nitems_trecall_mean', 'em_fits_nitems_trecall_mean_arrays', 'em_fits_subjects_nitems', 'em_fits_subjects_nitems_arrays', 'em_fits_subjects_nitems_trecall', 'em_fits_subjects_nitems_trecall_arrays',])
        ## Save item in a nice format for the model fit
        # self.generate_data_to_fit()

        # Perform Vtest for circular uniformity
        # self.compute_vtest()

        # Do per subject and nitems, get average histogram
        # self.compute_average_histograms()


    def extract_target_nontargets_columns(self, data, probe):
        '''
            Given an array NxK, where K is the number of items,
            should return the column corresponding to the target/probe, and the others columns
            When probe != 0, this is a bit annoying
        '''

        indices_columns = np.arange(data.shape[1])
        target_data = data[:, indices_columns == probe-1].flatten()
        nontarget_data = data[:, indices_columns != probe-1]

        return target_data, nontarget_data



    def create_subject_arrays(self, double_precision=True   ):
        '''
            Create arrays with errors per subject and per num_target
            also create an array with the precision per subject and num_target directly
        '''

        unique_subjects = np.unique(self.dataset['subject'])
        unique_n_items = np.unique(self.dataset['n_items'])

        self.dataset['errors_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_all_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_nontarget_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['sizes_subject_nitems_trecall'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size))
        self.dataset['precision_subject_nitems_trecall_bays'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size))
        self.dataset['precision_subject_nitems_trecall_theo'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size))
        self.dataset['precision_subject_nitems_trecall_theo_nochance'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size))
        self.dataset['precision_subject_nitems_trecall_bays_notreatment'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size))

        self.dataset['response_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['item_angle_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['target_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['nontargets_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)


        self.dataset['errors_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_all_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_nontarget_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['response_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['item_angle_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['target_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['nontargets_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)

        self.dataset['precision_nitems_trecall_bays'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))
        self.dataset['precision_nitems_trecall_theo'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))
        self.dataset['precision_nitems_trecall_theo_nochance'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))
        self.dataset['precision_nitems_trecall_bays_notreatment'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))



        for n_items_i, n_items in enumerate(unique_n_items):
            for subject_i, subject in enumerate(unique_subjects):
                for trecall_i, trecall in enumerate(np.arange(1, n_items+1)):
                    ids_filtered = ((self.dataset['subject']==subject) & (self.dataset['n_items'] == n_items) & (self.dataset['probe'] == trecall) & (self.dataset.get('masked', False) == False)).flatten()

                    # Invert the order of storage, 0 -> last item probed, 1 -> second to last item probe, etc...
                    # trecall_i = n_items - trecall

                    # Get the errors
                    self.dataset['errors_all_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['errors_all'][ids_filtered]

                    self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['errors_nontarget_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.extract_target_nontargets_columns(self.dataset['errors_all_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], trecall)


                    # Get the responses and correct item angles
                    self.dataset['response_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['response'][ids_filtered].flatten()
                    self.dataset['item_angle_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['item_angle'][ids_filtered]

                    # Save target item and nontargets as well
                    self.dataset['target_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['nontargets_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.extract_target_nontargets_columns(self.dataset['item_angle'][ids_filtered], trecall)

                    # Get the number of samples per conditions
                    self.dataset['sizes_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i].size

                    # Compute the precision
                    self.dataset['precision_subject_nitems_trecall_bays'][subject_i, n_items_i, trecall_i] = self.compute_precision(self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], remove_chance_level=True, correct_orientation=True, use_wrong_precision=True)
                    self.dataset['precision_subject_nitems_trecall_theo'][subject_i, n_items_i, trecall_i] = self.compute_precision(self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], remove_chance_level=False, correct_orientation=True, use_wrong_precision=False)
                    self.dataset['precision_subject_nitems_trecall_theo_nochance'][subject_i, n_items_i, trecall_i] = self.compute_precision(self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], remove_chance_level=True, correct_orientation=False, use_wrong_precision=False)
                    self.dataset['precision_subject_nitems_trecall_bays_notreatment'][subject_i, n_items_i, trecall_i] = self.compute_precision(self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], remove_chance_level=False, correct_orientation=True, use_wrong_precision=True)

        # if double_precision:
        #     precision_subject_nitems *= 2.
        #     precision_subject_nitems_theo *= 2.
        #     # self.dataset['precision_subject_nitems_theo_nochance'] *= 2.
        #     self.dataset['precision_subject_nitems_bays_notreatment'] *= 2.


        # self.dataset['errors_nitems_trecall'] = np.array([utils.flatten_list(self.dataset['errors_subject_nitems_trecall'][:, n_items_i]) for n_items_i in xrange(unique_n_items.size)])


        # Store all/average subjects data
        for n_items_i, n_items in enumerate(unique_n_items):
            for trecall_i, trecall in enumerate(np.arange(1, n_items+1)):
                self.dataset['errors_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['errors_subject_nitems_trecall'][:, n_items_i, trecall_i]))
                self.dataset['errors_all_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['errors_all_subject_nitems_trecall'][:, n_items_i, trecall_i]))
                self.dataset['errors_nontarget_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['errors_nontarget_subject_nitems_trecall'][:, n_items_i, trecall_i]))

                # Responses, target, nontarget
                self.dataset['response_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['response_subject_nitems_trecall'][:, n_items_i, trecall_i]))
                self.dataset['target_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['target_subject_nitems_trecall'][:, n_items_i, trecall_i]))
                self.dataset['nontargets_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['nontargets_subject_nitems_trecall'][:, n_items_i, trecall_i]))
                self.dataset['item_angle_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['item_angle_subject_nitems_trecall'][:, n_items_i, trecall_i]))


                # Precision over all subjects errors (not average of precisions)
                self.dataset['precision_nitems_trecall_bays'][n_items_i, trecall_i] = self.compute_precision(self.dataset['errors_nitems_trecall'][n_items_i, trecall_i], remove_chance_level=True, correct_orientation=True, use_wrong_precision=True)

                # self.dataset['precision_nitems_trecall_bays'] = np.mean(self.dataset['precision_subject_nitems_trecall_bays'], axis=0)
                self.dataset['precision_nitems_trecall_theo'] = np.mean(self.dataset['precision_subject_nitems_trecall_theo'], axis=0)
                self.dataset['precision_nitems_trecall_theo_nochance'] = np.mean(self.dataset['precision_subject_nitems_trecall_theo_nochance'], axis=0)
                self.dataset['precision_nitems_trecall_bays_notreatment'] = np.mean(self.dataset['precision_subject_nitems_trecall_bays_notreatment'], axis=0)



    def compute_vtest(self):
        self.dataset['vtest_nitems'] = np.empty(self.dataset['n_items_size'])*np.nan
        for n_items_i, n_items in enumerate(np.unique(self.dataset['n_items'])):
            if n_items > 1:
                self.dataset['vtest_nitems'][n_items_i] = utils.V_test(utils.dropnan(self.dataset['errors_nontarget_nitems'][n_items_i]).flatten())['pvalue']



    def fit_mixture_model(self):
        unique_subjects = np.unique(self.dataset['subject'])
        unique_n_items = np.unique(self.dataset['n_items'])

        # Initialize empty arrays
        em_fits_keys = ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'K', 'aic', 'bic']

        self.dataset['em_fits'] = dict()
        for k in em_fits_keys:
            self.dataset['em_fits'][k] = np.nan*np.empty(self.dataset['probe'].size)

        self.dataset['em_fits']['resp_target'] = np.nan*np.empty(self.dataset['probe'].size)
        self.dataset['em_fits']['resp_nontarget'] = np.nan*np.empty(self.dataset['probe'].size)
        self.dataset['em_fits']['resp_random'] = np.nan*np.empty(self.dataset['probe'].size)

        self.dataset['em_fits_subjects_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['em_fits_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['em_fits_subjects_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)

        # for subject_i, subject in enumerate(unique_subjects):
        #     self.dataset['em_fits_subjects_nitems_trecall'][subject] = dict()
        #     for n_items_i, n_items in enumerate(unique_n_items):
        #         self.dataset['em_fits_subjects_nitems_trecall'][subject][n_items] = dict()

        self.dataset['em_fits_nitems_trecall_mean'] = dict(mean=dict(), std=dict(), values=dict())

        # Compute mixture model fits per n_items, subject and trecall
        for n_items_i, n_items in enumerate(unique_n_items):
            for subject_i, subject in enumerate(unique_subjects):
                for trecall_i, trecall in enumerate(np.arange(1, n_items + 1)):
                    ids_filtered = ((self.dataset['subject']==subject) & (self.dataset['n_items'] == n_items) & (self.dataset['probe'] == trecall) & (self.dataset.get('masked', False) == False)).flatten()
                    # Invert the order of storage, 0 -> last item probed, 1 -> second to last item probe, etc...
                    # trecall_i = n_items - trecall

                    print "Fit mixture model, %d items, subject %d, trecall %d, %d datapoints (%d)" % (n_items, subject, trecall, np.sum(ids_filtered), self.dataset['sizes_subject_nitems_trecall'][subject_i, n_items_i, trecall_i])

                    params_fit = em_circular_mixture_to_use.fit(self.dataset['response_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['target_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['nontargets_subject_nitems_trecall'][subject_i, n_items_i, trecall_i])
                    params_fit['mixt_nontargets_sum'] = np.sum(params_fit['mixt_nontargets'])
                    # print self.dataset['response'][ids_filtered, 0].shape, self.dataset['item_angle'][ids_filtered, 0].shape, self.dataset['item_angle'][ids_filtered, 1:].shape

                    # cross_valid_outputs = em_circularmixture.cross_validation_kfold(self.dataset['response'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 1:], K=10, shuffle=True, debug=False)
                    # params_fit = cross_valid_outputs['best_fit']
                    resp = em_circular_mixture_to_use.compute_responsibilities(self.dataset['response_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['target_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['nontargets_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], params_fit)

                    for k, v in params_fit.iteritems():
                        self.dataset['em_fits'][k][ids_filtered] = v

                    # params_fit['responsibilities'] = resp

                    self.dataset['em_fits']['resp_target'][ids_filtered] = resp['target']
                    self.dataset['em_fits']['resp_nontarget'][ids_filtered] = np.sum(resp['nontargets'], axis=1)
                    self.dataset['em_fits']['resp_random'][ids_filtered] = resp['random']

                    self.dataset['em_fits_subjects_nitems_trecall'][subject_i, n_items_i, trecall_i] = params_fit

                # Do not look at trecall (weird but whatever)
                params_fit = em_circular_mixture_to_use.fit(np.array(utils.flatten_list(self.dataset['response_subject_nitems_trecall'][subject_i, n_items_i, :n_items_i+1])), np.array(utils.flatten_list(self.dataset['target_subject_nitems_trecall'][subject_i, n_items_i, :n_items_i+1])), np.array(utils.flatten_list(self.dataset['nontargets_subject_nitems_trecall'][subject_i, n_items_i, :n_items_i+1])))

                self.dataset['em_fits_subjects_nitems'][subject_i, n_items_i] = params_fit


        for n_items_i, n_items in enumerate(unique_n_items):
            for k in ['mean', 'std', 'values']:
                self.dataset['em_fits_nitems_trecall_mean'][k][n_items] = dict()

            for trecall_i, trecall in enumerate(np.arange(1, n_items + 1)):
                for k in ['mean', 'std', 'values']:
                    self.dataset['em_fits_nitems_trecall_mean'][k][n_items][trecall] = dict()

                ## Now compute mean/std em_fits per n_items, trecall
                # Refit the model mixing all subjects together (not sure how we could get sem, 1-held?)
                params_fit = em_circular_mixture_to_use.fit(self.dataset['response_nitems_trecall'][n_items_i, trecall_i], self.dataset['target_nitems_trecall'][n_items_i, trecall_i], self.dataset['nontargets_nitems_trecall'][n_items_i, trecall_i])
                self.dataset['em_fits_nitems_trecall'][n_items_i, trecall_i] = params_fit

                # Need to extract the values for a subject/nitems pair, for all keys of em_fits. Annoying dictionary indexing needed
                for key in em_fits_keys:
                    fits_persubjects = [self.dataset['em_fits_subjects_nitems_trecall'][subject_i, n_items_i, trecall_i][key] for subject in np.unique(unique_subjects)]

                    self.dataset['em_fits_nitems_trecall_mean']['mean'][n_items][trecall][key] = np.mean(fits_persubjects)
                    self.dataset['em_fits_nitems_trecall_mean']['std'][n_items][trecall][key] = np.std(fits_persubjects)
                    self.dataset['em_fits_nitems_trecall_mean']['values'][n_items][trecall][key] = fits_persubjects

        ## Construct array versions of the em_fits_nitems mixture proportions, for convenience
        self.construct_arrays_em_fits()



    def construct_arrays_em_fits(self):
        unique_subjects = np.unique(self.dataset['subject'])
        unique_n_items = np.unique(self.dataset['n_items'])

        # Check if mixt_nontargets in array or not
        if 'mixt_nontargets_sum' in self.dataset['em_fits_nitems_trecall_mean']['mean'].values()[0]:
            emkeys = ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL']
        else:
            emkeys = ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random', 'train_LL']


        if 'em_fits_nitems_trecall_mean_arrays' not in self.dataset:
            self.dataset['em_fits_nitems_trecall_mean_arrays'] = dict()

            self.dataset['em_fits_subjects_nitems_trecall_arrays'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size, len(emkeys)))
            self.dataset['em_fits_nitems_trecall_arrays'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size, len(emkeys)))
            self.dataset['em_fits_subjects_nitems_arrays'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, len(emkeys)))
            self.dataset['em_fits_nitems_mean_arrays'] = dict(mean=np.nan*np.empty((unique_n_items.size, len(emkeys))), std=np.nan*np.empty((unique_n_items.size, len(emkeys))), sem=np.nan*np.empty((unique_n_items.size, len(emkeys))))

            unique_n_items = np.unique(self.dataset['n_items'])

            self.dataset['em_fits_nitems_trecall_mean_arrays']['mean'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size, len(emkeys)))
            self.dataset['em_fits_nitems_trecall_mean_arrays']['std'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size, len(emkeys)))
            self.dataset['em_fits_nitems_trecall_mean_arrays']['sem'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size, len(emkeys)))

            for n_items_i, n_items in enumerate(unique_n_items):
                for trecall_i, trecall in enumerate(np.arange(1, n_items+1)):
                    self.dataset['em_fits_nitems_trecall_mean_arrays']['mean'][n_items_i, trecall_i] = np.array([self.dataset['em_fits_nitems_trecall_mean']['mean'][n_items][trecall][em_key] for em_key in emkeys])
                    self.dataset['em_fits_nitems_trecall_mean_arrays']['std'][n_items_i, trecall_i] = np.array([self.dataset['em_fits_nitems_trecall_mean']['std'][n_items][trecall][em_key] for em_key in emkeys])
                    self.dataset['em_fits_nitems_trecall_mean_arrays']['sem'][n_items_i, trecall_i] = self.dataset['em_fits_nitems_trecall_mean_arrays']['std'][n_items_i, trecall_i]/np.sqrt(self.dataset['subject_size'])

                    self.dataset['em_fits_nitems_trecall_arrays'][n_items_i, trecall_i] = np.array([self.dataset['em_fits_nitems_trecall'][n_items_i, trecall_i][em_key] for em_key in emkeys])

                for subject_i, subject in enumerate(unique_subjects):
                    self.dataset['em_fits_subjects_nitems_arrays'][subject_i, n_items_i] = np.array([self.dataset['em_fits_subjects_nitems'][subject_i, n_items_i][em_key] for em_key in emkeys])


            # get some mean/std for nitems sequentially, averaging over subjects, not taking trecall into account
            self.dataset['em_fits_nitems_mean_arrays']['mean'] = np.mean(self.dataset['em_fits_subjects_nitems_arrays'], axis=0)
            self.dataset['em_fits_nitems_mean_arrays']['std'] = np.std(self.dataset['em_fits_subjects_nitems_arrays'], axis=0)
            self.dataset['em_fits_nitems_mean_arrays']['sem'] = self.dataset['em_fits_nitems_mean_arrays']['std']/np.sqrt(self.dataset['subject_size'])


