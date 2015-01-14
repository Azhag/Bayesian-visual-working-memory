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

        self.dataset['n_items_size'] = np.unique(self.dataset['n_items']).size
        self.dataset['subject_size'] = np.unique(self.dataset['subject']).size


        # Will remove delayed trials
        self.dataset['masked'] = self.dataset['delayed'] == 1

        # Compute additional errors, between the response and all items
        self.compute_all_errors()

        # Create arrays per subject
        self.create_subject_arrays()


        # Fit the mixture model, and save the responsibilities per datapoint.
        # self.fit_mixture_model()

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
        self.dataset['errors_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_all_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_nontarget_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['precision_nitems_trecall_bays'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))
        self.dataset['precision_nitems_trecall_theo'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))
        self.dataset['precision_nitems_trecall_theo_nochance'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))
        self.dataset['precision_nitems_trecall_bays_notreatment'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))



        for n_items_i, n_items in enumerate(unique_n_items):
            for subject_i, subject in enumerate(unique_subjects):
                for trecall_i, trecall in enumerate(np.arange(1, n_items+1)):
                    ids_filtered = ((self.dataset['subject']==subject) & (self.dataset['n_items'] == n_items) & (self.dataset['probe'] == trecall) & (self.dataset.get('masked', False) == False)).flatten()

                    # Invert the order of storage, 0 -> last item probed, 1 -> second to last item probe, etc...
                    trecall_i = n_items - trecall

                    # Get the errors
                    self.dataset['errors_all_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['errors_all'][ids_filtered]

                    self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['errors_nontarget_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.extract_target_nontargets_columns(self.dataset['errors_all_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], trecall)


                    # Get the responses and correct item angles
                    self.dataset['response_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['response'][ids_filtered]
                    self.dataset['item_angle_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['item_angle'][ids_filtered]

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


        # self.dataset['errors_nitems_trecall'] = np.array([utils.flatten_list(self.dataset['errors_subject_nitems_trecall'][:, n_item_i]) for n_item_i in xrange(unique_n_items.size)])


        # Average subjects away
        for n_items_i, n_items in enumerate(unique_n_items):
            for trecall in np.arange(n_items):
                self.dataset['errors_nitems_trecall'][n_items_i, trecall] = np.array(utils.flatten_list(self.dataset['errors_subject_nitems_trecall'][:, n_items_i, trecall]))
                self.dataset['errors_all_nitems_trecall'][n_items_i, trecall] = np.array(utils.flatten_list(self.dataset['errors_all_subject_nitems_trecall'][:, n_items_i, trecall]))
                self.dataset['errors_nontarget_nitems_trecall'][n_items_i, trecall] = np.array(utils.flatten_list(self.dataset['errors_nontarget_subject_nitems_trecall'][:, n_items_i, trecall]))

                self.dataset['precision_nitems_trecall_bays'][n_items_i, trecall] = self.compute_precision(self.dataset['errors_nitems_trecall'][n_items_i, trecall], remove_chance_level=True, correct_orientation=True, use_wrong_precision=True)

                # self.dataset['precision_nitems_trecall_bays'] = np.mean(self.dataset['precision_subject_nitems_trecall_bays'], axis=0)
                self.dataset['precision_nitems_trecall_theo'] = np.mean(self.dataset['precision_subject_nitems_trecall_theo'], axis=0)
                self.dataset['precision_nitems_trecall_theo_nochance'] = np.mean(self.dataset['precision_subject_nitems_trecall_theo_nochance'], axis=0)
                self.dataset['precision_nitems_trecall_bays_notreatment'] = np.mean(self.dataset['precision_subject_nitems_trecall_bays_notreatment'], axis=0)



    def compute_vtest(self):
        self.dataset['vtest_nitems'] = np.empty(self.dataset['n_items_size'])*np.nan
        for n_items_i, n_items in enumerate(np.unique(self.dataset['n_items'])):
            if n_items > 1:
                self.dataset['vtest_nitems'][n_items_i] = utils.V_test(utils.dropnan(self.dataset['errors_nontarget_nitems'][n_items_i]).flatten())['pvalue']


