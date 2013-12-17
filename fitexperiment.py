#!/usr/bin/env python
# encoding: utf-8
"""
fitexperiment.py

Created by Loic Matthey on 2013-09-26.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import os

import numpy as np

import scipy as sp
import scipy.optimize as spopt
import scipy.stats as spst

import matplotlib.pyplot as plt

from datagenerator import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from datapbs import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *

import progress

import launchers

import load_experimental_data as loader_exp_data


class FitExperiment:
    '''
        Loads experimental data, set up DataGenerator and associated RFN, Sampler to optimize parameters
    '''

    def __init__(self, parameters={}):
        '''
            FitExperiment takes a parameters dict as input
            Specific fields:
            - experiment_id:  identifier for a specific experiment to fit. Automatically assigns a few variables that way
            - experiment_params:  extra parameters dict
        '''
        self.dataset_experiment = None

        # Experiment to fit
        self.experiment_id = parameters.get("experiment_id", "doublerecall")

        # Extra parameters
        self.experiment_params = parameters.get("experiment_params", dict(n_items_to_fit=3))

        self.parameters = parameters

        if self.parameters == {}:
            # Create default arguments dict
            self.parameters = {
                     'N': 100,
                     'K': 2,
                     'R': 2,
                     'M': 200,
                     'M_layer_one': 400,
                     'T': 3,
                     'alpha': 1.0,
                     'autoset_parameters': True,
                     'code_type': 'mixed',
                     'distribution_weights': 'exponential',
                     'enforce_first_stimulus': False,
                     'enforce_min_distance': 0.17,
                     'feat_ratio': 40.0,
                     'inference_method': 'none',
                     'input_filename': '',
                     'label': 'fitexperiment_mixed_ratiosigmax',
                     'normalise_weights': 1,
                     'num_repetitions': 1,
                     'num_samples': 100,
                     'output_both_layers': False,
                     'output_directory': 'Data/',
                     'parameters_filename': '',
                     'ratio_conj': 0.84,
                     'rc_scale': 4.0,
                     'rc_scale2': 0.4,
                     'selection_method': 'median',
                     'selection_num_samples': 1,
                     'sigma_weights': 1.0,
                     'sigmax': 0.1,
                     'sigmay': 0.0001,
                     'sparsity': 1.0,
                     'stimuli_generation_recall': 'random',
                     'use_theoretical_cov': False}

        # Load experimental data. Force some parameters to be used later while doing so
        experiment_data_dir = self.parameters.get('experiment_data_dir', '../../experimental_data/')
        self.load_experiment(data_dir=experiment_data_dir)


    def load_experiment(self, data_dir = '../../experimental_data/'):
        '''
            Load a specific human experiment.
        '''

        fit_mixturemodel = self.experiment_params.get('fit_mixturemodel', False)
        n_items_to_fit = self.experiment_params.get("n_items_to_fit", 3)

        if self.experiment_id == 'doublerecall':
            experiment_descriptor = dict(filename=os.path.join(data_dir, 'DualRecall_Bays', 'rate_data.mat'), preprocess=loader_exp_data.preprocess_doublerecall, parameters=dict(fit_mixturemodel=fit_mixturemodel))
            print "Using Double Recall dataset"
        elif self.experiment_id == 'gorgo_simult':
            experiment_descriptor = dict(filename='Exp2_withcolours.mat', preprocess=loader_exp_data.preprocess_simultaneous, parameters=dict(datadir=os.path.join(data_dir, 'Gorgoraptis_2011'), fit_mixturemodel=fit_mixturemodel))
            print "Using Gorgo simult dataset"

        # Load the data
        self.dataset_experiment = loader_exp_data.load_dataset(**experiment_descriptor)

        # Get shortcuts to the stimuli and responses, only actual things we use from the dataset
        self.data_stimuli = self.dataset_experiment['data_to_fit'][n_items_to_fit]['item_features']
        self.data_responses = self.dataset_experiment['data_to_fit'][n_items_to_fit]['response']

        # Force some parameters
        self.parameters['N'] = self.dataset_experiment['data_to_fit'][n_items_to_fit]['N']
        self.parameters['T'] = n_items_to_fit
        self.parameters['cued_feature_time'] = self.dataset_experiment['data_to_fit'][n_items_to_fit]['probe'][0]


    def instantiate_everything(self):
        '''
            Instantiate the objects with the appropriate parameters

            Don't forget to set the DataGenerator with the experimental data
        '''

        # Build the random network
        random_network = launchers.init_random_network(self.parameters)

        # Construct the real dataset
        time_weights_parameters = dict(weighting_alpha=self.parameters['alpha'], weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')

        # Specifically force the stimuli to be the human experimental ones
        data_gen = DataGeneratorRFN(self.parameters['N'], self.parameters['T'], random_network, sigma_y=self.parameters['sigmay'], sigma_x=self.parameters['sigmax'], time_weights_parameters=time_weights_parameters, cued_feature_time=self.parameters['cued_feature_time'], stimuli_generation=None, stimuli_to_use=self.data_stimuli)

        # Measure the noise structure, random stimuli here
        data_gen_noise = DataGeneratorRFN(5000, self.parameters['T'], random_network, sigma_y=self.parameters['sigmay'], sigma_x=self.parameters['sigmax'], time_weights_parameters=time_weights_parameters, cued_feature_time=self.parameters['cued_feature_time'], stimuli_generation=self.parameters['stimuli_generation_recall'])
        stat_meas = StatisticsMeasurer(data_gen_noise)

        self.sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=self.parameters['cued_feature_time'])


    def compute_likelihood_data_responses(self):
        '''
            Set the responses of the experimental data in the Sampler, and compute the loglikelihood under the current model
        '''

        # Set responses
        self.sampler.set_theta(self.data_responses)

        # Compute loglikelihood
        loglikelihood = self.sampler.compute_loglikelihood()

        # if self.parameters['verbose']:
        print("Loglikelihood: %s" % loglikelihood)

        return loglikelihood


    def estimate_likelihood_multiple_models(self, num_models=5):
        '''
            Reinstantiate the model {num_models} times, estimating the loglikelihood each time.

            This integrates out the noise in the model population code representation
        '''

        self.all_loglikelihood = np.empty(num_models)

        for curr_model_i in xrange(num_models):
            # Instantiate objects properly
            self.instantiate_everything()

            # Compute likelihood for the experimentally derived responses
            self.all_loglikelihood[curr_model_i] = self.compute_likelihood_data_responses()

        return np.mean(self.all_loglikelihood), np.std(self.all_loglikelihood)


    #####

    def plot_likelihood_misfit_datapoints(self, max_plots = 10):
        '''
            Plot the posterior distributions of the datapoints that are badly classified.
        '''

        # Set responses
        self.sampler.set_theta(self.data_responses)

        # Compute data loglikelihood
        data_llh = self.sampler.compute_loglikelihood_N()

        # Find misfit points
        data_llh_outliers = np.argsort(data_llh)[:max_plots]
        # data_llh_outliers = np.nonzero(data_llh <= (np.mean(data_llh) - 3.*np.std(data_llh)))[0]

        # Check where they are and their relation to the posterior distribution
        for outlier_i in data_llh_outliers:
            self.sampler.plot_likelihood_comparison(n=outlier_i)


    def plot_distribution_loglikelihoods(self, bins=50):
        '''
            Plot the distribution of data loglikelihoods obtained

            Can show the mean llh, and how many outliers they are
        '''

        # Set responses
        self.sampler.set_theta(self.data_responses)

        # Compute data loglikelihood
        data_llh = self.sampler.compute_loglikelihood_N()

        # Show the distribution of data llh as a boxplot and as the distribution directly
        _, axes = plt.subplots(1, 2)

        axes[0].boxplot(data_llh)
        axes[1].hist(data_llh, bins=bins)


    def plot_comparison_error_histograms(self, bins=50):
        '''
            Compare the distribution of errors of the human data,
            and from samples from the model
        '''

        _, axes = plt.subplots(2, 1)

        # Sample from model
        self.sampler.sample_theta(num_samples=100, burn_samples=100)

        # Plot distribution of samples first (so that we keep human data in theta)
        print 'Sampling...'
        self.sampler.plot_histogram_errors(bins=bins, ax_handle=axes[1])

        # Reput the data
        self.sampler.set_theta(self.data_responses)

        # Plot distribution of data
        self.sampler.plot_histogram_errors(bins=bins, ax_handle=axes[0])

        axes[0].set_title('Human data error distribution')
        axes[1].set_title('Model samples error distribution')

        # plt.hist(self.dataset_experiment['errors_angle_all'][self.dataset_experiment['3_items_trials'] & self.dataset_experiment['angle_trials'], 0], bins=50)

    ####

    def fit_parameter_pbs(self):
        '''
            Get the loglikelihood for a specific set of parameters

            Most likely inherited from a big PBS run
        '''
        # Compute likelihood for the experimentally derived responses
        return self.estimate_likelihood_multiple_models(num_models=self.parameters['num_repetitions'])



    def fit_parameter(self):
        '''
            Vary some parameters around and see how this affects the experimental responses loglikelihoods
        '''

        # param_space = (np.arange(1, 21, 1)**2.).astype(int)  # M
        # param_space = np.linspace(0.5, 10., 10)  # kappa
        param_space = np.linspace(0.01, 0.2, 20)

        # self.llh_fullspace_mean = np.empty(M_space.size)
        # self.llh_fullspace_std = np.empty(M_space.size)

        self.llh_fullspace_mean = np.empty(param_space.size)
        self.llh_fullspace_std = np.empty(param_space.size)

        # self.llh_fullspace_mean = np.empty((M_space.size, param_space.size))
        # self.llh_fullspace_std = np.empty((M_space.size, param_space.size))

        # search_progress = progress.Progress(M_space.size*param_space.size)
        search_progress = progress.Progress(param_space.size)

        # for i, M in enumerate(param_space):
        # for i, kappa in enumerate(param_space):
        for i, sigmax in enumerate(param_space):
            # if search_progress.percentage() % 5.0 < 0.0001:
            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            # print "Fit for M=%d" % M
            # print "Fit for kappa=%f" % kappa
            print "Fit for sigmax=%.2f" % sigmax

            # Update parameter
            # self.parameters['M'] = M
            # self.parameters['rc_scale'] = kappa
            self.parameters['sigmax'] = sigmax

            # Compute the loglikelihood
            self.llh_fullspace_mean[i], self.llh_fullspace_std[i] = self.estimate_likelihood_multiple_models(num_models=3)

            search_progress.increment()

        # Plot the result
        plot_mean_std_area(param_space, self.llh_fullspace_mean, self.llh_fullspace_std)
        # pcolor_2d_data(self.llh_fullspace_mean, M_space, param_space, "M", "kappa")


    def fit_parameters_2d(self):
        '''
            Vary some parameters around and see how this affects the experimental responses loglikelihoods
        '''

        M_space = (np.arange(1, 17, 2)**2.).astype(int)
        kappa_space = np.linspace(0.5, 20., 7)

        # self.llh_fullspace_mean = np.empty(M_space.size)
        # self.llh_fullspace_std = np.empty(M_space.size)

        # self.llh_fullspace_mean = np.empty(kappa_space.size)
        # self.llh_fullspace_std = np.empty(kappa_space.size)

        self.llh_fullspace_mean = np.empty((M_space.size, kappa_space.size))
        self.llh_fullspace_std = np.empty((M_space.size, kappa_space.size))

        search_progress = progress.Progress(M_space.size*kappa_space.size)

        for i, M in enumerate(M_space):
            for j, kappa in enumerate(kappa_space):
                # if search_progress.percentage() % 5.0 < 0.0001:
                print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                print "Fit for M=%d" % M
                print "Fit for kappa=%f" % kappa

                # Update parameter
                self.parameters['M'] = M
                self.parameters['rc_scale'] = kappa

                # Compute the loglikelihood
                self.llh_fullspace_mean[i, j], self.llh_fullspace_std[i, j] = self.estimate_likelihood_multiple_models(num_models=3)

                search_progress.increment()

        # Plot the result
        # plot_mean_std_area(M_space, self.llh_fullspace_mean, self.llh_fullspace_std)
        pcolor_2d_data(self.llh_fullspace_mean, M_space, kappa_space, "M", "kappa")


    def fit_parameters_mixed(self):
        '''
            Fit a mixed code, for ratio, sigmax, rc_scale, rc_scale2
        '''

        # variables_to_save = ['param1_space', 'param2_space', 'self']
        # dataio = DataIO(output_folder=self.parameters['output_directory'], label=self.parameters['label'])

        # param1_space = np.linspace(0.001, 1., 20)  # ratio conj
        # param2_space = np.linspace(0.01, 0.4, 20)  # sigmax
        param1_space = np.linspace(0.01, 10, 10)    # kappa conj
        param2_space = np.linspace(0.01, 50., 10)   # kappa feat

        self.llh_fullspace_mean = np.empty((param1_space.size, param2_space.size))
        self.llh_fullspace_std = np.empty((param1_space.size, param2_space.size))

        search_progress = progress.Progress(param1_space.size*param2_space.size)

        k = 0

        # for i, ratio_conj in enumerate(param1_space):
            # for j, sigmax in enumerate(param2_space):
        for i, rc_scale in enumerate(param1_space):
            for j, rc_scale2 in enumerate(param2_space):
                print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                # print "Fit for ratio=%.1f, sigmax=%.2f" % (ratio_conj, sigmax)
                print "Fit for rc_scale=%.2f, rc_scale2=%.2f" % (rc_scale, rc_scale2)

                # Update parameter
                # self.parameters['ratio_conj'] = ratio_conj
                # self.parameters['sigmax'] = sigmax
                # self.parameters['rc_scale'] = rc_scale
                # self.parameters['rc_scale2'] = rc_scale2

                # Compute the loglikelihood
                self.llh_fullspace_mean[i, j], self.llh_fullspace_std[i, j] = self.estimate_likelihood_multiple_models(num_models=self.parameters['num_repetitions'])

                search_progress.increment()

                # if k % 4 == 0:
                    # dataio.save_variables(variables_to_save, locals())

                k += 1

        # Plot the result
        # plot_mean_std_area(M_space, self.llh_fullspace_mean, self.llh_fullspace_std)
        # dataio.save_variables(variables_to_save, locals())
        # pcolor_2d_data(self.llh_fullspace_mean, param1_space, param2_space, "ratio_conj", "sigmax")
        # dataio.save_current_figure("fitexperiment_mixed_ratiosigmax_{unique_id}.pdf")





if __name__ == '__main__':
    fit_exp = FitExperiment()

