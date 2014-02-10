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

import progress

import experimentlauncher
import datagenerator
import launchers
import load_experimental_data
import utils


class FitExperiment:
    '''
        Loads experimental data, set up DataGenerator and associated RFN, Sampler to optimize parameters
    '''

    def __init__(self, sampler, parameters={}, debug=True):
        '''
            FitExperiment takes a sampler and a parameters dict as input
            Specific fields:
            - experiment_ids:  list of identifiers for experiments to fit.
            - experiment_params:  extra parameters dict
        '''
        self.sampler = sampler
        self.parameters = parameters
        self.debug = debug

        self.experiment_ids = parameters.get('experiment_ids', [])
        self.data_dir = parameters.get('experiment_data_dir', os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data/')))

        self.experimental_datasets = dict()
        self.experiment_enforced = ''
        self.data_responses = None

        # Load experimental data
        self.load_experiments()


    def load_experiments(self):
        '''
            Load a specific human experiment.
        '''

        fit_mixture_model = self.parameters.get('fit_mixturemodel', False)

        # Load each experimental dataset
        for experiment_id in self.experiment_ids:
            if experiment_id == 'bays09':
                self.experimental_datasets[experiment_id] = load_experimental_data.load_data_bays2009(data_dir=self.data_dir, fit_mixture_model=fit_mixture_model)

                if self.debug:
                    print "Loading Gorgo11 simult dataset"
            elif experiment_id == 'dualrecall':
                self.experimental_datasets[experiment_id] = load_experimental_data.load_data_dualrecall(data_dir=self.data_dir)

                if self.debug:
                    print "Loading double Recall dataset"

            elif experiment_id == 'gorgo11':
                self.experimental_datasets[experiment_id] = load_experimental_data.load_data_simult(data_dir=self.data_dir, fit_mixture_model=fit_mixture_model)

                if self.debug:
                    print "Loading Gorgo11 simult dataset"


    def force_experimental_stimuli(self, experiment_id):
        '''
            Will create a DataGenerator that uses the experimental data, in order to evaluate the model.
        '''

        print "Using {} dataset".format(experiment_id)

        if self.experiment_enforced != experiment_id:

            # Use this cued feature time (should be scalar)
            cued_feature_time = self.experimental_datasets[experiment_id]['data_to_fit'][self.sampler.T]['probe'][0]

            # Specifically force the stimuli to be the human experimental ones
            data_gen = datagenerator.DataGeneratorRFN(self.experimental_datasets[experiment_id]['data_to_fit'][self.sampler.T]['N'], self.sampler.T, self.sampler.random_network, sigma_y=self.sampler.data_gen.sigma_y, sigma_x=self.sampler.data_gen.sigma_x, time_weights=self.sampler.time_weights, cued_feature_time=cued_feature_time, stimuli_to_use=self.experimental_datasets[experiment_id]['data_to_fit'][self.sampler.T]['item_features'])

            # Use this new Data_gen, reinit a few things, hopefully it works..
            self.sampler.init_from_data_gen(data_gen, tc=cued_feature_time)

            # Set responses
            self.data_responses = self.experimental_datasets[experiment_id]['data_to_fit'][self.sampler.T]['response']
            self.sampler.set_theta(self.data_responses)

            self.experiment_enforced = experiment_id


    #####

    def compute_loglik_dataset(self, experiment_id):
        '''
            Set the responses of the experimental data in the Sampler to the provided dataset id, and compute the loglikelihood under the current model.

            Use self.sampler.T as the number of items
        '''

        if self.sampler.T not in self.experimental_datasets[experiment_id]['data_to_fit']['n_items']:
            # This dataset does not have sampler.T items
            return np.nan

        # Set dataset
        self.force_experimental_stimuli(experiment_id)

        # Compute loglikelihood
        loglikelihood = self.sampler.compute_loglikelihood()

        if self.debug:
            print "{0}> Loglikelihood: {1}".format(experiment_id, loglikelihood)

        return loglikelihood


    def compute_bic_dataset(self, experiment_id, K=None):
        '''
            Compute the BIC score of a model given a specific dataset

            If K is not set, assume K= (M, sigmax, ratio_conj if mixed)
        '''

        if self.sampler.T not in self.experimental_datasets[experiment_id]['data_to_fit']['n_items']:
            # This dataset does not have sampler.T items
            return np.nan

        # Set dataset
        self.force_experimental_stimuli(experiment_id)

        # Compute bic
        bic = self.sampler.compute_bic(K=K)

        if self.debug:
            print "{0}> BIC: {1}".format(experiment_id, bic)

        return bic


    def compute_loglik_all_datasets(self):
        '''
            Compute the loglikelihood of the current sampler for all datasets currently loaded
        '''

        loglikelihood_all = dict()
        for experiment_id in self.experimental_datasets:
            loglikelihood_all[experiment_id] = self.compute_loglik_dataset(experiment_id)

        return loglikelihood_all


    def compute_sum_loglik_all_datasets(self):
        '''
            Compute the sum of the loglikelihood obtained on all datasets.
        '''

        loglike_all = self.compute_loglik_all_datasets()

        return np.nansum([val for key, val in loglike_all.iteritems()])


    def compute_bic_all_datasets(self):
        '''
            Compute the BIC scores for all datasets
        '''
        bic_all = dict()
        for experiment_id in self.experimental_datasets:
            bic_all[experiment_id] = self.compute_bic_dataset(experiment_id)

        return bic_all


    def compute_bic_loglik_datasets(self, experiment_id):
        '''
            Compute both BIC and Loglikelihood
        '''
        result = dict()

        result['bic'] = self.compute_bic_dataset(experiment_id)
        result['LL'] = self.compute_loglik_dataset(experiment_id)

        return result


    def compute_bic_loglik_all_datasets(self):
        '''
            Compute both BIC and Loglikelihood for all datasets
        '''

        result_all = dict()
        for experiment_id in self.experimental_datasets:
            result_all[experiment_id] = self.compute_bic_loglik_datasets(experiment_id)

        return result_all

    #####

    def plot_loglik_misfit_datapoints(self, max_plots = 10):
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
            print outlier_i
            self.sampler.plot_likelihood_comparison(n=outlier_i)


    def plot_distribution_loglik(self, bins=50):
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
        if self.debug:
            print 'Sampling...'
        self.sampler.plot_histogram_errors(bins=bins, ax_handle=axes[1])

        # Reput the data
        self.sampler.set_theta(self.data_responses)

        # Plot distribution of data
        self.sampler.plot_histogram_errors(bins=bins, ax_handle=axes[0])

        axes[0].set_title('Human data error distribution')
        axes[1].set_title('Model samples error distribution')

        # plt.hist(self.dataset_experiment['errors_angle_all'][self.dataset_experiment['3_items_trials'] & self.dataset_experiment['angle_trials'], 0], bins=50)


def test_fit_experiment():

    # Load a sampler
    experiment_parameters = dict(action_to_do='launcher_do_simple_run',
                                  inference_method='none',
                                  T=3,
                                  M=200,
                                  N=200,
                                  num_samples=500,
                                  selection_method='last',
                                  sigmax=0.15,
                                  sigmay=0.0001,
                                  code_type='mixed',
                                  ratio_conj=0.8,
                                  output_directory='.',
                                  autoset_parameters=None)
    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=experiment_parameters)

    sampler = experiment_launcher.all_vars['sampler']

    # Now let's build a FitExperiment
    parameters = dict(experiment_ids=['gorgo11', 'dualrecall'], fit_mixture_model=True)
    fit_exp = FitExperiment(sampler, parameters)

    # Now compute some loglikelihoods
    # print fit_exp.compute_loglik_all_datasets()
    # print fit_exp.compute_sum_loglik_all_datasets()

    # Compute BIC
    # print fit_exp.compute_bic_all_datasets()

    print fit_exp.compute_bic_loglik_all_datasets()

    return locals()



if __name__ == '__main__':
    if True:
        all_vars = test_fit_experiment()


    for key, val in all_vars.iteritems():
        locals()[key] = val

