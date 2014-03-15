#!/usr/bin/env python
# encoding: utf-8
"""
fitexperiment.py

Created by Loic Matthey on 2013-09-26.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import os

import numpy as np

# import scipy as sp
# import scipy.optimize as spopt
# import scipy.stats as spst

import matplotlib.pyplot as plt

# import progress

import experimentlauncher
import datagenerator
# import launchers
import load_experimental_data
# import utils


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
                self.experimental_datasets[experiment_id] = load_experimental_data.load_data_bays09(data_dir=self.data_dir, fit_mixture_model=fit_mixture_model)

                if self.debug:
                    print "Loading Bays09 dataset"

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

    def apply_fct_dataset(self, experiment_id, fct_infos):
        '''
            Apply a function after having forced a specific dataset

            the function will be called as follows:

            result = fct_infos['fct'](sampler, fct_infos['parameters'])
        '''

        if self.sampler.T not in self.experimental_datasets[experiment_id]['data_to_fit']['n_items']:
            # This dataset does not have sampler.T items
            return np.nan

        # Set dataset
        self.force_experimental_stimuli(experiment_id)

        # Apply function
        result = fct_infos['fct'](self.sampler, fct_infos['parameters'])

        return result


    def apply_fct_all_datasets(self, fct_infos):
        '''
            Apply a function on all datasets

            result = fct_infos['fct'](sampler, fct_infos['parameters'])
        '''

        result_all = dict()
        for experiment_id in self.experimental_datasets:
            result_all[experiment_id] = self.apply_fct_dataset(experiment_id, fct_infos)

        return result_all


    def compute_bic_all_datasets(self, K=None):
        '''
            Compute the BIC scores for all datasets
        '''
        def compute_bic(sampler, parameters):
            bic = sampler.compute_bic(K=parameters['K'])
            return bic

        fct_infos = dict(fct=compute_bic, parameters=dict(K=K))

        return self.apply_fct_all_datasets(fct_infos)


    def compute_loglik_all_datasets(self):
        '''
            Compute the loglikelihood of the current sampler for all datasets currently loaded
        '''
        def compute_loglik(sampler, parameters):
            loglikelihood = sampler.compute_loglikelihood()
            return loglikelihood
        fct_infos = dict(fct=compute_loglik, parameters=None)

        return self.apply_fct_all_datasets(fct_infos)


    def compute_bic_loglik_all_datasets(self, K=None):
        '''
            Compute both the BIC and loglikelihood on all datasets
        '''
        def compute_bic(sampler, parameters):
            bic = sampler.compute_bic(K=parameters['K'])
            return bic
        def compute_loglik(sampler, parameters):
            loglikelihood = sampler.compute_loglikelihood()
            return loglikelihood
        def compute_loglik90percent(sampler, parameters):
            return sampler.compute_loglikelihood_top90percent()

        def compute_both(sampler, parameters):
            result = dict(bic=compute_bic(sampler, parameters), LL=compute_loglik(sampler, parameters), LL90=compute_loglik90percent(sampler, parameters))
            return result

        fct_infos = dict(fct=compute_both, parameters=dict(K=K))

        return self.apply_fct_all_datasets(fct_infos)


    def compute_sum_loglik_all_datasets(self):
        '''
            Compute the sum of the loglikelihood obtained on all datasets.
        '''

        loglike_all = self.compute_loglik_all_datasets()

        return np.nansum([val for key, val in loglike_all.iteritems()])



    #####

    def plot_model_human_loglik_comparison(self):
        '''
            Get the LL for human responses and model samples
            plot model wrt human, to see how they compare
        '''

        def compute_model_human_loglik(sampler, parameters):
            # First compute the human LL, which is easy as the responses are already set
            human_ll = sampler.compute_loglikelihood()

            # Now sample from the model
            sampler.sample_theta(num_samples=300, burn_samples=300, slice_width=0.07, selection_method='last')

            # Now recompute the LL
            model_ll = sampler.compute_loglikelihood()

            return dict(human=human_ll, model=model_ll)

        fct_infos = dict(fct=compute_model_human_loglik, parameters=None)

        return self.apply_fct_all_datasets(fct_infos)


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


    def plot_distribution_loglik(self, bins=50, enforce_response=False):
        '''
            Plot the distribution of data loglikelihoods obtained

            Can show the mean llh, and how many outliers they are
        '''

        # Set responses
        if enforce_response:
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
                                  T=1,
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
    parameters = dict(experiment_ids=['gorgo11', 'bays09','dualrecall'], fit_mixture_model=True)
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

