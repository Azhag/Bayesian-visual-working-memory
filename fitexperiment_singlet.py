#!/usr/bin/env python
# encoding: utf-8
"""
fitexperimentSingleT.py

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
import launchers
import load_experimental_data
# import utils


class FitExperimentSingleT:
    '''
        Loads experimental data, set up DataGenerator and associated RFN, Sampler to optimize parameters.

        This version expects a unique time/index of recall T and handles multiple datasets simultaneously.
    '''

    def __init__(self, sampler, parameters={}, debug=True):
        '''
            FitExperimentSingleT takes a sampler and a parameters dict as input
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
            data_gen = datagenerator.DataGeneratorRFN(self.experimental_datasets[experiment_id]['data_to_fit'][self.sampler.T]['N'], self.sampler.T, self.sampler.random_network, sigma_y=self.sampler.data_gen.sigma_y, sigma_x=self.sampler.data_gen.sigma_x, sigma_baseline=self.sampler.data_gen.sigma_baseline, renormalize_sigma=self.data_gen.renormalize_sigma, time_weights=self.sampler.time_weights, cued_feature_time=cued_feature_time, stimuli_to_use=self.experimental_datasets[experiment_id]['data_to_fit'][self.sampler.T]['item_features'])

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


    # def compute_bic_all_datasets(self, K=None):
    #     '''
    #         Compute the BIC scores for all datasets
    #     '''
    #     def compute_bic(sampler, parameters):
    #         bic = sampler.compute_bic(K=parameters['K'])
    #         return bic

    #     fct_infos = dict(fct=compute_bic, parameters=dict(K=K))

    #     return self.apply_fct_all_datasets(fct_infos)


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
            raise ValueError("This K for BIC is wrong here")
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


    def compute_bic_loglik_noise_convolved_all_datasets(self, precision=150):
        '''
            Compute BIC and LL, based on the
        '''

        def compute_everything(sampler, parameters):
            loglikelihood_conv_N = sampler.compute_loglikelihood_N_convolved_output_noise(precision=parameters['precision'])

            loglikelihood_conv = np.nansum(loglikelihood_conv_N)
            loglikelihood90_conv = sampler.compute_loglikelihood_top90percent(all_loglikelihoods=loglikelihood_conv_N)
            bic = sampler.compute_bic(LL=loglikelihood_conv)

            result = dict(LL=loglikelihood_conv, LL90=loglikelihood90_conv, bic=bic)

            return result

        fct_infos = dict(fct=compute_everything, parameters=dict(precision=precision))

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

    # Now let's build a FitExperimentSingleT
    parameters = dict(experiment_ids=['gorgo11', 'bays09','dualrecall'], fit_mixture_model=True)
    fit_exp = FitExperimentSingleT(sampler, parameters)

    # Now compute some loglikelihoods
    # print fit_exp.compute_loglik_all_datasets()
    # print fit_exp.compute_sum_loglik_all_datasets()

    # Compute BIC
    # print fit_exp.compute_bic_all_datasets()

    print fit_exp.compute_bic_loglik_all_datasets()

    return locals()


def test_loglike_fit():
    '''
        Check if the LL computation is correct

        Use specific data, generated from a given model. This model should then have max LL.
    '''

    # Get a specific model, with given ratio and sigmax
    experiment_parameters = dict(action_to_do='launcher_do_simple_run',
                                  inference_method='sample',
                                  T=2,
                                  M=200,
                                  N=400,
                                  num_samples=500,
                                  selection_method='last',
                                  sigmax=0.15,
                                  sigmay=0.0001,
                                  code_type='mixed',
                                  ratio_conj=0.6,
                                  output_directory='.',
                                  stimuli_generation_recall='random',
                                  autoset_parameters=None)
    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=experiment_parameters)
    experiment_parameters_full = experiment_launcher.args_dict
    sampler = experiment_launcher.all_vars['sampler']

    # Keep its dataset and responses
    stimuli_correct_to_force = sampler.data_gen.stimuli_correct.copy()
    response_to_force = sampler.theta[:, 0].copy()
    LL_target = sampler.compute_loglikelihood()

    experiment_parameters_full['stimuli_to_use'] = stimuli_correct_to_force

    ratio_space = np.linspace(0.0, 1.0, 31.)

    LL_all_new = np.zeros(ratio_space.shape)

    for ratio_conj_i, ratio_conj in enumerate(ratio_space):

        experiment_parameters_full['ratio_conj'] = ratio_conj

        _, _, _, sampler = launchers.init_everything(experiment_parameters_full)

        # Set responses
        sampler.set_theta(response_to_force)

        # Compute LL
        # LL_all_new[ratio_conj_i] = sampler.compute_loglikelihood()
        LL_all_new[ratio_conj_i] = sampler.compute_loglikelihood_top90percent()

        # Print result
        print LL_all_new[ratio_conj_i]

    print LL_target
    print ratio_space, LL_all_new
    print ratio_space[np.argmax(LL_all_new)]

    return locals()


def test_noiseoutput_loglike():
    '''
        Check if the LL computation given noise output is correct
    '''

    # Get a specific model, with given ratio and sigmax
    experiment_parameters = dict(action_to_do='launcher_do_simple_run',
                                  inference_method='none',
                                  T=2,
                                  M=200,
                                  N=400,
                                  num_samples=500,
                                  selection_method='last',
                                  sigmax=0.15,
                                  sigmay=0.0001,
                                  code_type='mixed',
                                  ratio_conj=0.6,
                                  output_directory='.',
                                  sigma_output=0.1,
                                  stimuli_generation_recall='random',
                                  autoset_parameters=None)
    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=experiment_parameters)
    sampler = experiment_launcher.all_vars['sampler']

    # Now let's build a FitExperimentSingleT
    parameters = dict(experiment_ids=['gorgo11', 'bays09'], fit_mixture_model=True)
    fit_exp = FitExperimentSingleT(sampler, parameters)

    if False:
        ## Check precision required for the convolved likelihood
        precision_space = np.linspace(50, 500, 7)
        convolved_ll = np.empty(precision_space.size)

        fit_exp.force_experimental_stimuli(experiment_id='bays09')

        for precision_i, precision in enumerate(precision_space):
            convolved_ll[precision_i] = fit_exp.sampler.compute_loglikelihood_convolved_output_noise(precision=precision)
        plt.plot(precision_space, convolved_ll, precision_space, np.ones(precision_space.size)*fit_exp.sampler.compute_loglikelihood())
        plt.legend(('Convolved', 'Classic'))
        plt.xlabel('Size of finite support')

    if True:
        # Now compute everything!
        logliks_nonoise = fit_exp.compute_bic_loglik_all_datasets()
        logliks_noise = fit_exp.compute_bic_loglik_noise_convolved_all_datasets()

        print "No noise: ", logliks_nonoise
        print "Noise convolved:", logliks_noise

    return locals()



if __name__ == '__main__':
    if False:
        all_vars = test_fit_experiment()
    if False:
        all_vars = test_loglike_fit()
    if True:
        all_vars = test_noiseoutput_loglike()


    for key, val in all_vars.iteritems():
        locals()[key] = val

