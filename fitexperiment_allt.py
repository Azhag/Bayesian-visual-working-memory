#!/usr/bin/env python
# encoding: utf-8
"""
fitexperimentAllT.py

Created by Loic Matthey on 2013-09-26.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import os

import numpy as np

# import scipy as sp
# import scipy.optimize as spopt
# import scipy.stats as spst

# import progress

import experimentlauncher
import launchers
import load_experimental_data
import utils


class FitExperimentAllT:
    '''
        Loads experimental data, set up DataGenerator and associated RFN, Sampler to optimize parameters.

        This version loads a unique dataset and will automatically run processings over all the possible nitems in it.
    '''

    def __init__(self, parameters={}, debug=True):
        '''
            FitExperimentAllT takes a parameters dict, same as a full launcher_

            Will then instantiate a Sampler and force a specific DataGenerator with constrained data from human experimental data.

            Requires experiment_id to be set.
        '''

        self.parameters = parameters
        self.debug = debug

        self.experiment_id = parameters.get('experiment_id', '')
        self.data_dir = parameters.get('experiment_data_dir',
                                       os.path.normpath(os.path.join(
                                           os.environ['WORKDIR_DROP'],
                                           '../../experimental_data/'))
                                       )

        self.experimental_dataset = load_experimental_data.load_data(experiment_id=self.experiment_id, data_dir=self.data_dir, fit_mixture_model=True)
        self.experiment_data_to_fit = self.experimental_dataset['data_to_fit']
        self.T_space = self.experiment_data_to_fit['n_items']

        self.all_samplers = dict()
        self.enforced_T = -1
        self.sampler = None

        self.num_datapoints = int(self.experiment_data_to_fit['N_smallest'])

        # Handle limiting the number of datapoints
        self.init_filter_datapoints()

        if self.debug:
            print "Loaded %s dataset. %d datapoints" % ((self.experiment_id, self.num_datapoints))


    def init_filter_datapoints(self):
        '''
            To speed things up, we may want to limit how many datapoints we actually use (per T/n_items).

            Check in the parameters dict:
            1) filter_datapoints_size [float] [if <= 1, treated as percent of total dataset size for given item]
            2) filter_datapoints_selection: {random, sequential}
            3) filter_datapoints_mask [array] direct mask to use. (If another FitExperiment already exist?)
        '''

        if 'filter_datapoints_mask' in self.parameters:
            self.filter_datapoints_mask = self.parameters['filter_datapoints_mask']
            self.num_datapoints = self.filter_datapoints_mask.size
        elif self.parameters.get('filter_datapoints_size', -1) > 0:
            selection_method = self.parameters.get('filter_datapoints_selection', 'sequential')
            selection_size = self.parameters.get('filter_datapoints_size', 1.)

            if selection_method == 'sequential':
                if selection_size > 1:
                    self.filter_datapoints_mask = np.arange(min(self.num_datapoints, int(selection_size)))
                else:
                    self.filter_datapoints_mask = np.arange(np.floor(self.num_datapoints*selection_size))
            elif selection_method == 'random':
                if selection_size > 1:
                    self.filter_datapoints_mask = np.random.permutation(np.arange(self.num_datapoints))[:int(selection_size)]
                else:
                    self.filter_datapoints_mask = np.random.permutation(np.arange(self.num_datapoints))[:np.floor(self.num_datapoints*selection_size)]

            self.num_datapoints = self.filter_datapoints_mask.size
        else:
            self.filter_datapoints_mask = slice(0, self.num_datapoints)


    def setup_experimental_stimuli_T(self, T):
        '''
            Setup everything needed (Sampler, etc) and then force a human experimental dataset.

            If already setup correctly, do nothing.
        '''

        assert T in self.T_space, "T=%d not possible. %s" % (T, self.T_space)

        if self.enforced_T != T:
            self.enforced_T = T

            if T not in self.all_samplers:
                print "\n>>> Setting up {} nitems, {} datapoints".format(T, self.num_datapoints)

                # Update parameters
                self.parameters['T'] = T
                self.parameters['N'] = self.num_datapoints
                self.parameters['fixed_cued_feature_time'] = self.experiment_data_to_fit[T]['probe'][0]  # should be scalar

                self.parameters['stimuli_to_use'] = self.experiment_data_to_fit[T]['item_features'][self.filter_datapoints_mask]

                # Instantiate everything
                (_, _, _, newSampler) = launchers.init_everything(self.parameters)

                # Fix responses to the human ones
                newSampler.set_theta(self.experiment_data_to_fit[T]['response'][self.filter_datapoints_mask])

                # Store it
                self.all_samplers[self.enforced_T] = newSampler

            self.sampler = self.all_samplers[self.enforced_T]


    def apply_fct_dataset_T(self, T, fct_infos):
        '''
            Apply a function after having forced a specific dataset
            The function will be called as follows:

            result = fct_infos['fct'](self, fct_infos['parameters'])
        '''

        # Set dataset
        self.setup_experimental_stimuli_T(T)

        # Apply function
        result = fct_infos['fct'](self, fct_infos.get('parameters', {}))

        return result


    def apply_fct_datasets_allT(self, fct_infos, return_array=False):
        '''
            Apply a function on all datasets

            TODO(lmatthey) if setting up dataset is costly, might need to provide a list of fcts and avoid reconstructing the DataGenerator each time.

            result = fct_infos['fct'](self, fct_infos['parameters'])
        '''

        result_allT = []
        for T in self.T_space:
            result_allT.append(self.apply_fct_dataset_T(T, fct_infos))

        if return_array:
            # Bit stupid for now. Might want to handle list of dictionaries, but meh
            result_allT_array = np.array([res for res in result_allT])

            return result_allT_array
        else:
            return result_allT


    def compute_dist_experimental_em_fits_currentT(self, model_fits):
        '''
            Given provided model_fits array, compute the distance to
            the loaded Experimental Data em fits.

            Inputs:
            * model_fits:   [array 6x1] ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'bic']


            Returns dict:
            * kappa MSE
            * mixture prop MSE
            * summed MSE
            * mixture prop KL divergence
        '''

        distances = dict()

        # TODO (lmatthey): use the EM Fits on the actual current subset here, instead of the full dataset!
        data_mixture_means = self.experimental_dataset['em_fits_nitems_arrays']['mean']

        curr_T_i = np.nonzero(self.T_space == self.enforced_T)[0][0]
        distances['all_mse'] = (data_mixture_means[:4, curr_T_i] - model_fits[:4])**2.
        distances['mixt_kl'] = utils.KL_div(data_mixture_means[1:4, curr_T_i], model_fits[1:4])

        return distances


    def recompute_em_fits_current_dataset(self):
        '''
            When we filter how data is used by our model, we cannot use the LL and BIC
            of the groundtruth EM Fits (also the parameters may be different...)

            So refit the EM mixture model on this subset and check how good it is.

            TODO(lmatthey) Should handle multiple subjects though...
        '''

        # First do the collapsed, it's easier to call...
        # params_fit = em_circularmixture_parametrickappa.fit(self.T_space, subject_data_dict['responses'], subject_data_dict['targets'], subject_data_dict['nontargets'], debug=False)

        # self.experiment_data_to_fit[T]['item_features'][self.filter_datapoints_mask]

        # params_fit = em_circularmixture_allitems_uniquekappa.fit(self.dataset['response'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 1:])



###########################################################################


def test_fit_experimentallt():

    # Set some parameters and let the others default
    experiment_parameters = dict(action_to_do='launcher_do_simple_run',
                                 inference_method='none',
                                 experiment_id='bays09',
                                 M=100,
                                 filter_datapoints_size=500,
                                 filter_datapoints_selection='random',
                                 num_samples=500,
                                 selection_method='last',
                                 sigmax=0.1,
                                 renormalize_sigmax=None,
                                 sigmay=0.0001,
                                 code_type='mixed',
                                 slice_width=0.07,
                                 burn_samples=200,
                                 ratio_conj=0.7,
                                 stimuli_generation_recall='random',
                                 autoset_parameters=None,
                                 label='test_fit_experimentallt'
                                 )
    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=experiment_parameters)
    experiment_parameters_full = experiment_launcher.args_dict

    # Now let's build a FitExperimentAllT
    fit_exp = FitExperimentAllT(experiment_parameters_full)

    # Now compute some loglikelihoods
    fct_infos = dict(fct=lambda s, p: s.sampler.compute_loglikelihood_N())

    ll_n = fit_exp.apply_fct_datasets_allT(fct_infos, return_array=True)
    print ll_n

    # print fit_exp.compute_loglik_all_datasets()
    # print fit_exp.compute_sum_loglik_all_datasets()

    # Compute BIC
    # print fit_exp.compute_bic_all_datasets()

    return locals()


def test_loglike_modelselection():
    '''
        Check if the LL computation is correct for model selection

        Use specific data, generated from a given model. This model should then have max LL.
    '''

    # Set some parameters and let the others default
    experiment_parameters = dict(action_to_do='launcher_do_simple_run',
                                 inference_method='sample',
                                 experiment_id='bays09',
                                 M=100,
                                 N=500,
                                 filter_datapoints_size=500,
                                 filter_datapoints_selection='random',
                                 num_samples=500,
                                 selection_method='last',
                                 sigmax=0.1,
                                 sigma_output=0.5,
                                 renormalize_sigmax=None,
                                 sigmay=0.0001,
                                 code_type='mixed',
                                 slice_width=0.07,
                                 burn_samples=200,
                                 ratio_conj=0.7,
                                 stimuli_generation_recall='random',
                                 autoset_parameters=None,
                                 label='test_fit_experimentallt'
                                 )
    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=experiment_parameters)
    experiment_parameters_full = experiment_launcher.args_dict
    sampler = experiment_launcher.all_vars['sampler']

    # Keep its dataset and responses
    stimuli_correct_to_force = sampler.data_gen.stimuli_correct.copy()
    response_to_force = sampler.theta[:, 0].copy()
    LL_target = sampler.compute_loglikelihood()

    experiment_parameters_full['stimuli_to_use'] = stimuli_correct_to_force

    sigmaoutput_space = np.linspace(0.0, 1.0, 10)

    LL_all_new = np.empty(sigmaoutput_space.size)
    LL_all_conv_new = np.empty(sigmaoutput_space.size)

    for sigmaout_i, sigma_output in enumerate(sigmaoutput_space):

        experiment_parameters_full['sigma_output'] = sigma_output

        _, _, _, samplerbis = launchers.init_everything(experiment_parameters_full)

        # Set responses
        samplerbis.set_theta(response_to_force)

        # Compute LL
        LL_all_new[sigmaout_i] = samplerbis.compute_loglikelihood()
        LL_all_conv_new[sigmaout_i] = samplerbis.compute_loglikelihood_convolved_output_noise()


        # Print result
        print LL_all_new[sigmaout_i], LL_all_conv_new[sigmaout_i]

    print LL_target
    print sigma_output, LL_all_new, LL_all_conv_new
    print sigmaoutput_space[np.argmax(LL_all_new)]
    print sigmaoutput_space[np.argmax(LL_all_conv_new)]

    return locals()


def test_noiseoutput_loglike():
    '''
        Check if the LL computation given noise output is correct
    '''

    # Set some parameters and let the others default
    experiment_parameters = dict(action_to_do='launcher_do_simple_run',
                                 inference_method='none',
                                 experiment_id='bays09',
                                 M=100,
                                 filter_datapoints_size=500,
                                 filter_datapoints_selection='random',
                                 num_samples=500,
                                 selection_method='last',
                                 sigmax=0.1,
                                 sigma_output=0.5,
                                 renormalize_sigmax=None,
                                 sigmay=0.0001,
                                 code_type='mixed',
                                 slice_width=0.07,
                                 burn_samples=200,
                                 ratio_conj=0.7,
                                 stimuli_generation_recall='random',
                                 autoset_parameters=None,
                                 label='test_fit_experimentallt'
                                 )
    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=experiment_parameters)
    experiment_parameters_full = experiment_launcher.args_dict

    # Now let's build a FitExperimentAllT
    fit_exp = FitExperimentAllT(experiment_parameters_full)

    # Now compute some loglikelihoods
    def compute_stats(self, params):
        loglik_N = self.sampler.compute_loglikelihood_N()
        loglik_conv_N = self.sampler.compute_loglikelihood_N_convolved_output_noise(precision=100)
        return np.array([loglik_N, loglik_conv_N])


    fct_infos = dict(fct=compute_stats)

    ll_outs = fit_exp.apply_fct_datasets_allT(fct_infos, return_array=True)
    print ll_outs.shape

    print "No noise: ", ll_outs[:, 0]
    print "Noise convolved:", ll_outs[:, 1]

    return locals()



if __name__ == '__main__':
    if True:
        all_vars = test_fit_experimentallt()
    if False:
        all_vars = test_loglike_modelselection()
    if False:
        all_vars = test_noiseoutput_loglike()


    for key, val in all_vars.iteritems():
        locals()[key] = val

