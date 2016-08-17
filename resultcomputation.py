#!/usr/bin/env python
# encoding: utf-8
"""
resultcomputations.py

Created by Loic Matthey on 2014-01-17
Copyright (c) 2014 . All rights reserved.
"""

import numpy as np
import load_experimental_data
import utils
import scipy.stats.mstats as mstats


class ResultComputation():
    """
        Class computing some result out of arrays.

        This is called from JobWrapper at the end of its compute() function. It receives all arrays created by an ExperimentLauncher. So use meaningful and constant variable names if you want to reuse different ResultComputations

        In generator_*, should specify the type of ResultComputation to use as an argument, everything will be instantiated appropriately later.

    """

    def __init__(self, computation_name, debug=True):

        self.computation_fct = None
        self.computation_name = None
        self.debug = debug

        # Check that the computation name is correct
        self.check_set_computation(computation_name)

        if self.debug:
            print "ResultComputation initialised\n > %s" % (self.computation_name)

    def __str__(self):
        '''
            Write which ResultComputation you are
        '''
        return 'ResultComputation %s' % self.computation_name


    def check_set_computation(self, computation_name):
        '''
            Look at the functions defined here, and verify that one matches the computation_name provided.

            Looks for compute_result_{computation_name}()
        '''

        # Duck-typing check it
        try:
            fct_found = getattr(self, "compute_result_%s" % computation_name)

            # All good.
            self.computation_name = computation_name
            self.computation_fct = fct_found

        except AttributeError:
            raise ValueError('ResultComputation %s not implemented' % computation_name)


    def compute_result(self, all_variables):
        '''
            Dispatching method for different compute_result_* functions.

            self.computation_name can be:
                - distemfits: looks at result_em_fits, computes the distance to specific datasets.

            TODO Find how to handle parameters... Should either load them from a file, or provide them as ExperimentLauncher arguments, but this is slightly tedious
        '''

        return self.computation_fct(all_variables)



    ##########################################################################

    def compute_result_random(self, all_variables):
        '''
            Dummy result computation, where you just return a random value
        '''

        return np.random.rand()


    def compute_result_distemfits_dataset(self, all_variables, experiment_id='bays09', cache_array_name='result_dist_bays09', variable_selection_slice=slice(None, 4), variable_selection_slice_cache=slice(None, None), metric='mse'):
        '''
            Result is the distance (sum squared) to experimental data fits

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
                - metric = 'mse' | 'kl'
                - variable_selection_slice: slice(0, 1) for kappa, slice(1, 4) for mixt proportions, slice(none, 4) for all EM params
        '''

        assert metric == 'mse' or metric == 'kl', "Metric should be {mse, kl}"

        repetitions_axis = all_variables.get('repetitions_axis', -1)

        if cache_array_name in all_variables:
            # If already computed, use it
            result_dist_allT = utils.nanmean(all_variables[cache_array_name][:, variable_selection_slice_cache], axis=repetitions_axis)

        elif 'result_em_fits' in all_variables:
            # Do some annoying slice manipulation
            slice_valid_indices = variable_selection_slice.indices(all_variables['result_em_fits'].shape[1])

            # Create output variables
            if metric == 'mse':
                result_dist_allT = np.nan*np.empty((all_variables['T_space'].size, slice_valid_indices[1] - slice_valid_indices[0]))
            elif metric == 'kl':
                result_dist_allT = np.nan*np.empty((all_variables['T_space'].size))

            ### Result computation
            if experiment_id == 'bays09':
                data_loaded = load_experimental_data.load_data_bays09(fit_mixture_model=True)
            elif experiment_id == 'gorgo11':
                data_loaded = load_experimental_data.load_data_gorgo11(fit_mixture_model=True)
            else:
                raise ValueError('wrong experiment_id {}'.format(experiment_id))

            experimental_mixtures_mean = data_loaded['em_fits_nitems_arrays']['mean']
            experimental_T_space = np.unique(data_loaded['n_items'])
            curr_result = np.nan

            for T_i, T in enumerate(all_variables['T_space']):
                if T in experimental_T_space:
                    if metric == 'mse':
                        curr_result = (experimental_mixtures_mean[variable_selection_slice, experimental_T_space == T] - all_variables['result_em_fits'][T_i, variable_selection_slice])**2.
                    elif metric == 'kl':
                        curr_result = utils.KL_div(all_variables['result_em_fits'][T_i, variable_selection_slice], experimental_mixtures_mean[variable_selection_slice, experimental_T_space==T], axis=0)

                    result_dist_allT[T_i] = utils.nanmean(curr_result, axis=repetitions_axis)
        else:
            raise ValueError('array {}/result_em_fits not present, bad'.format(cache_array_name))

        print result_dist_allT

        # return the overall distance, over all parameters and number of items
        return np.nansum(result_dist_allT)


    def compute_result_distemfits_bays09(self, all_variables):
        '''
            Result is the distance (sum squared) to experimental data fits

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''

        return self.compute_result_distemfits_dataset(all_variables, experiment_id='bays09', cache_array_name='result_dist_bays09', variable_selection_slice=slice(None, 4), metric='mse')

    def compute_result_distemfits_gorgo11(self, all_variables):
        '''
            Result is the distance (sum squared) to experimental data fits

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''

        return self.compute_result_distemfits_dataset(all_variables, experiment_id='gorgo11', cache_array_name='result_dist_gorgo11', variable_selection_slice=slice(None, 4), variable_selection_slice_cache=slice(None, 4), metric='mse')


    def compute_result_distemkappa_bays09(self, all_variables):
        '''
            Result is the distance (sum squared) to experimental data kappa

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''

        return self.compute_result_distemfits_dataset(all_variables, experiment_id='bays09', cache_array_name='result_dist_bays09', variable_selection_slice=slice(0, 1), variable_selection_slice_cache=slice(0, 1), metric='mse')

    def compute_result_distemkappalog_bays09(self, all_variables):
        '''
            Compute the distance to the experimental data kappa, but now take the log afterwards.

            Should be better behaved.

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''
        return np.log(self.compute_result_distemkappa_bays09(all_variables))

    def compute_result_distemkappa_gorgo11(self, all_variables):
        '''
            Result is the distance (sum squared) to experimental data kappa

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''

        return self.compute_result_distemfits_dataset(all_variables, experiment_id='gorgo11', cache_array_name='result_dist_gorgo11', variable_selection_slice=slice(0, 1), variable_selection_slice_cache=slice(0, 1), metric='mse')

    def compute_result_distemkappalog_gorgo11(self, all_variables):
        '''
            Compute the distance to the experimental data kappa, but now take the log afterwards.

            Should be better behaved.

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''
        return np.log(self.compute_result_distemkappa_gorgo11(all_variables))


    def compute_result_distemmixtKL_bays09(self, all_variables):
        '''
            Result is the distance (KL) to experimental mixture proportions

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''

        return self.compute_result_distemfits_dataset(all_variables, experiment_id='bays09', cache_array_name='result_dist_bays09_emmixt_KL', variable_selection_slice=slice(1, 4), variable_selection_slice_cache=slice(None, None), metric='kl')


    def compute_result_distemmixtKL_gorgo11(self, all_variables):
        '''
            Result is the distance (KL) to experimental mixture proportions

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''

        return self.compute_result_distemfits_dataset(all_variables, experiment_id='gorgo11', cache_array_name='result_dist_gorgo11_emmixt_KL', variable_selection_slice=slice(1, 4), variable_selection_slice_cache=slice(None, None), metric='kl')


    def compute_result_distem_logkappa_mixtKL_bays09(self, all_variables, normaliser_logkappa=1.0, normaliser_mixtKL=1.0):
        '''
            Result is the sum of the emkappa_log and emmixtKL distances.

            Should be normalized later, not sure how to pass it on.

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''
        return self.compute_result_distemkappalog_bays09(all_variables)/normaliser_logkappa + self.compute_result_distemmixtKL_bays09(all_variables)/normaliser_mixtKL

    def compute_result_distem_logkappa_mixtKL_gorgo11(self, all_variables, normaliser_logkappa=1.0, normaliser_mixtKL=1.0):
        '''
            Result is the sum of the emkappa_log and emmixtKL distances.

            Should be normalized later, not sure how to pass it on.

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''
        return self.compute_result_distemkappalog_gorgo11(all_variables)/normaliser_logkappa + self.compute_result_distemmixtKL_gorgo11(all_variables)/normaliser_mixtKL

    def compute_result_distfitexpbic(self, all_variables):
        '''
            Result is the summed BIC score of the FitExperiment result on all datasets
        '''

        if 'result_fitexperiments' in all_variables:
            # We have result_fitexperiments, that's good.

            # Make it work for either fixed T or multiple T.

            if len(all_variables['result_fitexperiments'].shape) == 2:
                # Single T. Average over axis -1
                bic_summed = utils.nanmean(all_variables['result_fitexperiments'][0])
            elif len(all_variables['result_fitexperiments'].shape) == 3:
                # Multiple T. Average over axis -1, then sum
                bic_summed = np.nansum(utils.nanmean(all_variables['result_fitexperiments'][:, 0], axis=-1))
            else:
                raise ValueError("wrong shape for result_fitexperiments: {}".format(all_variables['result_fitexperiments'].shape))
        else:
            # We do not have it, could instantiate a FitExperiment with "normal" parameters and work from there instead
            raise NotImplementedError('version without result_fitexperiments not implemented yet')

        return bic_summed


    def compute_result_distfit_bays09_bic(self, all_variables):
        '''
            Result is summed BIC score of FitExperiment to Bays09
        '''
        return self.compute_result_distfit_givenexp(all_variables, experiment_id='bays09', metric_index=0)

    def compute_result_distfit_gorgo11_bic(self, all_variables):
        '''
            Result is summed BIC score of FitExperiment to Gorgo11
        '''
        return self.compute_result_distfit_givenexp(all_variables, experiment_id='gorgo11', metric_index=0)


    def compute_result_distfit_bays09_ll90(self, all_variables):
        '''
            Result is summed negative LL, only top 90% each time.
        '''
        return -self.compute_result_distfit_givenexp(all_variables, experiment_id='bays09', metric_index=2)


    def compute_result_distfit_gorgo11_ll90(self, all_variables):
        '''
            Result is summed negative LL, only top 90% each time.
        '''
        return -self.compute_result_distfit_givenexp(all_variables, experiment_id='gorgo11', metric_index=2)


    def compute_result_distfit_givenexp(self, all_variables, experiment_id='bays09', metric_index=0, target_array_name='result_fitexperiments_all'):
        '''
            Result is the summed BIC score of the FitExperiment result on a given dataset
        '''

        if target_array_name in all_variables:
            # We have target_array_name (result_fitexperiments_all or result_fitexperiments_noiseconv_all say), that's good.

            # Extract only the Bays09 result
            experiment_index = all_variables['all_parameters']['experiment_ids'].index(experiment_id)

            # Make it work for either fixed T or multiple T.

            if len(all_variables[target_array_name].shape) == 3:
                # Single T. Average over axis -1
                bic_summed = utils.nanmean(all_variables[target_array_name][metric_index, experiment_index])
            elif len(all_variables[target_array_name].shape) == 4:
                # Multiple T. Average over axis -1, then sum
                bic_summed = np.nansum(utils.nanmean(all_variables[target_array_name][:, metric_index, experiment_index], axis=-1))
            else:
                raise ValueError("wrong shape for result_fitexperiments_all: {}".format(all_variables[target_array_name].shape))
        else:
            # We do not have it, could instantiate a FitExperiment with "normal" parameters and work from there instead
            raise NotImplementedError('version without result_fitexperiments_all not implemented yet')

        return bic_summed


    def compute_result_distfit_noiseconv_gorgo11_bic(self, all_variables):
        '''
            Result is summed BIC score of FitExperiment to Gorgo11, using a posterior convolved with a noise output distribution
        '''
        return self.compute_result_distfit_givenexp(all_variables, experiment_id='gorgo11', metric_index=0, target_array_name='result_fitexperiments_noiseconv_all')

    def compute_result_distfit_noiseconv_bays09_bic(self, all_variables):
        '''
            Result is summed BIC score of FitExperiment to Bays09, using a posterior convolved with a noise output distribution
        '''
        return self.compute_result_distfit_givenexp(all_variables, experiment_id='bays09', metric_index=0, target_array_name='result_fitexperiments_noiseconv_all')


    def compute_result_filenameoutput(self, all_variables):
        '''
            Result is filename of the outputted data.

            Looks weird, but is actually useful :)

            Assume that:
                - dataio exists.
        '''

        variables_required = ['dataio']

        if not set(variables_required) <= set(all_variables.keys()):
            print "Error, missing variables for compute_result_distemfits: \nRequired: %s\nPresent%s" % (variables_required, all_variables.keys())
            return np.nan

        # Extract the filename
        return all_variables['dataio'].filename


    def _compute_dist_llvariant(self, all_variables, variant='ll'):
        '''
            Given outputs from FitExperimentAllT, will compute the summed LL,
            as this seems to be an acceptable metric for data fitting.
        '''

        res_variant = 'result_%s_sum' % variant
        if res_variant in all_variables:
            # Average over repetitions and sum over the rest.
            repetitions_axis = all_variables.get('repetitions_axis', -1)
            result_dist = np.nansum(utils.nanmean(-all_variables[res_variant], axis=repetitions_axis))

            print result_dist
            return result_dist
        else:
            raise ValueError("%s was not found in the outputs" % res_variant)


    def compute_result_dist_ll_allt(self, all_variables):
        '''
            Given outputs from FitExperimentAllT, will compute the summed LL,
            as this seems to be an acceptable metric for data fitting.
        '''
        return self._compute_dist_llvariant(all_variables,
                                            variant='ll')


    def compute_result_dist_ll90_allt(self, all_variables):
        '''
            Given outputs from FitExperimentAllT, will compute the summed
            LL90.
            Discards the most outliers.
        '''

        return self._compute_dist_llvariant(all_variables,
                                            variant='ll90')


    def compute_result_dist_ll95_allt(self, all_variables):
        '''
            Given outputs from FitExperimentAllT, will compute the summed
            LL90.
            Discards the most outliers.
        '''

        return self._compute_dist_llvariant(all_variables,
                                            variant='ll95')


    def compute_result_dist_ll97_allt(self, all_variables):
        '''
            Given outputs from FitExperimentAllT, will compute the summed
            LL90.
            Discards the most outliers.
        '''

        return self._compute_dist_llvariant(all_variables,
                                            variant='ll97')


    def compute_result_dist_prodll_allt(self, all_variables):
        '''
            Given outputs from FitExperimentAllT, will compute the geometric mean of the LL.

            UGLY HACK: in order to keep track of the minLL, we return it here.
            You should have a cma_iter_function that cleans it before cma_es.tell() is called...
        '''
        if 'result_ll_sum' in all_variables:
            repetitions_axis = all_variables.get('repetitions_axis', -1)

            # Shift to get LL > 0 always
            currMinLL = np.min(all_variables['result_ll_sum'])
            if currMinLL < all_variables['all_parameters']['shiftMinLL']:
                all_variables['all_parameters']['shiftMinLL'] = currMinLL

            # Remove the current minLL, to make sure fitness > 0
            print 'Before: ', all_variables['result_ll_sum']
            all_variables['result_ll_sum'] -= all_variables['all_parameters']['shiftMinLL']
            all_variables['result_ll_sum'] += 0.001
            print 'Shifted: ', all_variables['result_ll_sum']

            result_dist_nll_geom = -mstats.gmean(utils.nanmean(all_variables['result_ll_sum'], axis=repetitions_axis), axis=-1)

            print result_dist_nll_geom
            return np.array([result_dist_nll_geom, all_variables['all_parameters']['shiftMinLL']])
        else:
            raise ValueError('result_ll_sum was not found in the outputs')



######################################################
## Unit tests
######################################################
def test_dummyresult_computation():
    '''
        Test if the dummy result computation works
    '''

    rc = ResultComputation('random')

    all_variables = {}
    print rc.compute_result(all_variables)


if __name__ == '__main__':
    test_dummyresult_computation()




