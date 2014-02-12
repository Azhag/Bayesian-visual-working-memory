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


    def compute_result_distemfits(self, all_variables):
        '''
            Result is the distance (sum squared) to experimental data fits

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''

        variables_required = ['result_em_fits', 'all_parameters', 'T_space']
        repetitions_axis = all_variables.get('repetitions_axis', -1)

        if not set(variables_required) <= set(all_variables.keys()):
            print "Error, missing variables for compute_result_distemfits: \nRequired: %s\nPresent%s" % (variables_required, all_variables.keys())
            return np.nan


        # Create output variables
        result_dist_bays09 = np.nan*np.empty((all_variables['T_space'].size, 4))

        ### Result computation
        data_bays2009 = load_experimental_data.load_data_bays2009(fit_mixture_model=True)
        bays09_experimental_mixtures_mean = data_bays2009['em_fits_nitems_arrays']['mean']
        bays09_T_space = np.unique(data_bays2009['n_items'])

        for repet_i in xrange(all_variables['all_parameters']['num_repetitions']):
            for T_i, T in enumerate(all_variables['T_space']):
                if T in bays09_T_space:
                    result_dist_bays09[T_i, :] = utils.nanmean((bays09_experimental_mixtures_mean[:, bays09_T_space == T] - all_variables['result_em_fits'][T_i, :4])**2., axis=repetitions_axis)

        print result_dist_bays09

        # return the overall distance, over all parameters and number of items
        return np.nansum(result_dist_bays09)


    def compute_result_distemkappa(self, all_variables):
        '''
            Result is the distance (sum squared) to experimental data kappa

            Assume that:
                - result_em_fits exists. Does an average over repetitions_axis and sums over all others.
        '''

        variables_required = ['result_em_fits', 'all_parameters', 'T_space']
        repetitions_axis = all_variables.get('repetitions_axis', -1)

        if not set(variables_required) <= set(all_variables.keys()):
            print "Error, missing variables for compute_result_distemfits: \nRequired: %s\nPresent%s" % (variables_required, all_variables.keys())
            return np.nan


        # Create output variables
        result_dist_bays09 = np.nan*np.empty(all_variables['T_space'].size)

        ### Result computation
        data_bays2009 = load_experimental_data.load_data_bays2009(fit_mixture_model=True)
        bays09_experimental_mixtures_mean = data_bays2009['em_fits_nitems_arrays']['mean']
        bays09_T_space = np.unique(data_bays2009['n_items'])

        for repet_i in xrange(all_variables['all_parameters']['num_repetitions']):
            for T_i, T in enumerate(all_variables['T_space']):
                if T in bays09_T_space:
                    result_dist_bays09[T_i] = utils.nanmean((bays09_experimental_mixtures_mean[0, bays09_T_space == T].flatten() - all_variables['result_em_fits'][T_i, 0])**2., axis=repetitions_axis)

                    # print bays09_experimental_mixtures_mean[0, bays09_T_space == T].flatten()
                    # print all_variables['result_em_fits'][T_i, 0]


        print result_dist_bays09

        # return the overall distance, over all parameters and number of items
        return np.nansum(result_dist_bays09)


    def compute_result_distfitexpbic(self, all_variables):
        '''
            Result is the summed BIC score of the FitExperiment result on all datasets
        '''

        if 'result_fitexperiments' in all_variables:
            # We have result_fitexperiments, that's good.
            # Average over axis -1, then sum
            bic_summed = utils.nanmean(all_variables['result_fitexperiments'][0])
        else:
            # We do not have it, could instantiate a FitExperiment with "normal" parameters and work from there instead
            raise NotImplementedError('version without result_fitexperiments not implemented yet')

        return bic_summed


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




