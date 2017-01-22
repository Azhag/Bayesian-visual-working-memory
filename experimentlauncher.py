#!/usr/bin/env python
# encoding: utf-8
"""
experiment_launcher.py


Created by Loic Matthey on 2012-07-10
Copyright (c) 2011 . All rights reserved.
"""

import argparse
import sys
import matplotlib.pyplot as plt
import glob
import inspect
import os
import numpy as np


from utils import say_finished


class ExperimentLauncher(object):
    """
        Takes parameters, either on the command line or from a file, and launch the appropriate commands
    """

    def __init__(self, run=True, arguments_dict={}, job_wrapped=False):
        self.actions = None
        self.args = None
        self.all_vars = None
        self.has_run = False
        self.job_wrapped = job_wrapped

        # Init the launchers (should be imported already)
        self.init_possible_launchers()

        # Create the argument parser, parse the inputs
        self.create_argument_parser()

        # Complete or overwrite the current arguments with arguments as function parameters
        self.add_arguments_parameters(arguments_dict)

        # Change some parameters if a best_parameters_file is given
        if not self.job_wrapped and self.args_dict['best_parameters_file']:
            self.load_extra_parameters_from_file(self.args_dict['best_parameters_file'], force_all=self.args_dict['load_all_from_parameters_file'])

        # Run the launcher if desired
        if run:
            self.run_launcher()


    def __str__(self):
        return 'ExperimentLauncher, action: %s, finished: %d' % (self.args.action_to_do, self.has_run)

    def init_possible_launchers(self):

        # List of possible launchers
        self.possible_launchers = {}

        # Get all the launchers available
        this_file = inspect.getfile(inspect.currentframe())
        current_folder = os.path.split(this_file)[0]

        # Do that by just globbing for launchers*.py files
        for launch_module_filename in glob.glob(os.path.join(current_folder, 'launchers*.py')):
            # Load the current module (remove the path, and splitext to remove the .py)
            launch_module = __import__(os.path.splitext(os.path.split(launch_module_filename)[1])[0])

            # Getmembers returns a list of tuples (f.__name__, f)
            all_functions = inspect.getmembers(launch_module, inspect.isfunction)

            # Only keep the functions starting with "launcher_" which denote one of our launcher
            for (func_name, func) in all_functions:
                if func_name.startswith('launcher_'):

                    # Fill the dictionary of callable launchers
                    self.possible_launchers[func_name] = func


    def create_argument_parser(self):

        print 'Arguments:', sys.argv[1:]

        parser = argparse.ArgumentParser(description='Sample a model of Visual working memory.')
        parser.add_argument('--seed', type=int, default=None,
            help='Random seed to fix')
        parser.add_argument('--label',
            help='label added to output files', default='')
        parser.add_argument('--output_directory', nargs='?', default='Data/')
        parser.add_argument('--action_to_do', choices=self.possible_launchers.keys(), default='launcher_do_simple_run',
            help='Launcher to run, actual code executed')
        parser.add_argument('--job_action', choices=self.possible_launchers.keys(), default='launcher_do_simple_run',
            help='When using a JobWrapper, this action will be executed')
        parser.add_argument('--result_computation', default='',
            help='When using a JobWrapper, ResultComputation.compute_result_{} will be used.')
        parser.add_argument('--input_filename', default='',
            help='Some input file, depending on context')
        parser.add_argument('--parameters_filename', default='',
            help='Some file to be imported containing parameters (and/or functions)')
        parser.add_argument('--num_repetitions', type=int, default=1,
            help='For search actions, number of repetitions to average on')
        parser.add_argument('--N', default=100, type=int,
            help='Number of datapoints')
        parser.add_argument('--T', default=1, type=int,
            help='Number of items. Used as Max number as well.')
        parser.add_argument('--T_min', default=1, type=int,
            help='Minimum number of items')
        parser.add_argument('--K', default=2, type=int,
            help='Number of representated features')  # Warning: Need more data for bigger matrix
        parser.add_argument('--D', default=32, type=int,
            help='Dimensionality of features')
        parser.add_argument('--M', default=300, type=int,
            help='Dimensionality of data/memory')
        parser.add_argument('--M_layer_one', default=400, type=int,
            help='Dimensionality of first layer for hierarchical networks')
        parser.add_argument('--R', default=2, type=int,
            help='Number of population codes')
        parser.add_argument('--num_samples', type=int, default=20,
            help='Number of samples to use')
        parser.add_argument('--num_sampling_passes', type=int, default=1,
            help='Number of passes to do over the thetas. If negative, do it till convergence.')
        parser.add_argument('--cued_feature_type', choices=['single', 'all'], default='single', help='In case of R>2, should we cue only feature R=1 or all R>0?')
        parser.add_argument('--fixed_cued_feature_time', type=int, default=-1,
            help='Index/time of item to try to recall. For simultaneous, should be set to T-1.')
        parser.add_argument('--burn_samples', type=int, default=100,
            help='Number of samples to use for burn in')
        parser.add_argument('--selection_num_samples', type=int, default=1,
            help='While selecting the new sample from a set of samples, consider the P last samples only. (if =1, return last sample)')
        parser.add_argument('--selection_method', choices=['median', 'last'], default='last',
            help='How the new sample is chosen from a set of samples. Median is closer to the ML value but could have weird effects.')
        parser.add_argument('--slice_width', type=float, default=np.pi/40.,
            help='Size of bin width for Slice Sampler. Smaller usually better but slower.')
        parser.add_argument('--stimuli_generation', choices=['constant', 'random', 'random_smallrange', 'constant_separated', 'separated', 'specific_stimuli'],
            default='random',
            help='How to generate the dataset.')
        parser.add_argument('--stimuli_generation_recall', choices=['constant', 'random', 'random_smallrange', 'constant_separated', 'separated'], default='random',
            help='Dataset generation used for the recall model.')
        parser.add_argument('--enforce_min_distance', type=float, default=0.17,
            help='Minimal distance between items of the same array')
        parser.add_argument('--enforce_distance_cued_feature_only', action='store_true', default=False,
            help='Enforce minimum distance on the cued feature only.')
        parser.add_argument('--specific_stimuli_random_centers', dest='specific_stimuli_random_centers', action='store_true', default=False,
            help='Should the centers in the specific stimuli be moved randomly?')
        parser.add_argument('--specific_stimuli_asymmetric', dest='specific_stimuli_asymmetric', action='store_true', default=False,
            help='Should the specific stimuli be asymmetric?')
        parser.add_argument('--alpha', default=1.0, type=float,
            help='Weighting of the decay through time')
        parser.add_argument('--code_type', choices=['conj', 'feat', 'mixed', 'wavelet', 'hierarchical'], default='conj',
            help='Select the type of code used by the Network')
        parser.add_argument('--rc_scale', type=float, default=0.5,
            help='Scale of receptive fields')
        parser.add_argument('--rc_scale2', type=float, default=0.4,
            help='Scale of receptive fields, second population (e.g. feature for mixed population)')
        parser.add_argument('--autoset_parameters', dest='autoset_parameters', action='store_true', default=False,
            help='Automatically attempt to set the rc_scale/ratio to cover the space evenly, depending on the number of neurons')
        parser.add_argument('--response_maxout', dest='response_maxout', action='store_true', default=False,
            help='Change the network response to be max = 1. Changes many things...')
        parser.add_argument('--type_layer_one', choices=['conjunctive', 'feature'], default='feature',
            help='Select the type of population code for an hierarchical network')
        parser.add_argument('--sparsity', type=float, default=1.0,
            help='Sets the sparsity of the hierarchical network sampling')
        parser.add_argument('--sigma_weights', type=float, default=1.0,
            help='Sets distribution parameter for the hierarchical sampling matrix')
        parser.add_argument('--distribution_weights', choices=['exponential', 'randn'], default='exponential',
            help='Select the distribution of the sampling weights for an hierarchical network')
        parser.add_argument('--normalise_weights', type=int, default=1,
            help='Decide if the sampling weights should normalise to 1 for an hierarchical network. Ill-defined for positive+negative weights distributions.')
        parser.add_argument('--threshold', type=float, default=0.0,
            help='Sets the threshold of the hierarchical network activation function')
        parser.add_argument('--output_both_layers', dest='output_both_layers', action='store_true', default=False,
            help='Allow both layers of hierarchical network to be read')
        parser.add_argument('--feat_ratio', type=float, default=40.,
            help='Ratio between eigenvectors for feature code')
        parser.add_argument('--sigmax', type=float, default=0.2,
            help='Noise per object')
        parser.add_argument('--sigmay', type=float, default=0.02,
            help='Noise along time')
        parser.add_argument('--sigma_baseline', type=float, default=0.0001,
            help='Baseline noise in memory.')
        parser.add_argument('--sigma_output', type=float, default=0.0,
            help='Noise added when outputting samples. Cheap lapse-like process')
        parser.add_argument('--lapse_rate', type=float, default=0.0,
            help='Probability of randomly lapsing, not looking at memory and sampling in U[-pi, pi] instead. Quite drastic...')
        parser.add_argument('--renormalize_sigma', action='store_true', default=False, help='If set, all sigmas are considered a proportion of the maximum activation of the network. Best for R>2.')
        parser.add_argument('--renormalize_sigma_output', action='store_true', default=False, help='If set, sigma_output is considered a proportion of the maximum activation of the network. Not sure if really meaningful actually.')
        parser.add_argument('--ratio_conj', type=float, default=0.2,
            help='Ratio of conjunctive/field subpopulations for mixed network')
        parser.add_argument('--ratio_hierarchical', type=float, default=None,
            help='Ratio of layer two/layer one subpopulations for hierarchical network')
        parser.add_argument('--inference_method', choices=['sample', 'max_lik', 'none'], default='sample',
            help='Method used to infer the responses. Either sample (default) or set the maximum likelihood/posterior values directly.')
        parser.add_argument('--subaction', default='',
            help='Some actions have multiple possibilities.')
        parser.add_argument('--collect_responses', dest='collect_responses', action='store_true', default=False,
            help='Some actions can store all sampler responses if desired, so that we can fit models later.')
        parser.add_argument('--search_type', choices=['random', 'grid'], default='random',
            help='When performing a parameter search, should we do a grid-search or random search?')
        parser.add_argument('--use_theoretical_cov', action='store_true', default=False,
            help='Use the theoretical KL approximation to the noise covariance matrix.')
        parser.add_argument('--say_completed', action='store_true', default=False,
            help='Will use the "say" command to indicate when the launcher has completed.')
        parser.add_argument('--enforce_first_stimulus', dest='enforce_first_stimulus', action='store_true', default=False,
            help='Force some datapoints to known values.')
        parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
            help='Prints more messages')
        parser.add_argument('--experiment_data_dir', dest='experiment_data_dir', default="../../experimental_data/",
            help="Base directory containing the experimental data")
        parser.add_argument('--pylab', dest='pylab', default=True, action='store_true', help='Ipython was invoked with --pylab. Let it be allowed.')
        parser.add_argument('--session_id', dest='session_id', default='',
            help='String used by JobWrapper for Result sync files. Used to avoid overwriting result files when running a job with the same parameters again.')
        parser.add_argument('--job_name', dest='job_name', default='',
            help='Unique job name, constructed from parameter values. Could be rebuilt on the fly, but easier to pass it if it exists.')
        parser.add_argument('--best_parameters_file', dest='best_parameters_file', default='',
            help='Reload parameters from a .npy file, created from PBS/SLURM runs. Expects some known dictionaries.')
        parser.add_argument('--load_all_from_parameters_file', default=False, action='store_true', help='If a best parameter file is given, should we force all used parameters or just the optimized/best ones?')
        parser.add_argument('--plot_while_running', dest='plot_while_running', default=False, action='store_true', help='If set, will plot while the simulation is going for chosen launchers.')

        parser.add_argument('--experiment_id', dest='experiment_id', choices=['bays09', 'gorgo11', 'gorgo11_sequential', 'dualrecall'], default='bays09',
            help='Experiment id to use for FitExperimentAllT (or possibly ExperimentalLoader)')
        parser.add_argument('--filter_datapoints_size', type=float, default=-1,
            help='If >0, will limit the number of datapoints used in FitExperiments. [0-1] uses percent total data, >1 use absolute number of samples.')
        parser.add_argument('--filter_datapoints_selection', dest='filter_datapoints_selection', choices=['sequential', 'random'], default='sequential',
            help='If filter_datapoints_size > 0, sets the method to choose which datapoints to use.')
        parser.add_argument('--experiment_subject', default=0, type=int,
            help='Subject to use when loading subset of experimental data. Unused in other situations.')
        parser.add_argument('--bic_K', type=float, default=None,
            help='If set, will fix the number of parameters for the computation of the BIC score. Should be set appropriately.')

        parser.add_argument('--shiftMinLL', type=float, default=0,
            help='Value to enforce LL > 0, when computing geometric means instead of arithmetic mean. Used by ResultCompute -> dist_prodll_allt')

        # Ipython notebook compatibility stuff
        parser.add_argument('-f')
        parser.add_argument('--profile-dir')


        self.args = parser.parse_args()
        self.args_dict = self.args.__dict__


    def add_arguments_parameters(self, other_arguments_dict):
        '''
            Add those parameters into the original self.args variable

            Assume that those parameters should overwrite the command-line ones.
        '''

        # Handle cases with param_value=None, which really mean they should be set to true
        for key, value in other_arguments_dict.iteritems():
            if value is None:
                other_arguments_dict[key] = True

        self.args_dict.update(other_arguments_dict)


    def load_extra_parameters_from_file(self, filename, force_best=True, force_all=False):
        '''
            Take a file (assumed .npy), load it and check for known dictionary names
            that should contain some variables values to force.

            Updates self.args/self.args_dict directly
        '''

        loaded_file = np.load(filename).item()

        if 'parameters' in loaded_file:
            if force_best and 'best_parameters' in loaded_file['parameters']:
                print "+++ Reloading best parameters from {} +++".format(filename)
                for key, value in loaded_file['parameters']['best_parameters'].iteritems():
                    print "\t> {} : {}".format(key, value)
                    self.args_dict[key] = value
            if force_all and 'parameters' in loaded_file['parameters']:
                print "+++ Reloading all previous parameters from {} +++".format(filename)
                for key, value in loaded_file['parameters']['parameters'].iteritems():
                    if value is None:
                        print "\t> {} : True".format(key)
                        self.args_dict[key] = True
                    else:
                        print "\t> {} : {}".format(key, value)
                        self.args_dict[key] = value





    def run_launcher(self):

        # Print the docstring
        print self.possible_launchers[self.args.action_to_do].__doc__

        # Fix seed
        if self.args.seed:
            np.random.seed(self.args.seed)

        # Run the launcher
        self.all_vars = self.possible_launchers[self.args.action_to_do](self.args)

        # Talk when completed if desired
        if self.args.say_completed:
            say_finished()

        self.has_run = True





if __name__ == '__main__':


    experiment_launcher = ExperimentLauncher(run=True)


    # Re-instantiate some variables
    #   Ugly but laziness prevails...
    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio', 'post_processing_outputs', 'fit_exp', 'all_outputs_data']


    if 'dataio' in experiment_launcher.all_vars:
        # Reinstantiate the variables saved automatically.
        dataio_variables_auto = experiment_launcher.all_vars['dataio'].__dict__.get('saved_variables', [])

        if not dataio_variables_auto:
            if 'variables_to_save' in experiment_launcher.all_vars:
                # Also reinstantiate the variables we saved
                variables_to_reinstantiate.extend(experiment_launcher.all_vars['variables_to_save'])
        else:
            variables_to_reinstantiate.extend(dataio_variables_auto)


    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]


    plt.show()

