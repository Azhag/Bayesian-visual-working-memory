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
import imp


from utils import say_finished


class ExperimentLauncher(object):
    """
        Takes parameters, either on the command line or from a file, and launch the appropriate commands
    """

    def __init__(self, run=True, arguments_dict={}):
        self.actions = None
        self.args = None
        self.all_vars = None

        # Init the launchers (should be imported already)
        self.init_possible_launchers()

        # Create the argument parser, parse the inputs
        self.create_argument_parser()

        # Complete or overwrite the current arguments with arguments as function parameters
        self.add_arguments_parameters(arguments_dict)

        # Run the launcher if desired
        if run:
            self.run_launcher()


    def init_possible_launchers(self):
        
        # List of possible launchers
        self.possible_launchers = {}

        # Get all the launchers available
        # Do that by just globbing for launchers*.py files
        for launch_module_filename in glob.glob('launchers*.py'):
            # Load the current module (splitext to remove the .py)
            # imp.load_source(os.path.splitext(launcher)[0], launcher)
            launch_module = imp.load_source(launch_module_filename, launch_module_filename)

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
        parser.add_argument('--label', help='label added to output files', default='')
        parser.add_argument('--output_directory', nargs='?', default='Data/')
        parser.add_argument('--action_to_do', choices=self.possible_launchers.keys(), default='launcher_do_simple_run')
        parser.add_argument('--input_filename', default='', help='Some input file, depending on context')
        parser.add_argument('--parameters_filename', default='', help='Some file to be imported containing parameters (and/or functions)')
        parser.add_argument('--num_repetitions', type=int, default=1, help='For search actions, number of repetitions to average on')
        parser.add_argument('--N', default=100, type=int, help='Number of datapoints')
        parser.add_argument('--T', default=1, type=int, help='Number of times')
        parser.add_argument('--K', default=2, type=int, help='Number of representated features')  # Warning: Need more data for bigger matrix
        parser.add_argument('--D', default=32, type=int, help='Dimensionality of features')
        parser.add_argument('--M', default=300, type=int, help='Dimensionality of data/memory')
        parser.add_argument('--M_layer_one', default=400, type=int, help='Dimensionality of first layer for hierarchical networks')
        parser.add_argument('--R', default=2, type=int, help='Number of population codes')
        parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to use')
        parser.add_argument('--selection_num_samples', type=int, default=1, help='While selecting the new sample from a set of samples, consider the P last samples only. (if =1, return last sample)')
        parser.add_argument('--selection_method', choices=['median', 'last'], default='median', help='How the new sample is chosen from a set of samples. Median is closer to the ML value but could have weird effects.')
        parser.add_argument('--stimuli_generation', choices=['constant', 'random', 'random_smallrange', 'constant_separated'], default='random', help='How to generate the dataset.')
        parser.add_argument('--enforce_min_distance', type=float, default=0.17, help='Minimal distance between items of the same array')
        parser.add_argument('--alpha', default=1.0, type=float, help='Weighting of the decay through time')
        parser.add_argument('--code_type', choices=['conj', 'feat', 'mixed', 'wavelet', 'hierarchical'], default='conj', help='Select the type of code used by the Network')
        parser.add_argument('--rc_scale', type=float, default=0.5, help='Scale of receptive fields')
        parser.add_argument('--rc_scale2', type=float, default=0.4, help='Scale of receptive fields, second population (e.g. feature for mixed population)')
        parser.add_argument('--autoset_parameters', action='store_true', default=False, help='Automatically attempt to set the rc_scale/ratio to cover the space evenly, depending on the number of neurons')
        parser.add_argument('--type_layer_one', choices=['conjunctive', 'feature'], default='conjunctive', help='Select the type of population code for an hierarchical network')
        parser.add_argument('--sparsity', type=float, default=1.0, help='Sets the sparsity of the hierarchical network sampling')
        parser.add_argument('--sigma_weights', type=float, default=1.0, help='Sets distribution parameter for the hierarchical sampling matrix')
        parser.add_argument('--feat_ratio', type=float, default=40., help='Ratio between eigenvectors for feature code')
        parser.add_argument('--sigmax', type=float, default=0.2, help='Noise per object')
        parser.add_argument('--sigmay', type=float, default=0.02, help='Noise along time')
        parser.add_argument('--ratio_conj', type=float, default=0.2, help='Ratio of conjunctive/field subpopulations for mixed network')
        parser.add_argument('--inference_method', choices=['sample', 'max_lik', 'none'], default='sample', help='Method used to infer the responses. Either sample (default) or set the maximum likelihood/posterior values directly.')
        parser.add_argument('--subaction', default='', help='Some actions have multiple possibilities.')
        parser.add_argument('--search_type', choices=['random', 'grid'], default='random', help='When performing a parameter search, should we do a grid-search or random search?')
        parser.add_argument('--use_theoretical_cov', action='store_true', default=False, help='Use the theoretical KL approximation to the noise covariance matrix.')
        parser.add_argument('--say_completed', action='store_true', default=False, help='Will use the "say" command to indicate when the launcher has completed.')

        self.args = parser.parse_args()
        self.args_dict = self.args.__dict__


    def add_arguments_parameters(self, other_arguments_dict):
        '''
            Add those parameters into the original self.args variable

            Assume that those parameters should overwrite the command-line ones.
        '''

        self.args_dict.update(other_arguments_dict)


    def run_launcher(self):

        # Print the docstring
        print self.possible_launchers[self.args.action_to_do].__doc__
        
        # Run the launcher
        self.all_vars = self.possible_launchers[self.args.action_to_do](self.args)

        # Talk when completed if desired
        if self.args.say_completed:
            say_finished()


    


if __name__ == '__main__':
    
    
    experiment_launcher = ExperimentLauncher(run=True)


    # Re-instantiate some variables
    #   Ugly but laziness prevails...
    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio']

    if 'variables_to_save' in experiment_launcher.all_vars:
        # Also reinstantiate the variables we saved
        variables_to_reinstantiate.extend(experiment_launcher.all_vars['variables_to_save'])

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]


    plt.show()

