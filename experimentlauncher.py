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
import inspect

import launchers
import launchers_profile
import launchers_memorycurves
import launchers_parametersweeps
import launchers_fisherinformation
import launchers_experimentalvolume

launchers_modules = [launchers, launchers_profile, launchers_memorycurves, launchers_parametersweeps, launchers_fisherinformation, launchers_experimentalvolume]


class ExperimentLauncher(object):
    """
        Takes parameters, either on the command line or from a file, and launch the appropriate commands
    """

    def __init__(self, run=True):
        self.actions = None
        self.args = None
        self.all_vars = None

        # Init the launchers (should be imported already)
        self.init_possible_launchers()

        # Create the argument parser, parse the inputs
        self.create_argument_parser()

        # Run the launcher if desired
        if run:
            self.run_launcher()


    def init_possible_launchers(self):
        
        # List of possible launchers
        self.possible_launchers = {}

        # Get all the launchers available
        for launch_module in launchers_modules:

            # Getmembers returns a list of tuples (f.__name__, f)
            all_functions = inspect.getmembers(launch_module, inspect.isfunction)

            # Only keep the functions starting with "launcher_" which denote one of our launcher
            for (func_name, func) in all_functions:
                if func_name.startswith('launcher_'):
                    
                    # Fill the dictionary of callable launchers
                    self.possible_launchers[func_name] = func



    def create_argument_parser(self):

        print sys.argv[1:]
    
        parser = argparse.ArgumentParser(description='Sample a model of Visual working memory.')
        parser.add_argument('--label', help='label added to output files', default='')
        parser.add_argument('--output_directory', nargs='?', default='Data/')
        parser.add_argument('--action_to_do', choices=self.possible_launchers.keys(), default='do_simple_run')
        parser.add_argument('--input_filename', default='', help='Some input file, depending on context')
        parser.add_argument('--num_repetitions', type=int, default=1, help='For search actions, number of repetitions to average on')
        parser.add_argument('--N', default=100, type=int, help='Number of datapoints')
        parser.add_argument('--T', default=1, type=int, help='Number of times')
        parser.add_argument('--K', default=2, type=int, help='Number of representated features')  # Warning: Need more data for bigger matrix
        parser.add_argument('--D', default=32, type=int, help='Dimensionality of features')
        parser.add_argument('--M', default=300, type=int, help='Dimensionality of data/memory')
        parser.add_argument('--R', default=2, type=int, help='Number of population codes')
        parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to use')
        parser.add_argument('--selection_num_samples', type=int, default=1, help='While selecting the new sample from a set of samples, consider the P last samples only. (if =1, return last sample)')
        parser.add_argument('--selection_method', choices=['median', 'last'], default='median', help='How the new sample is chosen from a set of samples. Median is closer to the ML value but could have weird effects.')
        parser.add_argument('--stimuli_generation', choices=['constant', 'random'], default='random', help='How to generate the dataset.')
        parser.add_argument('--alpha', default=1.0, type=float, help='Weighting of the decay through time')
        parser.add_argument('--code_type', choices=['conj', 'feat', 'mixed', 'wavelet'], default='conj', help='Select the type of code used by the Network')
        parser.add_argument('--rc_scale', type=float, default=0.5, help='Scale of receptive fields')
        parser.add_argument('--rc_scale2', type=float, default=0.4, help='Scale of receptive fields, second population (e.g. feature for mixed population)')
        parser.add_argument('--sigmax', type=float, default=0.2, help='Noise per object')
        parser.add_argument('--sigmay', type=float, default=0.02, help='Noise along time')
        parser.add_argument('--ratio_conj', type=float, default=0.2, help='Ratio of conjunctive/field subpopulations for mixed network')
        parser.add_argument('--inference_method', choices=['sample', 'max_lik', 'none'], default='sample', help='Method used to infer the responses. Either sample (default) or set the maximum likelihood/posterior values directly.')
        parser.add_argument('--subaction', default='', help='Some actions have multiple possibilities.')
        parser.add_argument('--search_type', choices=['random', 'grid'], default='random', help='When performing a parameter search, should we do a grid-search or random search?')

        
        self.args = parser.parse_args()


    def run_launcher(self):

        # Print the docstring
        print self.possible_launchers[self.args.action_to_do].__doc__
        
        # Run the launcher
        self.all_vars = self.possible_launchers[self.args.action_to_do](self.args)

    


if __name__ == '__main__':
    
    
    experiment_launcher = ExperimentLauncher(run=True)


    # Re-instantiate some variables
    #   Ugly but laziness prevails...
    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs']
    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]


    plt.show()

