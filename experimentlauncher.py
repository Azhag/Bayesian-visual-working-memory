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

from launchers import *
from launchers_profile import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *


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
        # Switch on different actions
        self.actions = dict([(x.__name__, x) for x in 
            [do_simple_run, 
                profile_me,
                do_size_receptive_field,
                do_neuron_number_precision,
                do_size_receptive_field_number_neurons,
                plot_neuron_number_precision,
                plot_size_receptive_field,
                plot_size_receptive_field_number_neurons,
                do_memory_curve,
                do_multiple_memory_curve,
                do_multiple_memory_curve_simult,
                plot_multiple_memory_curve,
                plot_multiple_memory_curve_simult,
                do_mixed_ratioconj,
                do_mixed_two_scales,
                do_save_responses_simultaneous,
                do_fisher_information_estimation
                ]])


    def create_argument_parser(self):

        print sys.argv[1:]
    
        parser = argparse.ArgumentParser(description='Sample a model of Visual working memory.')
        parser.add_argument('--label', help='label added to output files', default='')
        parser.add_argument('--output_directory', nargs='?', default='Data/')
        parser.add_argument('--action_to_do', choices=self.actions.keys(), default='do_simple_run')
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


        self.args = parser.parse_args()


    def run_launcher(self):

        # Run the launcher
        self.all_vars = self.actions[self.args.action_to_do](self.args)

    


if __name__ == '__main__':
    
    
    experiment_launcher = ExperimentLauncher(run=True)


    # Re-instantiate some variables
    #   Ugly but laziness prevails...
    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network']
    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]


    plt.show()

