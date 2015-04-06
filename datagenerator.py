#!/usr/bin/env python
# encoding: utf-8
"""
datagenerator.py

Created by Loic Matthey on 2011-06-10.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

# from scaledimage import *
import matplotlib.pyplot as plt
import matplotlib.ticker as plttic
import matplotlib.patches as plt_patches
import numpy as np
from scipy.spatial.distance import pdist

# from populationcode import *
# from randomnetwork import *
from randomfactorialnetwork import *

class DataGenerator:
    def __init__(self, N, T, random_network, sigma_y = 0.05, time_weights=None, time_weights_parameters = {}):
        self.N = N

        # For now, assume T is known and fixed
        self.T = T

        # Use this random network to construct the data
        self.random_network = random_network

        # Number of feature populations
        self.R = random_network.R

        self.sigma_y = sigma_y

        assert self.random_network.network_initialised, "Initialise the possible orientations in the Random Network first"

        # Initialise the weights for time decays if needed
        if time_weights is None:
            self.initialise_time_weights(time_weights_parameters)
            # prior=weight_prior, weighting_alpha=weighting_alpha, weighting_beta=weighting_beta, specific_weighting=specific_weighting)
        else:
            self.time_weights = time_weights


    def initialise_time_weights(self, time_weights_parameters):
        '''
            Initialises the weights used for mixing through time in the final 'memory'

            Could be:
                - Uniform
                - Prior for primacy

            format: [alpha_t ; beta_t], alpha_t mix the past, beta_t mix the current pattern
        '''

        try:
            weight_prior = time_weights_parameters['weight_prior']
            weighting_alpha = time_weights_parameters['weighting_alpha']
            weighting_beta = time_weights_parameters['weighting_beta']
            specific_weighting = time_weights_parameters.get("specific_weighting", 0.0)

        except TypeError:
            raise ValueError('Time_weights_parameter doesnt contain proper keys: weight_prior, weighting_alpha, weighting_beta, [specific_weighting]')

        self.time_weights = np.zeros((2, self.T))

        if weight_prior == 'uniform':
            self.time_weights[0] = weighting_alpha*np.ones(self.T)
            self.time_weights[1] = weighting_beta*np.ones(self.T)
        elif weight_prior == 'primacy':
            self.time_weights[0] = weighting_alpha*np.ones(self.T)
            self.time_weights[1] = weighting_beta*(np.ones(self.T) + specific_weighting*np.arange(self.T)[::-1])
        elif weight_prior =='recency':
            self.time_weights[0] = weighting_alpha*np.ones(self.T)
            self.time_weights[1] = weighting_beta*(np.ones(self.T) + specific_weighting*np.arange(self.T))
        elif weight_prior == 'normalised':
            self.time_weights[0] = weighting_alpha*np.ones(self.T)
            self.time_weights[1] = np.power(weighting_alpha, np.arange(self.T))
        else:
            raise ValueError('Prior for time weights unknown')


    def plot_data(self, nb_to_plot=-1):
        '''
            Show all datapoints
        '''
        if nb_to_plot < 0:
            nb_to_plot = self.N

        f = plt.figure()
        N_sqrt = np.sqrt(nb_to_plot).astype(np.int32)
        for i in xrange(N_sqrt):
            for j in xrange(N_sqrt):
                print "Plotting %d" % (N_sqrt*i+j+1)
                subax = f.add_subplot(N_sqrt, N_sqrt, N_sqrt*i+j+1)
                subax.plot(np.linspace(0., np.pi, self.random_network.M, endpoint=False), self.Y[N_sqrt*i+j])
                subax.xaxis.set_major_locator(plttic.NullLocator())
                subax.yaxis.set_major_locator(plttic.NullLocator())



class DataGeneratorBinary(DataGenerator):
    '''
        Generate a dataset. ('binary' 1-of-K code)
    '''
    def __init__(self, N, T, random_network, sigma_y = 0.05, time_weights=None, time_weights_parameters = dict(weighting_alpha=0.3, weighting_beta = 1.0, specific_weighting = 0.3, weight_prior='uniform')):
        '''
                       N:   number of datapoints
                       T:   number of patterns per datapoint
            time_weights:   [alpha_t ; beta_t] for all t=1:T
                 sigma_y:   Noise on the memory markov chain
        '''

        DataGenerator.__init__(self, N, T, random_network, sigma_y = sigma_y, time_weights = time_weights, time_weights_parameters = time_weights_parameters)

        # Build the dataset
        self.build_dataset()



    def build_dataset(self):
        '''
            Creates the dataset
                For each datapoint, choose T possible orientations ('binary' 1-of-K code),
                then combine them together, with time decay

            Z_true:             N x T x R x K
            Y :                 N x M
            all_Y:              N x T x M
            chosen_orientation: N x T x R
        '''

        ## Create Z, assigning a feature to each time for each datapoint

        self.Z_true = np.zeros((self.N, self.T, self.R, self.random_network.K))

        self.chosen_orientations = np.zeros((self.N, self.T, self.R), dtype='int')

        # Initialise Y (keep intermediate y_t as well)
        self.all_Y = np.zeros((self.N, self.T, self.random_network.M))
        self.Y = np.zeros((self.N, self.random_network.M))

        assert self.T <= self.random_network.possible_objects.size, "Unique objects needed"

        #print self.time_weights

        for i in xrange(self.N):

            # Choose T random orientations, uniformly
            self.chosen_orientations[i] = np.random.permutation(self.random_network.possible_objects_indices)[:self.T]

            # Activate those features for the current datapoint
            for r in xrange(self.R):
                self.Z_true[i, np.arange(self.T), r, self.chosen_orientations[i][:,r]] = 1.0

            # Get the 'x' samples (here from the population code, with correlated covariance, but whatever)
            x_samples_sum = self.random_network.sample_network_response_indices(self.chosen_orientations[i].T)

            # Combine them together
            for t in xrange(self.T):
                self.Y[i] = self.time_weights[0, t]*self.Y[i].copy() + self.time_weights[1, t]*x_samples_sum[t] + self.sigma_y*np.random.randn(self.random_network.M)
                self.all_Y[i, t] = self.Y[i]



class DataGeneratorDiscrete(DataGenerator):
    '''
        Generate a dataset. ('discrete' Z=k code)
    '''
    def __init__(self, N, T, random_network, sigma_y = 0.05, time_weights=None, time_weights_parameters = dict(weighting_alpha=0.3, weighting_beta = 1.0, specific_weighting = 0.3, weight_prior='uniform')):
        '''
                       N:   number of datapoints
                       T:   number of patterns per datapoint
            time_weights:   [alpha_t ; beta_t] for all t=1:T
                 sigma_y:   Noise on the memory markov chain
        '''
        DataGenerator.__init__(self, N, T, random_network, sigma_y = sigma_y, time_weights = time_weights, time_weights_parameters = time_weights_parameters)

        # Build the dataset
        self.build_dataset()



    def build_dataset(self):
        '''
            Creates the dataset
                For each datapoint, choose T possible orientations ('discrete' Z=k),
                then combine them together, with time decay

            Z_true:             N x T x R
            Y :                 N x M
            all_Y:              N x T x M
            chosen_orientation: N x T x R
        '''

        ## Create Z, assigning a feature to each time for each datapoint

        self.Z_true = np.zeros((self.N, self.T, self.R), dtype='int')

        self.chosen_orientations = np.zeros((self.N, self.T, self.R), dtype='int')

        # Initialise Y (keep intermediate y_t as well)
        self.all_Y = np.zeros((self.N, self.T, self.random_network.M))
        self.Y = np.zeros((self.N, self.random_network.M))

        assert self.T <= self.random_network.possible_objects_indices.size, "Unique objects needed"

        #print self.time_weights

        for i in xrange(self.N):

            # Choose T random orientations, uniformly
            self.chosen_orientations[i] = np.random.permutation(self.random_network.possible_objects_indices)[:self.T]

            self.Z_true[i] = self.chosen_orientations[i]

            # Get the 'x' samples (here from the population code, with correlated covariance, but whatever)
            x_samples_sum = self.random_network.sample_network_response_indices(self.chosen_orientations[i].T)

            # Combine them together
            for t in xrange(self.T):
                self.Y[i] = self.time_weights[0, t]*self.Y[i].copy() + self.time_weights[1, t]*x_samples_sum[t] + self.sigma_y*np.random.randn(self.random_network.M)
                self.all_Y[i, t] = self.Y[i]




    # def show_features(self):
    #         '''
    #             Show all features
    #         '''
    #         f = plt.figure()
    #
    #         for k in xrange(self.K):
    #             subaxe=f.add_subplot(1, self.K, k)
    #             scaledimage(self.features[k], ax=subaxe)



class DataGeneratorContinuous(DataGenerator):

    def __init__(self, N, T, random_network, sigma_y = 0.05, time_weights=None, time_weights_parameters = dict(weighting_alpha=0.3, weighting_beta = 1.0, specific_weighting = 0.3, weight_prior='uniform'), cued_feature_time=0):

        assert isinstance(random_network, RandomNetworkContinuous) or isinstance(random_network, RandomNetworkFactorialCode), "Use a RandomNetworkContinuous/RandomNetworkFactorialCode with this DataGeneratorContinuous"

        DataGenerator.__init__(self, N, T, random_network, sigma_y = sigma_y, time_weights = time_weights, time_weights_parameters = time_weights_parameters)

        # Build the dataset
        self.build_dataset(cued_feature_time=cued_feature_time)


    def build_dataset(self, input_orientations = None, cued_feature_time=0):
        '''
            Creates the dataset
                For each datapoint, choose T possible orientations ('discrete' Z=k),
                then combine them together, with time decay

            Y :                 N x M
            all_Y:              N x T x M
            chosen_orientation: N x T x R
            cued_features:      N x 2 (r_c, t_c)
        '''

        # Assign the correct orientations (i.e. orientation/color for each object)
        if input_orientations is None:
            self.chosen_orientations = np.zeros((self.N, self.T, self.R), dtype='float')
        else:
            self.chosen_orientations = input_orientations
        # Select which item should be recalled (and thus cue one/multiple of the other feature)

        self.cued_features = np.zeros((self.N, 2), dtype='int')

        # Initialise Y (keep intermediate y_t as well)
        self.all_Y = np.zeros((self.N, self.T, self.random_network.M))
        self.Y = np.zeros((self.N, self.random_network.M))
        self.all_X = np.zeros((self.N, self.T, self.random_network.M))

        assert self.T <= self.random_network.possible_objects_indices.size, "Unique objects needed"

        # TODO Hack for now, add the time contribution
        # self.time_contribution = 0.06*np.random.randn(self.T, self.random_network.M)

        for i in xrange(self.N):

            if input_orientations is None:
                # Choose T random orientations, uniformly
                self.chosen_orientations[i] = np.random.permutation(self.random_network.possible_objects)[:self.T]

            # For now, always cued the second code (i.e. color) and retrieve the first code (i.e. orientation)
            self.cued_features[i, 0] = 1

            # Randomly recall one of the times
            # self.cued_features[i, 1] = np.random.randint(self.T)
            self.cued_features[i, 1] = cued_feature_time

            # Get the 'x' samples (here from the population code, with correlated covariance, but whatever)
            x_samples_sum = self.random_network.sample_network_response(self.chosen_orientations[i])

            # Combine them together
            for t in xrange(self.T):
                self.Y[i] = self.time_weights[0, t]*self.Y[i].copy() + self.time_weights[1, t]*x_samples_sum[t] + self.sigma_y*np.random.randn(self.random_network.M)
                # self.Y[i] /= np.sum(np.abs(self.Y[i]))
                # self.Y[i] /= fast_1d_norm(self.Y[i])
                self.all_Y[i, t] = self.Y[i]
                self.all_X[i, t] = x_samples_sum[t]









if __name__ == '__main__':
    N = 1000
    T = 2
    K = 25
    M = int(14**2.)
    D = 50
    R = 2

    # random_network = RandomNetwork.create_instance_uniform(K, M, D=D, R=R, W_type='identity', W_parameters=[0.1, 0.5])
    # random_network = RandomNetworkContinuous.create_instance_uniform(K, M, D=D, R=R, W_type='identity', W_parameters=[0.1, 0.5], sigma=0.1, gamma=0.002, rho=0.002)
    # random_network = RandomNetworkFactorialCode.create_instance_uniform(K, D=D, R=R, sigma=0.02)
    # random_network = RandomFactorialNetwork(M, R=R)
    # ratio_concentration = 2.
    # random_network.assign_random_eigenvectors(scale_parameters=(10., 1/150.), ratio_parameters=(ratio_concentration, 4./(3.*ratio_concentration)), reset=True)
    # random_network.plot_coverage_feature_space()

    random_network = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(2.0, 0.01), ratio_moments=(1.0, 0.05))
    # random_network = RandomFactorialNetwork.create_full_features(M, R=R, scale=0.8, ratio=40., nb_feature_centers=1)


    # data_gen = DataGeneratorDiscrete(N, T, random_network, time_weights_parameters = dict(weighting_alpha=0.8, weighting_beta = 1.0, specific_weighting = 0.2, weight_prior='recency'))
    # data_gen = DataGeneratorContinuous(N, T, random_network, sigma_y = 0.02, time_weights_parameters = dict(weighting_alpha=0.7, weighting_beta = 1.0, specific_weighting = 0.2, weight_prior='uniform'))
    data_gen = DataGeneratorRFN(N, T, random_network, sigma_y = 0.02, sigma_x = 0.1, time_weights_parameters = dict(weighting_alpha=1.0, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform'), enforce_min_distance=0.1)


    # data_gen.plot_data(16)

    #print data_gen.X.shape

    # plt.figure()
    # plt.plot(np.mean(np.apply_along_axis(fast_1d_norm, 2, data_gen.all_Y), axis=0))

    plt.show()


from datageneratorrfn import *
