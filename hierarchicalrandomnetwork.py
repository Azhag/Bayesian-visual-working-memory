#!/usr/bin/env python
# encoding: utf-8
"""
hierarchicalrandomnetwork.py

Created by Loic Matthey on 2013-05-09.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import pylab as plt
import numpy as np
# import scipy as sp
# import scipy.special as scsp
# import progress

from randomfactorialnetwork import *
from utils import *

class HierarchialRandomNetwork(RandomFactorialNetwork):
    '''
        Network built hiearchically.
        Consist of two layers:
        - The first one provides some smooth basis on a conjunctive space of features.
        - The second samples the first randomly and computes a non-linear weighted sum of them
    '''

    #### Constructors

    def __init__(self, M, R=2, gain=1.0, M_layer_one=100, type_layer_one='conjunctive', output_both_layers=False, optimal_coverage=False, rcscale_layer_one=5.0, ratio_layer_one=200.0, gain_layer_one=4.*np.pi**2., nonlinearity_fct='positive_linear', threshold=0.0, sparsity_weights=0.7, distribution_weights='exponential', sigma_weights=0.5, normalise_weights=True, debug=True):
        assert R == 2, 'HiearchialRandomNetwork defined over two features for now'

        self.M_layer_two = M
        self.M_layer_one = M_layer_one
        if output_both_layers:
            self.M = self.M_layer_two + self.M_layer_one
        else:
            self.M = self.M_layer_two

        self.rcscale_layer_one = rcscale_layer_one
        self.ratio_layer_one = ratio_layer_one
        self.gain_layer_one = gain_layer_one
        self.optimal_coverage = optimal_coverage
        self.sigma_weights = sigma_weights
        self.normalise_weights = normalise_weights
        self.distribution_weights = distribution_weights
        self.type_layer_one = type_layer_one
        self.output_both_layers = output_both_layers

        self._ALL_NEURONS = np.arange(M)

        self.R = R
        self.gain = gain

        self.layer_one_network = None
        self.current_layer_one_response = None
        self.current_layer_two_response = None

        self.debug = debug

        if self.debug:
            print "-> Building HierarchicalRandomNetwork"

        # Initialise everything
        self.construct_layer_one(type_layer=type_layer_one)
        self.construct_nonlinearity_fct(fct=nonlinearity_fct, threshold=threshold)
        self.construct_A_sampling(sparsity=sparsity_weights, distribution_weights=distribution_weights, sigma_weights=sigma_weights, normalise=normalise_weights)

        self.population_code_type = 'hierarchical'
        self.coordinates = 'full_angles_sym'
        self.network_initialised = True


    def construct_layer_one(self, type_layer='conjunctive'):
        '''
            Initialises the first layer of the hiearchical network

            Consists of another RFN, makes everything simpler and more logical
        '''
        if type_layer == 'conjunctive':
            
            self.layer_one_network = RandomFactorialNetwork.create_full_conjunctive(self.M_layer_one, R=self.R, rcscale=self.rcscale_layer_one, autoset_parameters=self.optimal_coverage, response_type='bivariate_fisher', gain=self.gain_layer_one)

        elif type_layer == 'feature':
            
            self.layer_one_network = RandomFactorialNetwork.create_full_features(self.M_layer_one, R=self.R, scale=self.rcscale_layer_one, ratio=self.ratio_layer_one, autoset_parameters=self.optimal_coverage, gain=self.gain_layer_one, nb_feature_centers=1)
        else:
            raise NotImplementedError('type_layer is conjunctive only for now')


    def construct_nonlinearity_fct(self, fct='exponential', threshold=0.0):
        '''
            Set a nonlinearity function for the second layer. Not sure if needed, but let's do it.

            Input:
                fct: if function, used as it is. If string, switch between
                    exponential, identity, rectify
        '''

        if is_function(fct):
            # Function given, just use that
            self.nonlinearity_fct = fct
        else:

            # Switch based on some supported functions
            if fct == 'exponential':
                self.nonlinearity_fct = np.exp
            elif fct == 'identity':
                self.nonlinearity_fct = lambda x: x
            elif fct == 'positive_linear':
                self.threshold = threshold
                
                def positive_linear(x):
                    return (x - self.threshold).clip(0.0)

                self.nonlinearity_fct = positive_linear


    def construct_A_sampling(self, sparsity=0.1, distribution_weights='randn', sigma_weights=0.1, normalise=False):
        '''
            Creates the sampling matrix A for the network.

            Should have a small (sparsity amount) of non-zero weights. Weights are sampled independently from a gaussian distribution.
        '''

        if distribution_weights == 'randn':
            self.A_sampling = sigma_weights*np.random.randn(self.M_layer_two, self.M_layer_one)*(np.random.rand(self.M_layer_two, self.M_layer_one) <= sparsity)
        elif distribution_weights == 'exponential':
            self.A_sampling = np.random.exponential(sigma_weights, (self.M_layer_two, self.M_layer_one))*(np.random.rand(self.M_layer_two, self.M_layer_one) <= sparsity)
        else:
            raise ValueError('distribution_weights should be randn/exponential')

        if normalise == 1:
            # Normalise the rows to get a maximum activation level per neuron
            # self.A_sampling = self.A_sampling/np.sum(self.A_sampling, axis=0)
            self.A_sampling = (self.A_sampling.T/(np.sum(self.A_sampling, axis=1))).T
        elif normalise == 2:
            # Normalise the network activity by the number of layer one neurons
            self.gain /= self.M_layer_one*sigma_weights


    ##### Network behaviour

    def get_network_response(self, stimulus_input=None, specific_neurons=None, params={}):
        '''
            Output the activity of the network for the provided input.

            Can return either layer 2 or layer 1+2.
        '''

        if stimulus_input is None:
            stimulus_input = (0.0,)*self.R

        # Get the response of layer one to the stimulus
        layer_one_response = self.get_layer_one_response(stimulus_input)

        # Combine those responses according the the sampling matrices
        layer_two_response = self.get_layer_two_response(layer_one_response, specific_neurons=specific_neurons)

        if self.output_both_layers and specific_neurons is None:
            # Should return the activity of both layers collated
            # (handle stupid specific_neurons filter case in the cheapest way possible: don't support it)
            return np.r_[layer_two_response, layer_one_response]
        else:
            # Only layer two is relevant
            return layer_two_response


    def get_layer_one_response(self, stimulus_input=None, specific_neurons=None):
        '''
            Compute/updates the response of the first layer to the given stimulus

            The first layer is a normal RFN, so simply query it for its response
        '''
        self.current_layer_one_response = self.layer_one_network.get_network_response(stimulus_input=stimulus_input, specific_neurons=specific_neurons)

        return self.current_layer_one_response


    def get_layer_two_response(self, layer_one_response, specific_neurons=None):
        '''
            Compute/updates the response of the second layer, based on the response of the first layer

            The activity is given by:

                x_2 = f(A x_1)

            Where:
                - x_1 is the response of the first layer
                - A is the sampling matrix, random and sparse usually
                - f is a nonlinear function
        '''

        if specific_neurons is None:
            self.current_layer_two_response = self.gain*self.nonlinearity_fct(np.dot(self.A_sampling, layer_one_response))
        else:    
            self.current_layer_two_response = self.gain*self.nonlinearity_fct(np.dot(self.A_sampling[specific_neurons], layer_one_response))

        return self.current_layer_two_response


   

    ##### Helper behaviour function

    

    ##### Plots

    def plot_network_activity(self, stimulus_input=None):
        '''
            Plot the activity of the whole network on a specific stimulus.

            Shows activations of both layers
        '''

        if stimulus_input is None:
            stimulus_input = (0,)*self.R

        # Compute activity of network on the stimulus
        self.get_network_response(stimulus_input=stimulus_input)

        # Do a subplot, second layer on top, first layer on bottom
        plt.figure()
        ax_layertwo = plt.subplot(2, 1, 1)
        ax_layerone = plt.subplot(2, 1, 2)

        # Plot the level two activation, use a bar, easier to read
        ax_layertwo.bar(np.arange(self.M_layer_two), self.current_layer_two_response)

        # Plot the activation of the level one subnetwork (and of the individual responses at level two)
        M_sqrt = int(self.M_layer_one**0.5)
        
        # Level one
        im = ax_layerone.imshow(self.current_layer_one_response.reshape(M_sqrt, M_sqrt).T, origin='lower', aspect='equal', interpolation='nearest')
        im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        ax_layerone.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax_layerone.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax_layerone.set_yticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax_layerone.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

        e = Ellipse(xy=stimulus_input, width=0.4, height=0.4)
        
        ax_layerone.add_artist(e)
        e.set_clip_box(ax_layerone.bbox)
        e.set_alpha(0.5)
        e.set_facecolor('white')
        e.set_transform(ax_layerone.transData)

        plt.show()


    def plot_neuron_activity(self, neuron_index=0, precision=100):
        '''
            Plot the activity of one specific neuron over the whole input space.
        '''
        
        activity, feature_space1, feature_space2 = self.get_neuron_activity(neuron_index, precision=precision, return_axes_vect=True)
        
        # Plot it
        pcolor_2d_data(activity, x=feature_space1, y=feature_space2, ticks_interpolate=5)




def test_hierarchical_conjunctive():
    print 'Small test of the components of the hierarchical network'

    M = 100
    hrn = HierarchialRandomNetwork(M, sparsity_weights=0.5, normalise_weights=False)
    
    # Get one output of the network
    hrn.get_network_response(stimulus_input=(0.0, 0.0))

    # Get the activation of one neuron over its space
    hrn.get_neuron_activity(0, precision=100)


if __name__ == '__main__':
    test_hierarchical_conjunctive()

    M = 100.0
    # hrn = HierarchialRandomNetwork(M, distribution_weights='exponential', sigma_weights=0.5, sparsity_weights=0.5, normalise_weights=True, rcscale_layer_one=5.)
    hrn1 = HierarchialRandomNetwork(M, optimal_coverage=True, M_layer_one=100)
    hrn2 = HierarchialRandomNetwork(M, optimal_coverage=True, M_layer_one=30*30)
    hrn3 = HierarchialRandomNetwork(M, optimal_coverage=True, M_layer_one=15*15, output_both_layers=True)

    hrn_feat = HierarchialRandomNetwork(M, sigma_weights=1.0, sparsity_weights=0.5, normalise_weights=True, type_layer_one='feature', optimal_coverage=True, M_layer_one=7*7, distribution_weights='randn', threshold=0.5)

    hrn1.plot_neuron_activity()
    hrn1.plot_network_activity()

    hrn2.plot_neuron_activity()

    hrn_feat.plot_network_activity()
    hrn_feat.plot_neuron_activity(0)



    ## Try to PCA everything
    try:
        import sklearn.decomposition as skdec

        samples_pca = [hrn1.sample_network_response(np.random.uniform(-np.pi, np.pi, (2)), sigma=0.2) for i in xrange(100)]
        samples_pca.extend([hrn1.sample_network_response(np.random.uniform(-np.pi, np.pi, (2)), sigma=0.2) for i in xrange(100)])
        samples_pca.extend([hrn1.sample_network_response(np.random.uniform(-np.pi, np.pi, (2)), sigma=0.2) for i in xrange(100)])
        samples_pca = np.array(samples_pca)
        
        pca = skdec.PCA()
        samples_pca_transf = pca.fit(samples_pca).transform(samples_pca)
        print pca.explained_variance_ratio_

    except:
        pass

    plt.show()


