#!/usr/bin/env python
# encoding: utf-8
"""
randomfactorialnetwork.py

Created by Loic Matthey on 2011-11-09.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import pylab as plt
import numpy as np
from matplotlib.patches import Ellipse
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
import scipy.special as scsp

from utils import *
from statisticsmeasurer import *
from datagenerator import *


class RandomFactorialNetwork():
    '''
        Modified paradigm for this Network. Uses a factorial representation of K features, and samples them using K-dimensional gaussian receptive fields.
            Randomness is in the distribution of orientations and radii of those gaussians.
    '''
    def __init__(self, M, R=1, response_type = 'wrong_wrap'):
        
        assert R == 2, "RandomFactorialNetwork only implemented for R=2 for now"

        self.M = M
        self.K = 0
        self.R = R
        
        self.network_initialised = False
        
        self.neurons_preferred_stimulus = None
        self.neurons_sigma = None
        self.mask_neurons_unset = None
        
        self._ALL_NEURONS = np.arange(M)

        # self.response_type = 'bivariate_fisher'
        self.response_type = response_type
        if response_type == 'wrong_wrap' or response_type == 'bivariate_fisher':
            self.coordinates = 'full_angles_sym'
        elif response_type == 'fisher':
            self.coordinates = 'spherical'

        # Need to assign to each of the M neurons a preferred stimulus (tuple(orientation, color) for example)
        # By default, random
        self.assign_prefered_stimuli(tiling_type='conjunctive')
        self.assign_random_eigenvectors()

        # Used to stored cached network response statistics. Mean_theta(mu(theta)) and Cov_theta(mu(theta))
        self.network_response_statistics = None

        self.network_initialised = True
    
    
    def assign_prefered_stimuli(self, tiling_type='conjunctive', specified_neurons=None, reset=False, nb_feature_centers=3, scales_number=3):
        '''
            For all M factorial neurons, assign them a prefered stimulus tuple (e.g. (orientation, color) )
        '''
        
        if self.neurons_preferred_stimulus is None or reset:
            # Handle uninitialized neurons
            self.neurons_preferred_stimulus = np.nan*np.ones((self.M, self.R))
        
        if specified_neurons is None:
            specified_neurons = self._ALL_NEURONS
            
        if specified_neurons.size != 0:
            # Only do something if non-empty
            
            if tiling_type == 'conjunctive':
                self.assign_prefered_stimuli_conjunctive(specified_neurons)
            
            elif tiling_type == '2_features':
                self.assign_prefered_stimuli_2_features(specified_neurons, nb_feature_centers=nb_feature_centers)
            elif tiling_type == 'wavelet':
                self.assign_prefered_stimuli_wavelet(specified_neurons, scales_number=scales_number)
            
            # Handle uninitialized neurons
            #   check if still some nan, any on the first axis.
            self.mask_neurons_unset = np.any(np.isnan(self.neurons_preferred_stimulus), 1) 

            # Compute the vector representation
            self.compute_preferred_vectors()
    
    

    def assign_prefered_stimuli_conjunctive(self, neurons_indices):
        '''
            Tile conjunctive neurons along the space of possible angles.

            TODO: currently wrong, as this is in spherical coordinates, should cover the surface of the hypersphere instead.
        '''
        # Cover the space uniformly
        N_sqrt = np.floor(np.power(neurons_indices.size, 1./self.R))

        # coverage_1D = np.linspace(-np.pi, np.pi, N_sqrt, endpoint=False)
        coverage_1D = np.linspace( -np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)
        
        new_stim = np.array(cross(self.R*[coverage_1D.tolist()]))

        # Change the range?
        if self.coordinates == 'spherical':
            new_stim[:, 0] = new_stim[:, 0] + np.pi
            new_stim[:, 1] = (new_stim[:, 1] + np.pi)/2.

        # Assign the preferred stimuli
        #   Unintialized neurons will get masked out down there.
        self.neurons_preferred_stimulus[neurons_indices[:new_stim.shape[0]]] = new_stim
    
    def assign_prefered_stimuli_2_features(self, neurons_indices, nb_feature_centers=3):
        '''
            Tile feature neurons in 2D. Assigns nb_feature_centers neurons for each possible feature value.

            TODO: currently wrong, as this is in spherical coordinates, should cover the surface of the hypersphere instead.
        '''
        N = np.round(neurons_indices.size/self.R)
                
        # Arbitrary put them along (x, 0) and (0, y) axes
        # coverage_1D = np.linspace(-np.pi, np.pi, N, endpoint=False)
        # self.neurons_preferred_stimulus[neurons_indices[0:N], 0] = coverage_1D
        # self.neurons_preferred_stimulus[neurons_indices[0:N], 1] = 0.
        # self.neurons_preferred_stimulus[neurons_indices[N:self.R*N], 0] = 0.
        # self.neurons_preferred_stimulus[neurons_indices[N:self.R*N], 1] = coverage_1D

        # Distribute the cells along nb_feature_centers centers. Distributed evenly, should have 0 in them.
        self.nb_feature_centers = nb_feature_centers
        centers = np.linspace(0.0, 2*np.pi, nb_feature_centers, endpoint=False) - 2*np.pi/nb_feature_centers*int(nb_feature_centers/2.)

        # centers = [-2.*np.pi/3., 0.0, 2.*np.pi/3.]
        # centers = [0.0]

        sub_N = N/nb_feature_centers

        # coverage_1D = np.linspace( -np.pi, np.pi, sub_N, endpoint=False)
        coverage_1D = np.linspace( -np.pi + np.pi/sub_N, np.pi + np.pi/sub_N, sub_N, endpoint=False)

        for center_i in np.arange(nb_feature_centers):
            self.neurons_preferred_stimulus[neurons_indices[center_i*sub_N:(center_i+1)*sub_N], 0] = coverage_1D
            self.neurons_preferred_stimulus[neurons_indices[center_i*sub_N:(center_i+1)*sub_N], 1] = centers[center_i]
            self.neurons_preferred_stimulus[neurons_indices[N+center_i*sub_N:(N+(center_i+1)*sub_N)], 0] = centers[center_i]
            self.neurons_preferred_stimulus[neurons_indices[N+center_i*sub_N:(N+(center_i+1)*sub_N)], 1] = coverage_1D

    
    def assign_prefered_stimuli_wavelet(self, neurons_indices, scales_number=3):
        '''
            Tile conjunctive neurons in a multiscale manner.
        
        '''

        # Generate the positions of centers
        centers = []
        new_stim = None
        new_scales = None
        for k in np.arange(scales_number):
            centers.append(np.linspace(-np.pi + np.pi/(2.**k), np.pi - np.pi/(2.**k), 2**k))

            crossed_centers = np.array(cross(self.R*[centers[k].tolist()]))
            if new_stim is None:
                new_stim = crossed_centers
            else:
                new_stim = np.r_[new_stim, crossed_centers]

            if new_scales is None:
                new_scales = np.array([k])
            else:
                new_scales = np.r_[new_scales, (k)*np.ones(crossed_centers.shape[0])]

        # Change the range?
        if self.coordinates == 'spherical':
            new_stim[:, 0] = new_stim[:, 0] + np.pi
            new_stim[:, 1] = (new_stim[:, 1] + np.pi)/2.

        # Assign the preferred stimuli
        #   Unintialized neurons will get masked out down there.
        min_size = np.min((new_stim.shape[0], neurons_indices.size))
        self.neurons_preferred_stimulus[neurons_indices[:min_size]] = new_stim[:min_size]
        self.neurons_scales = new_scales[:min_size]


    def assign_aligned_eigenvectors(self, scale=1.0, ratio=1.0, specified_neurons=None, reset=False):
        '''
            Each neuron has a gaussian receptive field, defined by its eigenvectors/eigenvalues (principal axes and scale)
            Uses the same eigenvalues for all of those
            
            input:
                ratio:  1/-1   : circular gaussian.
                        >1      : ellipse, width>height
                        <-1     : ellipse, width<height
                        [-1, 1] : not used
        '''
        
        if self.neurons_sigma is None or reset:
            self.neurons_sigma = np.zeros((self.M, self.R))
            self.neurons_angle = np.zeros(self.M)
        
        if specified_neurons is None:
            specified_neurons = self._ALL_NEURONS
        
        assert ratio <= -1 or ratio >= 1, "respect my authority! Use ratio >= 1 or <=-1"
        
        if ratio>0:
            self.neurons_sigma[specified_neurons, 1] = ratio*scale
            self.neurons_sigma[specified_neurons, 0] = scale
            self.neurons_angle[specified_neurons] = 0.0
            
        elif ratio <0:
            self.neurons_sigma[specified_neurons, 1] = scale
            self.neurons_sigma[specified_neurons, 0] = -ratio*scale
            self.neurons_angle[specified_neurons] = 0.0
        
        # Update parameters
        self.compute_2d_parameters(specified_neurons=specified_neurons)
        
    
    
    def assign_random_eigenvectors(self, scale_parameters=(10.0, 1./10.), ratio_parameters=(1.0, 1.0), specified_neurons = None, reset=False):
        '''
            Each neuron has a gaussian receptive field, defined by its eigenvectors/eigenvalues (principal axes and scale)
            Get those randomly:
                - Draw random scale ~ Gamma(.) , mean at given scale
                - Draw random ratios ~ Gamma(.) , use given concentration parameters
        '''
        if self.R > 2:
            raise NotImplementedError('Not assign_random_eigenvectors not done for R>2')
        
        if self.neurons_sigma is None or reset:
            self.neurons_sigma = np.zeros((self.M, self.R))
            self.neurons_angle = np.zeros(self.M)
            self.neurons_params = np.zeros((self.M, 3))
        
        if specified_neurons is None:
            specified_neurons = self._ALL_NEURONS
        
        # Sample eigenvalues: Draw scale and ratios
        scale_rnd = np.random.gamma(scale_parameters[0], scale_parameters[1], size=specified_neurons.size)
        ratio_rnd = np.random.gamma(ratio_parameters[0], ratio_parameters[1], size=specified_neurons.size)
        self.neurons_sigma[specified_neurons, 0] = np.sqrt(scale_rnd/ratio_rnd)
        self.neurons_sigma[specified_neurons, 1] = ratio_rnd*self.neurons_sigma[specified_neurons, 0]
        
        # Sample eigenvectors (only need to sample a rotation matrix)
        # TODO only for R=2 for now, should find a better way.
        # self.neurons_angle[specified_neurons] = np.random.random_integers(0, 1, size=specified_neurons.size)*np.pi/2.
        self.neurons_angle[specified_neurons] = np.pi*np.random.random(size=specified_neurons.size)
        
        self.compute_2d_parameters(specified_neurons=specified_neurons)

    
    def assign_scaled_eigenvectors(self, scale_parameters = (100.0, 0.001), ratio_parameters = (100., 0.001), specified_neurons = None, reset = False):
        '''
            Each neuron gets a gaussian receptive field, with a scale that becomes smaller as it's associate scale goes down as well.
        '''

        if self.R > 2:
            raise NotImplementedError('Not assign_random_eigenvectors not done for R>2')
        
        if self.neurons_sigma is None or reset:
            self.neurons_sigma = np.zeros((self.M, self.R))
            self.neurons_angle = np.zeros(self.M)
            self.neurons_params = np.zeros((self.M, 3))
        
        if specified_neurons is None:
            specified_neurons = self._ALL_NEURONS
        
        # First create a normal conjunctive population. Then will shrink it appropriately
        scale_rnd = np.random.gamma(scale_parameters[0], scale_parameters[1], size=specified_neurons.size)
        ratio_rnd = np.random.gamma(ratio_parameters[0], ratio_parameters[1], size=specified_neurons.size)

        self.neurons_sigma[specified_neurons, 0] = np.sqrt(scale_rnd/ratio_rnd)
        self.neurons_sigma[specified_neurons, 1] = ratio_rnd*self.neurons_sigma[specified_neurons, 0]

        # Shrink neurons according to their associated scale
        self.neurons_sigma[:self.neurons_scales.size, :] = self.neurons_sigma[:self.neurons_scales.size, :]/(2.**self.neurons_scales[:, np.newaxis])

        # Assign angles
        self.neurons_angle[specified_neurons] = np.pi*np.random.random(size=specified_neurons.size)
        
        # Compute parameters
        self.compute_2d_parameters(specified_neurons=specified_neurons)


    


    def compute_preferred_vectors(self, should_plot=False):
        '''
            Now using hyperspheres to wrap around, so all neurons' preferred stimulus are represented by unit vectors.
            Compute those from the given preferred angles (even though we should do something different now, if we want uniform coverage of the sphere)
        '''
        # Compute the vector version (to be precomputed)
        self.neurons_preferred_stimulus_vect = spherical_to_vect_array(self.neurons_preferred_stimulus)

        if should_plot:
            # Plot them, pretty
            self.plot_coverage_preferred_stimuli_3d_sphere()

    

    def compute_2d_parameters(self, specified_neurons=None):
        '''
            Assuming R=2, we have simple fast formulas to get the gaussian responses of neurons
        '''
        
        if specified_neurons is None:
            specified_neurons = self._ALL_NEURONS
        
        # Compute the 3 components of the covariance of the gaussian filter [a, b; b, c]
        for m in np.arange(specified_neurons.size):
            
            self.neurons_params[specified_neurons[m], 0] = np.cos(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 0]**2.) + np.sin(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 1]**2.)
            self.neurons_params[specified_neurons[m], 1] = -np.sin(2.*self.neurons_angle[specified_neurons[m]])/(4.*self.neurons_sigma[specified_neurons[m], 0]**2.) + np.sin(2.*self.neurons_angle[specified_neurons[m]])/(4.*self.neurons_sigma[specified_neurons[m], 1]**2.)
            self.neurons_params[specified_neurons[m], 2] = np.sin(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 0]**2.) + np.cos(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 1]**2.)
        
    
    #########################################################################################################


    def get_network_response(self, stimulus_input, params={}):
        '''
            Function hook for the current way to get the network response.
        '''
        if self.response_type == 'fisher':
            return self.get_network_response_vonmisesfisher(stimulus_input, params=params)
        elif self.response_type == 'wrong_wrap':
            return self.get_network_response_wrongwrap(stimulus_input, params=params)
        elif self.response_type == 'bivariate_fisher':
            return self.get_network_response_bivariatefisher(stimulus_input, params=params)
        
        
    
    def get_network_response_vonmisesfisher(self, stimulus_input, params={}):
        '''
            Compute the response of the network.

            Use a Von Mises-Fisher general distribution.

            Now computes intermediate vectors, could change everything to use vectors only.
        '''

        # Unpack parameters
        if 'kappa' in params:
            kappa = params['kappa']
        else:
            kappa = 10.
        if 'beta' in params:
            beta = params['beta']
        else:
            beta = 0.5
        if 'function_type' in params:
            function_type = params['function_type']
        else:
            function_type = 'vonmisesfisher'
        
        # Vector version of the stimulus
        # stimulus_input = np.array(stimulus_input)
        # stimulus_input[1] += np.pi
        stimulus_input_vect = spherical_to_vect(stimulus_input)
        
        # Diff
        # deltaV = (self.neurons_preferred_stimulus_vect-stimulus_input_vect)
        
        # Get the response
        normalisation = kappa/(2.*np.pi*(np.exp(kappa) - np.exp(-kappa)))
        # normalisation = 1.0

        # M_rot = create_2D_rotation_matrix(self.neurons_angle[0])
        # M_scale = np.diag((1.0, 1.0))
        # A = np.dot(M_rot, np.dot(M_scale, M_rot.T))
        
        if function_type == 'vonmisesfisher':
            # Von Mises-Fisher
            output = normalisation*np.exp(kappa*np.dot(self.neurons_preferred_stimulus_vect, stimulus_input_vect))
        elif function_type == 'kent_5':
            # 5-param Kent
            
            print "Work in progress..."

            axis1 = gs_ortho(np.array((0., 0., 1.)), self.neurons_preferred_stimulus_vect[7])
            axis2 = gs_ortho(np.array((0., 1., 0.)), axis1)

            output = normalisation*np.exp(kappa*np.dot(self.neurons_preferred_stimulus_vect, stimulus_input_vect) + beta*(np.dot(stimulus_input_vect, axis1)**2. - np.dot(stimulus_input_vect, axis2)**2.))
        elif function_type == 'kent_3':
            # 3-param Kent, simplified. From \cite{Kent2005}
            #  exp(k cos(theta) + beta sin^2(theta) cos(2 gamma)
            return np.ones(self.M)*np.exp(kappa*np.cos(stimulus_input[1]) + beta*np.sin(stimulus_input[1])**2.*np.cos(2.*stimulus_input[0]))

        else:
            raise ValueError('function_type unknown')

        # Von Mises-Fisher extension with rotation
        # TODO Doesn't work...
        # M_scale = np.array([[1.0, 0., 0.], [0., 1., 0.], [0., 0., 1.0]])
        # modified_stimulus = np.dot(M_scale, stimulus_input_vect)
        # modified_stimulus /= np.linalg.norm(modified_stimulus)
        # M_rot = create_3D_rotation_around_vector(np.array([0., 0., 1.]), -np.pi/3.0)
        # modified_stimulus = np.dot(M_rot, stimulus_input_vect)
        # output = normalisation*np.exp(kappa*np.dot(self.neurons_preferred_stimulus_vect, modified_stimulus))

        output[self.mask_neurons_unset] = 0.0
        
        return output
    

    def get_network_response_bivariatefisher(self, stimulus_input, params={}, variant='sin'):
        '''
            Compute the response of the network.

            Use a Von Mises-Fisher general distribution.

            Now computes intermediate vectors, could change everything to use vectors only.
        '''

        # Unpack params
        if 'kappas' in params:
            kappas = params['kappas']
        else:
            kappas = [1.0, 1.0, 0.0]
        
        # Diff angles
        dtheta = (stimulus_input[0] - self.neurons_preferred_stimulus[:, 0])
        dgamma = (stimulus_input[1] - self.neurons_preferred_stimulus[:, 1])

        # Get the response
        # normalisation = kappa/(2.*np.pi*(np.exp(kappa) - np.exp(-kappa)))
        normalisation = 1.0

        if variant == 'cos':
            output = normalisation*np.exp(kappas[0]*np.cos(dtheta) + kappas[1]*np.cos(dgamma) - kappas[2]*np.cos(dtheta-dgamma))
        elif variant == 'sin':
            output = normalisation*np.exp(kappas[0]*np.cos(dtheta) + kappas[1]*np.cos(dgamma) + kappas[2]*np.sin(dtheta)*np.sin(dgamma))
        else:
            raise ValueError("variant parameter should be either 'cos' or 'sin'")

        output[self.mask_neurons_unset] = 0.0

        # return np.ones(self.M)*np.exp(kappa*np.cos(stimulus_input[1]) + beta*np.sin(stimulus_input[1])**2.*(np.cos(stimulus_input[0])**2. - np.sin(stimulus_input[0])**2.)*np.sin(stimulus_input[1]))

        return output



    def get_network_response_wrongwrap(self, stimulus_input, params={}):
        '''
            Compute the response of the network.
            
            Could be optimized, for now only for one stimulus tuple
            
        '''

        # Unpack parameters
        # if 'weight' in params:
        #     weight = params['weight']
        # else:
        #     weight = 6.
        # if 'periodicity' in params:
        #     periodicity = params['periodicity']
        # else:
        #     periodicity = 0.5

        
        normalisation = 1.
        
        dx = (self.neurons_preferred_stimulus[:, 0] - stimulus_input[0])
        dy = (self.neurons_preferred_stimulus[:, 1] - stimulus_input[1])

        # TODO Change wrap around.

        # Wrap using sin
        # 0.5 adapts the spatial frequency
        # (actually already a grid cell like code, for other values of this)
        dx = 6.*np.sin(0.5*dx)
        dy = 6.*np.sin(0.5*dy)

        # cosdx = 1.*np.cos(-dx)
        # sindx = 1.*np.sin(-dx)
        # cosdy = 1.*np.cos(-dy)
        # sindy = 1.*np.sin(-dy)

        # Wrap differently
        # dx = np.fmin(dx, 2.*np.pi - dx)
        # dy = np.fmin(dy, 2.*np.pi - dy)
        
        
        if self.R == 2:
            output = normalisation*np.exp(-self.neurons_params[:, 0]*dx**2.0 - 2.*self.neurons_params[:, 1]*dx*dy - self.neurons_params[:, 2]*dy**2.0)
            # output = normalisation*np.exp(vonmises_param[0]*cosdx + vonmises_param[1]*cosdy + vonmises_param[2]*sindx*sindy)
            output[self.mask_neurons_unset] = 0.0
            return output
        elif self.R == 3:
            raise NotImplementedError('R=3 for factorial code...')
        else:
            # not unrolled
            raise NotImplementedError('R>3 for factorial code...')
    
    ####

    def compute_network_response_statistics(self, precision = 20, params = {}):
        '''
            Will compute the mean and covariance of the network output.
            These are used in some analytical expressions.

            They are currently estimated from samples, there might be a closed-form solution...
        '''

        if self.network_response_statistics is None:
            # Should compute it
            
            # Get the theta spaces
            feature_space1 = np.linspace(-np.pi, np.pi, precision, endpoint=False)

            # Sample responses to measure the statistics on
            responses = np.zeros((feature_space1.size**2, self.M))
            for theta1_i in xrange(feature_space1.size):
                for theta2_i in xrange(feature_space1.size):
                    responses[theta1_i*feature_space1.size + theta2_i] = self.get_network_response((feature_space1[theta1_i], feature_space1[theta2_i]), params=params)

            # Compute the mean and covariance
            computed_mean = np.mean(responses, axis=0)
            computed_cov = np.cov(responses.T)

            # Cache them
            self.network_response_statistics = {'mean': computed_mean, 'cov': computed_cov}

        # Return the cached values
        return self.network_response_statistics
        
    
    ########################################################################################################################


    def get_neuron_response(self, neuron_index, stimulus_input, params={}):
        '''
            Get the output of one specific neuron, for a specific stimulus
        '''

        return self.get_network_response(stimulus_input, params=params)[neuron_index]


    def sample_network_response(self, stimulus_input, sigma=0.2, params={}):
        '''
            Get a random response for the given stimulus.
            
            return: M
        '''
        
        return self.get_network_response(stimulus_input, params=params) + sigma*np.random.randn(self.M)
    

    def sample_multiple_network_response(self, stimuli_input, sigma=0.2, params={}):
        '''
            Get a set of random responses for multiple stimuli
            
            return: N x M
        '''
        
        nb_samples = stimuli_input.shape[0]
        net_samples = np.zeros((nb_samples, self.M))


        for i in np.arange(nb_samples):
            net_samples[i] = self.sample_network_response(stimuli_input[i], sigma=sigma, params=params)
        
        return net_samples

    
    def compute_covariance_stimulus(self, stimulus_input, N=2000, sigma=0.2, params={}):
        '''
            Compute the covariance for a given stimulus.
        '''

        # Same stim for all
        all_stim = np.tile(stimulus_input, (N, 1))

        # Get samples
        samples = self.sample_multiple_network_response(all_stim, sigma=sigma, params=params)

        # Get covariance
        return np.cov(samples.T)

    def compute_covariance_KL(self, precision=100, sigma_2=0.2, beta=1.0, params={}, should_plot= False):
        '''
            Compute the covariance of the Gaussian approximation (through a KL) to the averaged object.

            Sigma* = sigma^2 I + beta^2 Cov( mu(theta))_p(theta)
        '''

        # Get the statistics of the network population code
        network_response_statistics = self.compute_network_response_statistics(precision = precision, params=params)

        # The actual computation
        covariance = beta**2.*network_response_statistics['cov'] + beta**2.*sigma_2*np.eye(self.M)

        if should_plot  == True:
            plt.figure()
            plt.imshow(covariance, interpolation='nearest')
            plt.show()

        # Output it
        return covariance


    def compute_fisher_information(self, stimulus_input, sigma=0.01, cov_stim=None, params={}):
        '''
            Compute and return the Fisher information for the given stimulus.
            Assume we are looking for the FI in coordinate 1, fixing the other (in 2D).

            Assuming that Cov_stim ~ cst, we use:
            I = f' Cov_stim^-1 f'
        '''

        if cov_stim is None:
            # The covariance for the stimulus
            cov_stim = self.compute_covariance_stimulus(stimulus_input, sigma=sigma, params=params)


        # Compute the derivative of the receptive field
        dx = (self.neurons_preferred_stimulus[:, 0] - stimulus_input[0])
        dy = (self.neurons_preferred_stimulus[:, 1] - stimulus_input[1])

        sindx = np.sin(dx)
        coshalfdx = np.cos(0.5*dx)
        sinhalfdy = np.sin(0.5*dy)

        der_f = 6.**2.*(self.neurons_params[:, 0]*0.5*sindx + self.neurons_params[:, 1]*coshalfdx*sinhalfdy)*self.get_network_response(stimulus_input, params=params)

        der_f[np.isnan(der_f)] = 0.0

        # Now get the Fisher information
        return np.dot(der_f, np.linalg.solve(cov_stim, der_f))


    def compute_fisher_information_fullspace(self, sigma=0.01, cov_stim=None, precision=100, params={}):
        
        feature_space = np.linspace(-np.pi, np.pi, precision)
        
        activity = np.zeros((feature_space.size, feature_space.size))
        
        if cov_stim is None:
            cov_stim = self.compute_covariance_stimulus((0., 0.), sigma=sigma, params=params)

        # Compute the activity of that neuron over the whole space
        for i in np.arange(feature_space.size):
            for j in np.arange(feature_space.size):
                activity[i, j] = self.compute_fisher_information((feature_space[i], feature_space[j]), sigma=sigma, cov_stim=cov_stim, params=params)
            
        return activity
        

    def get_neuron_activity_fullspace(self, neuron_index, precision=100, return_axes_vect = False, params={}):
        '''
            Returns the activity of a specific neuron over the entire space.
        '''

        (feature_space1, feature_space2, activity) = self.init_feature_cover_matrices(precision)
        
        # Compute the activity of that neuron over the whole space
        for i in np.arange(feature_space1.size):
            for j in np.arange(feature_space2.size):
                activity[i, j] = self.get_neuron_response(neuron_index, (feature_space1[i], feature_space2[j]), params=params)
        
        if return_axes_vect:
            return activity, feature_space1, feature_space2
        else:
            return activity
        

    ######################## 

    def init_feature_cover_matrices(self, precision=20):
        '''
            Helper function, creating appropriate linspaces, depending on the chosen coordinate system.
        '''

        if self.coordinates == 'full_angles':
            feature_space1 = np.linspace(0., 2.*np.pi, precision)
            feature_space2 = np.linspace(0., 2.*np.pi, precision)
            cross_array = np.zeros((feature_space1.size, feature_space2.size))
        if self.coordinates == 'full_angles_sym':
            feature_space1 = np.linspace(-np.pi, np.pi, precision)
            feature_space2 = np.linspace(-np.pi, np.pi, precision)
            cross_array = np.zeros((feature_space1.size, feature_space2.size))
        if self.coordinates == 'spherical':
            # feature_space1 = np.linspace(-np.pi, np.pi, precision)
            # feature_space2 = np.linspace(0, np.pi, precision)
            feature_space1 = np.linspace(0., 2.*np.pi, precision)
            feature_space2 = np.linspace(0., np.pi, precision)
            cross_array = np.zeros((feature_space1.size, feature_space2.size))
        if self.coordinates == 'spherical_sym':
            feature_space1 = np.linspace(-np.pi, np.pi, precision)
            feature_space2 = np.linspace(0, np.pi, precision)
            cross_array = np.zeros((feature_space1.size, feature_space2.size))
        
        return (feature_space1, feature_space2, cross_array)


    ######################## PLOTS ######################################

    def plot_coverage_feature_space(self, nb_stddev=0.7, specified_neurons=None, alpha_ellipses=0.5, facecolor='rand', ax=None):
        '''
            Show the features (R=2 only)
        '''
        assert self.R == 2, "only works for R=2"
        
        if specified_neurons is None:
            specified_neurons = self._ALL_NEURONS
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        
        ells = [Ellipse(xy=self.neurons_preferred_stimulus[m], width=nb_stddev*self.neurons_sigma[m, 0], height=nb_stddev*self.neurons_sigma[m, 1], angle=-np.degrees(self.neurons_angle[m])) for m in specified_neurons if ~self.mask_neurons_unset[m]]
        
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(alpha_ellipses)
            if facecolor is 'rand':
                e.set_facecolor(np.random.rand(3))
            elif facecolor is False or facecolor is None or facecolor == 'none' or facecolor == 'None':
                e.set_facecolor('none')
            else:
                e.set_facecolor(facecolor)
            e.set_transform(ax.transData)
        
        # ax.autoscale_view()
        ax.set_xlim(-1.4*np.pi, 1.3*np.pi)
        ax.set_ylim(-1.4*np.pi, 1.3*np.pi)
        
        ax.set_xlabel('Color', fontsize=14)
        ax.set_ylabel('Orientation', fontsize=14)
        
        plt.show()

        return ax
    

    def plot_mean_activity(self, specified_neurons=None, params={}):
        '''
            Plot \sum_i \phi_i(x) at all x
        '''
        
        assert self.R == 2, "Only implemented for R=2"
        
        if specified_neurons is None:
            specified_neurons = self._ALL_NEURONS
        
        precision = 100

        (feature_space1, feature_space2, mean_activity) = self.init_feature_cover_matrices(precision)
        
        for feat1_i in np.arange(feature_space1.size):
            for feat2_i in np.arange(feature_space2.size):
                all_activity = self.get_network_response((feature_space1[feat1_i], feature_space2[feat2_i]), params=params)
                mean_activity[feat1_i, feat2_i] = np.sum(all_activity[specified_neurons])
        
        print "%.3f %.5f" % (np.mean(mean_activity), np.std(mean_activity.flatten()))
        f = plt.figure()
        ax = f.add_subplot(111)
        im = ax.imshow(mean_activity.T, origin='lower')
        im.set_extent((feature_space1.min(), feature_space1.max(), feature_space2.min(), feature_space2.max()))
        ax.set_xlabel('Color')
        ax.set_ylabel('Orientation')
        # im.set_interpolation('nearest')
        f.colorbar(im)
        
        plt.show()
        
    
    
    def plot_neuron_activity(self, neuron_index, nb_stddev=1., precision=100, params={}):
        '''
            Plot the activity of one specific neuron over the whole input space.
        '''
        
        activity, feature_space1, feature_space2 = self.get_neuron_activity_fullspace(neuron_index, precision=precisions, return_axes_vect=True, params=params)
        
        # Plot it
        f = plt.figure()
        ax = f.add_subplot(111)
        im= ax.imshow(activity.T, origin='lower')
        # im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        im.set_extent((feature_space1.min(), feature_space1.max(), feature_space2.min(), feature_space2.max()))
        ax.set_xlabel('Color')
        ax.set_ylabel('Orientation')
        # im.set_interpolation('nearest')
        f.colorbar(im)
        
        # Plot the ellipse showing one standard deviation
        e = Ellipse(xy=self.neurons_preferred_stimulus[neuron_index], width=nb_stddev*self.neurons_sigma[neuron_index, 0], height=nb_stddev*self.neurons_sigma[neuron_index, 1], angle=-np.degrees(self.neurons_angle[neuron_index]))
        
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.5)
        e.set_facecolor('white')
        e.set_transform(ax.transData)        
        
        plt.show()


    def plot_network_activity(self, stimulus_input, nb_stddev=1., params={}):
        '''
            Plot the activity of the network to a specific stimulus.
        '''
        M_sqrt = np.floor(self.M**0.5)

        # Get the network response
        activity = np.reshape(self.get_network_response(stimulus_input, params=params)[:int(M_sqrt**2.)], (M_sqrt, M_sqrt))
        
        
        # Plot it
        f = plt.figure()
        ax = f.add_subplot(111)
        im= ax.imshow(activity.T, origin='lower')
        im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        # im.set_extent((feature_space1.min(), feature_space1.max(), feature_space2.min(), feature_space2.max()))
        ax.set_xlabel('Color')
        ax.set_ylabel('Orientation')
        im.set_interpolation('nearest')
        f.colorbar(im)
        
        plt.show()    
    

    def plot_fisher_info_fullspace(self, sigma=0.01, cov_stim=None, precision=100, params={}):
        activity = self.compute_fisher_information_fullspace(sigma=sigma, cov_stim=cov_stim, precision=precision, params=params)
        # Plot it
        f = plt.figure()
        ax = f.add_subplot(111)
        im= ax.imshow(activity.T, origin='lower')
        im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        ax.set_xlabel('Color')
        ax.set_ylabel('Orientation')
        # im.set_interpolation('nearest')
        f.colorbar(im)

    
    def plot_coverage_preferred_stimuli_3d(self):
        '''
            Show the preferred stimuli coverage on a sphere/torus.
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.neurons_preferred_stimulus_vect[~self.mask_neurons_unset, 0], self.neurons_preferred_stimulus_vect[~self.mask_neurons_unset, 1], self.neurons_preferred_stimulus_vect[~self.mask_neurons_unset, 2])
        plt.show()  
    
    
    def plot_neuron_activity_3d(self, neuron_index, precision=20, weight_deform=0.5, params={}, draw_colorbar=True):
        '''
            Plot the activity of a neuron on the sphere/torus
        '''

        (feature_space1, feature_space2, activity) = self.init_feature_cover_matrices(precision)
        
        # Compute the activity of that neuron over the whole space
        for i in np.arange(feature_space1.size):
            for j in np.arange(feature_space2.size):
                activity[i, j] = self.get_neuron_response(neuron_index, (feature_space1[i], feature_space2[j]), params=params)
                # activity[i,j] = self.get_neuron_response(neuron_index, (feature_space[i], feature_space[j]))
        
        if self.coordinates == 'spherical':
            plot_sphere(feature_space1, feature_space2, activity, weight_deform=weight_deform)
        elif self.coordinates == 'full_angles':
            plot_torus(feature_space1, feature_space2, activity, weight_deform=weight_deform, draw_colorbar=draw_colorbar)
    

    def plot_mean_activity_3d(self, precision=20, weight_deform=0.5, params={}):
        '''
            Plot the mean activity of the network on a sphere/torus
        '''

        (feature_space1, feature_space2, activity) = self.init_feature_cover_matrices(precision)
        
        # Compute the activity of that neuron over the whole space
        for i in np.arange(feature_space1.size):
            for j in np.arange(feature_space2.size):
                activity[i, j] = np.sum(self.get_network_response((feature_space1[i], feature_space2[j]), params=params))
                
        if self.coordinates == 'spherical':
            plot_sphere(feature_space1, feature_space2, activity, weight_deform=weight_deform)
        elif self.coordinates == 'full_angles':
            plot_torus(feature_space1, feature_space2, activity, weight_deform=weight_deform)


    ##########################

    @classmethod
    def create_full_conjunctive(cls, M, R=2, sigma=0.2, scale_parameters = None, ratio_parameters = None, scale_moments=None, ratio_moments=None, debug=False):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code
        '''

        if debug:
            print "create conjunctive network"

        if scale_parameters is None or ratio_parameters is None:
            scale_parameters = (100., 0.01)
            ratio_parameters = (3.33333, 0.3)
        
        if scale_moments is not None:
            # We are given the desired mean and variance of the scale. Convert to appropriate Gamma parameters
            scale_parameters = (scale_moments[0]**2./scale_moments[1], scale_moments[1]/scale_moments[0])
        
        if ratio_moments is not None:
            # same
            ratio_parameters = (ratio_moments[0]**2./ratio_moments[1], ratio_moments[1]/ratio_moments[0])
        
        rn = RandomFactorialNetwork(M, R=R)

        rn.assign_random_eigenvectors(scale_parameters=scale_parameters, ratio_parameters=ratio_parameters, reset=True)
        
        rn.population_code_type = 'conjunctive'

        return rn

    
    @classmethod
    def create_full_features(cls, M, R=2, sigma=0.2, scale=0.3, ratio=40., nb_feature_centers=3, response_type = 'wrong_wrap'):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code

            TODO Only for R=2 here
        '''
        print "create feature network"

        rn = RandomFactorialNetwork(M, R=R, response_type=response_type)

        rn.assign_prefered_stimuli(tiling_type='2_features', reset=True, nb_feature_centers=nb_feature_centers)
        rn.assign_aligned_eigenvectors(scale=scale, ratio=ratio, specified_neurons = np.arange(M/2), reset=True)
        rn.assign_aligned_eigenvectors(scale=scale, ratio=-ratio, specified_neurons = np.arange(M/2, M))

        rn.population_code_type = 'features'

        return rn
    
    @classmethod
    def create_mixed(cls, M, R=2, sigma=0.2, ratio_feature_conjunctive = 0.5, conjunctive_parameters=None, feature_parameters=None):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code
        '''
        print "create mixed network"
        
        if conjunctive_parameters is None:
            ratio_concentration = 1.
            conj_scale_parameters = (200., 1/150.)
            conj_ratio_parameters = (ratio_concentration, 4./(3.*ratio_concentration))
        else:
            if 'scale_moments' in conjunctive_parameters:
                conj_scale_parameters = (conjunctive_parameters['scale_moments'][0]**2./conjunctive_parameters['scale_moments'][1], conjunctive_parameters['scale_moments'][1]/conjunctive_parameters['scale_moments'][0])
                conj_ratio_parameters = (conjunctive_parameters['ratio_moments'][0]**2./conjunctive_parameters['ratio_moments'][1], conjunctive_parameters['ratio_moments'][1]/conjunctive_parameters['ratio_moments'][0])
            else:
                conj_scale_parameters = conjunctive_parameters['scale_parameters']
                conj_ratio_parameters = conjunctive_parameters['ratio_parameters']
        
        if feature_parameters is None:
            feat_scale = 0.3
            feat_ratio = 20.0
        else:
            feat_scale = feature_parameters['scale']
            feat_ratio = feature_parameters['ratio']

            if 'nb_feature_centers' in feature_parameters:
                nb_feature_centers = feature_parameters['nb_feature_centers']
            else:
                nb_feature_centers = 3
        
        conj_subpop_size = int(M*ratio_feature_conjunctive)
        feat_subpop_size = M - conj_subpop_size

        print "Population sizes: conj: %d, feat: %d" % (conj_subpop_size, feat_subpop_size)
        
        rn = RandomFactorialNetwork(M, R=R)

        # Create the conjunctive subpopulation
        rn.assign_prefered_stimuli(tiling_type='conjunctive', reset=True, specified_neurons = np.arange(conj_subpop_size))
        rn.assign_random_eigenvectors(scale_parameters=conj_scale_parameters, ratio_parameters=conj_ratio_parameters, specified_neurons = np.arange(conj_subpop_size), reset=True)


        # Create the feature subpopulation        
        rn.assign_prefered_stimuli(tiling_type='2_features', specified_neurons = np.arange(conj_subpop_size, M), nb_feature_centers=nb_feature_centers)
        rn.assign_aligned_eigenvectors(scale=feat_scale, ratio=feat_ratio, specified_neurons = np.arange(conj_subpop_size, int(feat_subpop_size/2.+conj_subpop_size)))
        rn.assign_aligned_eigenvectors(scale=feat_scale, ratio=-feat_ratio, specified_neurons = np.arange(int(feat_subpop_size/2.+conj_subpop_size), M))
        
        rn.population_code_type = 'mixed'

        return rn
    
    @classmethod
    def create_wavelet(cls, M, R=2, scales_number=3, scale_parameters = None, ratio_parameters = None, scale_moments=(85.0, 0.001), ratio_moments=(1.0, 0.001)):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code
        '''
        print "create wavelet network"

        if scale_parameters is None or ratio_parameters is None:
            scale_parameters = (100., 0.01)
            ratio_parameters = (10000.0, 0.0001)

        if scale_moments is not None:
            # We are given the desired mean and variance of the scale. Convert to appropriate Gamma parameters
            scale_parameters = (scale_moments[0]**2./scale_moments[1], scale_moments[1]/scale_moments[0])
        
        if ratio_moments is not None:
            # same
            ratio_parameters = (ratio_moments[0]**2./ratio_moments[1], ratio_moments[1]/ratio_moments[0])
        
        rn = RandomFactorialNetwork(M, R=R)

        rn.assign_prefered_stimuli(tiling_type='wavelet', reset=True, scales_number = scales_number)
        rn.assign_scaled_eigenvectors(scale_parameters=scale_parameters, ratio_parameters=ratio_parameters, reset=True)
        
        rn.population_code_type = 'wavelet'

        return rn

    

if __name__ == '__main__':
    R = 2
    M = int(20**R)
    
    
    # Pure conjunctive code
    if False:
        rn = RandomFactorialNetwork.create_full_features(M, R=R, scale=0.3)
        ratio_concentration = 1.
        rn.assign_random_eigenvectors(scale_parameters=(100., 1/150.), ratio_parameters=(ratio_concentration, 4./(3.*ratio_concentration)), reset=True)
        rn.plot_coverage_feature_space()
        
        rn.plot_mean_activity()
    
    # Pure feature code
    if False:
        rn = RandomFactorialNetwork.create_full_features(M, R=R, scale=0.3)
        rn.assign_prefered_stimuli(tiling_type='2_features', reset=True, specified_neurons = np.arange(M/4), nb_feature_centers=1)
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=20.0, specified_neurons = np.arange(M/8), reset=True)
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=-20.0, specified_neurons = np.arange(M/8, M/4))
        
        rn.plot_coverage_feature_space(specified_neurons=np.arange(M/4))
        
        rn.plot_mean_activity(specified_neurons=np.arange(M/4))
    
    # Mix of two population
    if False:
        rn = RandomFactorialNetwork.create_full_features(M, R=R, scale=0.3)
        ratio_concentration= 2.0
        rn.assign_prefered_stimuli(tiling_type='conjunctive', reset=True, specified_neurons = np.arange(M/2))
        rn.assign_random_eigenvectors(scale_parameters=(5., 1/150.), ratio_parameters=(ratio_concentration, 4./(3.*ratio_concentration)), specified_neurons = np.arange(M/2), reset=True)
        
        rn.assign_prefered_stimuli(tiling_type='2_features', specified_neurons = np.arange(M/2, M))
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=20.0, specified_neurons = np.arange(M/2, 3*M/4))
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=-20.0, specified_neurons = np.arange(3*M/4, M))
        
        rn.plot_coverage_feature_space(specified_neurons=np.arange(M, step=1), no_facecolor=True)
        
        rn.plot_mean_activity()
    
    # Test different wrap-around
    if False:
        M = 350
        R = 2
        sigma_x = 0.1
        # Moments of scale: mean = volume of receptive field directly.
        rn = RandomFactorialNetwork.create_full_conjunctive(M, R=R, sigma=sigma_x, scale_moments=(2.0, 0.1), ratio_moments=(1.0, 0.2))
        # rn = RandomFactorialNetwork.create_full_features(M, R=R, scale=1.0, ratio=4.0, response_type = 'bivariate_fisher')
        # rn = RandomFactorialNetwork.create_wavelet(M, R=R, scale_moments=(85.0, 0.001), ratio_moments=(1.0, 0.001), scales_number=5)
        
        # params=dict(kappa=10., beta=0., kappas=[8.0, 1.0, 0.0])
        # rn.plot_neuron_activity(7, params=params)
        # rn.plot_neuron_activity_3d(7, precision=20, params=params, weight_deform=0.5)
        # rn.plot_neuron_activity(175)
        # rn.plot_neuron_activity_3d(175, precision=40)
        # rn.plot_neuron_activity(0)
        # rn.plot_neuron_activity_3d(0)
        # rn.plot_coverage_feature_space()
        # rn.plot_mean_activity()

        # plt.rcParams['font.size'] = 17
        # ax = rn.plot_coverage_feature_space(alpha_ellipses=0.2, nb_stddev=0.7, facecolor='green')
        # ax.set_xlim(-3.5, 3.5)
        # ax.set_ylim(-3.5, 3.5)
        # ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        # ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        # ax.set_yticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        # ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        # plt.xlabel('')
        # plt.ylabel('')
        plt.show()
    
    # Compute covariance
    if False:
        M = 300
        R =2
        sigma_x = 0.1
        
        rn.plot_mean_activity()
        cc = rn.compute_covariance_stimulus((0.0, 0.0), sigma=sigma_x)

    # Fisher information test
    if False:
        N_sqrt = 10.

        sigma_x_2 = (0.5)**2.
        kappa1 = 3.0
        kappa2 = 5.0

        # Get the mu and gamma (means of receptive fields)
        coverage_1D = np.linspace( -np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)
        means = np.array(cross(2*[coverage_1D.tolist()]))

        mu = means[:, 0]
        gamma = means[:, 1]

        precision = 1.
        stim_space = np.linspace(0, 2.*np.pi, precision, endpoint = False)
        stim_space = np.array(cross(stim_space, stim_space))


        # Check that \sum_i f_i() is ~constant
        print np.mean(np.sum(np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))
        print np.std(np.sum(np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))
        # plt.imshow(np.reshape(np.sum(np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0), (precision, precision)))
        # plt.colorbar()
        
        # Now check the dependence on N_sqrt
        all_N_sqrt = np.arange(1, 21)
        all_FI_11 = np.zeros(all_N_sqrt.size)
        all_FI_12 = np.zeros(all_N_sqrt.size)
        all_FI_22 = np.zeros(all_N_sqrt.size)
        all_FI_11_limN = np.zeros(all_N_sqrt.size)
        all_FI_22_limN = np.zeros(all_N_sqrt.size)

        for i, N_sqrt in enumerate(all_N_sqrt):
            # Get the mu and gamma (means of receptive fields)
            coverage_1D = np.linspace( -np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)
            means = np.array(cross(2*[coverage_1D.tolist()]))

            mu = means[:, 0]
            gamma = means[:, 1]

            precision = 1.
            stim_space = np.linspace(0, 2.*np.pi, precision, endpoint = False)
            stim_space = np.array(cross(stim_space, stim_space))

            # Full FI
            all_FI_11[i] = 1./sigma_x_2 * kappa1**2.*np.mean(np.sum(np.sin(stim_space[:, 0] - mu[:, np.newaxis])**2.*np.exp(2.*kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + 2.*kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))

            all_FI_12[i] = 1./sigma_x_2 * kappa1*kappa2*np.mean(np.sum(np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.sin(stim_space[:, 1] - gamma[:, np.newaxis])*np.exp(2.*kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + 2.*kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))
            
            all_FI_22[i] = 1./sigma_x_2 * kappa2**2.*np.mean(np.sum(np.sin(stim_space[:, 1] - gamma[:, np.newaxis])**2.*np.exp(2.*kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + 2.*kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))

            # Large N limit
            density = N_sqrt**2./(4.*np.pi**2.)
            all_FI_11_limN[i] = 1./sigma_x_2*density*2.*np.pi**2.*kappa1**2.0*scsp.i0(2*kappa2)*(scsp.i0(2*kappa1) - scsp.iv(2, 2*kappa1))
            all_FI_22_limN[i] = 1./sigma_x_2*density*2.*np.pi**2.*kappa2**2.0*scsp.i0(2*kappa1)*(scsp.i0(2*kappa2) - scsp.iv(2, 2*kappa2))


        plt.plot(all_N_sqrt**2., all_FI_11, all_N_sqrt**2., all_FI_11_limN, all_N_sqrt**2., all_FI_12, all_N_sqrt**2., all_FI_22, all_N_sqrt**2., all_FI_22_limN)
        plt.xlabel('Number of neurons')
        plt.ylabel('Fisher Information')
        # pcolor( (np.reshape(np.sum(np.sin(stim_space[:,0] - mu[:, np.newaxis])**2.*np.exp(2.*kappa1*np.cos(stim_space[:,0] - mu[:, np.newaxis]) + 2.*kappa2*np.cos(stim_space[:,1] - gamma[:, np.newaxis])), axis=0), (100,100))))


        if True:
            ####
            # Getting the 'correct' Fisher information: check the FI for an average object
            N_sqrt = 10.

            sigma_x_2 = (2.)**2.
            kappa1 = 3.0
            kappa2 = 4.0
            alpha = 0.9

            # Get the mu and gamma (means of receptive fields)
            coverage_1D = np.linspace( -np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)
            means = np.array(cross(2*[coverage_1D.tolist()]))

            mu = means[:, 0]
            gamma = means[:, 1]

            R = 2
            K = 2
            sigma_x = 2.0
            sigma_y = 0.2

            precision = 2.
            stim_space = np.linspace(0, 2.*np.pi, precision, endpoint = False)
            stim_space = np.array(cross(stim_space, stim_space))

            density = N_sqrt**2./(4.*np.pi**2.)

            T_all = np.arange(1, 2)
            FI_Tt = np.zeros((T_all.size, T_all.size))
            all_FI_11_Tt = np.zeros((T_all.size, T_all.size))
            covariance_all = np.zeros((T_all.size, T_all.size, int(N_sqrt**2.), int(N_sqrt**2.)))
            for i, T in enumerate(T_all):
                time_weights_parameters = dict(weighting_alpha=alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')
                rn = RandomFactorialNetwork.create_full_conjunctive(int(N_sqrt**2.), R=R, sigma=sigma_x, scale_moments=(2.0, 0.1), ratio_moments=(1.0, 0.2))
                data_gen_noise = DataGeneratorRFN(4000, T, rn, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=T-1, nb_stimulus_per_feature=K, enforce_min_distance=0.0)
                stat_meas = StatisticsMeasurer(data_gen_noise)
                covariance = stat_meas.model_parameters['covariances'][2][-1]
                    
                for j, t in enumerate(xrange(T)):
                    
                    covariance_all[i, j] = covariance

                    # Full FI
                    # all_FI_11_Tt[i, j] = alpha**(2.*T-2.*t)*1./sigma_x_2 * kappa1**2.*np.mean(np.sum(np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))*np.linalg.solve(covariance, np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))), axis=0))
                    all_FI_11_Tt[i, j] = alpha**(2.*T-2.*t + 2.)*1./sigma_x_2 * kappa1**2.*np.mean(np.sum(np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))*np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))

                    FI_Tt[i, j] = alpha**(2.*T-2.*t)*1./sigma_x_2*density*2.*np.pi**2.*kappa1**2.0*scsp.i0(2*kappa2)*(scsp.i0(2*kappa1) - scsp.iv(2, 2*kappa1))        

            all_FI_11_Tt[all_FI_11_Tt == 0.0] = np.nan

            plt.figure()
            plt.imshow(all_FI_11_Tt, interpolation='nearest', origin='left')
            plt.colorbar()


        plt.show()

    if True:
        # Compute KL approx of mixture by Gausian
        alpha = 0.9
        N_sqrt = 20.
        N = int(N_sqrt**2.)
        T = 1
        sigma_x = 1.0
        sigma_y = 0.2
        beta = 2.0

        time_weights_parameters = dict(weighting_alpha=alpha, weighting_beta = beta, specific_weighting = 0.1, weight_prior='uniform')
        rn = RandomFactorialNetwork.create_full_conjunctive(N, R=2, sigma=sigma_x, scale_moments=(2.0, 0.01), ratio_moments=(1.0, 0.01))
        data_gen_noise = DataGeneratorRFN(15000, T, rn, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=T-1, enforce_min_distance=0.0)
        stat_meas = StatisticsMeasurer(data_gen_noise)

        measured_cov = stat_meas.model_parameters['covariances'][-1][-1]
        computed_cov = rn.compute_covariance_KL(sigma_2=(sigma_x**2. + sigma_y**2.), precision=50, beta=beta)

        # Plot covariance
        plt.figure()
        plt.imshow(measured_cov)
        plt.colorbar()
        plt.title('Measured covariance')

        plt.figure()
        plt.imshow(computed_cov)
        plt.colorbar()
        plt.title('Computed covariance')

        # plot without the diag
        plt.figure()
        plt.imshow(measured_cov - np.diag(np.diag(measured_cov)))
        plt.colorbar()
        plt.title('Measured covariance (no diagonal)')

        leftover_stddev = np.std(measured_cov - np.diag(np.diag(measured_cov)))
        
        plt.figure()
        plt.imshow(computed_cov - np.diag(np.diag(computed_cov)))
        plt.colorbar()
        plt.title('Computed covariance (no diagonal)')

        plt.figure()
        plt.imshow(computed_cov - np.diag(np.diag(computed_cov)) + leftover_stddev*np.random.randn(N, N))
        plt.colorbar()
        plt.title('Computed covariance (no diagonal), added background noise')

        plt.show()


        # Denoise?
        im = measured_cov - np.diag(np.diag(measured_cov))
        # Compute the 2d FFT of the input image
        F = np.fft.fft2(im)

        # In the lines following, we'll make a copy of the original spectrum and
        # truncate coefficients.  NO immediate code is to be written right here.
        # Define the fraction of coefficients (in each direction) we keep
        keep_fraction = 0.08
        # Call ff a copy of the original transform.  Numpy arrays have a copy
        # method for this purpose.
        ff = F.copy() 
        # Set r and c to be the number of rows and columns of the array.
        #@ Hint: use the array's shape attribute.
        r, c = ff.shape

        # Set to zero all rows with indices between r*keep_fraction and
        # r*(1-keep_fraction):
        ff[r*keep_fraction:r*(1-keep_fraction)] = 0
        # ff[:r*keep_fraction, :] = 0
        # ff[(1.-keep_fraction)*r:, :] = 0
    
        # Similarly with the columns:
        ff[:, c*keep_fraction:c*(1-keep_fraction)] = 0
        # ff[:, :c*keep_fraction] = 0
        # ff[:, c*(1.-keep_fraction):] = 0

        # Reconstruct the denoised image from the filtered spectrum, keep only the
        # real part for display.
        #@ Hint: There's an inverse 2d fft in the np.fft module as well (don't
        #@ forget that you only want the real part).
        #@ Call the result im_new, 
        im_new = np.fft.ifft2(ff).real

        # Show the results
        #@ The code below already works, if you did everything above right.
        def plot_spectrum(F, amplify=1000):
            """Normalise, amplify and plot an amplitude spectrum."""

            # Note: the problem here is that we have a spectrum whose histogram is
            # *very* sharply peaked at small values.  To get a meaningful display, a
            # simple strategy to improve the display quality consists of simply
            # amplifying the values in the array and then clipping.

            # Compute the magnitude of the input F (call it mag).  Then, rescale mag by
            # amplify/maximum_of_mag.  Numpy arrays can be scaled in-place with ARR *=
            # number.  For the max of an array, look for its max method.
            mag = abs(F)
            mag *= amplify/mag.max()

            mag[mag > 1] = 1

            # Display: this one already works, if you did everything right with mag
            plt.imshow(mag, plt.cm.Blues)

        plt.figure()
        plt.subplot(221)
        plt.title('Original image')
        plt.imshow(im, plt.cm.gray)
        plt.subplot(222)
        plt.title('Fourier transform')
        plot_spectrum(F)
        plt.subplot(224)
        plt.title('Filtered Spectrum')
        plot_spectrum(ff)
        plt.subplot(223)
        plt.title('Reconstructed Image')
        plt.imshow(im_new, plt.cm.gray)
        # Adjust the spacing between subplots for readability 
        plt.subplots_adjust(hspace=0.32)
        plt.show()


