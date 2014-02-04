#!/usr/bin/env python
# encoding: utf-8
"""
randomfactorialnetwork.py

Created by Loic Matthey on 2011-11-09.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import pylab as plt
import numpy as np
import scipy as sp
from matplotlib.patches import Ellipse
import sys
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
import scipy.special as scsp
from scipy.spatial.distance import pdist

import progress

from utils import *


class RandomFactorialNetwork():
    """
        Modified paradigm for this Network. Uses a factorial representation of K features, and samples them using K-dimensional gaussian receptive fields.
            Randomness is in the distribution of orientations and radii of those gaussians.
    """
    def __init__(self, M, R=2, response_type = 'wrong_wrap', gain=1.0):

        assert R == 2, "RandomFactorialNetwork only implemented for R=2 for now"

        self.M = M
        self.K = 0
        self.R = R

        self.network_initialised = False
        self.population_code_type = None

        self.neurons_preferred_stimulus = None
        self.neurons_sigma = None
        self.mask_neurons_unset = None
        self.normalisation = None
        self.gain = gain

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

        # (hacky) Store the noise covariance matrix, for ease of use. Will be set by the Sampler.
        self.noise_covariance = None

        self.network_initialised = True


    def assign_prefered_stimuli(self, tiling_type='conjunctive', specific_neurons=None, reset=False, nb_feature_centers=3, scales_number=3):
        '''
            For all M factorial neurons, assign them a prefered stimulus tuple (e.g. (orientation, color) )
        '''

        if self.neurons_preferred_stimulus is None or reset:
            # Handle uninitialized neurons
            self.neurons_preferred_stimulus = np.nan*np.ones((self.M, self.R))

        if specific_neurons is None:
            specific_neurons = self._ALL_NEURONS

        if specific_neurons.size != 0:
            # Only do something if non-empty

            if tiling_type == 'conjunctive':
                self.assign_prefered_stimuli_conjunctive(specific_neurons)

            elif tiling_type == '2_features':
                self.assign_prefered_stimuli_2_features(specific_neurons, nb_feature_centers=nb_feature_centers)
            elif tiling_type == 'wavelet':
                self.assign_prefered_stimuli_wavelet(specific_neurons, scales_number=scales_number)

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

        for center_i in xrange(nb_feature_centers):
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
        for k in xrange(scales_number):
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


    def assign_aligned_eigenvectors(self, scale=1.0, ratio=1.0, specific_neurons=None, reset=False):
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

        if specific_neurons is None:
            specific_neurons = self._ALL_NEURONS

        assert ratio <= -1 or ratio >= 1, "respect my authority! Use ratio >= 1 or <=-1"

        if ratio>0:
            self.neurons_sigma[specific_neurons, 1] = ratio*scale
            self.neurons_sigma[specific_neurons, 0] = scale
            self.neurons_angle[specific_neurons] = 0.0

        elif ratio <0:
            self.neurons_sigma[specific_neurons, 1] = scale
            self.neurons_sigma[specific_neurons, 0] = -ratio*scale
            self.neurons_angle[specific_neurons] = 0.0

        # Update parameters
        self.precompute_parameters(specific_neurons=specific_neurons)


    def assign_random_eigenvectors(self, scale_parameters=(10.0, 1./10.), ratio_parameters=(1.0, 1.0), specific_neurons = None, reset=False):
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

        if specific_neurons is None:
            specific_neurons = self._ALL_NEURONS

        # Sample eigenvalues: Draw scale and ratios
        scale_rnd = np.random.gamma(scale_parameters[0], scale_parameters[1], size=specific_neurons.size)
        ratio_rnd = np.random.gamma(ratio_parameters[0], ratio_parameters[1], size=specific_neurons.size)
        self.neurons_sigma[specific_neurons, 0] = np.sqrt(scale_rnd/ratio_rnd)
        self.neurons_sigma[specific_neurons, 1] = ratio_rnd*self.neurons_sigma[specific_neurons, 0]

        # Sample eigenvectors (only need to sample a rotation matrix)
        # TODO only for R=2 for now, should find a better way.
        # self.neurons_angle[specific_neurons] = np.random.random_integers(0, 1, size=specific_neurons.size)*np.pi/2.
        self.neurons_angle[specific_neurons] = np.pi*np.random.random(size=specific_neurons.size)

        self.precompute_parameters(specific_neurons=specific_neurons)


    def assign_scaled_eigenvectors(self, scale_parameters = (100.0, 0.001), ratio_parameters = (100., 0.001), specific_neurons = None, reset = False):
        '''
            Each neuron gets a gaussian receptive field, with a scale that becomes smaller as it's associate scale goes down as well.
        '''

        if self.R > 2:
            raise NotImplementedError('Not assign_random_eigenvectors not done for R>2')

        if self.neurons_sigma is None or reset:
            self.neurons_sigma = np.zeros((self.M, self.R))
            self.neurons_angle = np.zeros(self.M)

        if specific_neurons is None:
            specific_neurons = self._ALL_NEURONS

        # First create a normal conjunctive population. Then will shrink it appropriately
        scale_rnd = np.random.gamma(scale_parameters[0], scale_parameters[1], size=specific_neurons.size)
        ratio_rnd = np.random.gamma(ratio_parameters[0], ratio_parameters[1], size=specific_neurons.size)

        self.neurons_sigma[specific_neurons, 0] = np.sqrt(scale_rnd/ratio_rnd)
        self.neurons_sigma[specific_neurons, 1] = ratio_rnd*self.neurons_sigma[specific_neurons, 0]

        # Shrink neurons according to their associated scale
        self.neurons_sigma[:self.neurons_scales.size, :] = self.neurons_sigma[:self.neurons_scales.size, :]/(2.**self.neurons_scales[:, np.newaxis])

        # Assign angles
        self.neurons_angle[specific_neurons] = np.pi*np.random.random(size=specific_neurons.size)

        # Compute parameters
        self.precompute_parameters(specific_neurons=specific_neurons)



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


    def precompute_parameters(self, specific_neurons=None):
        '''
            Function called to precompute different parameters to speed-up some computations
        '''
        if self.response_type == 'wrong_wrap':
            # The receptive fields are (wrong) gaussians, precompute its parameters
            self.compute_2d_parameters(specific_neurons=specific_neurons)

        elif self.response_type == 'bivariate_fisher':
            # Precompute the normalisation constant
            self.compute_normalising_constant_bivariatefisher(specific_neurons=specific_neurons)

            # Save the rc_scale used
            self.rc_scale = np.mean(self.neurons_sigma, axis=0)



    def compute_2d_parameters(self, specific_neurons=None):
        '''
            Assuming R=2, we have simple fast formulas to get the gaussian responses of neurons
        '''

        if specific_neurons is None:
            specific_neurons = self._ALL_NEURONS

        self.neurons_params = np.zeros((self.M, 3))

        # Compute the 3 components of the covariance of the gaussian filter [a, b; b, c]
        for m in xrange(specific_neurons.size):

            self.neurons_params[specific_neurons[m], 0] = np.cos(self.neurons_angle[specific_neurons[m]])**2./(2.*self.neurons_sigma[specific_neurons[m], 0]**2.) + np.sin(self.neurons_angle[specific_neurons[m]])**2./(2.*self.neurons_sigma[specific_neurons[m], 1]**2.)
            self.neurons_params[specific_neurons[m], 1] = -np.sin(2.*self.neurons_angle[specific_neurons[m]])/(4.*self.neurons_sigma[specific_neurons[m], 0]**2.) + np.sin(2.*self.neurons_angle[specific_neurons[m]])/(4.*self.neurons_sigma[specific_neurons[m], 1]**2.)
            self.neurons_params[specific_neurons[m], 2] = np.sin(self.neurons_angle[specific_neurons[m]])**2./(2.*self.neurons_sigma[specific_neurons[m], 0]**2.) + np.cos(self.neurons_angle[specific_neurons[m]])**2./(2.*self.neurons_sigma[specific_neurons[m], 1]**2.)


    def compute_normalising_constant_bivariatefisher(self, specific_neurons=None):
        '''
            Depending on neuron_sigma, we have different normalising constants per neurons.

            The full formula, for kappa3 != 0 is more complex, we do not use it for now:

            Z = 4 pi^2 \sum_{m=0}^\infty n_choose_k(2m, m) (\kappa_3^2/(4 kappa_1 kappa_2))^m I_m(kappa_1) I_m(kappa_2)

            Here, for \kappa_3=0, only m=0 used
        '''

        if self.normalisation is None:
            self.normalisation = np.zeros(self.M)

        # The normalising constant
        #   Overflows have happened, but they have no real consequence, as 1/inf = 0.0, appropriately.
        if specific_neurons is None:
            self.normalisation = 4.*np.pi**2.0*scsp.i0(self.neurons_sigma[:, 0])*scsp.i0(self.neurons_sigma[:, 1])/self.gain
        else:
            self.normalisation[specific_neurons] = 4.*np.pi**2.0*scsp.i0(self.neurons_sigma[specific_neurons, 0])*scsp.i0(self.neurons_sigma[specific_neurons, 1])/self.gain


    #########################################################################################################


    def get_network_response(self, stimulus_input=None, specific_neurons=None, params={}):
        '''
            Function hook for the current way to get the network response.
        '''

        if stimulus_input is None:
            stimulus_input = (0,)*self.R

        if self.response_type == 'fisher':
            print "Not bivariate, are you sure?"
            return self.get_network_response_vonmisesfisher(stimulus_input, specific_neurons=specific_neurons, params=params)
        elif self.response_type == 'wrong_wrap':
            print "Not bivariate, are you sure?"
            return self.get_network_response_wrongwrap(stimulus_input, specific_neurons=specific_neurons, params=params)
        elif self.response_type == 'bivariate_fisher':
            return self.get_network_response_bivariatefisher(stimulus_input, specific_neurons=specific_neurons, params=params)


    def get_network_response_vonmisesfisher(self, stimulus_input, specific_neurons=None, params={}):
        '''
            Compute the response of the network.

            Use a Von Mises-Fisher general distribution.

            Now computes intermediate vectors, could change everything to use vectors only.
        '''

        # TODO get_network_response_vonmisesfisher is not finished
        print "Unfinished function..."

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

        if specific_neurons is None:
            specific_neurons = self._ALL_NEURONS

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
            output = normalisation*np.exp(kappa*np.dot(self.neurons_preferred_stimulus_vect[specific_neurons], stimulus_input_vect))
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

        output[self.mask_neurons_unset[specific_neurons]] = 0.0

        return output


    def get_network_response_bivariatefisher(self, stimulus_input, specific_neurons=None, params={}, variant='cos'):
        '''
            Compute the response of the network.

            Use a Von Mises-Fisher general distribution.

            Now computes intermediate vectors, could change everything to use vectors only.
        '''

        if specific_neurons is None:
            # Diff angles
            dtheta = (stimulus_input[0] - self.neurons_preferred_stimulus[:, 0])
            dgamma = (stimulus_input[1] - self.neurons_preferred_stimulus[:, 1])

            # Get the response
            output = np.exp(self.neurons_sigma[:, 0]*np.cos(dtheta) + self.neurons_sigma[:, 1]*np.cos(dgamma))/self.normalisation

            output[self.mask_neurons_unset] = 0.0
        else:

            dtheta = (stimulus_input[0] - self.neurons_preferred_stimulus[specific_neurons, 0])
            dgamma = (stimulus_input[1] - self.neurons_preferred_stimulus[specific_neurons, 1])

            # Get the response
            output = np.exp(self.neurons_sigma[specific_neurons, 0]*np.cos(dtheta) + self.neurons_sigma[specific_neurons, 1]*np.cos(dgamma))/self.normalisation[specific_neurons]

            output[self.mask_neurons_unset[specific_neurons]] = 0.0

        return output

    def get_network_response_wrongwrap(self, stimulus_input, specific_neurons=None, params={}):
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

        if specific_neurons is None:
            specific_neurons = self._ALL_NEURONS

        normalisation = 1.

        dx = (self.neurons_preferred_stimulus[specific_neurons, 0] - stimulus_input[0])
        dy = (self.neurons_preferred_stimulus[specific_neurons, 1] - stimulus_input[1])

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
            output = normalisation*np.exp(-self.neurons_params[specific_neurons, 0]*dx**2.0 - 2.*self.neurons_params[specific_neurons, 1]*dx*dy - self.neurons_params[specific_neurons, 2]*dy**2.0)
            # output = normalisation*np.exp(vonmises_param[0]*cosdx + vonmises_param[1]*cosdy + vonmises_param[2]*sindx*sindy)
            output[self.mask_neurons_unset[specific_neurons]] = 0.0
            return output
        elif self.R == 3:
            raise NotImplementedError('R=3 for factorial code...')
        else:
            # not unrolled
            raise NotImplementedError('R>3 for factorial code...')


    def get_network_response_opt2d(self, theta1, theta2):
        '''
            Optimized version of the Bivariate fisher population code, for 2 angles
        '''
        # Get the response
        output = np.exp(self.neurons_sigma[:, 0]*np.cos((theta1 - self.neurons_preferred_stimulus[:, 0])) + self.neurons_sigma[:, 1]*np.cos((theta2 - self.neurons_preferred_stimulus[:, 1])))/self.normalisation

        output[self.mask_neurons_unset] = 0.0

        return output

    ####

    def compute_network_response_statistics(self, precision = 20, params = {}, ignore_cache=False):
        '''
            Will compute the mean and covariance of the network output.
            These are used in some analytical expressions.

            They are currently estimated from samples, there might be a closed-form solution...
        '''

        if ignore_cache or self.network_response_statistics is None:
            # Should compute it

            # Sample responses to measure the statistics on
            responses = self.collect_network_responses(precision=precision, params=params)

            responses.shape = (precision**2., self.M)

            # Compute the mean and covariance
            computed_mean = np.mean(responses, axis=0)
            computed_cov = np.cov(responses.T)

            # Cache them
            self.network_response_statistics = {'mean': computed_mean, 'cov': computed_cov}

        # Return the cached values
        return self.network_response_statistics


    def collect_network_responses(self, precision = 20, params = {}):
        '''
            Sample network responses (population code outputs) over the entire space, to be used for empirical estimates
        '''

        feature_space1 = np.linspace(-np.pi, np.pi, precision, endpoint=False)

        responses = np.zeros((feature_space1.size, feature_space1.size, self.M))
        for theta1_i in progress.ProgressDisplay(xrange(feature_space1.size), display=progress.SINGLE_LINE):
            for theta2_i in xrange(feature_space1.size):
                responses[theta1_i, theta2_i] = self.get_network_response((feature_space1[theta1_i], feature_space1[theta2_i]), params=params)

        return responses


    def compute_network_response_statistics_theoretical(self, marginalise_both_feature=False, theta2=0.0):
        '''
            Uses the theoretical expressions for the mean and covariance of the population code

            Either marginalise over 2 features, or only one with the other kept constant.
        '''

        if marginalise_both_feature:
            return self.compute_network_response_statistics_theoretical_marginalise_both()
        else:
            return self.compute_network_response_statistics_theoretical_marginalise_theta2(theta2=theta2)


    def compute_network_response_statistics_theoretical_marginalise_both(self):
        '''
            Uses the theoretical expressions for the mean and covariance of the population code

            Marginalize over the two features
        '''

        raise NotImplementedError()


    def compute_network_response_statistics_theoretical_marginalise_theta2(self, theta2=0.0):
        '''
            Uses the theoretical expressions for the mean and covariance of the population code

            Marginalize over theta1, keeping theta2 constant
        '''
        kappa1, kappa2 = self.rc_scale

        # Marginalize over theta1, keeping theta2 fixed
        mean_theoretical = np.exp(kappa2*np.cos(self.neurons_preferred_stimulus[:, 1] - theta2))/(4.*np.pi**2.*scsp.i0(kappa2))

        half_phij_minus_phii = (self.neurons_preferred_stimulus[:, 0][:, np.newaxis] + self.neurons_preferred_stimulus[:, 0])/2.
        half_psij_plus_psii = (self.neurons_preferred_stimulus[:, 1][:, np.newaxis] + self.neurons_preferred_stimulus[:, 1])/2.
        half_psij_minus_psii = (self.neurons_preferred_stimulus[:, 1][:, np.newaxis] - self.neurons_preferred_stimulus[:, 1])/2.
        cov_theoretical = (scsp.i0(2*kappa1*np.cos(half_phij_minus_phii)) - scsp.i0(kappa1)**2.)/(16.*np.pi**4.*scsp.i0(kappa1)**2.*scsp.i0(kappa2)**2.) * np.exp(2.*kappa2*np.cos(theta2 - half_psij_plus_psii)*np.cos(half_psij_minus_psii))

        network_response_statistics = {'mean': mean_theoretical, 'cov': cov_theoretical}

        return network_response_statistics


    ########################################################################################################################


    def get_neuron_response(self, neuron_index, stimulus_input, params={}):
        '''
            Get the output of one specific neuron, for a specific stimulus

            returns: Mx1 vector
        '''

        return self.get_network_response(stimulus_input, params=params, specific_neurons=np.array([neuron_index]))


    def sample_network_response(self, stimulus_input, sigma=0.2, params={}):
        '''
            Get a random response for the given stimulus.

            return: Mx1
        '''

        return self.get_network_response(stimulus_input, params=params) + sigma*np.random.randn(self.M)


    def sample_multiple_network_response(self, stimuli_input, sigma=0.2, params={}):
        '''
            Get a set of random responses for multiple stimuli

            return: N x M
        '''

        nb_samples = stimuli_input.shape[0]
        net_samples = np.zeros((nb_samples, self.M))


        for i in xrange(nb_samples):
            net_samples[i] = self.sample_network_response(stimuli_input[i], sigma=sigma, params=params)

        return net_samples


    def get_derivative_network_response(self, derivative_feature_target = 0, stimulus_input=None, kappa1=None, kappa2=None):
        '''
            Compute and return the derivative of the network response.
        '''

        if stimulus_input is None:
            stimulus_input = (0., 0.)

        if kappa1 is None:
            kappa1 = self.rc_scale[0]
        if kappa2 is None:
            kappa2 = self.rc_scale[1]

        der_f_0 = np.sin(stimulus_input[derivative_feature_target] - self.neurons_preferred_stimulus[:, derivative_feature_target])*np.exp(kappa1*np.cos(stimulus_input[0] - self.neurons_preferred_stimulus[:, 0]) + kappa2*np.cos(stimulus_input[1] - self.neurons_preferred_stimulus[:, 1]))

        der_f_0[self.mask_neurons_unset] = 0.0

        return der_f_0


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


    def compute_covariance_KL(self, precision=100, sigma_2=0.2, beta=1.0, T=1, params={}, should_plot=False, ignore_cache=False):
        '''
            Compute the covariance of the Gaussian approximation (through a KL) to the averaged object.

            Sigma* = T (sigma_y^2 + beta^2 sigma_x^2) I + T beta^2 Cov( mu(theta))_p(theta)
        '''

        # Get the statistics of the network population code
        network_response_statistics = self.compute_network_response_statistics(precision = precision, params=params, ignore_cache=ignore_cache)

        # The actual computation
        covariance = T*beta**2.*network_response_statistics['cov'] + T*sigma_2*np.eye(self.M)

        if should_plot:
            plt.figure()
            plt.imshow(covariance, interpolation='nearest')
            plt.show()

        # Output it
        return covariance


    def compute_fisher_information(self, stimulus_input=None, sigma=0.01, cov_stim=None, kappa_different=False, params={}):
        '''
            Compute and return the Fisher information for the given stimulus.
            Assume we are looking for the FI in coordinate 1, fixing the other (in 2D).

            Assuming that Cov_stim ~ cst, we use:
            I = f' Cov_stim^-1 f'
        '''

        if stimulus_input is None:
            stimulus_input = np.array([0.0, 0.0])

        if cov_stim is None:
            # The covariance for the stimulus
            cov_stim = self.compute_covariance_stimulus(stimulus_input, sigma=sigma, params=params)

        if kappa_different:
            # We need to keep the kappa_i_1 and kappa_i_2 in the sum, more general
            der_f_0 = self.neurons_sigma[:, 0]*np.sin(stimulus_input[0] - self.neurons_preferred_stimulus[:, 0])*np.exp(self.neurons_sigma[:, 0]*np.cos(stimulus_input[0] - self.neurons_preferred_stimulus[:, 0]) + self.neurons_sigma[:, 1]*np.cos(stimulus_input[1] - self.neurons_preferred_stimulus[:, 1]))/(4.*np.pi**2.0*scsp.i0(self.neurons_sigma[:, 0])*scsp.i0(self.neurons_sigma[:, 1]))

            der_f_0[self.mask_neurons_unset] = 0.0

            return np.dot(der_f_0, np.linalg.solve(cov_stim, der_f_0))

        else:
            # Same kappa for full code, easier.
            kappa1 = self.rc_scale[0]
            kappa2 = self.rc_scale[1]

            der_f_0 = self.get_derivative_network_response(stimulus_input=stimulus_input)

            return (kappa1**2./(16.*np.pi**4.*scsp.i0(kappa1)**2.*scsp.i0(kappa2)**2.))*np.dot(der_f_0, np.linalg.solve(cov_stim, der_f_0))

        # (kappa1**2./(sigma_x**2.*16.*np.pi**4.0*scsp.i0(kappa1)**2.0*scsp.i0(kappa2)**2.0)*np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))*np.linalg.solve(covariances_all[T_i], np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])))).mean(axis=-1).sum(axis=-1)



    def compute_fisher_information_theoretical(self, sigma=None, sigmax=None, kappa1=None, kappa2=None):
        '''
            Compute the theoretical, large N limit estimate of the Fisher Information
            This one assumes a diagonal covariance matrix, wrong for the complete model.
        '''

        if self.population_code_type == 'conjunctive':
            rho = 1./(4*np.pi**2/(self.M))
            # rho = 1./(2*np.pi/(self.M))
        elif self.population_code_type == 'feature':
            # M/2 neuron per 2pi dimension.
            rho = 1./(np.pi**2./self.M**2.)
        else:
            raise NotImplementedError('Fisher information not defined for population type ' + self.population_code_type)

        if sigmax is not None and sigma is None:
            # Compute sigma as sigma_x + cov(mu(\theta))
            computed_cov = self.compute_covariance_KL(precision=50, sigma_2=sigmax**2.)
            sigma = np.mean(np.diag(computed_cov))**0.5

        if kappa1 is None:
            kappa1 = self.rc_scale[0]

        if kappa2 is None:
            kappa2 = self.rc_scale[1]

        return kappa1**2.*rho*(scsp.i0(2*kappa1) - scsp.iv(2, 2*kappa1))*scsp.i0(2*kappa2)/(sigma**2.*8*np.pi**2.*scsp.i0(kappa1)**2.*scsp.i0(kappa2)**2.)
        # return kappa1**2.*rho*(scsp.i0(2*kappa1) - scsp.iv(2, 2*kappa1))*scsp.i0(2*kappa2)/(sigma**2.*16*np.pi**3.*scsp.i0(kappa1)**2.*scsp.i0(kappa2)**2.)


    def compute_fisher_information_fullspace(self, sigma=0.01, cov_stim=None, precision=100, params={}):

        feature_space = np.linspace(-np.pi, np.pi, precision)

        activity = np.zeros((feature_space.size, feature_space.size))

        if cov_stim is None:
            cov_stim = self.compute_covariance_stimulus((0., 0.), sigma=sigma, params=params)

        # Compute the activity of that neuron over the whole space
        for i in xrange(feature_space.size):
            for j in xrange(feature_space.size):
                activity[i, j] = self.compute_fisher_information(stimulus_input=np.array((feature_space[i], feature_space[j])), sigma=sigma, cov_stim=cov_stim, params=params)

        return activity


    def compute_inverse_fisher_info_kitems(self, additional_items_thetas, inv_cov_stim):
        '''
            Compute the Inverse Fisher Information matrix for K+1 items

            Assume the first item to be at (0, 0), returns the Inv FI for the first component of this first item.
        '''

        K = len(additional_items_thetas)

        # Set vector of all items/features derivatives
        deriv_mu = np.zeros((2*K+2, self.M))

        deriv_mu[0] = -self.neurons_sigma[:, 0]*np.sin(-self.neurons_preferred_stimulus[:, 0])*self.get_network_response_opt2d(0., 0.)
        deriv_mu[1] = -self.neurons_sigma[:, 1]*np.sin(-self.neurons_preferred_stimulus[:, 1])*self.get_network_response_opt2d(0., 0.)
        for i in xrange(K):
            deriv_mu[2*i+2] = -self.neurons_sigma[:, 0]*np.sin(additional_items_thetas[i][0] - self.neurons_preferred_stimulus[:, 0])*self.get_network_response_opt2d(additional_items_thetas[i][0], additional_items_thetas[i][1])
            deriv_mu[2*i+3] = -self.neurons_sigma[:, 1]*np.sin(additional_items_thetas[i][1] - self.neurons_preferred_stimulus[:, 1])*self.get_network_response_opt2d(additional_items_thetas[i][0], additional_items_thetas[i][1])

        deriv_mu[np.isnan(deriv_mu)] = 0.0

        # Compute the Fisher information matrix
        FI_nobj = np.dot(deriv_mu, np.dot(inv_cov_stim, deriv_mu.T))
        # FI_nobj = np.dot(deriv_mu, np.linalg.solve(cov_stim, deriv_mu.T))

        # print FI_nobj

        inv_FI_nobj = np.linalg.inv(FI_nobj)
        return inv_FI_nobj[0, 0], FI_nobj[0, 0]


    def compute_sample_inv_fisher_information_kitems(self, k_items, inv_cov_stim, min_distance=0.17):
        '''
            Compute one sample estimate of the Inverse Fisher Information, K_items.

            Those samples can then be averaged together in a Monte Carlo scheme to provide a better estimate of the Inverse Fisher Information for K items.

            Assume the first item is fixed at (0, 0), sample the other uniformly
        '''

        def get_sample_item():
            return -np.pi + 2*np.pi*np.random.random(2)

        all_items = [np.array([0.0, 0.0])]

        # Add extra items
        for item_i in xrange(k_items-1):
            new_item = get_sample_item()
            while not enforce_distance_set(new_item, all_items, min_distance):
                new_item = get_sample_item()

            all_items.append(new_item)

        inv_FI_allobj, FI_allobj = self.compute_inverse_fisher_info_kitems(all_items[1:], inv_cov_stim)

        return dict(inv_FI_kitems=inv_FI_allobj, FI_kitems=FI_allobj)


    def compute_marginal_inverse_FI(self, k_items, inv_cov_stim, max_n_samples=int(1e5), min_distance=0.1, convergence_epsilon = 1e-7, debug=False):
        '''
            Compute a Monte Carlo estimate of the Marginal Inverse Fisher Information
            Averages over stimuli values. Requires the inverse of the covariance of the memory.

            Returns dict(inv_FI, inv_FI_std, FI, FI_std)
        '''

        min_num_samples_std = int(2e3)

        inv_FI_cum = 0
        inv_FI_cum_var = 0
        FI_cum = 0
        FI_cum_var = 0
        inv_FI_estimate = 0
        inv_FI_std_estimate = 0
        FI_estimate = 0
        FI_std_estimate = 0
        epsilon = 0
        converged_times = 0

        previous_estimates = np.zeros(4)

        search_progress = progress.Progress(max_n_samples)
        for i in xrange(max_n_samples):

            if i % 1000 == 0 and debug:
                sys.stdout.write("%.1f%%, %s: %d %s %s %s %s %f\r" % (search_progress.percentage(), search_progress.time_remaining_str(), i, inv_FI_estimate, inv_FI_std_estimate, FI_estimate, FI_std_estimate, epsilon))
                sys.stdout.flush()

            # Get sample of invFI and FI
            sample_FI_dict = self.compute_sample_inv_fisher_information_kitems(k_items, inv_cov_stim, min_distance)

            # Compute mean
            inv_FI_cum += sample_FI_dict['inv_FI_kitems']/float(max_n_samples)
            FI_cum += sample_FI_dict['FI_kitems']/float(max_n_samples)

            # Compute std (wrong but oh well...)
            if i > min_num_samples_std:
                inv_FI_cum_var += (sample_FI_dict['inv_FI_kitems']-inv_FI_cum*max_n_samples/(i+1.))**2./float(max_n_samples)
                FI_cum_var += (sample_FI_dict['FI_kitems'] - FI_cum*max_n_samples/(i+1.))**2./float(max_n_samples)

            # Compute current running averages
            inv_FI_estimate         = inv_FI_cum*max_n_samples/(i+1.)
            inv_FI_std_estimate     = np.sqrt(inv_FI_cum_var*max_n_samples/(i+1.))/np.sqrt(i+1)
            FI_estimate             = FI_cum*max_n_samples/(i+1.)
            FI_std_estimate         = np.sqrt(FI_cum_var*max_n_samples/(i+1.))/np.sqrt(i+1)

            search_progress.increment()

            if i > 1.5*min_num_samples_std:
                # Check convergence
                new_estimates = np.array([inv_FI_estimate, inv_FI_std_estimate, FI_estimate, FI_std_estimate])
                epsilon = np.sum(np.abs(previous_estimates - new_estimates))

                if epsilon <= convergence_epsilon:
                    # Converged, stop the loop
                    if debug:
                        print "Converged after %d samples" % i

                    break

                previous_estimates = new_estimates

        if debug:
            print "\n"

        return dict(inv_FI=inv_FI_estimate, inv_FI_std=inv_FI_std_estimate, FI=FI_estimate, FI_std=FI_std_estimate)



    def get_neuron_activity(self, neuron_index, precision=100, return_axes_vect = False, params={}):
        '''
            Returns the activity of a specific neuron over the entire space.
        '''

        (feature_space1, feature_space2, activity) = self.init_feature_cover_matrices(precision)

        # Compute the activity of that neuron over the whole space
        for i in xrange(feature_space1.size):
            for j in xrange(feature_space2.size):
                activity[i, j] = self.get_neuron_response(neuron_index, (feature_space1[i], feature_space2[j]), params=params)


        if return_axes_vect:
            return activity, feature_space1, feature_space2
        else:
            return activity


    def get_mean_activity(self, precision=100, specific_neurons=None, return_axes_vect = False, params={}):
        '''
            Returns the mean activity of the network.
        '''


        (feature_space1, feature_space2, mean_activity) = self.init_feature_cover_matrices(precision)

        for feat1_i in xrange(feature_space1.size):
            for feat2_i in xrange(feature_space2.size):
                mean_activity[feat1_i, feat2_i] = np.mean(self.get_network_response((feature_space1[feat1_i], feature_space2[feat2_i]), specific_neurons=specific_neurons, params=params))

        if return_axes_vect:
            return mean_activity, feature_space1, feature_space2
        else:
            return mean_activity


    def compute_num_neurons_responsive_stimulus(self, percent_max=0.5, precision=100, should_plot=True, debug=True):
        '''
            Check the response of the network to a stimulus to find out how many neurons
            respond significantly to it.
            If this number is too small, then the coverage is too sparse or the receptive fields are too narrow.
        '''

        (feature_space1, feature_space2, _) = self.init_feature_cover_matrices(precision)
        network_activity = np.zeros((precision, precision, self.M))

        for feat1_i in xrange(feature_space1.size):
            for feat2_i in xrange(feature_space2.size):
                network_activity[feat1_i, feat2_i] = self.get_network_response((feature_space1[feat1_i], feature_space2[feat2_i]))

        num_responsive_neurons = np.sum(network_activity > network_activity.max()*percent_max, axis=-1)

        avg_num_responsive_neurons = np.mean(num_responsive_neurons)

        if debug:
            print "Average coverage: %.3f neurons responding at %.0f %%" % (avg_num_responsive_neurons, percent_max*100.)

        if should_plot:
            f = plt.figure()
            ax = f.add_subplot(111)
            im = ax.imshow(num_responsive_neurons.T, origin='lower')
            im.set_extent((feature_space1.min(), feature_space1.max(), feature_space2.min(), feature_space2.max()))
            ax.set_ylabel('Color')
            ax.set_xlabel('Orientation')
            # im.set_interpolation('nearest')
            f.colorbar(im, ticks=np.unique(num_responsive_neurons))
            ax.set_title('Number of neurons responding at %.0f %% of max' % (percent_max*100.))
            plt.show()

        return avg_num_responsive_neurons






    ########################

    def init_feature_cover_matrices(self, precision=20, endpoint=True):
        '''
            Helper function, creating appropriate linspaces, depending on the chosen coordinate system.
        '''

        if self.coordinates == 'full_angles':
            feature_space1 = self.init_feature_space(precision=precision, endpoint=endpoint)
            feature_space2 = self.init_feature_space(precision=precision, endpoint=endpoint)
            cross_array = np.zeros((feature_space1.size, feature_space2.size))
        if self.coordinates == 'full_angles_sym':
            feature_space1 = self.init_feature_space(precision=precision, endpoint=endpoint)
            feature_space2 = self.init_feature_space(precision=precision, endpoint=endpoint)
            cross_array = np.zeros((feature_space1.size, feature_space2.size))
        if self.coordinates == 'spherical':
            # feature_space1 = np.linspace(-np.pi, np.pi, precision)
            # feature_space2 = np.linspace(0, np.pi, precision)
            feature_space1 = self.init_feature_space(precision=precision, endpoint=endpoint)
            feature_space2 = self.init_feature_space(precision=precision, endpoint=endpoint)
            cross_array = np.zeros((feature_space1.size, feature_space2.size))
        if self.coordinates == 'spherical_sym':
            feature_space1 = self.init_feature_space(precision=precision, endpoint=endpoint)
            feature_space2 = self.init_feature_space(precision=precision, endpoint=endpoint)
            cross_array = np.zeros((feature_space1.size, feature_space2.size))

        return (feature_space1, feature_space2, cross_array)


    def init_feature_space(self, precision=20, endpoint=True):
        '''
            Initialise the appropriate 1D linspace depending on the coordinate system.
        '''

        if self.coordinates == 'full_angles':
            feature_space = np.linspace(0., 2.*np.pi, precision)
        if self.coordinates == 'full_angles_sym':
            feature_space = np.linspace(-np.pi, np.pi, precision, endpoint=endpoint)
        if self.coordinates == 'spherical':
            feature_space = np.linspace(0., 2.*np.pi, precision)
        if self.coordinates == 'spherical_sym':
            feature_space = np.linspace(-np.pi, np.pi, precision)

        return feature_space



    ######################## PLOTS ######################################

    def plot_coverage_feature_space(self, nb_stddev=1.0, specific_neurons=None, alpha_ellipses=0.5, facecolor='rand', ax=None, lim_factor=1.0, width_factor=1., height_factor=1.0):
        '''
            Show the features (R=2 only)
        '''
        assert self.R == 2, "only works for R=2"

        if specific_neurons is None:
            specific_neurons = self._ALL_NEURONS

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')


        ells = [Ellipse(xy=self.neurons_preferred_stimulus[m], width=width_factor*nb_stddev*kappa_to_stddev(self.neurons_sigma[m, 0]), height=height_factor*nb_stddev*kappa_to_stddev(self.neurons_sigma[m, 1]), angle=-np.degrees(self.neurons_angle[m])) for m in specific_neurons if ~self.mask_neurons_unset[m]]

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
        # ax.set_xlim(-1.4*np.pi, 1.3*np.pi)
        # ax.set_ylim(-1.4*np.pi, 1.3*np.pi)
        ax.set_xlim(-lim_factor*np.pi, lim_factor*np.pi)
        ax.set_ylim(-lim_factor*np.pi, lim_factor*np.pi)

        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=17)
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=17)

        ax.set_xlabel('Orientation', fontsize=14)
        ax.set_ylabel('Color', fontsize=14)

        plt.draw()

        plt.show()

        return ax


    def plot_mean_activity(self, precision=100, specific_neurons=None, params={}):
        '''
            Plot \sum_i \phi_i(x) at all x
        '''

        assert self.R == 2, "Only implemented for R=2"

        (mean_activity, feature_space1, feature_space2) =  self.get_mean_activity(precision=precision, specific_neurons=specific_neurons, params=params, return_axes_vect=True)

        print "%.3f %.5f" % (np.mean(mean_activity), np.std(mean_activity.flatten()))

        f = plt.figure()
        ax = f.add_subplot(111)
        im = ax.imshow(mean_activity.T, origin='lower')
        im.set_extent((feature_space1.min(), feature_space1.max(), feature_space2.min(), feature_space2.max()))
        ax.set_ylabel('Color')
        ax.set_xlabel('Orientation')
        # im.set_interpolation('nearest')
        f.colorbar(im)

        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=17)
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=17)

        plt.show()


    def plot_mean_activity_fft(self, precision=100, specific_neurons=None, params={}):
        '''
            Plot \sum_i \phi_i(x) at all x
        '''

        assert self.R == 2, "Only implemented for R=2"

        (mean_activity, feature_space1, feature_space2) =  self.get_mean_activity(precision=precision, specific_neurons=specific_neurons, params=params, return_axes_vect=True)

        plot_fft2_power(mean_activity)

        plt.show()


    def plot_neuron_activity(self, neuron_index=0, nb_stddev=1., precision=100, params={}):
        '''
            Plot the activity of one specific neuron over the whole input space.
        '''

        activity, feature_space1, feature_space2 = self.get_neuron_activity(neuron_index, precision=precision, return_axes_vect=True, params=params)

        # Plot it
        f = plt.figure()
        ax = f.add_subplot(111)
        im= ax.imshow(activity.T, origin='lower')
        # im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        im.set_extent((feature_space1.min(), feature_space1.max(), feature_space2.min(), feature_space2.max()))
        ax.set_ylabel('Color')
        ax.set_xlabel('Orientation')
        # im.set_interpolation('nearest')
        f.colorbar(im)

        # Plot the ellipse showing one standard deviation
        e = Ellipse(xy=self.neurons_preferred_stimulus[neuron_index], width=nb_stddev*kappa_to_stddev(self.neurons_sigma[neuron_index, 0]), height=nb_stddev*kappa_to_stddev(self.neurons_sigma[neuron_index, 1]), angle=-np.degrees(self.neurons_angle[neuron_index]))

        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.5)
        e.set_facecolor('white')
        e.set_transform(ax.transData)

        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=17)
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=17)

        # Change mouse over behaviour
        def report_pixel(x_mouse, y_mouse):
            # Extract loglik at that position
            try:
                x_display = x_mouse
                y_display = y_mouse

                x_i = np.argmin((feature_space1 - x_display)**2.)
                y_i = np.argmin((feature_space2 - y_display)**2.)

                return "x=%.2f y=%.2f value=%.2f" % (x_display, y_display, activity[x_i, y_i])
            except:
                return ""

        ax.format_coord = report_pixel

        plt.show()


    def plot_neuron_activity_1d(self, neuron_index=0, axis_to_vary=0, fix_preferred_stim=True, fixed_stim=None, precision=100., params={}, normalise=True):
        '''
            Plot the activity of a neuron along one dimension. Either provide the stimulus to fix, or let it
            be at its preferred stimulus.
        '''

        feature_space = self.init_feature_space(precision=precision)
        activity = np.zeros(feature_space.shape)

        # Fix the rest of the stimulus
        if fix_preferred_stim:
            stimulus = self.neurons_preferred_stimulus[neuron_index].copy()
        elif fixed_stim is not None:
            stimulus = fixed_stim
        else:
            stimulus = np.zeros(self.R)

        # Compute the response.
        for i in xrange(feature_space.size):
            stimulus[axis_to_vary] = feature_space[i]
            activity[i] = self.get_neuron_response(neuron_index, stimulus, params=params)

        # Check the gaussian fit
        gaussian_approx = spst.norm.pdf(feature_space, self.neurons_preferred_stimulus[neuron_index, axis_to_vary], kappa_to_stddev(self.neurons_sigma[neuron_index, axis_to_vary]))

        if normalise:
            activity /= np.max(activity)
            gaussian_approx /= np.max(gaussian_approx)

        # Plot everything
        plt.figure()
        plt.plot(feature_space, activity)
        plt.plot(feature_space, gaussian_approx)

        plt.legend(['Neuron activity', 'Gaussian approximation'])

        plt.show()



    def plot_network_activity(self, stimulus_input=(0.0, 0.0), nb_stddev=1., ax_handle=None, params={}):
        '''
            Plot the activity of the network to a specific stimulus.
        '''
        M_sqrt = np.floor(self.M**0.5)

        # Get the network response
        activity = np.reshape(self.get_network_response(stimulus_input, params=params)[:int(M_sqrt**2.)], (M_sqrt, M_sqrt))

        # Plot it
        if ax_handle is None:
            f = plt.figure()
            ax_handle = f.add_subplot(111)

        im= ax_handle.imshow(activity.T, origin='lower')
        im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        # im.set_extent((feature_space1.min(), feature_space1.max(), feature_space2.min(), feature_space2.max()))
        ax_handle.set_ylabel('Color')
        ax_handle.set_xlabel('Orientation')
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
        ax.set_ylabel('Color')
        ax.set_xlabel('Orientation')
        # im.set_interpolation('nearest')
        f.colorbar(im)


    def plot_coverage_preferred_stimuli_3d(self):
        '''
            Show the preferred stimuli coverage on a sphere/torus.
        '''

        if self.coordinates == 'full_angles_sym':
            scatter3d_torus(self.neurons_preferred_stimulus[:, 0], self.neurons_preferred_stimulus[:, 1])
        elif self.coordinates == 'spherical':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.neurons_preferred_stimulus_vect[~self.mask_neurons_unset, 0], self.neurons_preferred_stimulus_vect[~self.mask_neurons_unset, 1], self.neurons_preferred_stimulus_vect[~self.mask_neurons_unset, 2])
            plt.show()


    def plot_neuron_activity_3d(self, neuron_index=0, precision=20, weight_deform=0.5, params={}, draw_colorbar=True):
        '''
            Plot the activity of a neuron on the sphere/torus
        '''

        (feature_space1, feature_space2, activity) = self.init_feature_cover_matrices(precision)

        # Compute the activity of that neuron over the whole space
        for i in xrange(feature_space1.size):
            for j in xrange(feature_space2.size):
                activity[i, j] = self.get_neuron_response(neuron_index, (feature_space1[i], feature_space2[j]), params=params)
                # activity[i,j] = self.get_neuron_response(neuron_index, (feature_space[i], feature_space[j]))

        if self.coordinates == 'spherical':
            plot_sphere(feature_space1, feature_space2, activity, weight_deform=weight_deform)
        elif self.coordinates == 'full_angles' or self.coordinates == 'full_angles_sym':
            plot_torus(feature_space1, feature_space2, activity, weight_deform=weight_deform, draw_colorbar=draw_colorbar)
        else:
            raise ValueError('Unsupported self.coordinates')


    def plot_mean_activity_3d(self, precision=20, specific_neurons=None, weight_deform=0.5, params={}):
        '''
            Plot the mean activity of the network on a sphere/torus
        '''

        (mean_activity, feature_space1, feature_space2) =  self.get_mean_activity(precision=precision, specific_neurons=specific_neurons, params=params, return_axes_vect=True)

        if self.coordinates == 'spherical':
            plot_sphere(feature_space1, feature_space2, mean_activity, weight_deform=weight_deform)
        elif self.coordinates == 'full_angles' or self.coordinates == 'full_angles_sym':
            plot_torus(feature_space1, feature_space2, mean_activity, weight_deform=weight_deform)
        else:
            raise ValueError('Unsupported self.coordinates')


    ##########################

    @classmethod
    def create_full_conjunctive(cls, M, R=2, rcscale=None, autoset_parameters=False, scale_parameters = (100., 0.01), ratio_parameters = (3.333, 0.3), scale_moments=None, ratio_moments=None, debug=True, response_type = 'bivariate_fisher', gain=1.0):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code
        '''

        if debug:
            print "create conjunctive network, M %d, autoset: %d" % (M, autoset_parameters)

        rn = RandomFactorialNetwork(M, R=R, response_type=response_type, gain=gain)
        rn.population_code_type = 'conjunctive'

        ## Create receptive fields

        if autoset_parameters:
            # We use the optimum heuristic for the rc_scale: try to cover the space fully, assuming uniform coverage with squares of size 2*(2*kappa_to_stddev(kappa)). We assume that 2*stddev gives a good approximation to the appropriate coverage required.
            rcscale = stddev_to_kappa(2.*np.pi/int(M**0.5))

        if rcscale:
            # Assume we construct a conjunctive with ratio 1, no need to get random eigenvectors
            rn.assign_aligned_eigenvectors(scale=rcscale, ratio=1.0, reset=True)

        else:
            if scale_moments is not None:
                if response_type == 'bivariate_fisher':
                    # For bivariate fisher, we have theory that requires kappa to be of the good scale, not sqrt'ed here.
                    scale_moments = (scale_moments[0]**2., scale_moments[1])

                # We are given the desired mean and variance of the scale. Convert to appropriate Gamma parameters
                scale_parameters = (scale_moments[0]**2./scale_moments[1], scale_moments[1]/scale_moments[0])

            if ratio_moments is not None:
                # same
                ratio_parameters = (ratio_moments[0]**2./ratio_moments[1], ratio_moments[1]/ratio_moments[0])

            # Create random eigenvectors
            rn.assign_random_eigenvectors(scale_parameters=scale_parameters, ratio_parameters=ratio_parameters, reset=True)

        return rn

    @classmethod
    def create_full_features(cls, M, R=2, scale=0.3, ratio=40., autoset_parameters=False, nb_feature_centers=3, response_type = 'bivariate_fisher', gain=1.0):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code

            TODO Only for R=2 here
        '''
        print "create feature network, autoset: %d" % autoset_parameters

        rn = RandomFactorialNetwork(M, R=R, response_type=response_type, gain=gain)

        if autoset_parameters:
            # Use optimal values for the parameters. Be careful, this assumes M/2 and coverage of full 2 pi space
            # Assume one direction should cover width = pi, the other should cover M/2 * width/2. = 2pi
            # width = stddev_to_kappa(stddev)

            scale = stddev_to_kappa(np.pi)
            scale2 = stddev_to_kappa(2.*np.pi/int(M/2.))
            ratio = scale2/scale

        rn.assign_prefered_stimuli(tiling_type='2_features', reset=True, nb_feature_centers=nb_feature_centers)
        rn.assign_aligned_eigenvectors(scale=scale, ratio=-ratio, specific_neurons = np.arange(M/2), reset=True)
        rn.assign_aligned_eigenvectors(scale=scale, ratio=ratio, specific_neurons = np.arange(M/2, M))

        rn.population_code_type = 'feature'

        return rn

    @classmethod
    def create_mixed(cls, M, R=2, ratio_feature_conjunctive = 0.5, conjunctive_parameters=None, feature_parameters=None, response_type = 'bivariate_fisher', autoset_parameters=False):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code
        '''
        print "create mixed network"

        conj_scale = 0.0
        feat_scale = 0.0
        feat_ratio = 0.0

        if conjunctive_parameters is None:
            ratio_concentration = 1.
            conj_scale_parameters = (200., 1/150.)
            conj_ratio_parameters = (ratio_concentration, 4./(3.*ratio_concentration))
        else:
            if 'scale_moments' in conjunctive_parameters:
                if response_type == 'bivariate_fisher':
                    # For bivariate fisher, we have theory that requires kappa to be of the good scale, not sqrt'ed here.
                    conjunctive_parameters['scale_moments'][0] = conjunctive_parameters['scale_moments'][0]**2.

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
            feat_ratio = -feature_parameters['ratio']

            if 'nb_feature_centers' in feature_parameters:
                nb_feature_centers = feature_parameters['nb_feature_centers']
            else:
                nb_feature_centers = 3

        rn = RandomFactorialNetwork(M, R=R, response_type=response_type)

        rn.conj_subpop_size = int(M*ratio_feature_conjunctive)
        rn.feat_subpop_size = M - rn.conj_subpop_size


        if autoset_parameters:
            # Use optimal values for the parameters. Be careful, this assumes M/2 and coverage of full 2 pi space
            # Assume one direction should cover width = pi, the other should cover M/2 * width/2. = 2pi
            # width = stddev_to_kappa(stddev)
            if rn.conj_subpop_size > 0:
                conj_scale = stddev_to_kappa(2.*np.pi/int(rn.conj_subpop_size**0.5))
            if rn.feat_subpop_size > 0:
                feat_scale = stddev_to_kappa(np.pi)
                feat_ratio = -stddev_to_kappa(2.*np.pi/int(rn.feat_subpop_size/2.))/feat_scale


        print "Population sizes: ratio: %.1f conj: %d, feat: %d, autoset: %d" % (ratio_feature_conjunctive, rn.conj_subpop_size, rn.feat_subpop_size, autoset_parameters)

        # Create the conjunctive subpopulation
        rn.assign_prefered_stimuli(tiling_type='conjunctive', reset=True, specific_neurons = np.arange(rn.conj_subpop_size))
        if autoset_parameters:
            rn.assign_aligned_eigenvectors(scale=conj_scale, ratio=1.0, specific_neurons = np.arange(rn.conj_subpop_size), reset=True)
        else:
            rn.assign_random_eigenvectors(scale_parameters=conj_scale_parameters, ratio_parameters=conj_ratio_parameters, specific_neurons = np.arange(rn.conj_subpop_size), reset=True)


        # Create the feature subpopulation
        rn.assign_prefered_stimuli(tiling_type='2_features', specific_neurons = np.arange(rn.conj_subpop_size, M), nb_feature_centers=nb_feature_centers)
        rn.assign_aligned_eigenvectors(scale=feat_scale, ratio=feat_ratio, specific_neurons = np.arange(rn.conj_subpop_size, int(rn.feat_subpop_size/2.+rn.conj_subpop_size)))
        rn.assign_aligned_eigenvectors(scale=feat_scale, ratio=-feat_ratio, specific_neurons = np.arange(int(rn.feat_subpop_size/2.+rn.conj_subpop_size), M))

        rn.population_code_type = 'mixed'

        return rn

    @classmethod
    def create_wavelet(cls, M, R=2, scales_number=3, scale_parameters = None, ratio_parameters = None, scale_moments=(85.0, 0.001), ratio_moments=(1.0, 0.001), response_type='wrong_wrap'):
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

        rn = RandomFactorialNetwork(M, R=R, response_type=response_type)

        rn.assign_prefered_stimuli(tiling_type='wavelet', reset=True, scales_number = scales_number)
        rn.assign_scaled_eigenvectors(scale_parameters=scale_parameters, ratio_parameters=ratio_parameters, reset=True)

        rn.population_code_type = 'wavelet'

        return rn



if __name__ == '__main__':
    from statisticsmeasurer import *
    from datagenerator import *

    R = 2
    M = int(20**R)


    # Pure conjunctive code
    if False:

        rn = RandomFactorialNetwork.create_full_conjunctive(M, R=R, rcscale=100)

        rn.plot_coverage_feature_space()

        # rn.plot_mean_activity()
        rn.plot_neuron_activity(15)

    # Pure feature code
    if False:
        rn = RandomFactorialNetwork.create_full_features(M, R=R, scale=0.3)
        rn.assign_prefered_stimuli(tiling_type='2_features', reset=True, specific_neurons = np.arange(M/4), nb_feature_centers=1)
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=20.0, specific_neurons = np.arange(M/8), reset=True)
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=-20.0, specific_neurons = np.arange(M/8, M/4))

        rn.plot_coverage_feature_space(specific_neurons=np.arange(M/4))

        rn.plot_mean_activity(specific_neurons=np.arange(M/4))

    # Mix of two population
    if False:
        rn = RandomFactorialNetwork.create_full_features(M, R=R, scale=0.3)
        ratio_concentration= 2.0
        rn.assign_prefered_stimuli(tiling_type='conjunctive', reset=True, specific_neurons = np.arange(M/2))
        rn.assign_random_eigenvectors(scale_parameters=(5., 1/150.), ratio_parameters=(ratio_concentration, 4./(3.*ratio_concentration)), specific_neurons = np.arange(M/2), reset=True)

        rn.assign_prefered_stimuli(tiling_type='2_features', specific_neurons = np.arange(M/2, M))
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=20.0, specific_neurons = np.arange(M/2, 3*M/4))
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=-20.0, specific_neurons = np.arange(3*M/4, M))

        rn.plot_coverage_feature_space(specific_neurons=np.arange(M, step=1), no_facecolor=True)

        rn.plot_mean_activity()

    # Test different wrap-around
    if False:
        M = 350
        R = 2
        sigma_x = 0.1
        # Moments of scale: mean = volume of receptive field directly.
        rn = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(2.0, 0.1), ratio_moments=(1.0, 0.2))
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
                rn = RandomFactorialNetwork.create_full_conjunctive(int(N_sqrt**2.), R=R, scale_moments=(2.0, 0.1), ratio_moments=(1.0, 0.2))
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

    if False:
        print 'Compute KL approx of mixture by Gausian'

        alpha = 1.0
        N_sqrt = 30.
        N = int(N_sqrt**2.)
        T = 1
        sigma_x = 0.2
        sigma_y = 0.001
        beta = 1.0
        rc_scale = 5.0

        time_weights_parameters = dict(weighting_alpha=alpha, weighting_beta = beta, specific_weighting = 0.1, weight_prior='uniform')
        rn = RandomFactorialNetwork.create_full_conjunctive(N, R=2, rcscale=rc_scale, response_type='bivariate_fisher', gain=4*np.pi**2.)
        data_gen_noise = DataGeneratorRFN(15000, T, rn, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=T-1, enforce_min_distance=0.0)
        stat_meas = StatisticsMeasurer(data_gen_noise)

        # assert False

        measured_cov = stat_meas.model_parameters['covariances'][-1][-1]
        computed_cov = rn.compute_covariance_KL(sigma_2=(beta**2.0*sigma_x**2. + sigma_y**2.), T=T, beta=beta, precision=50)

        print np.mean(np.abs(measured_cov-computed_cov))


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


        # Check if measured covariance is Toeplitz
        # A_{i,j} == A_{i+1, j+1} (constant along all diagonals)
        toeplitz_std_all = np.zeros(2*N-1)
        toeplitz_pdist_all = np.zeros(2*N-1)
        for i, diag_i in enumerate(np.arange(-N+2, N-1)):
            toeplitz_std_all[i] = np.std(np.diag(measured_cov, diag_i))
            toeplitz_pdist_all[i] = np.mean(pdist(np.diag(measured_cov, diag_i)[:, np.newaxis]))

        print "Toeplitz: std: %f, dist: %f" % (np.mean(toeplitz_std_all), np.mean(toeplitz_pdist_all))

        # Check if measured covariance is circulant
        # A_{1, n} = A_{2, 1}
        # roll everything back, should get a row constant matrix
        measured_cov_rollback = np.zeros((N, N))
        for roll_amount in xrange(N):
            measured_cov_rollback[roll_amount] = np.roll(measured_cov[roll_amount], -roll_amount)

        circulant_dist = np.sum(np.diff(measured_cov_rollback, axis=0)**2.)

        print "Circulant: MSE %f" % circulant_dist

        plt.plot(np.mean(np.abs(np.fft.fft(measured_cov)), axis=0))

        c_tilde = np.fft.fft(computed_cov[0])
        freqs = np.fft.fftfreq(N)
        mu_tilde = np.fft.fft(rn.get_derivative_network_response())

        w, v = np.linalg.eig(computed_cov)

        # Theoretical eigendecomposition of circulant matrix
        c_tilde_ordered = np.sort((c_tilde))[::-1]
        F = np.exp(-1j*2.0*np.pi/N)**np.outer(np.arange(N, dtype=float), np.arange(N, dtype=float))

        # Construct true circulant
        c = computed_cov[0]
        circ_c = sp.linalg.circulant(c)
        circ_c_tilde = np.fft.fft(c)
        circ_c_tilde_bis = np.dot(F, c)
        circ_c_reconst = np.abs((np.dot(F.conj(), np.dot(np.diag(circ_c_tilde_bis), F))/N))

        # ISSUE: weird mirroring, plus diagonal off by 1. USE conj() and not .T ...
        cov_reconst_theo = np.abs((np.dot(F, np.dot(np.diag(c_tilde), F.conj()))/N))

        IF_original = np.dot(rn.get_derivative_network_response(), np.linalg.solve(computed_cov, rn.get_derivative_network_response()))
        IF_eigendec = np.abs(np.dot(rn.get_derivative_network_response(), np.dot(v, np.dot(np.diag(w**-1), np.dot(v.T, rn.get_derivative_network_response())))))
        IF_fourier_bis = np.abs(np.dot(mu_tilde.conj(), c_tilde**-1*mu_tilde))
        IF_fourier = np.abs(np.dot(rn.get_derivative_network_response(), np.fft.ifft(c_tilde**-1*mu_tilde)))

        IF_fourier_theo = np.abs(np.dot(mu_tilde.conj(), np.dot(np.diag(c_tilde**-1), mu_tilde)))/N

        # ## Verifying math for 2 feature cases. Here the mean of the Gaussian approx.
        # def mu_theta2(theta2, preferred_angles, kappa=3.0):
        #     return np.exp(kappa*np.cos(preferred_angles - theta2))/(4.*np.pi**2.*scsp.i0(kappa))

        # responses = np.array([rn.get_network_response(stimulus_input=(x, 0.0)) for x in np.linspace(-np.pi, np.pi, 1000.)])

        # plt.figure()
        # plt.plot(np.mean(responses, axis=0))
        # plt.plot(mu_theta2(0.0, rn.neurons_preferred_stimulus[:, 1], kappa=rc_scale))

        # plt.legend(['Empirical mean', 'Theoretical'])
        # plt.title('Comparison between empirical and theoretical conditional mean E[mu(theta1, theta2)]_{theta1} | theta2')

    if False:
        print 'Check evolution of diagonal elements of measured covariance matrix'
        # \strike{Shows a big discrepancy, the measured covariance goes as ~sqrt(T)...}
        # DO NOT PUT alpha != 1.0 if you want to compare them...
        alpha = 1.0
        N_sqrt = 30.
        N = int(N_sqrt**2.)
        sigma_x = 0.0001
        sigma_y = 0.0001
        beta = 1.0
        rc_scale = 4.0

        time_weights_parameters = dict(weighting_alpha=alpha, weighting_beta = beta, specific_weighting = 0.1, weight_prior='uniform')
        rn = RandomFactorialNetwork.create_full_conjunctive(N, R=2, scale_moments=(rc_scale, 0.01), ratio_moments=(1.0, 0.01), response_type='bivariate_fisher')

        T_all = np.arange(1, 6)
        meas_cov_diag = np.zeros(T_all.size)
        comp_cov_diag = np.zeros(T_all.size)
        all_measured_cov = np.zeros((T_all.size, N, N))
        all_computed_cov = np.zeros((T_all.size, N, N))

        for T_i, T in enumerate(T_all):
            print "T: %d" % T

            data_gen_noise = DataGeneratorRFN(5000, T, rn, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=T-1, enforce_min_distance=0.0)
            stat_meas = StatisticsMeasurer(data_gen_noise)

            all_measured_cov[T_i] = stat_meas.model_parameters['covariances'][-1][-1]
            all_computed_cov[T_i] = rn.compute_covariance_KL(sigma_2=(beta**2.0*sigma_x**2. + sigma_y**2.), T=T, beta=beta, precision=50)
            meas_cov_diag[T_i] = np.mean(np.diag(all_measured_cov[T_i]))
            comp_cov_diag[T_i] = np.mean(np.diag(all_computed_cov[T_i]))

        plt.plot(T_all, meas_cov_diag, T_all, comp_cov_diag)
        plt.xlabel('Number of items')
        plt.ylabel('Mean diagonal magnitude')
        plt.title('Evolution of diagonal magnitude for increasing number of items')
        plt.legend(['Measured', 'Computed'], loc='best')
        plt.axis('tight')

        plt.figure()
        plt.plot(T_all, np.mean(np.mean((all_measured_cov - all_computed_cov)**2., axis=-1), axis=-1))
        plt.xlabel('Number of items')
        plt.title('MSE between computed and measured covariance matrix elements')
        plt.axis('tight')
        plt.show()

    if False:
        # Check difference between kappa parameter for a bivariate_fisher and the previously used "scale" (which was more or less the standard deviation of a gaussian)
        # Do that by looking at the activation of one neuron over one dimension and fit a Gaussian to it.

        alpha = 0.999
        beta = 1.0
        N_sqrt = 20.
        N = int(N_sqrt**2.)
        time_weights_parameters = dict(weighting_alpha=alpha, weighting_beta = beta, specific_weighting = 0.1, weight_prior='uniform')
        precision = 100
        nb_params = 100

        # Define our Gaussian and error function
        gaussian = lambda p, x: 1./(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2./(2.*p[1]**2.))
        gaussian_bis = lambda p, x: p[2]*np.exp(-(x-p[0])**2./(2.*p[1]**2.))
        error_funct = lambda p, x, y: (y - gaussian(p, x))

        # Space
        xx = np.linspace(-np.pi, np.pi, precision)

        if False:
            # First actually get the relationship for the wrong_wrap method used before.
            rc_scale_space = np.linspace(0.001, 25., nb_params)
            std_dev_results = np.zeros(rc_scale_space.size)

            print "Doing for wrong wrap"
            for i, rc_scale in enumerate(rc_scale_space):
                print rc_scale
                rn = RandomFactorialNetwork.create_full_conjunctive(N, R=2, scale_moments=(rc_scale, 0.01), ratio_moments=(1.0, 0.01), response_type='wrong_wrap')

                selected_neuron = 209

                # Get the activity of one neuron (selected arbitrarily), then look at one axis only.
                mean_neuron_out = np.mean(rn.get_neuron_activity(selected_neuron), axis=0)
                mean_neuron_out /= np.sum(mean_neuron_out)

                # Fit a Gaussian to it.

                # pinit = [0.0, 1.0]
                # out = spopt.leastsq(error_funct, pinit, args=(xx, mean_neuron_out))
                # new_params = out[0]
                # std_dev_results[i] = new_params[1]

                # Estimate mean and variance instead
                curr_mean = np.sum(mean_neuron_out*xx)/np.sum(mean_neuron_out)
                curr_std = np.sqrt(np.abs(np.sum((xx-curr_mean)**2*mean_neuron_out)/np.sum(mean_neuron_out)))
                curr_max = mean_neuron_out.max()

                std_dev_results[i] = curr_std

            # Plots
            plt.figure()
            plt.plot(rc_scale_space, std_dev_results)
            plt.title('Wrong wrap, relationship between rc_scale and std dev')
            plt.xlabel('Receptive field scale')
            plt.ylabel('Standard deviation of fitted gaussian')


        if True:

            # Second, see the relationship with the new bivariate_fisher kappas
            kappa_space = np.linspace(0.001, 5., nb_params)
            std_dev_results_kappas = np.zeros(kappa_space.size)

            print "Doing for bivariate fisher"
            for i, kappa in enumerate(kappa_space):
                print kappa

                rn = RandomFactorialNetwork.create_full_conjunctive(N, R=2, scale_moments=(kappa, 0.001), ratio_moments=(1.0, 0.001))

                selected_neuron = 209

                # Get the activity of one neuron (selected arbitrarily), then look at one axis only.
                mean_neuron_out_ = np.mean(rn.get_neuron_activity(selected_neuron, precision=precision), axis=0)
                mean_neuron_out_ /= np.sum(mean_neuron_out_)

                # Fit a Gaussian to it.
                # pinit = [0.0, 1.0]
                # out = spopt.leastsq(error_funct, pinit, args=(xx, mean_neuron_out_))
                # new_params = out[0]
                # std_dev_results_kappas[i] = new_params[1]

                # Estimate mean and variance instead
                curr_mean = np.sum(mean_neuron_out_*xx)/np.sum(mean_neuron_out_)
                curr_std = np.sqrt(np.abs(np.sum((xx-curr_mean)**2*mean_neuron_out_)/np.sum(mean_neuron_out_)))
                curr_max = mean_neuron_out_.max()

                std_dev_results_kappas[i] = curr_std


            # Plot
            plt.figure()
            plt.plot(kappa_space, std_dev_results_kappas)
            plt.title('Bivariate Fisher, relationship between rc_scale and kappa')
            plt.xlabel('Kappa scale')
            plt.ylabel('Standard deviation of fitted gaussian')

            plt.show()


    if False:
        # Play with the normalising constant of the Bivariate Fisher
        kappa = 1.0
        N_sqrt = 20.
        N = int(N_sqrt**2.)
        precision = 100

        rn = RandomFactorialNetwork.create_full_conjunctive(N, R=2, scale_moments=(kappa, 0.001), ratio_moments=(1.0, 0.001))

        selected_neuron = 209

        # Get the activity of one neuron (selected arbitrarily), then look at one axis only.
        # => Those don't seem to give exactly same thing when marginalise one dimension
        normalisation = 4.*np.pi**2.*scsp.i0(kappa)*scsp.i0(kappa)
        mean_neuron_out_ = rn.get_neuron_activity(selected_neuron, precision=precision)/normalisation
        # mean_neuron_out_ /= np.sum(mean_neuron_out_)

        preferred_stim = np.array([0.0, 0.0])
        stimulus = np.array([0.0, 0.0])
        kappas = np.array([1.0, 1.0])

        dtheta = (stimulus[0] - preferred_stim[0])
        dgamma = (stimulus[1] - preferred_stim[1])

        # 2D versus 1D, should obtain same thing (approximately)
        normalisation = 4.*np.pi**2.*scsp.i0(kappas[0])*scsp.i0(kappas[1])
        output = np.exp(kappas[0]*np.cos(dtheta) + kappas[1]*np.cos(dgamma))/normalisation

        normalisation_1d = 2.*np.pi*scsp.i0(kappas[0])
        output_1d = np.exp(kappas[0]*np.cos(dtheta))/normalisation_1d

        # 1D multiple kappas
        kappas_space = np.linspace(0.01, 10, 100)
        all_out = np.zeros((kappas_space.size, precision))
        xx = np.linspace(-np.pi, np.pi, precision)

        for i, kappa in enumerate(kappas_space):
            print kappa
            normalisation = 2.*np.pi*scsp.i0(kappa)
            all_out[i] = np.exp(kappa*np.cos(xx))/normalisation

        # 2d multiple kappas
        kappas_space = np.linspace(0.01, 10, 100)
        all_out = np.zeros((kappas_space.size, precision, precision))
        x = np.linspace(-np.pi, np.pi, precision, endpoint=False)
        xx, yy = np.meshgrid(x, x)

        for i, kappa in enumerate(kappas_space):
            print kappa
            normalisation = 4.*np.pi**2.*scsp.i0(kappa)**2.
            all_out[i] = np.exp(kappa*np.cos(xx)+kappa*np.cos(yy))/normalisation


        # 2D Try multiple kappas, see if the normalisation works.
        kappas_space = np.linspace(10, 200, 50)
        all_neuron_out = np.zeros((kappas_space.size, precision, precision))
        selected_neuron = 209
        for i, kappa in enumerate(kappas_space):
            print "%.3f" % kappa
            rn = RandomFactorialNetwork.create_full_conjunctive(N, R=2, scale_moments=(kappa**2., 0.01), ratio_moments=(1.0, 0.01))
            # normalisation = 4.*np.pi**2.*scsp.i0(kappa)*scsp.i0(kappa)
            # all_neuron_out[i] = rn.get_neuron_activity(selected_neuron, precision=precision, params={'kappas': np.array([kappa, kappa, 0.0])})/normalisation
            all_neuron_out[i] = rn.get_neuron_activity(selected_neuron, precision=precision, params={'kappas': np.array([kappa, kappa, 0.0])})

        plt.figure()
        plt.plot(kappas_space, all_neuron_out.mean(axis=-1).sum(axis=-1))

        plt.figure()
        plt.plot(x, all_neuron_out.mean(axis=-1).T)

        plt.show()

    if False:
        # See how many neurons respond to a given stimulus, indication for wrong coverage/rc_size.

        kappa = 2.0
        N_sqrt = 10.
        N = int(N_sqrt**2.)
        precision = 30
        percent_max = 0.5

        rn = RandomFactorialNetwork.create_full_conjunctive(N, R=2, scale_moments=(kappa, 0.001), ratio_moments=(1.0, 0.001))

        # rn.plot_mean_activity()
        # rn.plot_mean_activity_fft()

        rn.compute_num_neurons_responsive_stimulus(percent_max=percent_max)

        # See the whole dependance
        kappa_space = np.linspace(0.05, 20, 10)
        num_neurons_responding = np.zeros((kappa_space.size), dtype=float)

        for i, kappa in enumerate(kappa_space):
            if i%10 == 0:
                print kappa

            rn = RandomFactorialNetwork.create_full_conjunctive(N, R=2, scale_moments=(kappa, 0.001), ratio_moments=(1.0, 0.001))
            num_neurons_responding[i] = rn.compute_num_neurons_responsive_stimulus(should_plot=False, precision=precision, percent_max=percent_max, debug=False)

        plt.figure()
        plt.plot(kappa_space, num_neurons_responding)
        plt.xlabel('kappa')
        plt.ylabel('Number of neurons responding at %.f %%' % (percent_max*100.))
        plt.show()

    if True:
        ## Check if covariance approximation with Toeplitz structure works.

        # Run %run experimentlauncher before.
        kappa = 4.0
        def cov_toeplitz(theta_n, theta_m, kappa):
            return (scsp.i0(2*kappa*np.cos((theta_n - theta_m)/2.)) - scsp.i0(kappa)**2.)/(4.*np.pi**2.*scsp.i0(kappa)**2.)















