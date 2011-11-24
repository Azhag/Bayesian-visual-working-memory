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

from utils import *

class RandomFactorialNetwork():
    '''
        Modified paradigm for this Network. Uses a factorial representation of K features, and samples them using K-dimensional gaussian receptive fields.
            Randomness is in the distribution of orientations and radii of those gaussians.
    '''
    def __init__(self, M, R=1):
        
        assert R == 2, "RandomFactorialNetwork only implemented for R=2 for now"

        self.M = M
        self.K = 0
        self.R = R
        
        self.network_initialised = False
        
        self.neurons_preferred_stimulus = None
        self.neurons_sigma = None
        self.mask_neurons_unset = None
        
        self._ALL_NEURONS = np.arange(M)
        
        # Need to assign to each of the M neurons a preferred stimulus (tuple(orientation, color) for example)
        # By default, random
        self.assign_prefered_stimuli(tiling_type='conjunctive')
        self.assign_random_eigenvectors()

        self.network_initialised = True
    
    
    def assign_prefered_stimuli(self, tiling_type='conjunctive', specified_neurons=None, reset=False, nb_feature_centers=3):
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
                # Cover the space uniformly
                N_sqrt = np.floor(np.power(specified_neurons.size, 1./self.R))

                # coverage_1D = np.linspace(-np.pi, np.pi, N_sqrt, endpoint=False)
                coverage_1D = np.linspace( -np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)
                
                new_stim = np.array(cross(self.R*[coverage_1D.tolist()]))

                # Assign the preferred stimuli
                #   Unintialized neurons will get masked out down there.
                self.neurons_preferred_stimulus[specified_neurons[:new_stim.shape[0]]] = new_stim
            
            elif tiling_type == '2_features':
                N = np.round(specified_neurons.size/self.R)
                
                # Arbitrary put them along (x, 0) and (0, y) axes
                # coverage_1D = np.linspace(-np.pi, np.pi, N, endpoint=False)
                # self.neurons_preferred_stimulus[specified_neurons[0:N], 0] = coverage_1D
                # self.neurons_preferred_stimulus[specified_neurons[0:N], 1] = 0.
                # self.neurons_preferred_stimulus[specified_neurons[N:self.R*N], 0] = 0.
                # self.neurons_preferred_stimulus[specified_neurons[N:self.R*N], 1] = coverage_1D

                # Distribute the cells along nb_feature_centers centers. Distributed evenly, should have 0 in them.
                centers = np.linspace(0.0, 2*np.pi, nb_feature_centers, endpoint=False) - 2*np.pi/nb_feature_centers*int(nb_feature_centers/2.)

                # centers = [-2.*np.pi/3., 0.0, 2.*np.pi/3.]
                # centers = [0.0]

                sub_N = N/nb_feature_centers

                # coverage_1D = np.linspace( -np.pi, np.pi, sub_N, endpoint=False)
                coverage_1D = np.linspace( -np.pi + np.pi/sub_N, np.pi + np.pi/sub_N, sub_N, endpoint=False)

                for center_i in np.arange(nb_feature_centers):
                    self.neurons_preferred_stimulus[specified_neurons[center_i*sub_N:(center_i+1)*sub_N], 0] = coverage_1D
                    self.neurons_preferred_stimulus[specified_neurons[center_i*sub_N:(center_i+1)*sub_N], 1] = centers[center_i]
                    self.neurons_preferred_stimulus[specified_neurons[N+center_i*sub_N:(N+(center_i+1)*sub_N)], 0] = centers[center_i]
                    self.neurons_preferred_stimulus[specified_neurons[N+center_i*sub_N:(N+(center_i+1)*sub_N)], 1] = coverage_1D
            
            # Handle uninitialized neurons
            self.mask_neurons_unset = np.any(np.isnan(self.neurons_preferred_stimulus), 1) # check if still some nan, any on the first axis.
        
    
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
        # self.neurons_angle[specified_neurons] = np.random.random_integers(0,1, size=specified_neurons.size)*np.pi/2.
        self.neurons_angle[specified_neurons] = np.pi*np.random.random(size=specified_neurons.size)
        
        self.compute_2d_parameters(specified_neurons=specified_neurons)
        
    
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
        
    
    
    def get_network_response(self, stimulus_input, vonmises_param=(0.2,0.05,0.005)):
        '''
            Compute the response of the network.
            
            Could be optimized, for now only for one stimulus tuple
            
        '''
        normalisation = 1.
        
        dx = (self.neurons_preferred_stimulus[:,0] - stimulus_input[0])
        dy = (self.neurons_preferred_stimulus[:,1] - stimulus_input[1])
        
        # Add wrap around in the [0; 2pi] sphere
        # dx = (np.mod((self.neurons_preferred_stimulus[:,0] - stimulus_input[0]) + np.pi, 2*np.pi) - np.pi)
        # dy = (np.mod((self.neurons_preferred_stimulus[:,1] - stimulus_input[1]) + np.pi, 2*np.pi) - np.pi)
        
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
            output = normalisation*np.exp(-self.neurons_params[:,0]*dx**2.0 - 2.*self.neurons_params[:,1]*dx*dy - self.neurons_params[:,2]*dy**2.0)
            # output = normalisation*np.exp(vonmises_param[0]*cosdx + vonmises_param[1]*cosdy + vonmises_param[2]*sindx*sindy)
            output[self.mask_neurons_unset] = 0.0
            return output
        elif self.R == 3:
            raise NotImplementedError('R=3 for factorial code...')
        else:
            # not unrolled
            raise NotImplementedError('R>3 for factorial code...')
        
    
    
    def get_neuron_response(self, neuron_index, stimulus_input):
        '''
            Get the output of one specific neuron, for a specific stimulus
        '''

        return self.get_network_response(stimulus_input)[neuron_index]

    
    
    def sample_network_response(self, stimulus_input, sigma=0.2):
        '''
            Get a random response for the given stimulus.
            
            return: M
        '''
        
        return self.get_network_response(stimulus_input) + sigma*np.random.randn(self.M)
    

    def sample_multiple_network_response(self, stimuli_input, sigma=0.2):
        '''
            Get a set of random responses for multiple stimuli
            
            return: N x M
        '''
        
        nb_samples = stimuli_input.shape[0]
        net_samples = np.zeros((nb_samples, self.M))


        for i in np.arange(nb_samples):
            net_samples[i] = self.sample_network_response(stimuli_input[i], sigma=sigma)
        
        return net_samples

    
    def compute_covariance_stimulus(self, stimulus_input, N=2000, sigma=0.2):
        '''
            Compute the covariance for a given stimulus.
        '''

        # Same stim for all
        all_stim = np.tile(stimulus_input, (N, 1))

        # Get samples
        samples = self.sample_multiple_network_response(all_stim, sigma=sigma)

        # Get covariance
        return np.cov(samples.T)


    def compute_fisher_information(self, stimulus_input, sigma=0.01, cov_stim=None):
        '''
            Compute and return the Fisher information for the given stimulus.
            Assume we are looking for the FI in coordinate 1, fixing the other (in 2D).

            Assuming that Cov_stim ~ cst, we use:
            I = f' Cov_stim^-1 f'
        '''

        if cov_stim is None:
            # The covariance for the stimulus
            cov_stim = self.compute_covariance_stimulus(stimulus_input, sigma=sigma)


        # Compute the derivative of the receptive field
        dx = (self.neurons_preferred_stimulus[:,0] - stimulus_input[0])
        dy = (self.neurons_preferred_stimulus[:,1] - stimulus_input[1])

        sindx = np.sin(dx)
        coshalfdx = np.cos(0.5*dx)
        sinhalfdy = np.sin(0.5*dy)

        der_f = 6.**2.*(self.neurons_params[:,0]*0.5*sindx + self.neurons_params[:,1]*coshalfdx*sinhalfdy)*self.get_network_response(stimulus_input)

        der_f[np.isnan(der_f)] = 0.0

        # Now get the Fisher information
        return np.dot(der_f, np.linalg.solve(cov_stim, der_f))


    def compute_fisher_information_fullspace(self, sigma=0.01, cov_stim=None, precision=100):
        
        feature_space = np.linspace(-np.pi, np.pi, precision)
        
        activity = np.zeros((feature_space.size, feature_space.size))
        
        if cov_stim is None:
            cov_stim = self.compute_covariance_stimulus((0.,0.), sigma=sigma)

        # Compute the activity of that neuron over the whole space
        for i in np.arange(feature_space.size):
            for j in np.arange(feature_space.size):
                activity[i,j] = self.compute_fisher_information((feature_space[i], feature_space[j]), sigma=sigma, cov_stim=cov_stim)
            
        return activity
        



    ######################## PLOTS ######################################

    def plot_coverage_feature_space(self, nb_stddev=1., specified_neurons=None, alpha_ellipses=0.5, facecolor=None, ax=None):
        '''
            Show the features (R=2 only)
        '''
        assert self.R == 2, "only works for R=2"
        
        if specified_neurons is None:
            specified_neurons = self._ALL_NEURONS
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        
        ells = [Ellipse(xy=self.neurons_preferred_stimulus[m], width=nb_stddev*self.neurons_sigma[m, 0], height=nb_stddev*self.neurons_sigma[m, 1], angle=-np.degrees(self.neurons_angle[m])) for m in specified_neurons]
        
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(alpha_ellipses)
            if facecolor is None:
                e.set_facecolor(np.random.rand(3))
            elif facecolor==False or facecolor == 'none' or facecolor == 'None':
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
        
    
    def plot_mean_activity(self, specified_neurons=None):
        '''
            Plot \sum_i \phi_i(x) at all x
        '''
        
        assert self.R == 2, "Only implemented for R=2"
        
        if specified_neurons is None:
            specified_neurons = self._ALL_NEURONS
        
        precision = 100
        feature_space = np.linspace(-np.pi, np.pi, precision)
        
        mean_activity = np.zeros((feature_space.size, feature_space.size))
        
        for feat1_i in np.arange(feature_space.size):
            for feat2_i in np.arange(feature_space.size):
                all_activity = self.get_network_response((feature_space[feat1_i], feature_space[feat2_i]))
                mean_activity[feat1_i, feat2_i] = np.sum(all_activity[specified_neurons])
        
        print "%.3f %.5f" % (np.mean(mean_activity), np.std(mean_activity.flatten()))
        f = plt.figure()
        ax = f.add_subplot(111)
        im= ax.imshow(mean_activity.T, origin='lower')
        im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        ax.set_xlabel('Color')
        ax.set_ylabel('Orientation')
        # im.set_interpolation('nearest')
        f.colorbar(im)
        
        plt.show()
        
    
    
    def plot_neuron_activity_fullspace(self, neuron_index, nb_stddev=1.):
        '''
            Plot the activity of one specific neuron over the whole input space.
        '''
        
        precision = 100
        feature_space = np.linspace(-np.pi, np.pi, precision)
        
        activity = np.zeros((feature_space.size, feature_space.size))
        
        # Compute the activity of that neuron over the whole space
        for i in np.arange(feature_space.size):
            for j in np.arange(feature_space.size):
                activity[i,j] = self.get_neuron_response(neuron_index, (feature_space[i], feature_space[j]))
        
        # Plot it
        f = plt.figure()
        ax = f.add_subplot(111)
        im= ax.imshow(activity.T, origin='lower')
        im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
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

    
    def plot_fisher_info_fullspace(self, sigma=0.01, cov_stim=None, precision=100):
        activity = self.compute_fisher_information_fullspace(sigma=sigma, cov_stim=cov_stim, precision=precision)
        # Plot it
        f = plt.figure()
        ax = f.add_subplot(111)
        im= ax.imshow(activity.T, origin='lower')
        im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        ax.set_xlabel('Color')
        ax.set_ylabel('Orientation')
        # im.set_interpolation('nearest')
        f.colorbar(im)

        
        
    


    ##########################

    @classmethod
    def create_full_conjunctive(cls, M, R=2, sigma=0.2, scale_parameters = None, ratio_parameters = None, scale_moments=None, ratio_moments=None):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code
        '''
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
        
        return rn

    @classmethod
    def create_full_features(cls, M, R=2, sigma=0.2, scale=0.3, ratio=40.):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code
        '''
        print "create feature network"

        rn = RandomFactorialNetwork(M, R=R)

        rn.assign_prefered_stimuli(tiling_type='2_features', reset=True)
        rn.assign_aligned_eigenvectors(scale=scale, ratio=ratio, specified_neurons = np.arange(M/2), reset=True)
        rn.assign_aligned_eigenvectors(scale=scale, ratio=-ratio, specified_neurons = np.arange(M/2, M))

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
        
        conj_subpop_size = int(M*ratio_feature_conjunctive)
        feat_subpop_size = M - conj_subpop_size

        print "Population sizes: %d %d" % (conj_subpop_size, feat_subpop_size)
        
        rn = RandomFactorialNetwork(M, R=R)

        # Create the conjunctive subpopulation
        rn.assign_prefered_stimuli(tiling_type='conjunctive', reset=True, specified_neurons = np.arange(conj_subpop_size))
        rn.assign_random_eigenvectors(scale_parameters=conj_scale_parameters, ratio_parameters=conj_ratio_parameters, specified_neurons = np.arange(conj_subpop_size), reset=True)


        # Create the feature subpopulation        
        rn.assign_prefered_stimuli(tiling_type='2_features', specified_neurons = np.arange(conj_subpop_size, M))
        rn.assign_aligned_eigenvectors(scale=feat_scale, ratio=feat_ratio, specified_neurons = np.arange(conj_subpop_size, int(feat_subpop_size/2.+conj_subpop_size)))
        rn.assign_aligned_eigenvectors(scale=feat_scale, ratio=-feat_ratio, specified_neurons = np.arange(int(feat_subpop_size/2.+conj_subpop_size), M))
        
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
    if True:
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
    
    # New creation method
    if False:
        M = 200
        R = 2
        sigma_x = 0.1
        # Moments of scale: mean = volume of receptive field directly.
        # rn = RandomFactorialNetwork.create_full_conjunctive(M, R=R, sigma=sigma_x, scale_moments=(2.0, 0.1), ratio_moments=(1.0, 0.2))
        rn = RandomFactorialNetwork.create_full_features(M, R=R, scale=0.3)
        
        rn.plot_neuron_activity_fullspace(10)
        rn.plot_neuron_activity_fullspace(50)
        rn.plot_neuron_activity_fullspace(100)
        rn.plot_coverage_feature_space(facecolor = True)
        rn.plot_mean_activity()
    
    # Compute covariance
    if False:
        M = 300
        R =2
        sigma_x = 0.1
        

        rn.plot_mean_activity()
        cc = rn.compute_covariance_stimulus((0.0,0.0), sigma=sigma_x)

    

    





    

    