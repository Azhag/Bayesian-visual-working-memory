#!/usr/bin/env python
# encoding: utf-8
"""
populationcode.py

Created by Loic Matthey on 2011-05-30.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import numpy as np
import pylab as plt

class PopulationCode:
    def __init__(self, N):
        self.N = N
    


class PopulationCodeAngle(PopulationCode):
    '''
        Implements a population for an angle. Correlated covariance matrix.
    '''
    
    def __init__(self, N, sigma=1., rho=0.3, a=0.1, gamma=0.7, neurons_angles=None, method_neurons_angles='uniform', max_angle=2.*np.pi):
        PopulationCode.__init__(self, N)
        
        self.sigma2 = sigma**2
        self.rho = rho
        self.a = a
        self.gamma = gamma
        self.max_angle = max_angle
        
        if neurons_angles is None:
            # Need to assign the neurons preferred angles. Do that uniformly by default
            self.neurons_angles = self.assign_neurons_angles(method=method_neurons_angles)
            
        # Create the covariance matrix
        self.create_covariance_matrix()
        
    
    def assign_neurons_angles(self, method='uniform'):
        '''
            Uses the current number of neurons to cover the [0:2pi] interval.
            Returns the created assignment
        '''
        if method == 'uniform':
            # Put neurons uniformly around the circle
            return np.linspace(0., self.max_angle, self.N, endpoint=False)
        
        elif method == 'random':
            # Randomly assign them
            return np.random.uniform(0., self.max_angle, size=self.N)
        
    
    def mean_response(self, theta_input, bias=0.0):
        '''
            Return the mean output of the population
        '''
        
        correction_wrapup = 2.*np.pi/self.max_angle
        
        if np.isscalar(theta_input):
            mean = bias+np.exp(1./self.sigma2*np.cos(correction_wrapup*(self.neurons_angles - theta_input)))
        
        else:
            mean = bias+np.exp(1./self.sigma2*np.cos(correction_wrapup*(np.tile(self.neurons_angles, (theta_input.size, 1)).T - theta_input))).T
        
        return mean/np.max(mean)
    
    def create_covariance_matrix(self):
        '''
            Create a covariance matrix with angle-sensitive cross terms
        '''
        self.covariance = np.zeros((self.N, self.N))
        
        all_angles = np.tile(self.neurons_angles, (self.N, 1))
        diff_angles = all_angles.T - all_angles
        correction_wrapup = 2.*np.pi/self.max_angle
        
        self.covariance = self.rho*np.exp(self.a*np.cos(correction_wrapup*diff_angles))
        np.fill_diagonal(self.covariance,  self.gamma)
        
        
    
    
    def sample_random_response(self, theta_input, nb_samples=1):
        '''
            Draw samples from the current population
            
            return:  nb_input_orientations x D
                     nb_samples x nb_input_orientations x D (nb_samples > 1)
        '''
        
        assert nb_samples > 0, 'Cannot sample 0 samples...'
        
        if np.isscalar(theta_input):
            response = np.random.multivariate_normal(self.mean_response(theta_input), self.covariance, size=nb_samples)
        else:
            response = np.zeros((nb_samples, theta_input.size, self.N))
            for theta_i in np.arange(theta_input.size):
                response[:, theta_i, :] = np.random.multivariate_normal(self.mean_response(theta_input[theta_i]), self.covariance, size=nb_samples)
            
        
        if nb_samples == 1:
            return response[0]
        else:
            return response
        
    
    def plot_population_representation(self, theta_s):
        
        fig, ax = plt.subplots(1)
        mean = self.mean_response(theta_s)
        samples = self.sample_random_response(theta_s, nb_samples=20)
        std_samples = np.std(samples, axis=0)
        mean_minus_std = mean - std_samples
        mean_plus_std = mean + std_samples
        
        # plt.plot(popcod.neurons_angles, mean)
        # plt.plot(popcod.neurons_angles, np.mean(samples, axis=1), '.')
        
        if np.isscalar(theta_s):
            ax.plot(self.neurons_angles, mean)
            ax.fill_between(self.neurons_angles, mean_minus_std, mean_plus_std, facecolor='blue', alpha=0.4,
                        label='1 sigma range')
        else:
            for mean_i in np.arange(mean.shape[0]):
                l = ax.plot(self.neurons_angles, mean[mean_i])
                ax.fill_between(self.neurons_angles, mean_minus_std[mean_i], mean_plus_std[mean_i], alpha=0.3,
                    facecolor=plt.rcParams['axes.color_cycle'][mean_i%len(plt.rcParams['axes.color_cycle'])], interpolate=True)
        
        ax.autoscale(tight=True)
        #plt.show()
    



if __name__ == '__main__':
    N = 200
    
    popcod = PopulationCodeAngle(N, sigma=0.2, rho=0.01, gamma=0.01, max_angle=2.*np.pi)
    
    theta = 0.
    popcod.plot_population_representation(theta)
    
    K = 30
    multiple_angles = np.linspace(0.0, 2.*np.pi, K, endpoint=False)
    popcod.plot_population_representation(multiple_angles)
    
    # fig, ax = plt.subplots(1)
    #     mean = popcod.mean_response(theta)
    #     samples = popcod.sample_random_response(theta, nb_samples=100)
    #     std_samples = np.std(samples, axis=1)
    #     mean_minus_std = mean - std_samples
    #     mean_plus_std = mean + std_samples
    #     
    #     ax.plot(popcod.neurons_angles, popcod.mean_response(theta), 'k')
    #     # ax.plot(popcod.neurons_angles, popcod.sample_random_response(theta, nb_samples=20), '.')
    #     ax.fill_between(popcod.neurons_angles, mean_minus_std, mean_plus_std, facecolor='blue', alpha=0.4,
    #                 label='1 sigma range')
    #     # plt.plot(popcod.neurons_angles, np.mean(popcod.sample_random_response(theta, nb_samples=1), axis=0), '.')
    #     # plt.plot(popcod.neurons_angles, np.mean(popcod.sample_random_response(theta, nb_samples=1), axis=0), '.')
    #     
    #     fig, ax = plt.subplots(1)
    #     multiple_angles = np.linspace(0.0, np.pi, 8, endpoint=False)
    #     mean = popcod.mean_response(multiple_angles)
    #     samples = popcod.sample_random_response(multiple_angles, nb_samples=50)
    #     std_samples = np.std(samples, axis=1)
    #     mean_minus_std = mean - std_samples
    #     mean_plus_std = mean + std_samples
    #     
    #     # plt.plot(popcod.neurons_angles, mean)
    #     # plt.plot(popcod.neurons_angles, np.mean(samples, axis=1), '.')
    #     for mean_i in np.arange(mean.shape[1]):
    #         l = ax.plot(popcod.neurons_angles, mean[:, mean_i])
    #         ax.fill_between(popcod.neurons_angles, mean_minus_std[:, mean_i], mean_plus_std[:, mean_i], alpha=0.3, facecolor=plt.rcParams['axes.color_cycle'][mean_i%len(plt.rcParams['axes.color_cycle'])], interpolate=True)
    #         
    #     
    plt.show()





