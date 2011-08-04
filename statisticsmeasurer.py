#!/usr/bin/env python
# encoding: utf-8
"""
statisticsmeasurer.py

Created by Loic Matthey on 2011-08-02.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import numpy as np
from datagenerator import *
from randomnetwork import *
from utils import *
import pylab as plt


class StatisticsMeasurer:
    def __init__(self, data_gen):
        self.data_gen = data_gen
        
        (self.N, self.T, self.M) = data_gen.all_Y.shape
        self.Y = data_gen.all_Y
        
        self.measure_moments()
        
        self.compute_collapsed_model_parameters()
        
        print "StatisticMeasurer executed successfully"
    
    
    def measure_moments(self):
        '''
            Compute the moments of Y for each time.
        '''
        
        self.means = np.zeros((self.T, self.M))
        self.covariances = np.zeros((self.T, self.M, self.M))
        
        for t in np.arange(self.T):
            self.means[t] = np.mean(self.Y[:,t,:], axis=0)
            self.covariances[t] = np.cov(self.Y[:,t,:].T)
        
    
    def plot_moments(self):
        '''
            Plot the fitted moments
        '''
        
        # Plot the means
        plt.figure()
        plt.plot(self.means.T)
        
        (f, subaxes) = pcolor_square_grid(self.covariances)
        
    
    def compute_collapsed_model_parameters(self):
        '''
            Compute m_t^s, m_t^e, \Sigma_t^s and \Sigma_t^e, for all possible times t_c.
            
                m_t^s : mean of the "starting" noise process
                m_t^e : mean of the "ending" noise process
                and their covariances
        '''
        
        model_means = np.zeros((2, self.T, self.M))
        model_covariances = np.zeros((2, self.T, self.M, self.M))
        
        # Mean and covariance of the starting noise is easy, just take the measured marginals
        model_means[0, 1:] = self.means[:self.T-1]
        
        # Covariance of ending noise is also easy
        model_covariances[0, 1:] = self.covariances[:self.T-1]
        
        # Mean and covariance of the ending noise requires a small mapping
        for t in np.arange(self.T-1):
            model_means[1, t] = self.means[self.T-1] - np.dot( np.power(self.data_gen.time_weights[0][t], self.T-t),  self.means[t])
            model_covariances[1, t] = self.covariances[self.T-1] - np.dot( np.power(self.data_gen.time_weights[0][t], self.T-t),  self.covariances[t])
        
        
        self.model_parameters = dict(means=model_means, covariances=model_covariances)


if __name__ == '__main__':
    K = 20
    D = 20
    M = 50
    R = 2
    T = 8
    N = 1000
    
    sigma_y = 0.02
    
    random_network = RandomNetworkContinuous.create_instance_uniform(K, M, D=D, R=R, W_type='dirichlet', W_parameters=[0.1, 0.01], sigma=0.2, gamma=0.005, rho=0.005)
    
    data_gen = DataGeneratorContinuous(N, T, random_network, sigma_y = sigma_y, time_weights_parameters = dict(weighting_alpha=0.85, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='recency'))
    
    stat_meas = StatisticsMeasurer(data_gen)
    # stat_meas.plot_moments()
    