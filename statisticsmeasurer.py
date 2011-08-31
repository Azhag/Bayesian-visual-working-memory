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
        
        print "StatisticMeasurer has measured"
    
    
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
        
        model_means = np.zeros((3, self.T, self.M))
        model_covariances = np.zeros((3, self.T, self.M, self.M))
        
        # Mean and covariance of the starting noise is easy, just take the measured marginals of the previous time, transform them once.
        for t in np.arange(1, self.T):
            model_means[0, t] = np.dot(self.data_gen.time_weights[0][t], self.means[t-1])
            model_covariances[0, t] = np.dot(self.data_gen.time_weights[0][t], np.dot(self.covariances[t-1], self.data_gen.time_weights[0][t].T))
        
        
        # Mean and covariance of the ending noise requires a small mapping
        for t in np.arange(self.T-1):
            ATmtc = np.power(self.data_gen.time_weights[0][t], self.T-1-t)
            model_means[1, t] = self.means[self.T-1] - np.dot(ATmtc,  self.means[t])
            model_covariances[1, t] = self.covariances[self.T-1] - np.dot(ATmtc,  np.dot(self.covariances[t], ATmtc.T))
        
        # Measured means and covariances
        model_means[2] = self.means
        model_covariances[2] = self.covariances
            
        self.model_parameters = dict(means=model_means, covariances=model_covariances)


if __name__ == '__main__':
    K = 20
    D = 32
    M = 128
    R = 2
    T = 3
    N = 1000
    
    sigma_y = 0.02
    
    random_network = RandomNetworkContinuous.create_instance_uniform(K, M, D=D, R=R, W_type='dirichlet', W_parameters=[0.1, 0.5], sigma=0.2, gamma=0.002, rho=0.002)
    # random_network = RandomNetworkFactorialCode.create_instance_uniform(K, D=D, R=R, sigma=0.05)
    
    data_gen = DataGeneratorContinuous(N, T, random_network, sigma_y = sigma_y, time_weights_parameters = dict(weighting_alpha=0.8, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform'))
    
    stat_meas = StatisticsMeasurer(data_gen)
    stat_meas.plot_moments()
    
    plt.figure()
    for t in np.arange(T)[::-1]:
        plt.hist(np.mean(stat_meas.Y[:,t,:] - np.mean(stat_meas.Y[:,t, :], axis=0), axis=1), bins=50)
    
    
    plt.show()
    