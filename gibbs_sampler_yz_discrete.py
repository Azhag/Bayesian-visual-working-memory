#!/usr/bin/env python
# encoding: utf-8
"""
sampler.py

Created by Loic Matthey on 2011-06-1.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import numpy as np
import scipy.special as scsp
import time
from datagenerator import *
from randomnetwork import *
from utils import *
import sys
import os.path
import argparse

class Sampler:
    '''
        Use Y and Z only, no intermediate X
        Use one \pi prior for all times., multiples for multiple populations
    '''
    def __init__(self, data_gen, dirichlet_alpha=0.5, sigma_to_sample=True, sigma_alpha=1.0, sigma_beta=2.0):
        '''
            Initialise the sampler
        '''
        
        # Get the data
        self.data_gen = data_gen
        self.random_network = data_gen.random_network
        self.YT = data_gen.Y
        
        # Get sizes
        (self.N, self.M) = self.YT.shape
        self.T = data_gen.T
        self.K = data_gen.random_network.K
        self.R = data_gen.random_network.R
        
        # Time weights
        self.time_weights = data_gen.time_weights
        
        # Initialise the feature-assignment matrix
        self.init_z(dirichlet_alpha)
        
        # Initial sigma_y
        self.sigma_to_sample = sigma_to_sample
        
        # Initial sigma_y
        self.init_sigma2y(sigma_alpha, sigma_beta)
        
        # Initialise the Y
        self.init_y()
        
        # Maximum exponent in current precision
        self.__maxexp__ = np.finfo('float').maxexp
    
    def init_z(self, dirichlet_alpha):
        '''
            Sample initial Z uniformly, for each population and time
            
            dir_alpha : K
            Z:          N x T x R
            A:          K x R
        '''
        
        self.dir_alpha = float(dirichlet_alpha)
        self.Z = np.random.randint(self.K, size=(self.N, self.T, self.R))
        
        self.lprob_zntrk = np.zeros(self.K)
        
        # Get the matrix of counts
        self.Akr = np.array([np.sum(np.sum(self.Z == k, axis=0), axis=0) for k in np.arange(self.K)])
        
    
    
    def init_sigma2y(self, sigmay_alpha, sigmay_beta):
        self.sigmay_alpha = sigmay_alpha
        self.sigmay_beta = sigmay_beta
        self.sigma2y = self.sample_invgamma(sigmay_alpha, sigmay_beta)
        # self.sigma2y = 0.01
    
    
    def init_y(self):
        '''
            Initialise the full Y
            the last t index is observed, and will not be modified
            
            network_representations: R x K x M
            Y:                    N x T x M
            Z:                    N x T x R
        '''
        
        features_combined = self.random_network.get_network_features_combined(self.Z)
        
        self.Y = np.zeros((self.N, self.T, self.M))
        
        # t=0 is different
        self.Y[:,0,:] = np.dot(features_combined[:,0,:], self.time_weights[1,0]) + np.sqrt(self.sigma2y)*np.random.randn(self.N, self.M)
        
        # t < T
        for t in np.arange(1, self.T-1):
            self.Y[:, t, :] = np.dot(self.Y[:, t-1, :], self.time_weights[0, t]) + np.dot(features_combined[:, t, :], self.time_weights[1, t]) + np.sqrt(self.sigma2y)*np.random.randn(self.N, self.M)
        
        # t= T is fixed
        self.Y[:, self.T-1, :] = self.YT
        
    
    ########
    
    def sample_invgamma(self, alpha, beta):
        '''
            Sample from an inverse gamma. numpy uses the shape/scale, not alpha/beta...
        '''
        return 1./np.random.gamma(alpha, 1./beta)
    
    def sample_log_bernoulli(self, lp1, lp0):
        '''
            Sample a bernoulli from log-transformed probabilities
        '''
        #print lp0-lp1
        if (lp0-lp1) < -500:
            p1 = 1.
        elif (lp0-lp1) > 500:
            p1 = 0.
        else:
            p1 = 1./(1+np.exp(lp0-lp1))
        
        return np.random.rand() < p1
    
    def sample_discrete_logp(self, log_prob):
        '''
            Use the logistic link function to get back to probabilities (thanks Sam Roweis)
            Also put a constant in it to avoid underflows
        '''
        
        b = - np.log(self.K) - np.max(log_prob)
        
        prob = np.exp(log_prob+b)/np.sum(np.exp(log_prob+b))
        cum_prob = np.cumsum(prob)
        
        return np.where(np.random.rand() < cum_prob)[0][0] # Slightly faster than np.find
    
    
    
    #######
    
    def sample_all(self):
        '''
            Do one full sweep of sampling
        '''
        
        # t = time.time()
        self.sample_z()
        # print "Sample_z time: %.3f" % (time.time()-t)
        
        # t = time.time()
        self.sample_y()
        # print "Sample_y time: %.3f" % (time.time()-t)
        
        if self.sigma_to_sample:
            # t = time.time()
            self.sample_sigmay()
            # print "Sample_sigmay time: %.3f" % (time.time()-t)
        
    
    def sample_z(self):
        '''
            Main method, get the new Z by Collapsed Gibbs sampling.
            Discrete variable, with Dirichlet prior, integrated over.
            
            Do everything in log-domain, to avoid numerical errors
            
        '''
        
        # Iterate over whole matrix
        permuted_datapoints = np.random.permutation(np.arange(self.N))
        
        for n in permuted_datapoints:
            # For each datapoint, need to resample the new z_ikt's
            
            permuted_time = np.random.permutation(np.arange(self.T))
            
            for t in permuted_time:
                
                permuted_population = np.random.permutation(np.arange(self.R))
                
                for r in permuted_population:
                    
                    # Update the counts
                    self.Akr[self.Z[n,t,r], r] -= 1
                    
                    for k in np.arange(self.K):
                        # Get the prior prob of z_n_t_k
                        self.lprob_zntrk[k] = np.log(self.dir_alpha + self.Akr[k,r]) - np.log(self.K*self.dir_alpha + self.N - 1.)
                        
                        # Get the likelihood of ynt using z_n_t = k
                        self.Z[n, t, r] = k
                        lik_ynt = self.compute_loglikelihood_ynt(n, t)
                        
                        self.lprob_zntrk[k] += lik_ynt
                        
                        # print "%d,%d,%d,%d: lik_ynt:%.3f" % (n,t,r,k, lik_ynt)
                    
                    
                    # Get the new sample
                    new_zntr = self.sample_discrete_logp(self.lprob_zntrk)
                    
                    # Increment the counts
                    self.Akr[new_zntr, r] += 1
                    
                    self.Z[n,t, r] = new_zntr
                
    
    
    def sample_y(self):
        '''
            Sample a new y_t, from posterior normal: P(y_t | y_{t-1}, x_t) P(y_{t+1} | y_t, x_{t+1})
        '''
        
        features_combined = self.random_network.get_network_features_combined(self.Z)
        
        # Iterate over whole datapoints
        permuted_datapoints = np.random.permutation(np.arange(self.N))
        
        # Do everything in log-domain, to avoid numerical errors
        for n in permuted_datapoints:
            
            # Y[n,T] is not sampled, it's observed, don't touch it
            permuted_time = np.random.permutation(np.arange(self.T-1))
            
            for t in permuted_time:
                
                # Posterior covariance
                invdelta_nosigma2y = (np.eye(self.M) + np.dot(self.time_weights[0, t+1], self.time_weights[0, t+1].T))
                
                # Posterior mean
                mu = np.dot(self.time_weights[0, t+1].T, (self.Y[n,t+1] - np.dot(self.time_weights[1, t+1], features_combined[n, t+1])))
                mu += np.dot(self.time_weights[1, t], features_combined[n,t])
                if t>0:
                    mu += np.dot(self.time_weights[0, t], self.Y[n, t-1])
                
                mu = np.linalg.solve(invdelta_nosigma2y, mu)
                
                # Sample the new Y[n,t]
                self.Y[n,t] = mu + np.sqrt(self.sigma2y)*np.dot(np.linalg.cholesky(invdelta_nosigma2y), np.random.randn(self.M))
                
                # OLD VERSION, easier A and B
                # # Posterior covariance
                # delta = self.sigma2y/(1. + self.time_weights[0, t+1]**2.)
                # 
                # # Posterior mean
                # mu = self.time_weights[0, t+1]*(self.Y[n,t+1] - self.time_weights[1, t+1]*features_combined[n, t+1])
                # mu += self.time_weights[1, t]*features_combined[n,t]
                # if t>0:
                #     mu += self.time_weights[0, t]*self.Y[n, t-1]
                # mu /= (1. + self.time_weights[0, t+1]**2)
                # 
                # # Sample the new Y[n,t]
                # self.Y[n,t] = mu + np.sqrt(delta)*np.random.randn(self.M)
            
        
        
    
    def sample_sigmay(self):
        '''
            Sample a new sigmay, assuming an inverse-gamma prior
            
            Y: N x T x M
        '''
        
        # TODO CHANGE THIS.
        
        features_combined = self.random_network.get_network_features_combined(self.Z)
        
        # Computes \sum_t \sum_n (y_n_t - beta_t y_n_{t-1} - alpha_t WP z_n_t), with G also
        Ytminus = np.zeros_like(self.Y)
        Ytminus[:, 1:, :] = self.Y[:, :-1, :]
        Y_proj = np.zeros_like(self.Y)
        for t in np.arange(self.T):
            Y_proj[:,t,:] = self.Y[:,t,:] - np.dot(features_combined[:,t,:], self.time_weights[1,t]) - np.dot(Ytminus[:,t,:], self.time_weights[0,t])
        
        self.sigma2y = self.sample_invgamma(self.sigmay_alpha + self.T*self.N*self.M/2., self.sigmay_beta + 0.5*np.tensordot(Y_proj, Y_proj, axes=3))
    
    
    # def compute_likelihood_ynt(self, znt, n, t):
    #         '''
    #             Compute the log-likelihood of one datapoint under the current parameters.
    #         '''
    #         
    #         ynt_proj = self.Y[n,t] - self.time_weights[1, t]*self.random_network.network_orientations[:,znt]
    #         if t>0:
    #             ynt_proj -= self.time_weights[0, t]*self.Y[n, t-1]
    #         
    #         return (2*np.pi*self.sigma2y)**(-self.M/2.)*np.exp((-0.5/self.sigma2y)*np.dot(ynt_proj, ynt_proj))
    #     
    
    def compute_loglikelihood_ynt(self, n, t):
        '''
            Compute the log-likelihood of one datapoint under the current parameters.
        '''
        
        features_combined = self.random_network.get_network_features_combined(self.Z[n,t])
        
        ynt_proj = self.Y[n,t] - np.dot(self.time_weights[1, t], features_combined)
        if t>0:
            ynt_proj -= np.dot(self.time_weights[0, t], self.Y[n, t-1])
            
        
        l = -0.5*self.M*np.log(2.*np.pi*self.sigma2y)
        l -= 0.5/self.sigma2y*np.dot(ynt_proj, ynt_proj)
        return l
        
    
    
    def compute_joint_loglike(self):
        '''
            Compute the joint loglikelihood 
        '''
        
        l = self.compute_loglike_z()
        l += self.compute_loglike_y()
        
        return l
        
    
    def compute_all_loglike(self):
        '''
            Compute the joint loglikelihood 
        '''
        # 
        ly = self.compute_loglike_y()
        lz = self.compute_loglike_z()
        
        return (ly, lz, ly+lz)
    
    
    def compute_loglike_y(self):
        '''
            Compute the log likelihood of P(Y | Y, X, sigma2, P)
        '''
        
        features_combined = self.random_network.get_network_features_combined(self.Z)
        
        Ytminus = np.zeros_like(self.Y)
        Ytminus[:, 1:, :] = self.Y[:, :-1, :]
        Y_proj = np.zeros_like(self.Y)
        for t in np.arange(self.T):
            Y_proj[:,t,:] = self.Y[:,t,:] - np.dot(features_combined[:,t,:], self.time_weights[1,t]) - np.dot(Ytminus[:,t,:], self.time_weights[0,t])
        
        l = -0.5*self.N*self.M*self.T*np.log(2.*np.pi*self.sigma2y)
        l -= 0.5/self.sigma2y*np.tensordot(Y_proj, Y_proj, axes=3)
        return l
    
    
    def compute_loglike_z(self):
        '''
            Compute the log probability of P(Z)
        '''
        
        
        l = self.R*scsp.gammaln(self.K*self.dir_alpha) - self.R*self.K*scsp.gammaln(self.dir_alpha)
        
        for r in np.arange(self.R):            
            for k in np.arange(self.K):
                l += scsp.gammaln(self.dir_alpha + self.Akr[k, r])
            l -= scsp.gammaln(self.K*self.dir_alpha + self.N)
        
        return l
        
    
    #########
    
    def run(self, iterations=10, verbose=True):
        '''
            Run the sampler for some iterations, print some information
            
            Running time: XXms * N * iterations
        '''
        
        # Get the original loglikelihoods
        log_y = np.zeros(iterations+1)
        log_z = np.zeros(iterations+1)
        log_joint = np.zeros(iterations+1)
        
        (log_y[0], log_z[0], log_joint[0]) = self.compute_all_loglike()
        
        if verbose:
            print "Initialisation: likelihoods = y %.3f, z %.3f, joint: %.3f" % (log_y[0], log_z[0], log_joint[0])
        
        for i in np.arange(iterations):
            # Do a full sampling sweep
            self.sample_all()
            
            # Get the likelihoods
            (log_y[i+1], log_z[i+1], log_joint[i+1]) = self.compute_all_loglike()
            
            # Print report
            if verbose:
                print "Sample %d: likelihoods = y %.3f, z %.3f, joint: %.3f" % (i+1, log_y[i+1], log_z[i+1], log_joint[i+1])
        
        return (log_z, log_y, log_joint)
    
    
    def compute_correctness_full(self):
        '''
            Compute the percentage of correct identification of features
        '''
        return (np.sum(self.data_gen.Z_true == self.Z).astype(float)/self.Z.size)
    
    def plot_datapoint_approximation(self, t, n):
        pass
    
    def compute_errors(self, Z):
        '''
            Similar to above, but with arbitrary matrices
        '''
        return (np.sum(self.data_gen.Z_true != Z).astype(float)/self.Z.size)
    
    def print_z_comparison(self):
        '''
            Print some measures and plots
        '''
        print "\nCorrectness: %.3f" % self.compute_correctness_full()
        
        data_misclassified = np.unique(np.nonzero(np.any(self.Z != self.data_gen.Z_true, axis=1))[0])
        
        if data_misclassified.size > 0:
            # Only if they are errors, print them :)
            
            print "Wrong datapoints: %s\n" % (data_misclassified)
            
            print "N\tZ_true | inferred Z"
            for data_p in data_misclassified:
                print "%d\t%s" % (data_p, array2string(self.data_gen.Z_true[data_p]))
                print "\t%s" % (array2string(self.Z[data_p]))
    
    
    def compute_metric_all(self):
        '''
            Get metric statistics for the whole dataset
            
            Z:  N x T x R
        '''
        
        (angle_errors_stats, angle_errors) = self.compute_angle_error()
        
        if self.T==0:
            (angle_mean_error_nomisbind, angle_std_dev_error_nomisbind, angle_mean_vector_nomisbind, avg_error_nomisbind, misbound_datapoints) = self.compute_misbinds(angle_errors)
        
            return (angle_errors_stats, (angle_mean_error_nomisbind, angle_std_dev_error_nomisbind, angle_mean_vector_nomisbind, avg_error_nomisbind, misbound_datapoints))
        
        return [angle_errors_stats, angle_errors]
    
    
    
    def compute_angle_error(self):
        '''
            Compute the mean angle error for the current assignment of Z
        '''
        # Compute the angle difference error between predicted and ground truth
        angle_errors = self.random_network.possible_angles[self.Z] - self.random_network.possible_angles[self.data_gen.Z_true]
        
        # Correct for obtuse angles
        angle_errors = self.smallest_angle_vectors(angle_errors)
        
        # Compute the statistics. Uses the spherical formulation of standard deviation
        return (self.compute_mean_std_circular_data(angle_errors), angle_errors)
    
    
    def compute_misbinds(self, angles_errors):
        # TODO Convert this
        
        ## Now check if misbinding could have occured
        # TODO Only for T=2 now
        if self.T == 2:
            # Possible swaps
            different_allocations = cross(np.arange(self.R), np.arange(self.R))
            
            Z_swapped = self.Z
            
            angle_errors_misbinding = np.zeros_like(angle_errors)
            angle_errors_misbinding[:,0] = self.random_network.possible_orientations[self.Z[:,1]] - self.random_network.possible_orientations[self.data_gen.Z_true[:,0]]
            angle_errors_misbinding[:,1] = self.random_network.possible_orientations[self.Z[:,0]] - self.random_network.possible_orientations[self.data_gen.Z_true[:,1]]
            
            angle_errors_misbinding = self.smallest_angle_vectors(angle_errors_misbinding)
            
            # Assume that a datapoint was misbound if the sum of errors is smaller in the other setting.
            tot_error_misbound = np.sum(np.abs(angle_errors_misbinding), axis=1)
            tot_error_original = np.sum(np.abs(angle_errors), axis=1)
            misbound_datapoints = tot_error_misbound < tot_error_original
            
            # Now get another result, correcting for misbinding
            angles_errors_nomisbind = angle_errors.copy()
            angles_errors_nomisbind[misbound_datapoints] = angle_errors_misbinding[misbound_datapoints]
            
            (angle_mean_error_nomisbind, angle_std_dev_error_nomisbind, angle_mean_vector_nomisbind, avg_error_nomisbind) = self.compute_mean_std_circular_data(angles_errors_nomisbind)
            
    
    
    def smallest_angle_vectors(self, angles):
        '''
            Get the smallest angle between the two responses
        '''
        while np.any(angles < -np.pi):
            angles[angles < -np.pi] += 2.*np.pi
        while np.any(angles > np.pi):
            angles[angles > np.pi] -= 2.*np.pi
        return angles
    
    
    def compute_mean_std_circular_data(self, angles):
        '''
            Compute the mean vector, the std deviation according to the Circular Statistics formula
            Assume a NxTxR matrix, averaging over N
        '''
        
        # Average error
        avg_error = np.mean(np.abs(angles), axis=0)
        
        # Angle population vector
        angle_mean_vector = np.mean(np.exp(1j*angles), axis=0)
        
        # Population mean
        angle_mean_error = np.angle(angle_mean_vector)
        
        # Circular standard deviation estimate
        angle_std_dev_error = np.sqrt(-2.*np.log(np.abs(angle_mean_vector)))
        
        return (angle_mean_error, angle_std_dev_error, angle_mean_vector, avg_error)
    
    

####################################

def profile_me():
    import cProfile
    import pstats
    
    cProfile.runctx('profiling_run()', globals(), locals(), filename='profile_sampler.stats')
    
    stat = pstats.Stats('profile_sampler.stats')
    stat.strip_dirs().sort_stats('cumulative').print_stats()
    


def profiling_run():
    
    N = 100
    T = 3
    K = 30
    D = 30
    M = 200
    R = 2
    
    random_network = RandomNetwork.create_instance_uniform(K, M, D=D, R=R, W_type='dirichlet', W_parameters=[0.2, 0.7])
    data_gen = DataGenerator(N, T, random_network, type_Z='discrete', weighting_alpha=0.6, time_weights_prior='recency', sigma_y = 0.02)
    sampler = Sampler(data_gen, dirichlet_alpha=0.5/K, sigma_to_sample=True, sigma_alpha=3, sigma_beta=0.5)
    
    (log_y, log_z, log_joint) = sampler.run(10, verbose=True)
    


####################################

def do_simple_run(args):
    
    print "Simple run"
    
    
    N = args.N
    T = args.T
    K = args.K
    D = args.D
    M = args.M
    R = args.R
    nb_samples = args.nb_samples
    
    random_network = RandomNetwork.create_instance_uniform(K, M, D=D, R=R, W_type='dirichlet', W_parameters=[0.1, 0.5], sigma=0.2, gamma=0.005, rho=0.005)
    data_gen = DataGenerator(N, T, random_network, type_Z='discrete', weighting_alpha=0.75, specific_weighting=0.3, time_weights_prior='recency', sigma_y = 0.05)
    sampler = Sampler(data_gen, dirichlet_alpha=1./K, sigma_to_sample=False, sigma_alpha=5, sigma_beta=0.5)
    
    if True:
        t = time.time()
        
        (log_y, log_z, log_joint) = sampler.run(nb_samples, verbose=True)
        
        print '\nElapsed time: %d' % (time.time()-t)
        
        print '\nSigma_y: %.3f' % np.sqrt(sampler.sigma2y)
        
        sampler.print_z_comparison()
        
        (stats_original, angle_errors) = sampler.compute_metric_all()
        
        print stats_original
        
        # Computed beforehand
        precision_guessing = 0.2
        
        if True:
            plt.figure()
            plt.plot(1./stats_original[1]-precision_guessing)
            plt.show()
        precisions = 1./stats_original[1] - precision_guessing
        
        mean_last_precision = np.mean(precisions[:,-1])
        avg_precision = np.mean(precisions)
        
        plt.show()
    
    return locals()
    

def do_search_dirichlet_alpha(args):
    print "Plot effect of Dirichlet_alpha"
    
    N = args.N
    T = args.T
    K = args.K
    D = args.D
    M = args.M
    R = args.R
    
    
    dir_alpha_space = np.array([0.01, 0.1, 0.5, 0.7, 1.5])
    nb_repetitions = 2
    
    mean_last_precision = np.zeros((dir_alpha_space.size, nb_repetitions))
    avg_precision = np.zeros((dir_alpha_space.size, nb_repetitions))
    
    for dir_alpha_i in np.arange(dir_alpha_space.size):
        print "Doing Dir_alpha %.3f" % dir_alpha_space[dir_alpha_i]
        
        for repet_i in np.arange(nb_repetitions):
            print "%d/%d" % (repet_i+1, nb_repetitions)
            
            random_network = RandomNetwork.create_instance_uniform(K, M, D=D, R=R, W_type='dirichlet', W_parameters=[0.1, dir_alpha_space[dir_alpha_i]], sigma=0.2, gamma=0.005, rho=0.005)
            data_gen = DataGenerator(N, T, random_network, type_Z='discrete', weighting_alpha=0.9, time_weights_prior='recency', sigma_y = 0.02)
            sampler = Sampler(data_gen, dirichlet_alpha=1./K, sigma_to_sample=False, sigma_alpha=2, sigma_beta=0.5)
            
            (log_y, log_z, log_joint) = sampler.run(5, verbose=False)
            
            (stats_original, angle_errors) = sampler.compute_metric_all()
            
            print stats_original
            
            # Computed beforehand
            precision_guessing = 0.2
            
            # Get the precisions
            precisions = 1./stats_original[1] - precision_guessing
            
            mean_last_precision[dir_alpha_i, repet_i] = np.mean(precisions[-1])
            avg_precision[dir_alpha_i, repet_i] = np.mean(precisions)
    
    
    mean_last_precision_avg = np.mean(mean_last_precision, axis=1)
    mean_last_precision_std = np.std(mean_last_precision, axis=1)
    avg_precision_avg = np.mean(avg_precision, axis=1)
    avg_precision_std = np.std(avg_precision, axis=1)
    
    ax = plot_std_area(dir_alpha_space, mean_last_precision_avg, mean_last_precision_std)
    plot_std_area(dir_alpha_space, avg_precision_avg, avg_precision_std, ax_handle=ax)
    
    return locals()
    

def do_search_alphat(args):
    print "Plot effect of alpha_t"
    
    N = args.N
    T = args.T
    K = args.K
    D = args.D
    M = args.M
    R = args.R
    
    alphat_space = np.array([0.3, 0.5, 0.7, 1., 1.5])
    nb_repetitions = 5
    
    mean_last_precision = np.zeros((alphat_space.size, nb_repetitions))
    other_precision = np.zeros((alphat_space.size, nb_repetitions))
    
    for alpha_i in np.arange(alphat_space.size):
        print "Doing alpha_t %.3f" % alphat_space[alpha_i]
        
        for repet_i in np.arange(nb_repetitions):
            print "%d/%d" % (repet_i+1, nb_repetitions)
            
            random_network = RandomNetwork.create_instance_uniform(K, M, D=D, R=R, W_type='dirichlet', W_parameters=[0.1, 0.5], sigma=0.2, gamma=0.005, rho=0.005)
            data_gen = DataGenerator(N, T, random_network, type_Z='discrete', weighting_alpha=alphat_space[alpha_i], time_weights_prior='recency', sigma_y = 0.02)
            sampler = Sampler(data_gen, dirichlet_alpha=1./K, sigma_to_sample=False, sigma_alpha=2, sigma_beta=0.5)
            
            (log_y, log_z, log_joint) = sampler.run(50, verbose=False)
            
            (stats_original, angle_errors) = sampler.compute_metric_all()
            
            #print stats_original
            
            # Computed beforehand
            precision_guessing = 0.2
            
            # Get the precisions
            precisions = 1./stats_original[1] - precision_guessing
            
            mean_last_precision[alpha_i, repet_i] = np.mean(precisions[-1])
            other_precision[alpha_i, repet_i] = np.mean(precisions[:-1])
    
    return locals()
    

####################################
if __name__ == '__main__':
    # Switch on different actions
    actions = [do_simple_run, do_search_dirichlet_alpha, do_search_alphat]
    
    print sys.argv[1:]
    
    parser = argparse.ArgumentParser(description='Sample a model of Visual working memory.')
    parser.add_argument('--label', help='label added to output files', default='')
    parser.add_argument('--output_directory', nargs='?', default='Data')
    parser.add_argument('--action_to_do', choices=np.arange(len(actions)), default=0)
    parser.add_argument('--nb_samples', default=20)
    parser.add_argument('--N', default=100, help='Number of datapoints')
    parser.add_argument('--T', default=2, help='Number of times')
    parser.add_argument('--K', default=10, help='Number of representated features')
    parser.add_argument('--D', default=20, help='Dimensionality of features')
    parser.add_argument('--M', default=200, help='Dimensionality of data/memory')
    parser.add_argument('--R', default=2, help='Number of population codes')
    
    args = parser.parse_args()
    
    should_save = True
    output_dir = os.path.join(args.output_directory, args.label)
    
    # Run it
    all_vars = actions[args.action_to_do](args)
    
    if 'data_gen' in all_vars:
        data_gen = all_vars['data_gen']
    if 'sampler' in all_vars:
        sampler = all_vars['sampler']
    if 'log_joint' in all_vars:
        log_joint = all_vars['log_joint']
    if 'log_z' in all_vars:
        log_z = all_vars['log_z']
    
    # Save the results
    if should_save:
        output_file = os.path.join(output_dir, 'all_vars.npy')
        np.save(output_file, all_vars)
    
    plt.show()
