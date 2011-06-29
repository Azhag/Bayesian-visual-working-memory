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

class Sampler:
    '''
        Use Y and Z only, no intermediate X.
        Use multiple \pi prior for different times (idea: enforce sparsity per time)
    '''
    def __init__(self, data_gen, pi_alpha=0.5, sigma_to_sample=True, sigma_alpha=1.0, sigma_beta=2.0):
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
        self.init_z(pi_alpha)
        
        # Initial sigma_y
        self.sigma_to_sample = sigma_to_sample
        
        # Initial sigma_y
        self.init_sigma2y(sigma_alpha, sigma_beta)
        
        # Initialise the Y
        self.init_y()
        
        # Maximum exponent in current precision
        self.__maxexp__ = np.finfo('float').maxexp
    
    def init_z(self, alpha):
        '''
            Sample initial Z
            
            Z:          N x T x R x K
            m:          T x R x K
        '''
        
        self.alpha_k = float(alpha)/self.K
        #self.Z = np.random.rand(self.N, self.T, self.K) < self.pi
        self.Z = np.random.rand(self.N, self.T, self.R, self.K) < 0.1
        # self.Z = np.zeros((self.N, self.T, self.K))
        
        # Get the matrix of counts
        self.m = np.sum(self.Z, axis=0)
        
    
    
    def init_sigma2y(self, sigmay_alpha, sigmay_beta):
        self.sigmay_alpha = sigmay_alpha
        self.sigmay_beta = sigmay_beta
        self.sigma2y = self.sample_invgamma(sigmay_alpha, sigmay_beta)
        # self.sigma2y = 0.01
    
    
    def init_y(self):
        '''
            Initialise the full Y
            the last t index is observed, and will not be modified
        '''
        
        features_combined = self.random_network.get_network_features_combined_binary(self.Z)
        
        self.Y = np.zeros((self.N, self.T, self.M))
        
        # t=0 is different
        self.Y[:,0,:] = self.time_weights[1,0]*features_combined[:,0,:] + np.sqrt(self.sigma2y)*np.random.randn(self.N, self.M)
        
        # t < T
        for t in np.arange(1, self.T-1):
            self.Y[:, t, :] = self.time_weights[0, t]*self.Y[:, t-1, :] + self.time_weights[1, t]*features_combined[:, t, :] + np.sqrt(self.sigma2y)*np.random.randn(self.N, self.M)
        
        # t= T is fixed
        self.Y[:, self.T-1, :] = self.YT
        
    
    ########
    
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
            Bernoulli variable, with Beta prior, integrated over.
        '''
        
        # Iterate over whole matrix
        permuted_datapoints = np.random.permutation(np.arange(self.N))
        
        # Do everything in log-domain, to avoid numerical errors
        for n in permuted_datapoints:
            # For each datapoint, need to resample the new z_ikt's
            
            permuted_time = np.random.permutation(np.arange(self.T))
            
            for t in permuted_time:
                
                permuted_population = np.random.permutation(np.arange(self.R))
                
                for r in permuted_population:
                    
                    permuted_features = np.random.permutation(np.arange(self.K))
                    
                    for k in permuted_features:
                        # Get the data and change the counts
                        m_notnt_k = (self.m[t,r,k] - self.Z[n, t, r,k])
                        
                        # print zi
                        
                        # Get the probabilities for znkt=1
                        lprob_znkt_1 = np.log(m_notnt_k + self.alpha_k) - np.log(self.N + self.alpha_k)
                        
                        # Get the likelihood of ynt
                        self.Z[n, t, r, k] = True
                        llik_1 = self.compute_loglikelihood_ynt(n, t)
                        
                        lprob_znktlik_1 = lprob_znkt_1 + llik_1
                        
                        # Get the probabilities for zik=0
                        lprob_znkt_0 = np.log(self.N - m_notnt_k) - np.log(self.N + self.alpha_k)
                        self.Z[n, t, r, k] = False
                        llik_0 = self.compute_loglikelihood_ynt(n, t)
                        lprob_znktlik_0 = lprob_znkt_0 + llik_0
                        
                        # Get the new sample
                        sampled_zi = self.sample_log_bernoulli(lprob_znktlik_1, lprob_znktlik_0)
                        
                        #print '%d %d' % (m_noti_k + sampled_zi, self.m[k])
                        
                        # Update the counts
                        self.m[t, r, k] = m_notnt_k + sampled_zi
                        self.Z[n,t,r,k] = sampled_zi
                        
                        # print self.Z[i]
    
    
    def sample_y(self):
        '''
            Sample a new y_t, from posterior normal: P(y_t | y_{t-1}, x_t) P(y_{t+1} | y_t, x_{t+1})
        '''
        
        features_combined = self.random_network.get_network_features_combined_binary(self.Z)
        
        # Iterate over whole datapoints
        permuted_datapoints = np.random.permutation(np.arange(self.N))
        
        # Do everything in log-domain, to avoid numerical errors
        for n in permuted_datapoints:
            
            # Y[n,T] is not sampled, it's observed, don't touch it
            permuted_time = np.random.permutation(np.arange(self.T-1))
            
            for t in permuted_time:
                
                # Posterior covariance
                delta = self.sigma2y/(1. + self.time_weights[0, t+1]**2)
                
                # Posterior mean
                mu = self.time_weights[0, t+1]*(self.Y[n,t+1] - self.time_weights[1, t+1]*features_combined[n, t+1])
                mu += self.time_weights[1, t]*features_combined[n,t]
                if t>0:
                    mu += self.time_weights[0, t]*self.Y[n, t-1]
                mu /= (1. + self.time_weights[0, t+1]**2)
                
                # Sample the new Y[n,t]
                self.Y[n,t] = mu + np.sqrt(delta)*np.random.randn(self.M)
            
        
        
    
    def sample_sigmay(self):
        '''
            Sample a new sigmay, assuming an inverse-gamma prior
        '''
        
        features_combined = self.random_network.get_network_features_combined_binary(self.Z)
        
        # Computes \sum_t \sum_n (y_n_t - beta_t y_n_{t-1} - alpha_t WP z_n_t), with tensor also
        Ytminus = np.zeros_like(self.Y)
        Ytminus[:, 1:, :] = self.Y[:, :-1, :]
        Y_proj = self.Y - (features_combined.transpose(0,2,1)*self.time_weights[1]).transpose(0,2,1) - (Ytminus.transpose(0,2,1)*self.time_weights[0]).transpose(0,2,1)
        
        self.sigma2y = self.sample_invgamma(self.sigmay_alpha + self.T*self.N*self.M/2., self.sigmay_beta + 0.5*np.tensordot(Y_proj, Y_proj, axes=3))
    
    
    def compute_loglikelihood_ynt(self, n, t):
        '''
            Compute the log-likelihood of one datapoint under the current parameters.
        '''
        
        features_combined = self.random_network.get_network_features_combined_binary(self.Z[n,t])
        
        ynt_proj = self.Y[n,t] - self.time_weights[1, t]*features_combined
        if t>0:
            ynt_proj -= self.time_weights[0, t]*self.Y[n, t-1]
            
        
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
        features_combined = self.random_network.get_network_features_combined_binary(self.Z)
        
        Ytminus = np.zeros_like(self.Y)
        Ytminus[:, 1:, :] = self.Y[:, :-1, :]
        Y_proj = self.Y.transpose(0,2,1) - (features_combined.transpose(0,2,1)*self.time_weights[1]) - (Ytminus.transpose(0,2,1)*self.time_weights[0])
        
        l = -0.5*self.N*self.M*self.T*np.log(2.*np.pi*self.sigma2y)
        l -= 0.5/self.sigma2y*np.tensordot(Y_proj, Y_proj, axes=3)
        return l
    
    
    def compute_loglike_z(self):
        '''
            Compute the log probability of P(Z)
        '''
        
        # print np.sum(self.m)
        
        l = self.K*self.T*np.log(self.alpha_k)
        for t in np.arange(self.T):
            for r in np.arange(self.R):
                for k in np.arange(self.K):
                    l += scsp.gammaln(self.N - self.m[t, r, k] + 1.)
                    l -= scsp.gammaln(self.N + self.alpha_k + 1.)
                    l += scsp.gammaln(self.m[t, r, k] + self.alpha_k)
        
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
        
        data_misclassified = np.unique(np.nonzero(np.any(np.any(self.Z != self.data_gen.Z_true, axis=3), axis=2))[0])
        
        if data_misclassified.size > 0:
            # Only if they are errors, print them :)
            plot_limit = 15
            data_misclassified_plot = data_misclassified[:plot_limit]
            
            max_nb_features = np.max((np.max(np.sum(self.Z[data_misclassified_plot], axis=1)), 
                                      np.max(np.sum(true_Z[data_misclassified_plot], axis=1))))
            
            print "Wrong datapoints: %s\n(plotting the first %d only)" % (data_misclassified, plot_limit)
            
    
    def compute_metric_all(self):
        # TODO CONVERT FOR BINARY
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
        # TODO CONVERT FOR BINARY
        '''
            Compute the mean angle error for the current assignment of Z
        '''
        # Compute the angle difference error between predicted and ground truth
        angle_errors = self.random_network.possible_angles[self.Z] - self.random_network.possible_angles[self.data_gen.Z_true]
        
        # Correct for obtuse angles
        angle_errors = self.smallest_angle_vectors(angle_errors)
        
        # Compute the statistics. Uses the spherical formulation of standard deviation
        return (self.compute_mean_std_circular_data(angle_errors), angle_errors)
    
    def smallest_angle_vectors(self, angles):
        # TODO CONVERT FOR BINARY
        '''
            Get the smallest angle between the two responses
        '''
        while np.any(angles < -np.pi):
            angles[angles < -np.pi] += 2.*np.pi
        while np.any(angles > np.pi):
            angles[angles > np.pi] -= 2.*np.pi
        return angles
    
    
    def compute_mean_std_circular_data(self, angles):
        # TODO CONVERT FOR BINARY
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
    
    
    
    
    ####
    
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
    
    

####################################

def profile_me():
    import cProfile
    import pstats
    
    print "Profiling the application..."
    
    cProfile.runctx('profiling_run()', globals(), locals(), filename='profile_sampler.stats')
    
    stat = pstats.Stats('profile_sampler.stats')
    stat.strip_dirs().sort_stats('cumulative').print_stats()
    
    return {}


def profiling_run():
    
    N = 100
    T = 2
    K = 20
    D = 30
    M = 100
    R = 2
    
    random_network = RandomNetwork.create_instance_uniform(K, M, D=D, R=R, W_type='dirichlet', W_parameters=[0.1, 0.5], sigma=0.2, gamma=0.005, rho=0.005)
    data_gen = DataGenerator(N, T, random_network, type_Z='binary', weighting_alpha=0.5, weight_prior='recency', sigma_y = 0.02)
    sampler = Sampler(data_gen, pi_alpha=1., sigma_to_sample=True, sigma_alpha=2, sigma_beta=0.5)
    
    (log_y, log_z, log_joint) = sampler.run(10, verbose=False)
    


####################################

def do_simple_run():
    
    
    print "Simple run"
    
    N = 100
    T = 2
    K = 20
    D = 50
    M = 100
    R = 2
    
    random_network = RandomNetwork.create_instance_uniform(K, M, D=D, R=R, W_type='dirichlet', W_parameters=[0.1, 0.5], sigma=0.2, gamma=0.005, rho=0.005)
    data_gen = DataGenerator(N, T, random_network, type_Z='binary', weighting_alpha=0.5, weight_prior='recency', sigma_y = 0.02)
    sampler = Sampler(data_gen, pi_alpha=1., sigma_to_sample=True, sigma_alpha=2, sigma_beta=0.5)
    
    if True:
        t = time.time()
        
        (log_y, log_z, log_joint) = sampler.run(10, verbose=True)
        
        print '\nElapsed time: %d' % (time.time()-t)
        
        print '\nSigma_y: %.3f' % np.sqrt(sampler.sigma2y)
        
        # sampler.print_z_comparison()
        #         
        #         (stats_original, angle_errors) = sampler.compute_metric_all()
        #         
        #         print stats_original
        #         
        #         # Computed beforehand
        #         precision_guessing = 0.2
        #         
        #         if False:
        #             plt.figure()
        #             plt.plot(1./stats_original[1]-precision_guessing)
        #             plt.show()
        #         precisions = 1./stats_original[1] - precision_guessing
        #         
        #         mean_last_precision = np.mean(precisions[:,-1])
        #         avg_precision = np.mean(precisions)
        #         
        #         plt.show()
    
    return locals()
    


######################################################


if __name__ == '__main__':
    
    # Switch on different actions
    action_to_do = 1
    actions = [do_simple_run, profile_me]
    
    # Run it
    all_vars = actions[action_to_do]()
    
    if 'data_gen' in all_vars:
        data_gen = all_vars['data_gen']
    if 'sampler' in all_vars:
        sampler = all_vars['sampler']
    
    plt.show()
    
