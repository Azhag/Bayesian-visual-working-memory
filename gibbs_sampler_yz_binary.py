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
        Use one \pi prior for everything
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
            Sample initial Z from a beta distribution
        '''
        
        self.alpha_k = float(alpha)/self.K
        self.pi = np.random.beta(self.alpha_k, 1., size=self.K)
        #self.Z = np.random.rand(self.N, self.T, self.K) < self.pi
        self.Z = np.random.rand(self.N, self.T, self.K) < 0.1
        # self.Z = np.zeros((self.N, self.T, self.K))
        
        # Get the matrix of counts
        self.m = np.sum(np.sum(self.Z, axis=0), axis=0)
        
    
    
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
        
        features = np.dot(self.Z, self.random_network.network_orientations.T)
        
        self.Y = np.zeros((self.N, self.T, self.M))
        
        # t=0 is different
        self.Y[:,0,:] = self.time_weights[1,0]*features[:,0,:] + np.sqrt(self.sigma2y)*np.random.randn(self.N, self.M)
        
        # t < T
        for t in np.arange(1, self.T-1):
            self.Y[:, t, :] = self.time_weights[0, t]*self.Y[:, t-1, :] + self.time_weights[1, t]*features[:, t, :] + np.sqrt(self.sigma2y)*np.random.randn(self.N, self.M)
        
        # t= T is fixed
        self.Y[:, self.T-1, :] = self.YT
        
    
    ########
    
    def sample_all(self):
        '''
            Do one full sweep of sampling
        '''
        
        self.sample_z()
        
        self.sample_y()
        
        if self.sigma_to_sample:
            self.sample_sigmay()
        
    
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
                
                permuted_features = np.random.permutation(np.arange(self.K))
                
                for k in permuted_features:
                    # Get the data and change the counts
                    m_notnt_k = (self.m - self.Z[n, t])[k]
                    
                    # print zi
                    
                    # Get the probabilities for znkt=1
                    lprob_znkt_1 = np.log(m_notnt_k + self.alpha_k) - np.log(self.N*self.T + self.alpha_k)
                    
                    # Get the likelihood of ynt
                    new_znt = self.Z[n, t].copy()
                    new_znt[k] = True
                    llik_1 = self.compute_loglikelihood_ynt(new_znt, n, t)
                    
                    lprob_znktlik_1 = lprob_znkt_1 + llik_1
                    
                    # Get the probabilities for zik=0
                    lprob_znkt_0 = np.log(self.N*self.T - m_notnt_k) - np.log(self.N*self.T + self.alpha_k)
                    new_znt[k] = False
                    llik_0 = self.compute_loglikelihood_ynt(new_znt, n, t)
                    lprob_znktlik_0 = lprob_znkt_0 + llik_0
                    
                    # Get the new sample
                    sampled_zi = self.sample_log_bernoulli(lprob_znktlik_1, lprob_znktlik_0)
                    
                    #print '%d %d' % (m_noti_k + sampled_zi, self.m[k])
                    
                    # Update the counts
                    self.m[k] = m_notnt_k + sampled_zi
                    self.Z[n,t,k] = sampled_zi
                    
                    # print self.Z[i]
    
    
    def sample_y(self):
        '''
            Sample a new y_t, from posterior normal: P(y_t | y_{t-1}, x_t) P(y_{t+1} | y_t, x_{t+1})
        '''
        
        features = np.dot(self.Z, self.random_network.network_orientations.T)
        
        # Iterate over whole datapoints
        permuted_datapoints = np.random.permutation(np.arange(self.N))
        
        # Do everything in log-domain, to avoid numerical errors
        for n in permuted_datapoints:
            
            # Y[n,T] is not sampled, it's observed, don't touch it
            permuted_time = np.random.permutation(np.arange(self.T-1))
            
            for t in permuted_time:
                
                # Posterior covariance
                delta = self.sigma2y/(1. + self.time_weights[1, t+1]**2)
                
                # Posterior mean
                mu = self.time_weights[1, t+1]*(self.Y[n,t+1] - self.time_weights[0, t+1]*features[n, t+1])
                mu += self.time_weights[0, t]*features[n,t]
                if t>0:
                    mu += self.time_weights[1, t]*self.Y[n, t-1]
                mu /= (1. + self.time_weights[1, t+1]**2)
                
                # Sample the new Y[n,t]
                self.Y[n,t] = mu + np.sqrt(delta)*np.random.randn(self.M)
            
        
        
    
    def sample_sigmay(self):
        '''
            Sample a new sigmay, assuming an inverse-gamma prior
        '''
        
        features = np.dot(self.Z, self.random_network.network_orientations.T)
        
        # Computes \sum_t \sum_n (y_n_t - beta_t y_n_{t-1} - alpha_t WP z_n_t), with tensor also
        Ytminus = np.zeros_like(self.Y)
        Ytminus[:, 1:, :] = self.Y[:, :-1, :]
        Y_proj = self.Y - (features.transpose(0,2,1)*self.time_weights[0]).transpose(0,2,1) - (Ytminus.transpose(0,2,1)*self.time_weights[1]).transpose(0,2,1)
        
        self.sigma2y = self.sample_invgamma(self.sigmay_alpha + self.T*self.N*self.M/2., self.sigmay_beta + 0.5*np.tensordot(Y_proj, Y_proj, axes=3))
    
    
    def compute_loglikelihood_ynt(self, znt, n, t):
        '''
            Compute the log-likelihood of one datapoint under the current parameters.
        '''
        ynt_proj = self.Y[n,t] - self.time_weights[0, t]*np.dot(self.random_network.network_orientations, znt)
        if t>0:
            ynt_proj -= self.time_weights[1, t]*self.Y[n, t-1]
            
        
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
        features = np.dot(self.Z, self.random_network.network_orientations.T)
        
        Ytminus = np.zeros_like(self.Y)
        Ytminus[:, 1:, :] = self.Y[:, :-1, :]
        Y_proj = self.Y.transpose(0,2,1) - (features.transpose(0,2,1)*self.time_weights[0]) - (Ytminus.transpose(0,2,1)*self.time_weights[1])
        
        l = -0.5*self.N*self.M*self.T*np.log(2.*np.pi*self.sigma2y)
        l -= 0.5/self.sigma2y*np.tensordot(Y_proj, Y_proj, axes=3)
        return l
    
    
    def compute_loglike_z(self):
        '''
            Compute the log probability of P(Z)
        '''
        
        # print np.sum(self.m)
        
        l = self.K*np.log(self.alpha_k)
        for k in np.arange(self.K):
            l += scsp.gammaln(self.N*self.T - self.m[k] + 1.)
            l -= scsp.gammaln(self.N*self.T + self.alpha_k + 1.)
            l += scsp.gammaln(self.m[k] + self.alpha_k)
            
        
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
    
    
    def compute_correctness_full(self, true_Z):
        '''
            Compute the percentage of correct identification of features
        '''
        return (np.sum(true_Z == self.Z).astype(float)/self.Z.size)
    
    def compute_errors(self, Z, true_Z):
        '''
            Similar to above, but with arbitrary matrices
        '''
        return (np.sum(true_Z != Z).astype(float)/self.Z.size)
    
    def print_z_comparison(self, true_Z):
        '''
            Print some measures and plots
        '''
        print "\nCorrectness: %.3f" % self.compute_correctness_full(true_Z)
        
        data_misclassified = np.nonzero(np.any(self.Z != true_Z, axis=1))[0]
        
        if data_misclassified.size > 0:
            # Only if they are errors, print them :)
            plot_limit = 15
            data_misclassified_plot = data_misclassified[:plot_limit]
            
            max_nb_features = np.max((np.max(np.sum(self.Z[data_misclassified_plot], axis=1)), 
                                      np.max(np.sum(true_Z[data_misclassified_plot], axis=1))))
            
            print "Wrong datapoints: %s\n(plotting the first %d only)" % (data_misclassified, plot_limit)
            
            # # Now show the misclassified data
            # f = plt.figure()
            # 
            # plt_cnt = 1
            # for (i, miss_data) in enumerate(data_misclassified_plot):
            #     #print 'Data %d' % miss_data
            #     plt_cnt = 1+i*(max_nb_features+1)
            #     
            #     # Plot the datapoint
            #     subax = f.add_subplot(data_misclassified_plot.size, max_nb_features+1, plt_cnt)
            #     scaledimage(self.X[miss_data], ax=subax)
            #     
            #     plt_cnt = 2+2*i*(max_nb_features+1)
            #     
            #     # Plot the inferred features
            #     inferred_features = np.nonzero(self.Z[miss_data])[0]
            #     first_plot = True
            #     for feat in inferred_features:
            #         subax = f.add_subplot(data_misclassified_plot.size*2, max_nb_features+1, plt_cnt)
            #         scaledimage(self.features[feat], ax=subax)
            #         
            #         if first_plot:
            #             subax.text(-0.25, 0.5, 'S', rotation='vertical', fontsize=9, horizontalalignment='center', 
            #                          verticalalignment='center', transform = subax.transAxes)
            #             first_plot = False
            #         
            #         plt_cnt += 1
            #     
            #     plt_cnt = 3+max_nb_features+2*i*(max_nb_features+1)
            #     
            #     # Plot the true features
            #     true_features = np.nonzero(true_Z[miss_data])[0]
            #     first_plot = True
            #     for feat in true_features:
            #         subax = f.add_subplot(data_misclassified_plot.size*2, max_nb_features+1, plt_cnt)
            #         scaledimage(self.features[feat], ax=subax)
            #         
            #         if first_plot:
            #             subax.text(-0.25, 0.5, 'Zt', rotation='vertical', fontsize=9, horizontalalignment='center', 
            #                          verticalalignment='center', transform = subax.transAxes)
            #             first_plot = False
            #         
            #         plt_cnt += 1
            # 
            # plt.subplots_adjust(hspace=0.05, wspace=0.01, left=0.02, right=0.98, bottom=0.01, top=0.99)
            # plt.show()
    
    
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
    
    



if __name__ == '__main__':
    
    N = 100
    T = 2
    K = 5
    D = 30
    M = 100
    R = 2
    
    
    random_network = RandomNetwork.create_instance_uniform(K, M, D=D, R=R, W_type='dirichlet', W_parameters=[0.1, 0.5], sigma=0.2, gamma=0.005, rho=0.005)
    
    data_gen = DataGenerator(N, T, random_network, type_Z='binary', weighting_alpha=0.5, weight_prior='recency', sigma_y = 0.02)
    sampler = Sampler(data_gen, pi_alpha=1.0, sigma_to_sample=False, sigma_alpha=4, sigma_beta=0.5)
    
    t = time.time()
    (log_y, log_z, log_joint) = sampler.run(50, verbose=True)
    
    print time.time()-t
    
    sampler.print_z_comparison(data_gen.Z_true)
    
    # 
    # #### Basic calls
    # data_gen = DataGenerator(N, M, 0.2, max_nb_features=6)
    # sampler = Sampler(data_gen.X, data_gen.features, pi_alpha = 10.)
    # 
    # t = time.time()
    # (log_x, log_z, log_joint) = sampler.run(20, verbose=True)
    # 
    # print time.time()-t
    # 
    # sampler.print_z_comparison(data_gen.Z)
    # 
    
    ### Gathering some stats
    # Effet of many features present
    # nb_features = np.arange(1, 9)
    #     nb_repetitions = 20
    # 
    #     score = np.zeros((nb_features.size, nb_repetitions))
    # 
    #     for (i, feat) in enumerate(nb_features):
    #         print "%d features..." % feat
    #         for repet in np.arange(nb_repetitions):
    #             print "%.0f %%" % (100*(repet+1.)/nb_repetitions)
    #             data_gen = DataGenerator(N, M, sigma, max_nb_features = feat)
    #             sampler = Sampler(data_gen.X, data_gen.features, pi_alpha = 2.)
    #             sampler.run(50, verbose=False)
    #             score[i,repet] = sampler.compute_correctness_full(data_gen.Z)
    #         
    #     
    #     print score
    #     score_mean = np.mean(score, axis=1)
    #     score_std = np.std(score, axis=1)
    #     
    #     plt.errorbar(nb_features, score_mean, yerr=score_std)
    #     
    #     plt.show()
    
    
    ### Optimise the hyperparameters
    # alpha = 2 is (marginally) better. But mostly ineffective
    # pi_alpha_space = np.arange(0.1, 20., 2.)
    #     nb_repetitions = 20
    #     score = np.zeros((pi_alpha_space.size, nb_repetitions))
    #     
    #     for (i, pi_alpha) in enumerate(pi_alpha_space):
    #         print "Pi alpha: %.1f" % pi_alpha
    #         for repet in np.arange(nb_repetitions):
    #             print "%.0f %%" % (100*(repet+1.)/nb_repetitions)
    #             data_gen = DataGenerator(N, M, sigma)
    #             sampler = Sampler(data_gen.X, data_gen.features, pi_alpha = pi_alpha)
    #             sampler.run(50, verbose=False)
    #             score[i,repet] = sampler.compute_correctness_full(data_gen.Z)
    #     
    #     
    #     print score
    #     score_mean = np.mean(score, axis=1)
    #     score_std = np.std(score, axis=1)
    #     
    #     plt.errorbar(pi_alpha_space, score_mean, yerr=score_std)
    #     
    #     plt.show()
    
    ### Check if the number of features in the memory makes performance decays already
    # nb_repetitions = 20
    #     K = 9
    #     
    #     score = np.zeros((K, nb_repetitions))
    #     cnt_nb_features = np.zeros(K)
    #     
    #     for repet in np.arange(nb_repetitions):
    #         print "%.0f %%" % (100*(repet+1.)/nb_repetitions)
    #         data_gen = DataGenerator(N, M, sigma)
    #         sampler = Sampler(data_gen.X, data_gen.features, pi_alpha = 2.)
    #         sampler.run(50, verbose=False)
    #         
    #         # Now get the score, but depending on how many feature each datapoint has
    #         features_datapoints = np.sum(data_gen.Z, axis=1)
    #         unique_nb_features = np.unique(features_datapoints)
    #         
    #         for nb_feature in unique_nb_features:
    #             # Now for the datapoints concerned, check the performance
    #             score[nb_feature, repet] = sampler.compute_errors(sampler.Z[features_datapoints == nb_feature], data_gen.Z[features_datapoints == nb_feature])
    #             cnt_nb_features[nb_feature] += np.sum(features_datapoints == nb_feature)
    #         
    #     
    #     print score
    #     score_mean = np.mean(score, axis=1)
    #     score_std = np.std(score, axis=1)
    #     
    #     plt.errorbar(np.arange(K), score_mean, yerr=score_std)
    #     
    #     plt.show()
