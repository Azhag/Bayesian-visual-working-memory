#!/usr/bin/env python
# encoding: utf-8
"""
sampler.py

Created by Loic Matthey on 2011-06-1.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import numpy as np
import scipy.special as scsp
from scipy.stats import vonmises as vm
import scipy.optimize as spopt
import scipy.interpolate as spint
import scipy.io as sio
import time
import sys
import os.path
import argparse
import matplotlib.patches as plt_patches
import matplotlib.collections as plt_collections

from datagenerator import *
from randomnetwork import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *


def loglike_theta_fct_vect(thetas, (datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, covariance_fixed_contrib)):
    '''
        Compute the loglikelihood of: theta_r | n_tc theta_r' tc
    '''
    
    like_mean = datapoint - mean_fixed_contrib - \
                np.dot(ATtcB, rn.get_network_response(thetas))
            
    # Using inverse covariance as param
    return theta_kappa*np.cos(thetas[sampled_feature_index] - theta_mu) - 0.5*np.dot(like_mean, np.dot(covariance_fixed_contrib, like_mean))
    # return -0.5*np.dot(like_mean, np.dot(covariance_fixed_contrib, like_mean))


def loglike_theta_fct_single(new_theta, (thetas, datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, covariance_fixed_contrib)):
    '''
        Compute the loglikelihood of: theta_r | n_tc theta_r' tc
    '''
    # Put the new proposed point correctly
    thetas[sampled_feature_index] = new_theta

    like_mean = datapoint - mean_fixed_contrib - \
                np.dot(ATtcB, rn.get_network_response(thetas))
    
    # Using inverse covariance as param
    return theta_kappa*np.cos(thetas[sampled_feature_index] - theta_mu) - 0.5*np.dot(like_mean, np.dot(covariance_fixed_contrib, like_mean))
    # return -0.5*np.dot(like_mean, like_mean)


def loglike_theta_fct_single_min(x, thetas, datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, covariance_fixed_contrib):
    return -loglike_theta_fct_single(x, (thetas, datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, covariance_fixed_contrib))


class Sampler:
    '''
        Continuous angles Theta, with Von Mise prior. 
        x | Theta ~ Normal. Using the population codes directly
        y_t | x_t, y_{t-1} ~ Normal
        
    '''
    def __init__(self, data_gen, tc=None, theta_kappa=0.1, n_parameters = dict()):
        '''
            Initialise the sampler
            
            n_parameters:         {means: T x M, covariances: T x M x M}
        '''
        
        # Get the data
        self.data_gen = data_gen
        self.random_network = data_gen.random_network
        self.YT = data_gen.Y
        
        # Get sizes
        (self.N, self.M) = self.YT.shape
        self.T = data_gen.T
        self.R = data_gen.random_network.R
        
        # Time weights
        self.time_weights = data_gen.time_weights
        
        # Initialise t_c
        self.init_tc(tc=tc)
        
        # Initialise latent angles
        self.init_theta(theta_kappa=theta_kappa)
        
        # Initialise n_T
        self.init_n(n_parameters)
        
        # Initialise a Slice Sampler for theta
        self.slicesampler = SliceSampler()

        # Precompute the parameters and cache them
        self.init_cache_parameters()
        
    
    
    def init_tc(self, tc=None):
        '''
            Initialise the time of recall
            
            tc = N x 1
            
            Could be sampled later, for now just fix it.
        '''
        
        if tc is None:
            # Start with first one.
            tc = np.zeros(self.N, dtype='int')
            # tc = np.random.randint(self.T)
        elif np.isscalar(tc):
            tc = tc*np.ones(self.N, dtype='int')
        
        self.tc = tc
        
    
    
    def init_theta(self, theta_gamma=0.0, theta_kappa = 2.0):
        '''
            Sample initial angles. Use a Von Mises prior, low concentration (~flat)
            
            Theta:          N x R
        '''
        
        self.theta_gamma = theta_gamma
        self.theta_kappa = theta_kappa
        self.theta = np.random.vonmises(theta_gamma, theta_kappa, size=(self.N, self.R))
        
        # Assign the cued ones now
        #   stimuli_correct: N x T x R
        #   cued_features:      N x (recall_feature, recall_time)
        self.theta[np.arange(self.N), self.data_gen.cued_features[:, 0]] = self.data_gen.stimuli_correct[np.arange(self.N), self.data_gen.cued_features[:, 1], self.data_gen.cued_features[:, 0]]
        
        # Construct the list of uncued features, which should be sampled
        self.theta_to_sample = np.array([[r for r in np.arange(self.R) if r != self.data_gen.cued_features[n, 0]] for n in np.arange(self.N)], dtype='int')
        
        if self.R == 2:
            # Just for convenience (in compute_angle_error), flatten the theta_to_sample
            self.theta_to_sample = self.theta_to_sample.flatten()

        
        
    
    
    def init_n(self, n_parameters):
        '''
            Initialise the background noise n_T. It actually is observed, so nothing really interesting there.
            
            N:                    N x M
            
            n_parameters:         {means: T x M, covariances: T x M x M}
        '''
        
        # Store the parameters and precompute the Cholesky decompositions
        self.n_means_start = n_parameters['means'][0]
        self.n_means_end = n_parameters['means'][1]
        self.n_covariances_start = n_parameters['covariances'][0]
        self.n_covariances_end = n_parameters['covariances'][1]
        self.n_covariances_start_chol = np.zeros_like(self.n_covariances_start)
        self.n_covariances_end_chol = np.zeros_like(self.n_covariances_end)
        self.n_means_measured = n_parameters['means'][2]
        self.n_covariances_measured = n_parameters['covariances'][2]
        
        # for t in np.arange(self.T):
        #             try:
        #                 self.n_covariances_start_chol[t] = np.linalg.cholesky(self.n_covariances_start[t])
        #             except np.linalg.linalg.LinAlgError:
        #                 # Not positive definite, most likely only zeros, don't care, leave the zeros.
        #                 pass
        #             
        #             try:
        #                 self.n_covariances_end_chol[t] = np.linalg.cholesky(self.n_covariances_end[t])
        #             except np.linalg.linalg.LinAlgError:
        #                 # Not positive definite, most likely only zeros, don't care, leave the zeros.
        #                 pass
        #             
        
        # Initialise N
        self.NT = np.zeros((self.N, self.M))
        self.NT = self.YT
        
    
    def init_cache_parameters(self, amplify_diag=1.0):
        '''
            Most of our multiplicative factors are fixed, so precompute them, for all tc.

            Computes:
                - ATtcB
                - mean_fixed_contrib
                - covariance_fixed_contrib
        '''

        
        self.ATtcB = np.zeros(self.T)
        self.mean_fixed_contrib = np.zeros((self.T, self.M))
        self.covariance_fixed_contrib = np.zeros((self.M, self.M))

        for t in np.arange(self.T):
            (self.ATtcB[t], self.mean_fixed_contrib[t], self.covariance_fixed_contrib) = self.precompute_parameters(t, amplify_diag=amplify_diag)
        
    
    
    def precompute_parameters(self, t, amplify_diag=1.0):
        '''
            Precompute some matrices to speed up the sampling.
        '''
        # Precompute the mean and covariance contributions.
        ATmtc = np.power(self.time_weights[0, t], self.T - t - 1.)
        mean_fixed_contrib = self.n_means_end[t] + np.dot(ATmtc, self.n_means_start[t])
        ATtcB = np.dot(ATmtc, self.time_weights[1, t])
        # covariance_fixed_contrib = self.n_covariances_end[t] + np.dot(ATmtc, np.dot(self.n_covariances_start[t], ATmtc))   # + np.dot(ATtcB, np.dot(self.random_network.get_network_covariance_combined(), ATtcB.T))
        covariance_fixed_contrib = self.n_covariances_measured[-1]
        
        # Weird, this solves it. Measured covariances are wrong for generation...
        covariance_fixed_contrib[np.arange(self.M), np.arange(self.M)] *= amplify_diag
        
        # Precompute the inverse, should speedup quite nicely
        covariance_fixed_contrib = np.linalg.inv(covariance_fixed_contrib)
        # covariance_fixed_contrib = np.eye(self.M)

        return (ATtcB, mean_fixed_contrib, covariance_fixed_contrib)
      

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
        
        return np.where(np.random.rand() < cum_prob)[0][0]  # Slightly faster than np.find
    
    
    
    #######
    
    def sample_all(self):
        '''
            Do one full sweep of sampling
        '''
        
        # t = time.time()
        self.sample_theta()
        # print "Sample_z time: %.3f" % (time.time()-t)
        # self.sample_tc()
        
    
    
    def sample_theta(self, num_samples=500, return_samples=False, burn_samples=100, integrate_tc_out = True, selection_method='median', selection_num_samples=10, debug=True):
        '''
            Sample the thetas
            Need to use a slice sampler, as we do not know the normalization constant.
            
            ASSUMES A_t = A for all t. Same for B.
        '''
        
        # Iterate over whole datapoints
        # permuted_datapoints = np.random.permutation(np.arange(self.N))
        permuted_datapoints = np.arange(self.N)
        # permuted_datapoints = np.arange(1)
        
        errors = np.zeros(permuted_datapoints.shape, dtype=float)

        if return_samples:
            all_samples = np.zeros((self.N, num_samples))

        curr_theta = np.zeros(self.R)
        
        # Do everything in log-domain, to avoid numerical errors
        for n in permuted_datapoints:
            curr_theta = self.theta[n].copy()
            
            # Sample all the non-cued features
            permuted_features = np.random.permutation(self.theta_to_sample[n, np.newaxis])
            
            for sampled_feature_index in permuted_features:
                if debug:
                    print "%d, %d" % (n, sampled_feature_index)
                
                # Get samples from the current distribution
                if integrate_tc_out:
                    samples = self.get_samples_theta_tc_integratedout(n, num_samples=num_samples, sampled_feature_index=sampled_feature_index, burn_samples=burn_samples)
                else:
                    (samples, _) = self.get_samples_theta_current_tc(n, num_samples=num_samples, sampled_feature_index=sampled_feature_index, burn_samples=burn_samples)
                

                # Keep all samples if desired
                if return_samples:
                    all_samples[n] = samples
                
                # Select the new orientation
                if selection_method == 'median':
                    sampled_orientation = np.median(samples[-selection_num_samples:], overwrite_input=True)
                elif selection_method == 'last':
                    sampled_orientation = samples[-1]
                else:
                    raise ValueError('wrong value for selection_method')
                
                # Save the orientation
                self.theta[n, sampled_feature_index] = wrap_angles(sampled_orientation)
                
                # # Plot some stuff
                # if False:
                #     x = np.linspace(-np.pi, np.pi, 1000)
                #     plt.figure()
                #     ll_x = np.array([(loglike_theta_fct_single(a, params)) for a in x])
                #     if should_exponentiate:
                #         ll_x = np.exp(ll_x)
                #     ll_x -= np.mean(ll_x)
                #     ll_x /= np.abs(np.max(ll_x))
                #     plt.plot(x, ll_x)
                #     plt.axvline(x=self.data_gen.stimuli_correct[n, self.data_gen.cued_features[n, 1], sampled_feature_index], color='r')
                    
                #     sample_h, left_x = np.histogram(samples, bins=x)
                #     plt.bar(x[:-1], sample_h/np.max(sample_h).astype('float'), facecolor='green', alpha=0.75, width=np.diff(x)[0])
            
        
        if return_samples:
            return all_samples
    
    def get_samples_theta_current_tc(self, n, num_samples=2000, sampled_feature_index=0, burn_samples=100):

        # Pack the parameters for the likelihood function.
        #   Here, as the loglike_function only varies one of the input, need to give the rest of the theta vector.
        params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[self.tc[n]], sampled_feature_index, self.mean_fixed_contrib[self.tc[n]], self.covariance_fixed_contrib)
                
        # Sample the new theta
        samples, llh = self.slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=burn_samples, widths=np.pi/5., loglike_fct_params=params, debug=False, step_out=True)
        # samples, llh = self.slicesampler.sample_1D_circular(1, self.theta[n, sampled_feature_index], loglike_theta_fct, burn=100, widths=np.pi/3., thinning=2, loglike_fct_params=params, debug=False, step_out=True)

        return (samples, llh)
                


    def get_samples_theta_tc_integratedout(self, n, num_samples=2000, sampled_feature_index=0, burn_samples=100):
        '''
            Sample theta, with tc integrated out.
            Use rejection sampling (or something), discarding some samples.

            Note: the actual number of samples returned is random, but around num_samples.
        '''    

        samples_integratedout = []
        for tc in np.arange(self.T):
            params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[tc], sampled_feature_index, self.mean_fixed_contrib[tc], self.covariance_fixed_contrib)
            
            # TODO> Should be starting from the previous sample here.
            samples, _ = self.slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=burn_samples, widths=np.pi/4., loglike_fct_params=params, debug=False, step_out=True)

            # Now keep only some of them, following p(tc)
            #   for now, p(tc) = 1/T
            filter_samples = np.random.random_sample(num_samples) < 1./self.T
            samples_integratedout.extend(samples[filter_samples])
        
        return np.array(samples_integratedout)

        
    def set_theta_max_likelihood(self, num_points=100, post_optimise=True, sampled_feature_index=0):
        '''
            Update theta to their Max Likelihood values.
            Should be faster than sampling.
        '''

        all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        llh = np.zeros(num_points)
        
        # Compute the array
        for n in np.arange(self.N):

            # Pack the parameters for the likelihood function
            params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[self.tc[n]], sampled_feature_index, self.mean_fixed_contrib[self.tc[n]], self.covariance_fixed_contrib)
                
            # Compute the loglikelihood for all possible first feature
            # Use this as initial value for the optimisation routine
            for i in np.arange(num_points):
                # Give the correct cued second feature
                llh[i] = loglike_theta_fct_single(all_angles[i], params)
            
            # opt_angles[n] = spopt.fminbound(loglike_theta_fct_single_min, -np.pi, np.pi, params, disp=3)
            # opt_angles[n] = spopt.brent(loglike_theta_fct_single_min, params)
            # opt_angles[n] = wrap_angles(np.array([np.mod(spopt.anneal(loglike_theta_fct_single_min, np.random.random_sample()*np.pi*2. - np.pi, args=params)[0], 2.*np.pi)]))
            
            if post_optimise:
                self.theta[n, sampled_feature_index] = spopt.fmin(loglike_theta_fct_single_min, all_angles[np.argmax(llh)], args=params, disp=False)[0]
            else:
                self.theta[n, sampled_feature_index] = all_angles[np.argmax(llh)]
    

    def set_theta_max_likelihood_tc_integratedout(self, num_points=100, post_optimise=True, sampled_feature_index=0):
        '''
            Update theta to their Max Likelihood values.
            Integrate out tc, by averaging the output of P(theta | yT, tc) for all tc.
        '''

        all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        llh = np.zeros(num_points)
        inferred_angles = np.zeros(self.T)
        
        for n in np.arange(self.N):

            for tc in np.arange(self.T):

                # Pack the parameters for the likelihood function
                params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[tc], sampled_feature_index, self.mean_fixed_contrib[tc], self.covariance_fixed_contrib)
                    
                # Compute the loglikelihood for all possible first feature
                # Use this as initial value for the optimisation routine
                for i in np.arange(num_points):
                    # Give the correct cued second feature
                    llh[i] = loglike_theta_fct_single(all_angles[i], params)
                
                # opt_angles[n] = spopt.fminbound(loglike_theta_fct_single_min, -np.pi, np.pi, params, disp=3)
                # opt_angles[n] = spopt.brent(loglike_theta_fct_single_min, params)
                # opt_angles[n] = wrap_angles(np.array([np.mod(spopt.anneal(loglike_theta_fct_single_min, np.random.random_sample()*np.pi*2. - np.pi, args=params)[0], 2.*np.pi)]))
                
                if post_optimise:
                    # Use a simple simplex method to converge closer to the solution and avoid aliasing effects.
                    inferred_angles[tc] = spopt.fmin(loglike_theta_fct_single_min, all_angles[np.argmax(llh)], args=params, disp=False)[0]
                else:
                    inferred_angles[tc] = all_angles[np.argmax(llh)]
        
        # This approximates the sum of gaussian by a gaussian. As they have the same mixture probabilities, this is alright here...
        self.theta[n, sampled_feature_index] = mean_angles(inferred_angles)
    

    def change_cued_features(self, t_cued):
        '''
            Change the cue.
                Modify time of cue, and pull it from data_gen again
        '''

        # The time of the cued feature
        self.data_gen.cued_features[:, 1]= t_cued

        # Reset the cued theta
        self.theta[np.arange(self.N), self.data_gen.cued_features[:, 0]] = self.data_gen.stimuli_correct[np.arange(self.N), self.data_gen.cued_features[:, 1], self.data_gen.cued_features[:, 0]]
        
    
    def sample_tc(self):
        '''
            Sample a new t_c. As t_c is discrete, this merely requires a few likelihood evaluations.
            
            Do everything in log-domain, to avoid numerical errors
            
        '''
        
        # TODO CHANGE THIS
        
        # Update A^{T-tc}
        # self.ATmtc = np.power(self.time_weights[0, self.tc], self.T-self.tc)
        # # Iterate over whole matrix
        #         permuted_datapoints = np.random.permutation(np.arange(self.N))
        #         
        #         for n in permuted_datapoints:
        #             # For each datapoint, need to resample the new z_ikt's
        #             
        #             permuted_time = np.random.permutation(np.arange(self.T))
        #             
        #             for t in permuted_time:
        #                 
        #                 permuted_population = np.random.permutation(np.arange(self.R))
        #                 
        #                 for r in permuted_population:
        #                     
        #                     # Update the counts
        #                     self.Akr[self.Z[n, t, r], r] -= 1
        #                     
        #                     for k in np.arange(self.K):
        #                         # Get the prior prob of z_n_t_k
        #                         self.lprob_zntrk[k] = np.log(self.dir_alpha + self.Akr[k, r]) - np.log(self.K*self.dir_alpha + self.N - 1.)
        #                         
        #                         # Get the likelihood of ynt using z_n_t = k
        #                         self.Z[n, t, r] = k
        #                         lik_ynt = self.compute_loglikelihood_ynt(n, t)
        #                         
        #                         self.lprob_zntrk[k] += lik_ynt
        #                         
        #                         # print "%d,%d,%d,%d, lik_ynt: %.3f" % (n, t, r, k, lik_ynt)
        #                     
        #                     
        #                     # Get the new sample
        #                     new_zntr = self.sample_discrete_logp(self.lprob_zntrk)
        #                     
        #                     # Increment the counts
        #                     self.Akr[new_zntr, r] += 1
        #                     
        #                     self.Z[n, t, r] = new_zntr
        pass
                
    
    def compute_loglikelihood_ynt(self, n, t):
        '''
            Compute the log-likelihood of one datapoint under the current parameters.
        '''

        raise NotImplementedError()
        
        features_combined = self.random_network.get_network_response(self.Z[n, t])
        
        ynt_proj = self.Y[n, t] - self.time_weights[1, t]*features_combined
        if t>0:
            ynt_proj -= self.time_weights[0, t]*self.Y[n, t-1]
            
        
        l = -0.5*self.M*np.log(2.*np.pi*self.sigma2y)
        l -= 0.5/self.sigma2y*np.dot(ynt_proj, ynt_proj)
        return l
        
    
    
    def compute_joint_loglike(self):
        '''
            Compute the joint loglikelihood 
        '''
        
        raise NotImplementedError()

        return self.compute_all_loglike()[-1]
        
    
    
    def compute_all_loglike(self):
        '''
            Compute the joint loglikelihood 
        '''
        
        raise NotImplementedError()

        ly = self.compute_loglike_y()
        lz = self.compute_loglike_z()
        
        return (ly, lz, ly+lz)
    
    
    def compute_loglike_y(self):
        '''
            Compute the log likelihood of P(Y | Y, X, sigma2, P)
        '''
        raise NotImplementedError()

        features_combined = self.random_network.get_network_response(self.Z)
        
        Ytminus = np.zeros_like(self.Y)
        Ytminus[:, 1:, :] = self.Y[:, :-1, :]
        Y_proj = self.Y.transpose(0, 2, 1) - (features_combined.transpose(0, 2, 1)*self.time_weights[1]) - (Ytminus.transpose(0, 2, 1)*self.time_weights[0])
        
        l = -0.5*self.N*self.M*self.T*np.log(2.*np.pi*self.sigma2y)
        l -= 0.5/self.sigma2y*np.tensordot(Y_proj, Y_proj, axes=3)
        return l
    
    
    def compute_loglike_z(self):
        '''
            Compute the log probability of P(Z)
        '''
        raise NotImplementedError()
        
        l = self.R*scsp.gammaln(self.K*self.dir_alpha) - self.R*self.K*scsp.gammaln(self.dir_alpha)
        
        for r in np.arange(self.R):            
            for k in np.arange(self.K):
                l += scsp.gammaln(self.dir_alpha + self.Akr[k, r])
            l -= scsp.gammaln(self.K*self.dir_alpha + self.N)
        
        return l
    

    ######################

    def plot_likelihood(self, n=0, t=0, amplify_diag = 1.0, should_sample=False, num_samples=2000, return_output=False, should_exponentiate = False, num_points=1000, sampled_feature_index = 0):
        

        # Pack the parameters for the likelihood function.
        #   Here, as the loglike_function only varies one of the input, need to give the rest of the theta vector.
        params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[t], sampled_feature_index, self.mean_fixed_contrib[t], self.covariance_fixed_contrib)
        
        x = np.linspace(-np.pi, np.pi, num_points)
        plt.figure()
        ll_x = np.array([(loglike_theta_fct_single(a, params)) for a in x])
        if should_exponentiate:
            ll_x = np.exp(ll_x)
        
        # ll_x -= np.mean(ll_x)
        # ll_x /= np.abs(np.max(ll_x))
        plt.plot(x, ll_x)
        plt.axvline(x=self.data_gen.stimuli_correct[n, self.data_gen.cued_features[n, 1], 0], color='r')
        
        if should_sample:
            samples, _ = self.slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=500, widths=np.pi/4., loglike_fct_params=params, debug=False, step_out=True)
            x_edges = x - np.pi/num_points  # np.histogram wants the left-right boundaries...
            x_edges = np.r_[x_edges, -x_edges[0]]  # the rightmost boundary is the mirror of the leftmost one
            sample_h, left_x = np.histogram(samples, bins=x_edges)
            plt.bar(x_edges[:-1], sample_h/np.max(sample_h).astype('float'), facecolor='green', alpha=0.75, width=np.pi/num_points)
        
        if return_output:
            if should_sample:
                return (ll_x, x, samples)
            else:
                return (ll_x, x)
    
    
    def plot_likelihood_variation_twoangles(self, num_points=100, amplify_diag=1.0, should_plot=True, should_return=False, should_exponentiate = False, n=0, t=0, sampled_feature_index = 0):
        '''
            Compute the likelihood, varying two angles around.
            Plot the result
        '''

        # Pack the parameters for the likelihood function
        params = (self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[t], sampled_feature_index, self.mean_fixed_contrib[t], self.covariance_fixed_contrib)
        

        all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        llh_2angles = np.zeros((num_points, num_points))
        
        # Compute the array
        for i in np.arange(num_points):
            print "%d%%" % (i/float(num_points)*100)
            for j in np.arange(num_points):
                llh_2angles[i, j] = loglike_theta_fct_vect(np.array([all_angles[i], all_angles[j]]), params)
        
        if should_exponentiate:
            llh_2angles = np.exp(llh_2angles)
        
        if should_plot:
            # Plot the obtained landscape
            f = plt.figure()
            ax = f.add_subplot(111)
            im= ax.imshow(llh_2angles.T, origin='lower')
            im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
            im.set_interpolation('nearest')
            f.colorbar(im)
            ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
            ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=15)
            ax.set_yticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
            ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=15)
            
            # Callback function when moving mouse around figure.
            def report_pixel(x, y): 
                # Extract loglik at that position
                x_i = (np.abs(all_angles-x)).argmin()
                y_i = (np.abs(all_angles-y)).argmin()
                v = llh_2angles[x_i, y_i] 
                return "x=%f y=%f value=%f" % (x, y, v) 
            
            ax.format_coord = report_pixel 
            
            # Indicate the correct solutions
            correct_angles = self.data_gen.stimuli_correct[n]
            
            colmap = plt.get_cmap('gist_rainbow')
            color_gen = [colmap(1.*(i)/self.T) for i in range(self.T)]  # use 22 colors

            for t in np.arange(self.T):
                w = plt_patches.Wedge((correct_angles[t, 0], correct_angles[t, 1]), 0.25, 0, 360, 0.03, color=color_gen[t], alpha=0.7)
                ax.add_patch(w)

            # plt.annotate('O', (correct_angles[1, 0], correct_angles[1, 1]), color='blue', fontweight='bold', fontsize=30, horizontalalignment='center', verticalalignment='center')
        
        
        if should_return:
            return llh_2angles
    
    
    def plot_likelihood_correctlycuedtimes(self, n=0, amplify_diag=1.0, num_points=500, should_plot=True, should_return=False, should_exponentiate = False, sampled_feature_index = 0):
        '''
            Plot the log-likelihood function, over the space of the sampled theta, keeping the other thetas fixed to their correct cued value.
        '''

        all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        llh_2angles = np.zeros((self.T, num_points))
        
        # Compute the array
        for t in np.arange(self.T):

            # Pack the parameters for the likelihood function
            params = (self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[t], sampled_feature_index, self.mean_fixed_contrib[t], self.covariance_fixed_contrib)
            
            # Compute the loglikelihood for all possible first feature
            for i in np.arange(num_points):
                # Give the correct cued second feature
                llh_2angles[t, i] = loglike_theta_fct_vect(np.array([all_angles[i], self.data_gen.stimuli_correct[n, t, 1]]), params)
        
        llh_2angles = llh_2angles.T
        
        # Center loglik
        llh_2angles -= np.mean(llh_2angles, axis=0)
        
        if should_exponentiate:
            # If desired, plot the likelihood, not the loglik
            llh_2angles = np.exp(llh_2angles)
        
        # Normalize loglik
        llh_2angles /= np.abs(np.max(llh_2angles, axis=0))
        
        opt_angles = np.argmax(llh_2angles, axis=0)
        
        # Move them a bit apart
        llh_2angles += 1.2*np.arange(self.T)*np.abs(np.max(llh_2angles, axis=0)-np.mean(llh_2angles, axis=0))
        
        # Plot the result
        f = plt.figure()
        ax = f.add_subplot(111)
        lines = ax.plot(all_angles, llh_2angles)
        ax.set_xlim((-np.pi, np.pi))

        legends = ['-%d' % x for x in np.arange(self.T)[::-1]]
        legends[-1] = 'Last'
        
        for t in np.arange(self.T):
            # Put the legends
            ax.legend(lines, legends, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=self.T, fancybox=True, shadow=True)

            # Put a vertical line at the true answer
            ax.axvline(x=self.data_gen.stimuli_correct[n, t, 0], color=lines[t].get_c())  # ax[t] returns the plotted line

        # Print the answers
        print "True angles: %s >> Inferred: %s" % (' | '.join(['%.3f' % x for x in self.data_gen.stimuli_correct[n, :, 0]]),  ' | '.join(['%.3f' % x for x in all_angles[opt_angles]]))

        # if sampler.T == 2:
        #     plt.legend(('First', 'Second'), loc='best')
        #     print "True angles: %.3f | %.3f >> Inferred: %.3f | %.3f" % (self.data_gen.stimuli_correct[n, 0, 0], self.data_gen.stimuli_correct[n, 1, 0], all_angles[opt_angles[0]], all_angles[opt_angles[1]])
        #     plt.axvline(x=self.data_gen.stimuli_correct[n, 1, 0], color='g')
        # elif sampler.T == 3:
        #     plt.axvline(x=self.data_gen.stimuli_correct[n, 2, 0], color='r')
        #     plt.legend(('First', 'Second', 'Third'), loc='best')
        #     print "True angles: %.3f | %.3f | %.3f >> Inferred: %.3f | %.3f | %.3f" % (self.data_gen.stimuli_correct[n, 0, 0], self.data_gen.stimuli_correct[n, 1, 0], self.data_gen.stimuli_correct[n, 2, 0], all_angles[opt_angles[0]], all_angles[opt_angles[1]], all_angles[opt_angles[2]])
        
        plt.show()


    def plot_likelihood_alltc(self, n=0, num_points=500, should_plot=True, should_sample=False, num_samples=2000, return_output=False, sampled_feature_index = 0):
        '''
            Plot the posterior for different values of tc
            Sample from all of them, reject samples to get sample from posterior with tc marginalized out.
        '''

        x = np.linspace(-np.pi, np.pi, num_points)
        ll_x = np.zeros((self.T, num_points))

        # Do everything for all tc = 1:T.
        for tc in np.arange(self.T):
            # Pack the parameters for the likelihood function.
            #   Here, as the loglike_function only varies one of the input, need to give the rest of the theta vector.
            params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[tc], sampled_feature_index, self.mean_fixed_contrib[tc], self.covariance_fixed_contrib)
            
            # Compute the loglikelihood
            i =0
            for a in x:
                ll_x[tc, i] = loglike_theta_fct_single(a, params)
                i+=1
            
        # Sample if desired.
        if should_sample:
            posterior_samples = self.get_samples_theta_tc_integratedout(n)

        if should_plot:
            plt.figure()
            
            # Center and 'normalize'
            ll_x -= np.mean(ll_x, axis=1)[:, np.newaxis]
            ll_x /= np.abs(np.max(ll_x, axis=1))[:, np.newaxis]

            # Plot
            plt.plot(x, ll_x.T)
            plt.axvline(x=self.data_gen.stimuli_correct[n, self.data_gen.cued_features[n, 1], 0], color='r')
            plt.axvline(x=mean_angles(x[np.argmax(ll_x, 1)]), color='b')

            if should_sample:
                x_edges = x - np.pi/num_points  # np.histogram wants the left-right boundaries...
                x_edges = np.r_[x_edges, -x_edges[0]]  # the rightmost boundary is the mirror of the leftmost one
                sample_h, left_x = np.histogram(posterior_samples, bins=x_edges)
                plt.bar(x_edges[:-1], sample_h/np.max(sample_h).astype('float'), facecolor='green', alpha=0.75, width=np.pi/num_points)
        
        if return_output:
            if should_sample:
                return (ll_x, x, posterior_samples)
            else:
                return (ll_x, x)
    
    
    #################
    
    def run(self, iterations=10, verbose=True):
        '''
            Run the sampler for some iterations, print some information
            
            Running time: XXms * N * iterations
        '''
        
        raise NotImplementedError()

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
    
    
    
    def compute_metric_all(self):
        '''
            Get metric statistics for the whole dataset
            
            Z:  N x T x R
        '''
        
        raise NotImplementedError()

        (angle_errors_stats, angle_errors) = self.compute_angle_error()
        
        if self.T==0:
            (angle_mean_error_nomisbind, angle_std_dev_error_nomisbind, angle_mean_vector_nomisbind, avg_error_nomisbind, misbound_datapoints) = self.compute_misbinds(angle_errors)
        
            return (angle_errors_stats, (angle_mean_error_nomisbind, angle_std_dev_error_nomisbind, angle_mean_vector_nomisbind, avg_error_nomisbind, misbound_datapoints))
        
        return [angle_errors_stats, angle_errors]
    
    
    def compute_angle_error(self, return_errors=False, return_groundtruth=False):
        '''
            Compute the mean angle error for the current assignment of Z
            output: (mean_std_)
        '''
        
        # Get the target angles
        true_angles = self.data_gen.stimuli_correct[np.arange(self.N), self.data_gen.cued_features[:, 1], self.theta_to_sample]

        # Compute the angle difference error between predicted and ground truth
        angle_errors = true_angles - self.theta[np.arange(self.N), self.theta_to_sample]
        
        # Correct for obtuse angles
        angle_errors = wrap_angles(angle_errors)
        
        # Compute the statistics. Uses the spherical formulation of standard deviation
        if return_errors:
            if return_groundtruth:
                return (self.compute_mean_std_circular_data(angle_errors), angle_errors, true_angles)
            else:
                return (self.compute_mean_std_circular_data(angle_errors), angle_errors)
        else:
            if return_groundtruth:
                return (self.compute_mean_std_circular_data(angle_errors), true_angles)
            else:
                return self.compute_mean_std_circular_data(angle_errors)
    
    
    def compute_misbinds(self, angles_errors):
        raise NotImplementedError()
    
    
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
    

    def get_precision(self, remove_chance_level=False):
        '''
            Compute the precision, inverse of the std dev of the errors.
            This is our target metric
        '''

        # Compute precision
        precision = 1./self.compute_angle_error()[1]

        if remove_chance_level:
            # Expected precision under uniform distribution
            x = np.logspace(-2, 2, 100)

            precision_uniform = np.trapz(self.N/(np.sqrt(x)*np.exp(x+self.N*np.exp(-x))), x)

            # Remove the chance level
            precision -= precision_uniform

        return precision
    

    def print_comparison_inferred_groundtruth(self, show_nontargets=True):
        '''
            Print the list of all inferred angles vs true angles, and some stats
        '''

        # Get the groundtruth and the errors
        (stats, angle_errors, groundtruth) = self.compute_angle_error(return_groundtruth=True, return_errors=True)

        print "======================================="

        if show_nontargets:

            # Get the non-target/distractor angles.
            nontargets = self.data_gen.stimuli_correct[np.arange(self.N), self.data_gen.nontargets_indices.T, self.theta_to_sample].T

            print "Target " + ''.join(["\t NT %d " % x for x in (np.arange(nontargets.shape[1])+1)]) + "\t Inferred \t Error"

            for i in np.arange(self.N):
                print "% 4.3f" % (groundtruth[i]) + ''.join(["\t\t% 4.3f" % x for x in nontargets[i]]) + "\t\t % 4.3f \t % 4.3f" % (self.theta[i, 0], angle_errors[i])
        else:
            print "Target \t Inferred \t Error"
            for i in np.arange(self.N):
                print "% 4.3f \t\t % 4.3f \t % 4.3f" % (self.theta[i, 0], groundtruth[i], angle_errors[i])           

        print "======================================="
        print "  Precision:\t %.3f" % (1./stats[1])
        print "======================================="
    

    def plot_histogram_errors(self, bins=20, in_degrees=False, norm='max'):
        '''
            Compute the errors and plot a histogram.

            Should see a Gaussian + background.
        '''

        (_, errors) = self.compute_angle_error(return_errors=True)

        histogram_angular_data(errors, bins=bins, title='Errors between response and best non-target', norm=norm, in_degrees=in_degrees)

    
    def collect_responses(self):
        '''
            Gather and return the responses, target angles and non-target angles

            return (responses, target, nontargets)
        '''
        # Current inferred responses
        responses = self.theta[np.arange(self.N), self.theta_to_sample]
        # Target angles. Funny indexing, maybe not the best place for t_r
        target    =  self.data_gen.stimuli_correct[np.arange(self.N), self.data_gen.cued_features[:, 1], self.theta_to_sample]
        # Non-target angles
        nontargets = self.data_gen.stimuli_correct[np.arange(self.N), self.data_gen.nontargets_indices.T, self.theta_to_sample].T

        return (responses, target, nontargets)
    

    def save_responses_tomatlab(self, filename='output_angles_formatlab'):
        '''
            Save the responses, target and nontarget angles, to be fitted with P. Bays' model in Matlab.
        '''

        (responses, target, nontargets) = self.collect_responses()

        # Save to .mat
        sio.savemat(filename, {'response': responses, 'target': target, 'nontargets': nontargets}, appendmat=True)
    
    
    def plot_histogram_bias_nontarget(self, bins=31, in_degrees=False):
        '''
            Get an histogram of the errors between the response and all non targets

            If biased towards 0-values, indicates misbinding errors.

            [from Ed Awh's paper]
        '''

        (responses, target, nontargets) = self.collect_responses()

        # Now check the error between the responses and nontargets.
        # Flatten everything, we want the full histogram.
        errors_nontargets = wrap_angles((responses[:, np.newaxis] - nontargets).flatten())

        # Errors between the response the best nontarget.
        errors_best_nontarget = wrap_angles((responses[:, np.newaxis] - nontargets))
        errors_best_nontarget = errors_best_nontarget[np.arange(errors_best_nontarget.shape[0]), np.argmin(np.abs(errors_best_nontarget), axis=1)]

        # Do the plots
        histogram_angular_data(errors_nontargets, bins=bins, title='Errors between response and non-targets', norm='sum')
        histogram_angular_data(errors_best_nontarget, bins=bins, title='Errors between response and best non-target', norm='sum')

        




####################################

def profile_me(args):
    print "-------- Profiling ----------"
    
    import cProfile
    import pstats
    
    cProfile.runctx('profiling_run()', globals(), locals(), filename='profile_sampler.stats')
    
    stat = pstats.Stats('profile_sampler.stats')
    stat.strip_dirs().sort_stats('cumulative').print_stats()
    
    return {}


def profiling_run():
    
    N = 100
    T = 2
    K = 25
    D = 64
    M = 128
    R = 2
    
    # random_network = RandomNetwork.create_instance_uniform(K, M, D=D, R=R, W_type='identity', W_parameters=[0.2, 0.7])
    #     data_gen = DataGenerator(N, T, random_network, type_Z='discrete', weighting_alpha=0.6, weight_prior='recency', sigma_y = 0.02)
    #     sampler = Sampler(data_gen, dirichlet_alpha=0.5/K, sigma_to_sample=True, sigma_alpha=3, sigma_beta=0.5)
    #     
    #     (log_y, log_z, log_joint) = sampler.run(10, verbose=True)
    
    N = args.N
    T = args.T
    K = args.K
    M = args.M
    R = args.R
    num_samples = args.num_samples
    weighting_alpha = args.alpha
    
    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.1
    time_weights_parameters = dict(weighting_alpha=0.9, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')
    cued_feature_time = T-1

    random_network = RandomFactorialNetwork.create_full_conjunctive(M, R=R)
    # random_network = RandomFactorialNetwork.create_full_features(M, R=R, sigma=sigma_x)
    # random_network = RandomFactorialNetwork.create_mixed(M, R=R, sigma=sigma_x, ratio_feature_conjunctive=0.2)
    
    # Construct the real dataset
    print "Building the database"
    data_gen = DataGeneratorRFN(N, T, random_network, sigma_y = sigma_y, time_weights_parameters = time_weights_parameters, cued_feature_time=cued_feature_time, sigma_x = sigma_x)
    
    # Measure the noise structure
    print "Measuring noise structure"
    data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y = sigma_y, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
    stat_meas = StatisticsMeasurer(data_gen_noise)
    # stat_meas = StatisticsMeasurer(data_gen)
    
    print "Sampling..."
    sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters, tc=cued_feature_time)

    sampler.plot_likelihood(sample=True, num_samples=1000)
    
    


####################################

def do_simple_run(args):
    
    print "Simple run"
    
    N = args.N
    T = args.T
    K = args.K
    M = args.M
    R = args.R
    weighting_alpha = args.alpha
    code_type = args.code_type
    rc_scale = args.rc_scale
    rc_scale2 = args.rc_scale2
    ratio_conj = args.ratio_conj
    sigma_x = args.sigmax
    sigma_y = args.sigmay


    # Build the random network
    # sigma_y = 0.02
    # sigma_y = 0.2
    # sigma_x = 0.1
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')
    cued_feature_time = T-1

    # 'conj', 'feat', 'mixed'
    if code_type == 'conj':
        random_network = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
    elif code_type == 'feat':
        random_network = RandomFactorialNetwork.create_full_features(M, R=R, scale=rc_scale, ratio=40.)
    elif code_type == 'mixed':
        conj_params = dict(scale_moments=(rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
        feat_params = dict(scale=rc_scale2, ratio=40.)

        random_network = RandomFactorialNetwork.create_mixed(M, R=R, ratio_feature_conjunctive=ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
    elif code_type == 'wavelet':
        random_network = RandomFactorialNetwork.create_wavelet(M, R=R, scales_number=5)
    else:
        raise ValueError('Code_type is wrong!')
    
    # Construct the real dataset
    print "Building the database"
    data_gen = DataGeneratorRFN(N, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters = time_weights_parameters, cued_feature_time=cued_feature_time)
    
    # Measure the noise structure
    print "Measuring noise structure"
    data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
    stat_meas = StatisticsMeasurer(data_gen_noise)
    # stat_meas = StatisticsMeasurer(data_gen)
    
    print "Sampling..."
    sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters, tc=cued_feature_time)
    
    print "Inferring optimal angles, for t=%d" % sampler.tc[0]
    # sampler.set_theta_max_likelihood(num_points=500, post_optimise=True)
    sampler.change_cued_features(sampler.T-1)
    sampler.sample_theta(num_samples=args.num_samples, burn_samples=20, selection_method='median', selection_num_samples=args.selection_num_samples, integrate_tc_out=False, debug=False)
    
    sampler.print_comparison_inferred_groundtruth()
    
    if False:
        t = time.time()
        
        (log_y, log_z, log_joint) = sampler.run(num_samples, verbose=True)
        
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
        
        mean_last_precision = np.mean(precisions[:, -1])
        avg_precision = np.mean(precisions)
        
        plt.show()
    
    return locals()
    


def do_neuron_number_precision(args):
    '''
        Check the effect of the number of neurons on the coding precision.
    '''
    N = args.N
    num_samples = args.num_samples
    weighting_alpha = args.alpha
    num_repetitions = args.num_repetitions
    T = args.T
    code_type = args.code_type
    output_dir = os.path.join(args.output_directory, args.label)
    
    R = 2
    param1_space = np.array([10, 20, 50, 100, 150, 200, 300, 500, 1000, 1500])
    # param1_space = np.array([10, 50, 100, 300])
    # param1_space = np.array([300])
    # num_repetitions = 5
    # num_repetitions = 3

    # After searching the big N/scale space, get some relation for scale = f(N)
    fitted_scale_space = np.array([4.0, 4.0, 1.5, 1.0, 0.8, 0.55, 0.45, 0.35, 0.2, 0.2])
    # fitted_scale_space = np.array([0.4])
    
    output_string = unique_filename(prefix=strcat(output_dir, 'neuron_number_precision'))

    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.2
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')  # alpha 0.8
    cued_feature_time = T-1

    print "Doing do_neuron_number_precision"
    print "param1_space: %s" % param1_space
    print "File: %s" % output_string

    all_precisions = np.zeros((param1_space.size, num_repetitions))
    for param1_i in np.arange(param1_space.size):
        for repet_i in np.arange(num_repetitions):
            #### Get multiple examples of precisions, for different number of neurons. #####
            print "Doing M=%d, %d/%d" % (param1_space[param1_i], repet_i+1, num_repetitions)

            if code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(param1_space[param1_i], R=R, scale_moments=(fitted_scale_space[param1_i], 0.001), ratio_moments=(1.0, 0.2))
            elif code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(param1_space[param1_i], R=R, scale=fitted_scale_space[param1_i])
            elif code_type == 'mixed':
                random_network = RandomFactorialNetwork.create_mixed(param1_space[param1_i], R=R, ratio_feature_conjunctive=0.2)

            # random_network = RandomFactorialNetwork.create_full_conjunctive(param1_space[param1_i], R=R, scale_moments=(0.4, 0.02), ratio_moments=(1.0, 0.05))
            # random_network = RandomFactorialNetwork.create_full_conjunctive(param1_space[param1_i], R=R, scale_moments=(fitted_scale_space[param1_i], 0.001), ratio_moments=(1.0, 0.2))
            # random_network = RandomFactorialNetwork.create_full_features(param1_space[param1_i], R=R, scale=0.3)
            # random_network = RandomFactorialNetwork.create_mixed(param1_space[param1_i], R=R, ratio_feature_conjunctive=0.2)
            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(N, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters = time_weights_parameters, cued_feature_time=cued_feature_time)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters, tc=cued_feature_time)
            
            # Cheat here, just use the ML value for the theta
            sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
            
            # Save the precision
            # all_precisions[param1_i, repet_i] = sampler.get_precision()
            all_precisions[param1_i, repet_i] = sampler.compute_angle_error()[1]

            print "-> %.5f" % all_precisions[param1_i, repet_i]
        
        # Save to disk, unique filename
        np.save(output_string, {'all_precisions': all_precisions, 'param1_space': param1_space, 'args': args, 'num_repetitions': num_repetitions, 'fitted_scale_space': fitted_scale_space, 'output_string': output_string})
            
    # Plot
    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(param1_space, np.mean(1./all_precisions, 1), np.std(1./all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Number of neurons in population')
    ax.set_ylabel('Precision [rad]')
    
    print all_precisions

    print "Done: %s" % output_string

    return locals()


def plot_neuron_number_precision(args):
    '''
        Plot from results of a do_neuron_number_precision
    '''
    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from do_neuron_number_precision"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    if 'M_space' in loaded_data:
        param1_space = loaded_data['M_space']
    elif 'param1_space' in loaded_data:
        param1_space = loaded_data['param1_space']

    
    # Do the plot(s)
    f = plt.figure()
    ax = f.add_subplot(111)
    if np.any(np.std(all_precisions, 1) == 0.0):
        plot_mean_std_area(param1_space, np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
        ax.set_ylabel('Std dev [rad]')
    else:
        plot_mean_std_area(param1_space, np.mean(1./all_precisions, 1), np.std(1./all_precisions, 1), ax_handle=ax)
        ax.set_ylabel('Precision [rad]^-1')
    ax.set_xlabel('Number of neurons in population')
    

    return locals()

    # For multiple plot, run this, then save all_precisions, and redo same for other parameters.
    #
    # all_precisions1 = all_vars['all_precisions']
    # all_precisions2 = all_vars['all_precisions']
    #
    # ax = semilogy_mean_std_area(param1_space1, np.mean(all_precisions1, 1), np.std(all_precisions1, 1))
    # ax = semilogy_mean_std_area(param1_space1, np.mean(all_precisions2, 1), np.std(all_precisions2, 1), ax_handle=ax)
    # legend(['Features', 'Conjunctive'])


def plot_multiple_neuron_number_precision(args):
    input_filenames = ['Data/Used/feat_new_neuron_number_precision-5067b28c-0fd1-4586-a1be-1a5ab0a820f4.npy', 'Data/Used/conj_good_neuron_number_precision-4da143e7-bbd4-432d-8603-195348dd7afa.npy']

    f = plt.figure()
    ax = f.add_subplot(111)

    for file_n in input_filenames:
        loaded_data = np.load(file_n).item()
        loaded_precision = loaded_data['all_precisions']
        if 'M_space' in loaded_data:
            param1_space = loaded_data['M_space']
        elif 'param1_space' in loaded_data:
            param1_space = loaded_data['param1_space']

        ax = semilogy_mean_std_area(param1_space, np.mean(loaded_precision, 1), np.std(loaded_precision, 1), ax_handle=ax)
    

def do_size_receptive_field(args):
    '''
        Check the effect of the size of the receptive fields of neurons on the coding precision.
    '''

    N = args.N
    M = args.M
    T = args.T
    num_samples = args.num_samples
    weighting_alpha = args.alpha
    num_repetitions = args.num_repetitions
    code_type = args.code_type
    rc_scale = args.rc_scale
    
    output_dir = os.path.join(args.output_directory, args.label)
    
    R = 2

    # param1_space = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 1.5])
    param1_space = np.logspace(np.log10(0.05), np.log10(1.5), 12)

    output_string = unique_filename(prefix=strcat(output_dir, 'size_receptive_field'))

    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.2
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')  # alpha 0.5
    cued_feature_time = T-1

    print "Doing do_size_receptive_field"
    print "Scale_space: %s" % param1_space
    print "File: %s" % output_string

    all_precisions = np.zeros((param1_space.size, num_repetitions))
    for param1_i in np.arange(param1_space.size):
        for repet_i in np.arange(num_repetitions):
            #### Get multiple examples of precisions, for different number of neurons. #####
            print "Doing Scale=%.2f, %d/%d" % (param1_space[param1_i], repet_i+1, num_repetitions)

            random_network = RandomFactorialNetwork.create_full_conjunctive(M, R=R,  scale_moments=(param1_space[param1_i], 0.001), ratio_moments=(1.0, 0.1))
            # random_network = RandomFactorialNetwork.create_full_features(M_space[param1_i], R=R)
            # random_network = RandomFactorialNetwork.create_mixed(M_space[param1_i], R=R, ratio_feature_conjunctive=0.2)
            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(N, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters = time_weights_parameters, cued_feature_time=cued_feature_time)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters, tc=cued_feature_time)
            
            # Cheat here, just use the ML value for the theta
            sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
            
            # Save the precision
            # all_precisions[param1_i, repet_i] = sampler.get_precision()
            all_precisions[param1_i, repet_i] = sampler.compute_angle_error()[1]

            print "-> %.5f" % all_precisions[param1_i, repet_i]
        
        # Save to disk, unique filename
        np.save(output_string, {'all_precisions': all_precisions, 'param1_space': param1_space, 'args': args, 'num_repetitions': num_repetitions, 'output_string': output_string})

    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(param1_space, np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Size of receptive fields')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string

    return locals()


def plot_size_receptive_field(args):
    '''
        Plot from results of a size_receptive_field
    '''

    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from size_receptive_field"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    param1_space = loaded_data['param1_space']

    # Do the plot(s)
    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(param1_space, np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Number of neurons in population')
    ax.set_ylabel('Precision [rad]^-1')

    return locals()

    # For multiple plot, run this, then save all_precisions, and redo same for other parameters.
    #
    # all_precisions1 = all_vars['all_precisions']
    # all_precisions2 = all_vars['all_precisions']
    #
    # ax = semilogy_mean_std_area(M_space1, np.mean(all_precisions1, 1), np.std(all_precisions1, 1))
    # ax = semilogy_mean_std_area(M_space1, np.mean(all_precisions2, 1), np.std(all_precisions2, 1), ax_handle=ax)
    # legend(['Features', 'Conjunctive'])


def do_size_receptive_field_number_neurons(args):
    '''
        Check the effect of the size of the receptive fields of neurons on the coding precision
        Also change the number of neurons.
    '''


    N = args.N
    T = args.T
    num_samples = args.num_samples
    weighting_alpha = args.alpha
    output_dir = os.path.join(args.output_directory, args.label)
    num_repetitions = args.num_repetitions
    code_type = args.code_type

    output_string = unique_filename(prefix=strcat(output_dir, 'size_receptive_field_number_neurons'))
    R = 2

    param1_space = np.array([10, 20, 50, 75, 100, 200, 300, 500, 1000])
    # param2_space = np.array([0.05, 0.1, 0.15, 0.17, 0.2, 0.25, 0.3, 0.5, 1.0, 1.5])
    # param2_space = np.array([0.05, 0.1, 0.15, 0.2, 0.5, 1.0])
    param2_space = np.logspace(np.log10(0.05), np.log10(4.0), 12)

    # num_repetitions = 10

    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.01
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')
    cued_feature_time = T-1

    print "Doing do_size_receptive_field_number_neurons"
    print "M_space: %s" % param1_space
    print "Scale_space: %s" % param2_space
    print "File: %s" % output_string

    all_precisions = np.zeros((param1_space.size, param2_space.size, num_repetitions))
    for param1_i in np.arange(param1_space.size):
        for param2_i in np.arange(param2_space.size):
            for repet_i in np.arange(num_repetitions):
                #### Get multiple examples of precisions, for different number of neurons. #####
                print "Doing M=%d, Scale=%.2f, %d/%d" % (param1_space[param1_i], param2_space[param2_i], repet_i+1, num_repetitions)

                if code_type == 'conj':
                    random_network = RandomFactorialNetwork.create_full_conjunctive(param1_space[param1_i], R=R, scale_moments=(param2_space[param2_i], 0.001), ratio_moments=(1.0, 0.2))
                elif code_type == 'feat':
                    random_network = RandomFactorialNetwork.create_full_features(param1_space[param1_i], R=R, scale=param2_space[param2_i])
                elif code_type == 'mixed':
                    random_network = RandomFactorialNetwork.create_mixed(param1_space[param1_i], R=R, ratio_feature_conjunctive=0.2)
                
                # Construct the real dataset
                data_gen = DataGeneratorRFN(N, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters = time_weights_parameters, cued_feature_time=cued_feature_time)
                
                # Measure the noise structure
                data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
                stat_meas = StatisticsMeasurer(data_gen_noise)
                # stat_meas = StatisticsMeasurer(data_gen)
                
                sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters, tc=cued_feature_time)
                
                # Cheat here, just use the ML value for the theta
                sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
                
                # Save the precision
                # all_precisions[param1_i, repet_i] = sampler.get_precision()
                all_precisions[param1_i, param2_i, repet_i] = sampler.compute_angle_error()[1]

                print "-> %.5f" % all_precisions[param1_i, param2_i, repet_i]

            # Save to disk, unique filename
            np.save(output_string, {'all_precisions': all_precisions, 'param1_space': param1_space, 'param2_space': param2_space, 'args': args, 'num_repetitions': num_repetitions, 'output_string': output_string})

            
    
    
    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(np.mean(1./all_precisions, 2).T, origin='lower', aspect='auto')
    # im.set_extent((param1_space.min(), param1_space.max(), param2_space.min(), param2_space.max()))
    im.set_interpolation('nearest')
    # ax.xaxis.set_major_locator(plttic.NullLocator())
    # ax.yaxis.set_major_locator(plttic.NullLocator())
    plt.xticks(np.arange(param1_space.size), param1_space)
    plt.yticks(np.arange(param2_space.size), np.around(param2_space, 2))
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Scale of receptive field')
    f.colorbar(im)

    # f = plt.figure()
    # ax = f.add_subplot(111)
    # plot_mean_std_area(param1_space, np.mean(1./all_precisions, 1), np.std(1./all_precisions, 1), ax_handle=ax)
    # ax.set_xlabel('Number of neurons in population')
    # ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string

    return locals()


def plot_size_receptive_field_number_neurons(args):

    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from do_size_receptive_field_number_neurons"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    param1_space = loaded_data['param1_space']
    param2_space = loaded_data['param2_space']

    ### Simple imshow
    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(np.mean(1./all_precisions, 2).T, origin='lower', aspect='auto')
    # im.set_extent((param1_space.min(), param1_space.max(), param2_space.min(), param2_space.max()))
    im.set_interpolation('nearest')
    # ax.xaxis.set_major_locator(plttic.NullLocator())
    # ax.yaxis.set_major_locator(plttic.NullLocator())
    plt.xticks(np.arange(param1_space.size), param1_space)
    plt.yticks(np.arange(param2_space.size), np.around(param2_space, 2))
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Scale of receptive field')
    f.colorbar(im)

    ### Fill a nicer plot, interpolating between the sampled points
    param1_space_int = np.linspace(param1_space.min(), param1_space.max(), 100)
    param2_space_int = np.linspace(param2_space.min(), param2_space.max(), 100)

    all_points = np.array(cross(param1_space, param2_space))
    all_precisions_flat = 1./np.mean(all_precisions, 2).flatten()

    all_precisions_int = spint.griddata(all_points, all_precisions_flat, (param1_space_int[None, :], param2_space_int[:, None]), method='nearest')

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    cs = ax1.contourf(param1_space_int, param2_space_int, all_precisions_int, 20)   # cmap=plt.cm.jet
    ax1.set_xlabel('Number of neurons')
    ax1.set_ylabel('Scale of receptive field')
    ax1.set_title('Precision wrt scale/number of neurons')
    ax1.scatter(all_points[:, 0], all_points[:, 1], marker='o', c='b', s=5)
    ax1.set_xlim(param1_space_int.min(), param1_space_int.max())
    ax1.set_ylim(param2_space_int.min(), param2_space_int.max())
    f1.colorbar(cs)

    ### Print the 1D plot for each N
    
    for i in np.arange(param1_space.size):
        f = plt.figure()
        ax = f.add_subplot(111)
        plot_mean_std_area(param2_space, np.mean(1./all_precisions[i], 1), np.std(1./all_precisions[i], 1), ax_handle=ax)
        ax.set_xlabel('Scale of filter')
        ax.set_ylabel('Precision [rad]^-1')
        ax.set_title('Optim scale for N=%d' % param1_space[i])
    # plot_square_grid(np.tile(param2_space, (param1_space.size, 1)), np.mean(1./all_precisions, 2))

    ### Plot the optimal scale for all N
    optimal_scale_N = param2_space[np.argmax(np.mean(1./all_precisions, 2), 1)]
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.plot(param1_space, optimal_scale_N)
    ax2.set_xlabel('Number of neurons')
    ax2.set_ylabel('Optimal scale')

    return locals()

    
def do_memory_curve(args):
    '''
        Get the memory curve
    '''

    
    output_dir = os.path.join(args.output_directory, args.label)
    output_string = unique_filename(prefix=strcat(output_dir, 'memory_curve'))

    # Build the random network
    time_weights_parameters = dict(weighting_alpha=args.alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')

    print "Doing do_memory_curve"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.ones((args.T, args.num_repetitions))*np.nan
    for repet_i in np.arange(args.num_repetitions):
        #### Get multiple examples of precisions, for different number of neurons. #####
        
        if args.code_type == 'conj':
            random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
        elif args.code_type == 'feat':
            random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=40.)
        elif args.code_type == 'mixed':
            conj_params = dict(scale_moments=(args.rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
            feat_params = dict(scale=args.rc_scale2, ratio=40.)
            random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
        else:
            raise ValueError('Code_type is wrong!')

        
        # Construct the real dataset
        data_gen = DataGeneratorRFN(args.N, args.T, random_network, sigma_y = args.sigmay, sigma_x=args.sigmax, time_weights_parameters = time_weights_parameters)
        
        # Measure the noise structure
        data_gen_noise = DataGeneratorRFN(3000, args.T, random_network, sigma_y = args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
        stat_meas = StatisticsMeasurer(data_gen_noise)
        # stat_meas = StatisticsMeasurer(data_gen)
        
        sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters)
        
        for t in np.arange(args.T):
            print "Doing T=%d,  %d/%d" % (t, repet_i+1, args.num_repetitions)

            # Change the cued feature
            sampler.change_cued_features(t)

            # Cheat here, just use the ML value for the theta
            # sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
            # sampler.set_theta_max_likelihood_tc_integratedout(num_points=200, post_optimise=True)

            # Sample the new theta
            sampler.sample_theta(num_samples=args.num_samples, burn_samples=20, selection_method='median', selection_num_samples=args.selection_num_samples, integrate_tc_out=False, debug=False)
            
            # Save the precision
            all_precisions[t, repet_i] = sampler.get_precision(remove_chance_level=True)
            # all_precisions[t, repet_i] = sampler.compute_angle_error()[1]

            print "-> %.5f" % all_precisions[t, repet_i]
    
        # Save to disk, unique filename
        np.save(output_string, {'all_precisions': all_precisions, 'args': args, 'num_repetitions': args.num_repetitions, 'T': args.T, 'output_string': output_string})

    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(np.arange(args.T), np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Object')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()


def do_multiple_memory_curve(args):
    '''
        Get the memory curves, for 1...T objects
    '''

    output_dir = os.path.join(args.output_directory, args.label)
    output_string = unique_filename(prefix=strcat(output_dir, 'multiple_memory_curve'))

    # Build the random network
    time_weights_parameters = dict(weighting_alpha=args.alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')

    print "Doing do_multiple_memory_curve"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.zeros((args.T, args.T, args.num_repetitions))

    # Construct different datasets, with t objects
    for t in np.arange(args.T):

        for repet_i in np.arange(args.num_repetitions):
            #### Get multiple examples of precisions, for different number of neurons. #####
            
            if args.code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
            elif args.code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=40.)
            elif args.code_type == 'mixed':
                conj_params = dict(scale_moments=(args.rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=args.rc_scale2, ratio=40.)
                random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
            else:
                raise ValueError('Code_type is wrong!')

            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(args.N, t+1, random_network, sigma_y = args.sigmay, sigma_x=args.sigmax, time_weights_parameters = time_weights_parameters)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, t+1, random_network, sigma_y = args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters)
            
            for tc in np.arange(t+1):
                print "Doing T=%d, Tc=%d,  %d/%d" % (t+1, tc, repet_i+1, args.num_repetitions)

                # Change the cued feature
                sampler.change_cued_features(tc)

                # Cheat here, just use the ML value for the theta
                # sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
                # sampler.set_theta_max_likelihood_tc_integratedout(num_points=200, post_optimise=True)

                # Sample thetas
                sampler.sample_theta(num_samples=args.num_samples, burn_samples=20, selection_method='median', selection_num_samples=args.num_samples, integrate_tc_out=False, debug=False)

                # Save the precision
                all_precisions[t, tc, repet_i] = sampler.get_precision(remove_chance_level=True)
                # all_precisions[t, tc, repet_i] = sampler.compute_angle_error()[1]

                print "-> %.5f" % all_precisions[t, tc, repet_i]
            
            # Save to disk, unique filename
            np.save(output_string, {'all_precisions': all_precisions, 'args': args, 'num_repetitions': args.num_repetitions, 'output_string': output_string})

    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    for t in np.arange(args.T):
        t_space_aligned_right = (args.T - np.arange(t+1))[::-1]
        plot_mean_std_area(t_space_aligned_right, np.mean(all_precisions[t], 1)[:t+1], np.std(all_precisions[t], 1)[:t+1], ax_handle=ax)
    ax.set_xlabel('Recall time')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()


def do_multiple_memory_curve_simult(args):
    '''
        Get the memory curves, for 1...T objects, simultaneous presentation
        (will force alpha=1, and only recall one object for each T, more independent)
    '''

    output_dir = os.path.join(args.output_directory, args.label)
    output_string = unique_filename(prefix=strcat(output_dir, 'multiple_memory_curve'))

    # Build the random network
    alpha = 1.
    time_weights_parameters = dict(weighting_alpha=alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')

    print "Doing do_multiple_memory_curve"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.zeros((args.T, args.num_repetitions))

    # Construct different datasets, with t objects
    for repet_i in np.arange(args.num_repetitions):

        for t in np.arange(args.T):

            #### Get multiple examples of precisions, for different number of neurons. #####
            
            if args.code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
            elif args.code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=40.)
            elif args.code_type == 'mixed':
                conj_params = dict(scale_moments=(args.rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=args.rc_scale2, ratio=40.)
                random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
            else:
                raise ValueError('Code_type is wrong!')

            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(args.N, t+1, random_network, sigma_y = args.sigmay, sigma_x=args.sigmax, time_weights_parameters = time_weights_parameters)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, t+1, random_network, sigma_y = args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters)
            
            print "  doing T=%d %d/%d" % (t+1, repet_i+1, args.num_repetitions)

            # Sample thetas
            sampler.sample_theta(num_samples=args.num_samples, burn_samples=20, selection_method='median', selection_num_samples=args.num_samples, integrate_tc_out=False, debug=False)

            # Save the precision
            all_precisions[t, repet_i] = sampler.get_precision(remove_chance_level=True)
            # all_precisions[t, repet_i] = 1./sampler.compute_angle_error()[1]

            print "-> %.5f" % all_precisions[t, repet_i]
            
            # Save to disk, unique filename
            np.save(output_string, {'all_precisions': all_precisions, 'args': args, 'num_repetitions': args.num_repetitions, 'output_string': output_string})
        
        # TODO CONTINUE WRITING HERE. Should compute the power law fit for the current number of repetitions available.
        xx = np.tile(np.arange(1, args.T+1, dtype='float'), (repet_i+1, 1)).T
        power_law_params = fit_powerlaw(xx, all_precisions[:, :(repet_i+1)], should_plot=True)

        print '====> Power law fits: exponent: %.4f, bias: %.4f' % (power_law_params[0], power_law_params[1])

    print all_precisions

    
    # Save to disk, unique filename
    np.save(output_string, {'all_precisions': all_precisions, 'args': args, 'num_repetitions': args.num_repetitions, 'power_law_params': power_law_params})

    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(np.arange(args.T), np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Number of objects')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()


def plot_multiple_memory_curve(args):

    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from do_multiple_memory_curve"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    T = loaded_data['args'].T
    
    f = plt.figure()
    ax = f.add_subplot(111)
    for t in np.arange(T):
        t_space_aligned_right = (T - np.arange(t+1))[::-1]
        # plot_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], np.std(1./all_precisions[t], 1)[:t+1], ax_handle=ax)
        # semilogy_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], np.std(1./all_precisions[t], 1)[:t+1], ax_handle=ax)
        # plt.semilogy(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], 'o-', linewidth=2, markersize=8)
        plt.plot(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], 'o-')
    x_labels = ['-%d' % x for x in np.arange(T)[::-1]]
    x_labels[-1] = 'Last'

    ax.set_xticks(t_space_aligned_right)
    ax.set_xticklabels(x_labels)
    ax.set_xlim((0.8, T+0.2))
    ax.set_ylim((1.01, 40))
    # ax.set_xlabel('Recall time')
    # ax.set_ylabel('Precision [rad]')
    legends=['%d items' % (x+1) for x in np.arange(T)]
    legends[0] = '1 item'
    plt.legend(legends, loc='best', numpoints=1, fancybox=True)

    return locals()




def plot_multiple_memory_curve_simult(args):
    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from do_multiple_memory_curve"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    T = loaded_data['args'].T
    
    # Average over repetitions, and then get mean across T
    mean_precision = np.zeros(T)
    std_precision = np.zeros(T)
    for t in np.arange(T):
        mean_precision[t] = np.mean(all_precisions[t][:t+1])
        std_precision[t] = np.std(all_precisions[t][:t+1])

    f = plt.figure()
    ax = f.add_subplot(111)
    plt.semilogy(np.arange(1, T+1), 1./mean_precision, 'o-')
    plt.xticks(np.arange(1, T+1))
    plt.xlim((0.9, T+0.1))
    ax.set_xlabel('Number of stored items')
    ax.set_ylabel('Precision [rad]')

    f = plt.figure()
    ax = f.add_subplot(111)
    plt.bar(np.arange(1, T+1), 1./mean_precision, align='center', log=False)
    ax.set_xlabel('Number of stored items')
    ax.set_ylabel('Precision [rad]')


    return locals()


def do_mixed_ratioconj(args):
    '''
        For a mixed population, check the effect of increasing the ratio of conjunctive cells
        on the performance.
    '''

    N = args.N
    M = args.M
    num_samples = args.num_samples
    weighting_alpha = args.alpha
    num_repetitions = args.num_repetitions
    T = args.T
    output_dir = os.path.join(args.output_directory, args.label)
    rc_scale = args.rc_scale
    rc_scale2 = args.rc_scale2

    R = 2
    args.R = 2
    args.code_type = 'mixed'
    param1_space = np.array([0.0, 0.1, 0.3, 0.5, 0.7])
    
    output_string = unique_filename(prefix=strcat(output_dir, 'mixed_ratioconj'))

    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.01
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')
    cued_feature_time = T-1

    print "Doing do_mixed_ratioconj"
    print "param1_space: %s" % param1_space
    print "rc_scales: %.3f %.3f" % (rc_scale, rc_scale2)
    print "File: %s" % output_string

    all_precisions = np.zeros((param1_space.size, num_repetitions))
    for param1_i in np.arange(param1_space.size):
        for repet_i in np.arange(num_repetitions):
            #### Get multiple examples of precisions, for different number of neurons. #####
            print "Doing M=%.3f, %d/%d" % (param1_space[param1_i], repet_i+1, num_repetitions)

            # Construct the network with appropriate parameters
            conj_params = dict(scale_moments=(rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
            feat_params = dict(scale=rc_scale2, ratio=40.)

            random_network = RandomFactorialNetwork.create_mixed(M, R=R, ratio_feature_conjunctive=param1_space[param1_i], conjunctive_parameters=conj_params, feature_parameters=feat_params)
            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(N, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters = time_weights_parameters, cued_feature_time=cued_feature_time)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters, tc=cued_feature_time)
            
            # Cheat here, just use the ML value for the theta
            sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
            
            # Save the precision
            # all_precisions[param1_i, repet_i] = sampler.get_precision()
            all_precisions[param1_i, repet_i] = sampler.compute_angle_error()[1]

            print "-> %.5f" % all_precisions[param1_i, repet_i]
        
        # Save to disk, unique filename
        np.save(output_string, {'all_precisions': all_precisions, 'param1_space': param1_space, 'args': args, 'num_repetitions': num_repetitions, 'output_string': output_string})
            
    # Plot
    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(param1_space, np.mean(1./all_precisions, 1), np.std(1./all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Number of neurons in population')
    ax.set_ylabel('Precision [rad]')
    
    print all_precisions

    print "Done: %s" % output_string

    return locals()


def do_mixed_two_scales(args):
    '''
        Search over the space of conjunctive scale and feature scale.
        Should be called with different values of conj_ratio (e.g. from PBS)
    '''

    N = args.N
    M = args.M
    K = args.K
    num_samples = args.num_samples
    weighting_alpha = args.alpha
    num_repetitions = args.num_repetitions
    T = args.T
    output_dir = os.path.join(args.output_directory, args.label)
    rc_scale = args.rc_scale
    rc_scale2 = args.rc_scale2
    ratio_conj = args.ratio_conj

    R = 2
    args.R = 2
    args.code_type = 'mixed'
    param1_space = np.logspace(np.log10(0.05), np.log10(4.0), num_samples)
    param2_space = np.logspace(np.log10(0.05), np.log10(4.0), num_samples)
    
    output_string = unique_filename(prefix=strcat(output_dir, 'mixed_two_scales'))

    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.01
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')
    cued_feature_time = T-1

    print "Doing do_mixed_ratioconj"
    print "param1_space: %s" % param1_space
    print "param2_space: %s" % param2_space
    print "ratio_conj: %.3f" % (ratio_conj)
    print "File: %s" % output_string

    all_precisions = np.zeros((param1_space.size, param2_space.size, num_repetitions))
    for param1_i in np.arange(param1_space.size):
        for param2_i in np.arange(param2_space.size):
            for repet_i in np.arange(num_repetitions):
                #### Get multiple examples of precisions, for different number of neurons. #####
                print "Doing scales=(%.3f, %.3f), %d/%d" % (param1_space[param1_i], param2_space[param2_i], repet_i+1, num_repetitions)

                # Construct the network with appropriate parameters
                conj_params = dict(scale_moments=(param1_space[param1_i], 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=param2_space[param2_i], ratio=40.)

                random_network = RandomFactorialNetwork.create_mixed(M, R=R, ratio_feature_conjunctive=ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
                
                # Construct the real dataset
                data_gen = DataGeneratorRFN(N, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters = time_weights_parameters, cued_feature_time=cued_feature_time)
                
                # Measure the noise structure
                data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
                stat_meas = StatisticsMeasurer(data_gen_noise)
                # stat_meas = StatisticsMeasurer(data_gen)
                
                sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters, tc=cued_feature_time)
                
                # Cheat here, just use the ML value for the theta
                sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
                
                # Save the precision
                # all_precisions[param1_i, repet_i] = sampler.get_precision()
                all_precisions[param1_i, param2_i, repet_i] = sampler.compute_angle_error()[1]

                print "-> %.5f" % all_precisions[param1_i, param2_i, repet_i]
            
            # Save to disk, unique filename
            np.save(output_string, {'all_precisions': all_precisions, 'param1_space': param1_space, 'param2_space': param2_space, 'args': args, 'num_repetitions': num_repetitions, 'output_string': output_string})
          
    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(np.mean(1./all_precisions, 2).T, origin='lower', aspect='auto')
    # im.set_extent((param1_space.min(), param1_space.max(), param2_space.min(), param2_space.max()))
    im.set_interpolation('nearest')
    # ax.xaxis.set_major_locator(plttic.NullLocator())
    # ax.yaxis.set_major_locator(plttic.NullLocator())
    plt.xticks(np.arange(param1_space.size), np.around(param1_space, 2))
    plt.yticks(np.arange(param2_space.size), np.around(param2_space, 2))
    ax.set_xlabel('Scale of conjunctive neurons')
    ax.set_ylabel('Scale of feature neurons')
    f.colorbar(im)
    
    print "Done: %s" % output_string

    return locals()


def do_save_responses_simultaneous(args):
    '''
        Simulate simultaneous presentations, with 1...T objects.
        Outputs the responses and target/non-targets, to be fitted in Matlab (TODO: convert EM fit in Python code)
    '''

    output_dir = os.path.join(args.output_directory, args.label)
    output_string = unique_filename(prefix=strcat(output_dir, 'save_responses_simultaneous'))

    # Build the random network
    time_weights_parameters = dict(weighting_alpha=args.alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')

    print "Doing do_save_responses_simultaneous"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.zeros((args.T, args.T, args.num_repetitions))
    all_responses = np.zeros((args.T, args.T, args.num_repetitions, args.N))
    all_targets = np.zeros((args.T, args.T, args.num_repetitions, args.N))
    all_nontargets = np.zeros((args.T, args.T, args.num_repetitions, args.N, args.T-1))
    
    for repet_i in np.arange(args.num_repetitions):
        
        # Construct different datasets, with t objects
        for t in np.arange(args.T):
            
            if args.code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.001))
            elif args.code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=40.)
            elif args.code_type == 'mixed':
                conj_params = dict(scale_moments=(args.rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=args.rc_scale2, ratio=40.)

                random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
            else:
                raise ValueError('Code_type is wrong!')

            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(args.N, t+1, random_network, sigma_y = args.sigmay, sigma_x = args.sigmax, time_weights_parameters = time_weights_parameters)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, t+1, random_network, sigma_y = args.sigmay, sigma_x = args.sigmax, time_weights_parameters=time_weights_parameters)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters)
            
            for tc in np.arange(t+1):
                print "Doing T=%d, Tc=%d,  %d/%d" % (t+1, tc, repet_i+1, args.num_repetitions)

                # Change the cued feature
                sampler.change_cued_features(tc)

                # Sample the new theta
                sampler.sample_theta(num_samples=args.num_samples, burn_samples=20, selection_method='median', selection_num_samples=args.selection_num_samples, integrate_tc_out=False, debug=False)

                # Save the precision
                # all_precisions[param1_i, repet_i] = sampler.get_precision()
                all_precisions[t, tc, repet_i] = sampler.compute_angle_error()[1]

                print "-> %.5f" % all_precisions[t, tc, repet_i]

                # Save the responses, targets and nontargets
                print t
                print tc
                (all_responses[t, tc, repet_i], all_targets[t, tc, repet_i], all_nontargets[t, tc, repet_i, :, :t]) = sampler.collect_responses()

        
            # Save to disk, unique filename
            np.save(output_string, {'all_precisions': all_precisions, 'all_responses': all_responses, 'all_targets': all_targets, 'all_nontargets': all_nontargets, 'args': args, 'num_repetitions': args.num_repetitions, 'T': args.T, 'output_string': output_string})
            sio.savemat(output_string, {'all_precisions': all_precisions, 'all_responses': all_responses, 'all_targets': all_targets, 'all_nontargets': all_nontargets, 'args': args, 'num_repetitions': args.num_repetitions, 'T': args.T, 'output_string': output_string})

    
    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    for t in np.arange(args.T):
        t_space_aligned_right = (args.T - np.arange(t+1))[::-1]
        semilogy_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], np.std(1./all_precisions[t], 1)[:t+1], ax_handle=ax)
    ax.set_xlabel('Recall time')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()




####################################
if __name__ == '__main__':
        
    # Switch on different actions
    actions = {x.__name__: x for x in 
            [do_simple_run, 
            profile_me,
            do_size_receptive_field,
            do_neuron_number_precision,
            do_size_receptive_field_number_neurons,
            plot_neuron_number_precision,
            plot_size_receptive_field,
            plot_size_receptive_field_number_neurons,
            do_memory_curve,
            do_multiple_memory_curve,
            do_multiple_memory_curve_simult,
            plot_multiple_memory_curve,
            plot_multiple_memory_curve_simult,
            do_mixed_ratioconj,
            do_mixed_two_scales,
            do_save_responses_simultaneous
            ]}
    
    print sys.argv[1:]
    
    parser = argparse.ArgumentParser(description='Sample a model of Visual working memory.')
    parser.add_argument('--label', help='label added to output files', default='')
    parser.add_argument('--output_directory', nargs='?', default='Data/')
    parser.add_argument('--action_to_do', choices=actions.keys(), default='do_simple_run')
    parser.add_argument('--input_filename', default='', help='Some input file, depending on context')
    parser.add_argument('--num_repetitions', type=int, default=1, help='For search actions, number of repetitions to average on')
    parser.add_argument('--N', default=200, type=int, help='Number of datapoints')
    parser.add_argument('--T', default=1, type=int, help='Number of times')
    parser.add_argument('--K', default=2, type=int, help='Number of representated features')  # Warning: Need more data for bigger matrix
    parser.add_argument('--D', default=32, type=int, help='Dimensionality of features')
    parser.add_argument('--M', default=300, type=int, help='Dimensionality of data/memory')
    parser.add_argument('--R', default=2, type=int, help='Number of population codes')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to use')
    parser.add_argument('--selection_num_samples', type=int, default=1, help='While selecting the new sample from a set of samples, consider the P last samples only. (if =1, return last sample)')
    parser.add_argument('--alpha', default=1.0, type=float, help='Weighting of the decay through time')
    parser.add_argument('--code_type', choices=['conj', 'feat', 'mixed', 'wavelet'], default='conj', help='Select the type of code used by the Network')
    parser.add_argument('--rc_scale', type=float, default=0.5, help='Scale of receptive fields')
    parser.add_argument('--rc_scale2', type=float, default=0.4, help='Scale of receptive fields, second population (e.g. feature for mixed population)')
    parser.add_argument('--sigmax', type=float, default=0.2, help='Noise per object')
    parser.add_argument('--sigmay', type=float, default=0.02, help='Noise along time')
    parser.add_argument('--ratio_conj', type=float, default=0.2, help='Ratio of conjunctive/field subpopulations for mixed network')


    args = parser.parse_args()
    
    should_save = False
    output_dir = os.path.join(args.output_directory, args.label)
    
    # Run it
    all_vars = actions[args.action_to_do](args)
    

    # Re-instantiate some variables
    #   Ugly but laziness prevails...
    variables_to_reinstantiate = ['data_gen', 'sampler', 'log_joint', 'log_z', 'stat_meas', 'random_network']
    for var_reinst in variables_to_reinstantiate:
        if var_reinst in all_vars:
            vars()[var_reinst] = all_vars[var_reinst]

    # Save the results
    if should_save:
        output_file = os.path.join(output_dir, 'all_vars.npy')
        np.save(output_file, all_vars)
    
    plt.show()
