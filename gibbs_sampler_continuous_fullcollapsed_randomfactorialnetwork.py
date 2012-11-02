#!/usr/bin/env python
# encoding: utf-8
"""
sampler.py

Created by Loic Matthey on 2011-06-1.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import numpy as np
import scipy.special as scsp
# from scipy.stats import vonmises as vm
import scipy.optimize as spopt
# import scipy.interpolate as spint
# import scipy.io as sio
import matplotlib.patches as plt_patches
# import matplotlib.collections as plt_collections
import matplotlib.pyplot as plt

# from datagenerator import *
# from randomnetwork import *
# from randomfactorialnetwork import *
# from statisticsmeasurer import *
from utils import *

from slicesampler import *
# from dataio import *
import progress


def loglike_theta_fct_single(new_theta, (thetas, datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, inv_covariance_fixed_contrib)):
    '''
        Compute the loglikelihood of: theta_r | n_tc theta_r' tc
    '''
    # Put the new proposed point correctly
    thetas[sampled_feature_index] = new_theta

    like_mean = datapoint - mean_fixed_contrib - \
                np.dot(ATtcB, rn.get_network_response(thetas))
    
    # Using inverse covariance as param
    # return theta_kappa*np.cos(thetas[sampled_feature_index] - theta_mu) - 0.5*np.dot(like_mean, np.dot(inv_covariance_fixed_contrib, like_mean))
    return -0.5*np.dot(like_mean, np.dot(inv_covariance_fixed_contrib, like_mean))
    # return -1./(2*0.2**2)*np.sum(like_mean**2.)


def loglike_theta_fct_single_min(x, thetas, datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, inv_covariance_fixed_contrib):
    return -loglike_theta_fct_single(x, (thetas, datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, inv_covariance_fixed_contrib))


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
                - inv_covariance_fixed_contrib
        '''

        
        self.ATtcB = np.zeros(self.T)
        self.mean_fixed_contrib = np.zeros((self.T, self.M))
        self.inv_covariance_fixed_contrib = np.zeros((self.M, self.M))
        self.noise_covariance = self.n_covariances_measured[-1]

        for t in np.arange(self.T):
            (self.ATtcB[t], self.mean_fixed_contrib[t], self.inv_covariance_fixed_contrib) = self.precompute_parameters(t, amplify_diag=amplify_diag)
        
    
    
    def precompute_parameters(self, t, amplify_diag=1.0):
        '''
            Precompute some matrices to speed up the sampling.
        '''
        # Precompute the mean and covariance contributions.
        ATmtc = np.power(self.time_weights[0, t], self.T - t - 1.)
        mean_fixed_contrib = self.n_means_end[t] + np.dot(ATmtc, self.n_means_start[t])
        ATtcB = np.dot(ATmtc, self.time_weights[1, t])
        # inv_covariance_fixed_contrib = self.n_covariances_end[t] + np.dot(ATmtc, np.dot(self.n_covariances_start[t], ATmtc))   # + np.dot(ATtcB, np.dot(self.random_network.get_network_covariance_combined(), ATtcB.T))
        inv_covariance_fixed_contrib = self.n_covariances_measured[-1]
        
        # Weird, this solves it. Measured covariances are wrong for generation...
        inv_covariance_fixed_contrib[np.arange(self.M), np.arange(self.M)] *= amplify_diag
        
        # Precompute the inverse, should speedup quite nicely
        inv_covariance_fixed_contrib = np.linalg.inv(inv_covariance_fixed_contrib)
        # inv_covariance_fixed_contrib = np.eye(self.M)

        return (ATtcB, mean_fixed_contrib, inv_covariance_fixed_contrib)
      

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
        
    
    
    def sample_theta(self, num_samples=500, return_samples=False, burn_samples=20, integrate_tc_out = True, selection_method='median', selection_num_samples=250, subset_theta=None, debug=False):
        '''
            Sample the thetas
            Need to use a slice sampler, as we do not know the normalization constant.
            
            ASSUMES A_t = A for all t. Same for B.
        '''
        
        if selection_num_samples > num_samples:
            # Limit selection_num_samples
            selection_num_samples = num_samples

        if subset_theta is not None:
            # Should only sample a subset of the theta
            permuted_datapoints = np.array(subset_theta)
        else:
            # Iterate over whole datapoints
            # permuted_datapoints = np.random.permutation(np.arange(self.N))
            permuted_datapoints = np.arange(self.N)
        
        # errors = np.zeros(permuted_datapoints.shape, dtype=float)

        if return_samples:
            all_samples = np.zeros((permuted_datapoints.size, num_samples))

        # curr_theta = np.zeros(self.R)
        
        # Do everything in log-domain, to avoid numerical errors
        i = 0
        for n in progress.ProgressDisplay(permuted_datapoints, display=progress.SINGLE_LINE):
            # curr_theta = self.theta[n].copy()
            
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
                    all_samples[i] = samples
                
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
            
            i+= 1

        if return_samples:
            return all_samples
    

    def get_samples_theta_current_tc(self, n, num_samples=2000, sampled_feature_index=0, burn_samples=200):

        # Pack the parameters for the likelihood function.
        #   Here, as the loglike_function only varies one of the input, need to give the rest of the theta vector.
        params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[self.tc[n]], sampled_feature_index, self.mean_fixed_contrib[self.tc[n]], self.inv_covariance_fixed_contrib)

        # Sample the new theta
        samples, llh = self.slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=burn_samples, widths=np.pi/8., loglike_fct_params=params, debug=False, step_out=True)
        # samples, llh = self.slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=burn_samples, widths=0.01, loglike_fct_params=params, debug=False, step_out=True)
        
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
            params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[tc], sampled_feature_index, self.mean_fixed_contrib[tc], self.inv_covariance_fixed_contrib)
            
            # TODO> Should be starting from the previous sample here.
            samples, _ = self.slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=burn_samples, widths=np.pi/8., loglike_fct_params=params, debug=False, step_out=True)

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
        for n in progress.ProgressDisplay(np.arange(self.N), display=progress.SINGLE_LINE):

            # Pack the parameters for the likelihood function
            params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[self.tc[n]], sampled_feature_index, self.mean_fixed_contrib[self.tc[n]], self.inv_covariance_fixed_contrib)
                
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
                params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[tc], sampled_feature_index, self.mean_fixed_contrib[tc], self.inv_covariance_fixed_contrib)
                    
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


    def compute_likelihood_fullspace(self, n=0, sampled_feature_index=0, all_angles=None, num_points=1000, normalize=False, remove_mean=True, should_exponentiate=False):
        '''
            Computes and returns the (log)likelihood evaluated for a given datapoint on the entire space (e.g. [-pi,pi]).
        '''

        if all_angles is None:
            all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        else:
            num_points = all_angles.size
        
        likelihood = np.zeros((self.T, num_points))
        
        # Compute the array
        for t in np.arange(self.T):

            # Compute the loglikelihood for all possible first feature
            for i in np.arange(num_points):

                # Pack the parameters for the likelihood function
                params = (np.array([all_angles[i], self.data_gen.stimuli_correct[n, t, 1]]), self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[t], sampled_feature_index, self.mean_fixed_contrib[t], self.inv_covariance_fixed_contrib)
            
                # Give the correct cued second feature
                likelihood[t, i] = loglike_theta_fct_single(all_angles[i], params)
                # likelihood[t, i] = loglike_theta_fct_vect(np.array([all_angles[i], self.data_gen.stimuli_correct[n, t, 1]]), params)
        
        likelihood = likelihood.T
        
        # Center loglik
        if remove_mean:
            likelihood -= np.mean(likelihood, axis=0)
        
        if should_exponentiate:
            # If desired, plot the likelihood, not the loglik
            likelihood = np.exp(likelihood)

            if normalize:
                # Normalise if required.
                dx = np.diff(all_angles)[0]
                likelihood /= np.sum(likelihood*dx)

        return likelihood
    

    ######################

    def plot_likelihood(self, n=0, t=0, amplify_diag = 1.0, should_sample=False, num_samples=2000, return_output=False, should_exponentiate = False, num_points=1000, sampled_feature_index = 0):
        

        # Pack the parameters for the likelihood function.
        #   Here, as the loglike_function only varies one of the input, need to give the rest of the theta vector.
        params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[t], sampled_feature_index, self.mean_fixed_contrib[t], self.inv_covariance_fixed_contrib)
        
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


        all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        llh_2angles = np.zeros((num_points, num_points))
        
        # Compute the array
        for i in np.arange(num_points):
            print "%d%%" % (i/float(num_points)*100)
            for j in np.arange(num_points):

                # Pack the parameters for the likelihood function
                params = (np.array([all_angles[i], all_angles[j]]), self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[t], sampled_feature_index, self.mean_fixed_contrib[t], self.inv_covariance_fixed_contrib)
        
                # llh_2angles[i, j] = loglike_theta_fct_vect(np.array([all_angles[i], all_angles[j]]), params)
                llh_2angles[i, j] = loglike_theta_fct_single(all_angles[i], params)
        
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
    
    
    def plot_likelihood_correctlycuedtimes(self, n=0, amplify_diag=1.0, all_angles=None, num_points=500, should_plot=True, should_return=False, should_exponentiate = False, sampled_feature_index = 0, debug=True):
        '''
            Plot the log-likelihood function, over the space of the sampled theta, keeping the other thetas fixed to their correct cued value.
        '''

        num_points = int(num_points)

        if all_angles is None:
            all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        
        # Compute the likelihood
        llh_2angles = self.compute_likelihood_fullspace(n=n, all_angles=all_angles, num_points=num_points, should_exponentiate=should_exponentiate, sampled_feature_index=sampled_feature_index)

        # Save it if we need to return it
        if should_return:
            llh_2angles_out = llh_2angles.copy()

        # Normalize loglik
        llh_2angles /= np.abs(np.max(llh_2angles, axis=0))
        
        opt_angles = np.argmax(llh_2angles, axis=0)
        
        # Move them a bit apart
        llh_2angles += 1.2*np.arange(self.T)*np.abs(np.max(llh_2angles, axis=0)-np.mean(llh_2angles, axis=0))
        
        # Plot the result
        if should_plot:
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
        if debug:
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

        if should_return:
            return llh_2angles_out


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
            params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[tc], sampled_feature_index, self.mean_fixed_contrib[tc], self.inv_covariance_fixed_contrib)
            
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
    

    def estimate_fisher_info_from_posterior(self, n=0, all_angles=None, num_points=500):
        '''
            Look at the curvature of the posterior to estimate the Fisher Information
        '''

        if all_angles is None:
            all_angles = np.linspace(-np.pi, np.pi, num_points)

        posterior = self.compute_likelihood_fullspace(n=n, all_angles=all_angles, num_points=num_points, should_exponentiate=True)[:, 0]

        # Look if it seems Gaussian enough

        log_posterior = np.log(posterior.T)
        log_posterior[np.isinf(log_posterior)] = 0.0
        log_posterior[np.isnan(log_posterior)] = 0.0

        dx = np.diff(all_angles)[0]

        posterior /= np.sum(posterior*dx)

        np.seterr(all='raise')
        try:
            # Incorrect here, see Issue #23
            # FI_estim_curv = np.trapz(-np.gradient(np.gradient(log_posterior))*posterior/dx**2., x)

            ml_index = np.argmax(posterior)
            curv_logp = -np.gradient(np.gradient(log_posterior))/dx**2.

            # take the curvature at the ML value
            FI_estim_curv = curv_logp[ml_index]
        except FloatingPointError:
            # print 'Overflow on n: %d' % n
            FI_estim_curv = np.nan

        np.seterr(all='warn')

        return FI_estim_curv



    def estimate_fisher_info_from_posterior_avg(self, num_points=500, full_stats=False):
        '''
            Estimate the Fisher Information from the curvature of the posterior.

            Takes the mean over all datapoints.
        '''

        mean_FI = np.zeros(self.N)

        all_angles = np.linspace(-np.pi, np.pi, num_points)

        for i in progress.ProgressDisplay(np.arange(self.N), display=progress.SINGLE_LINE):
            mean_FI[i] = self.estimate_fisher_info_from_posterior(n=i, all_angles=all_angles)

        if full_stats:
            return dict(mean=nanmean(mean_FI), std=nanstd(mean_FI), median=nanmedian(mean_FI), all=mean_FI)
        else:
            return nanmean(mean_FI)



    def estimate_precision_from_posterior(self, n=0, num_points=500):
        '''
            Look at the posterior to estimate the precision directly
        '''

        posterior = self.plot_likelihood_correctlycuedtimes(n=n, num_points=num_points, should_plot=False, should_return=True, should_exponentiate = True, debug=False)[:, 0]

        x = np.linspace(-np.pi, np.pi, num_points)
        dx = np.diff(x)[0]

        posterior /= np.sum(posterior*dx)

        np.seterr(all='raise')
        try:
            precision_estimated = 1./(-2.*np.log(np.abs(np.trapz(posterior*np.exp(1j*x), x))))
        except FloatingPointError:
            # print 'Overflow on n: %d' % n
            precision_estimated = np.nan

        np.seterr(all='warn')

        return precision_estimated



    def estimate_precision_from_posterior_avg(self, num_points=500, full_stats=False):
        '''
            Estimate the precision from the posterior.

            Takes the mean over all datapoints.
        '''

        precisions = np.zeros(self.N)

        for i in progress.ProgressDisplay(np.arange(self.N), display=progress.SINGLE_LINE):
            precisions[i] = self.estimate_precision_from_posterior(n=i, num_points=num_points)

        if full_stats:
            return dict(mean=nanmean(precisions), std=nanstd(precisions), median=nanmedian(precisions), all=precisions)
        else:
            return nanmean(precisions)


    def estimate_precision_from_samples(self, n=0, num_samples=1000, num_repetitions=1, selection_method='median'):
        '''
            Take samples of theta for a particular datapoint, and estimate the precision from their distribution.
        '''
        
        all_precisions = np.zeros(num_repetitions)

        for repet_i in xrange(num_repetitions):

            # Get samples
            samples = self.sample_theta(num_samples=num_samples, integrate_tc_out=False, return_samples=True, subset_theta=[n], selection_method=selection_method)[0]

            # Estimate the circular standard deviation of those samples
            circ_std_dev = self.compute_mean_std_circular_data(samples)[1]

            # And now get the precision (uncorrected for chance level)
            all_precisions[repet_i] = 1./circ_std_dev**2.

        return dict(mean=np.mean(all_precisions), std=np.std(all_precisions), all=all_precisions)


    def estimate_precision_from_samples_avg(self, num_samples=1000, num_repetitions=1, full_stats=False, selection_method='median'):
        '''
            Estimate precision from the samples. Get it for every datapoint.
        '''

        all_precision = np.zeros(self.N)
        all_precision_everything = np.zeros((self.N, num_repetitions))

        for i in xrange(self.N):
            print i
            res = self.estimate_precision_from_samples(n=i, num_samples=num_samples, num_repetitions=num_repetitions, selection_method=selection_method)
            (all_precision[i], all_precision_everything[i]) = (res['mean'], res['all'])

        if full_stats:
            return dict(mean=nanmean(all_precision), std=nanstd(all_precision), median=nanmedian(all_precision), all=all_precision_everything)
        else:
            return nanmean(all_precision)


    def estimate_truevariance_from_posterior(self, n=0, all_angles=None, num_points=500):
        '''
            Estimate the variance from the empirical posterior.
        '''

        if all_angles is None:
            all_angles = np.linspace(-np.pi, np.pi, num_points)

        posterior = self.compute_likelihood_fullspace(n=n, all_angles=all_angles, num_points=num_points, should_exponentiate=True, normalize=True)[:, 0]

        mean = np.trapz(posterior*all_angles, all_angles)
        variance = np.trapz(posterior*(all_angles - mean)**2., all_angles)

        return variance


    def estimate_truevariance_from_posterior_avg(self, all_angles=None, num_points=500, full_stats=False):
        '''
            Get the mean estimated variance from the empirical posterior
        '''

        if all_angles is None:
            all_angles = np.linspace(-np.pi, np.pi, num_points)

        truevariances = np.zeros(self.N)

        for i in progress.ProgressDisplay(np.arange(self.N), display=progress.SINGLE_LINE):
            # print i

            truevariances[i] = self.estimate_truevariance_from_posterior(n=i, all_angles=all_angles, num_points=num_points)

        if full_stats:
            return dict(mean=nanmean(truevariances), std=nanstd(truevariances), median=nanmedian(truevariances), all=truevariances)
        else:
            return nanmean(truevariances)


    def plot_comparison_samples_fit_posterior(self, n=0, samples=None, num_samples=1000, num_points=1000, selection_method='median'):
        '''
            Plot a series of samples (usually from theta), associated with the posterior generating them and a gaussian fit.

            Trying to see where the bias from the Slice Sampler comes from
        '''

        if samples is None:
            # no samples, get them
            samples = self.sample_theta(num_samples=num_samples, integrate_tc_out=False, return_samples=True, subset_theta=[n], selection_method=selection_method)[0]

        # Plot the samples and the fit
        fit_gaussian_samples(samples)

        # Get the posterior
        posterior = self.plot_likelihood_correctlycuedtimes(n=n, num_points=num_points, should_plot=False, should_return=True, should_exponentiate = True, debug=False)[:, 0]

        x = np.linspace(-np.pi, np.pi, num_points)

        plt.plot(x, posterior/posterior.max(), 'k')

        plt.legend(['Fit', 'Posterior'])
    

    
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
    

    def get_precision(self, remove_chance_level=False, correction_theo_fit=1.0):
        '''
            Compute the precision, inverse of the std dev of the errors.
            This is our target metric
        '''

        # Compute precision
        precision = 1./self.compute_angle_error()[1]**2.
        precision *= correction_theo_fit

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
        print "  Precision:\t %.3f" % (self.get_precision())
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

        assert self.T > 1, "No nontarget for a single object..."

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
if __name__ == '__main__':
    
    print "====> DEPRECATED, use experimentlauncher.py instead"

    import experimentlauncher
    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True)

    plt.show()


