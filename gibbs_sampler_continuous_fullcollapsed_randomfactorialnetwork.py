#!/usr/bin/env python
# encoding: utf-8
"""
sampler.py

Created by Loic Matthey on 2011-06-1.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import numpy as np
# import scipy.special as scsp
# from scipy.stats import vonmises as vm
import scipy.stats as spst
import scipy.optimize as spopt
import scipy.integrate as spintg
import scipy.io as sio
import matplotlib.patches as plt_patches
# import matplotlib.collections as plt_collections
import matplotlib.pyplot as plt

import sys

from utils import *

import em_circularmixture
import em_circularmixture_allitems_uniquekappa

import slicesampler

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

def loglike_theta_fct_opt2d(new_theta, theta2, datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, inv_covariance_fixed_contrib):
    '''
        Compute the loglikelihood of: theta_r | n_tc theta_r' tc
    '''

    like_mean = datapoint - mean_fixed_contrib - \
                ATtcB*rn.get_network_response_opt2d(new_theta, theta2)

    # Using inverse covariance as param
    # return theta_kappa*np.cos(thetas[sampled_feature_index] - theta_mu) - 0.5*np.dot(like_mean, np.dot(inv_covariance_fixed_contrib, like_mean))
    return -0.5*np.dot(like_mean, np.dot(inv_covariance_fixed_contrib, like_mean))
    # return -1./(2*0.2**2)*np.sum(like_mean**2.)

def like_theta_fct_opt2d(new_theta, theta2, datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, inv_covariance_fixed_contrib):
    return np.exp(loglike_theta_fct_opt2d(new_theta, theta2, datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, inv_covariance_fixed_contrib))

def loglike_theta_fct_single_min(x, thetas, datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, inv_covariance_fixed_contrib):
    return -loglike_theta_fct_single(x, (thetas, datapoint, rn, theta_mu, theta_kappa, ATtcB, sampled_feature_index, mean_fixed_contrib, inv_covariance_fixed_contrib))


class Sampler:
    '''
        Continuous angles Theta, with Von Mise prior.
        x | Theta ~ Normal. Using the population codes directly
        y_t | x_t, y_{t-1} ~ Normal

    '''
    def __init__(self, data_gen, tc=None, theta_prior_dict=dict(kappa=0.01, gamma=0.0), n_parameters = dict()):
        '''
            Initialise the sampler

            n_parameters:         {means: T x M, covariances: T x M x M}
        '''

        self.theta_prior_dict = theta_prior_dict

        # Initialise noise parameters
        self.set_noise_parameters(n_parameters)

        # Get the data
        self.init_from_data_gen(data_gen, tc=tc)


    def set_noise_parameters(self, n_parameters):
        '''
            Store the noise parameters, computed from a StatisticsMeasurer

            n_parameters:         {means: T x M, covariances: T x M x M}
        '''

        self.n_means_start = n_parameters['means'][0]
        self.n_means_end = n_parameters['means'][1]
        self.n_covariances_start = n_parameters['covariances'][0]
        self.n_covariances_end = n_parameters['covariances'][1]
        self.n_means_measured = n_parameters['means'][2]
        self.n_covariances_measured = n_parameters['covariances'][2]

        self.noise_covariance = self.n_covariances_measured[-1]



    def init_from_data_gen(self, data_gen, tc=None):
        '''

        '''

        self.data_gen = data_gen
        self.random_network = self.data_gen.random_network
        self.NT = self.data_gen.Y

        # Get sizes
        (self.N, self.M) = self.NT.shape
        self.T = self.data_gen.T
        self.R = self.data_gen.random_network.R

        # Time weights
        self.time_weights = self.data_gen.time_weights
        self.sampled_feature_index = 0

        # Initialise t_c
        self.init_tc(tc=tc)

        # Initialise latent angles
        self.init_theta()

        # Precompute the parameters and cache them
        self.init_cache_parameters()


    def init_tc(self, tc=None):
        '''
            Initialise the time of recall

            tc = N x 1

            Could be sampled later, for now just fix it.
        '''

        if np.isscalar(tc):
            self.tc = tc*np.ones(self.N, dtype='int')
        else:
            self.tc = np.zeros(self.N, dtype='int')


    def init_theta(self):
        '''
            Sample initial angles. Use a Von Mises prior, low concentration (~flat)

            Theta:          N x R
        '''

        self.theta_gamma = self.theta_prior_dict['gamma']
        self.theta_kappa = self.theta_prior_dict['kappa']
        self.theta = np.random.vonmises(self.theta_gamma, self.theta_kappa, size=(self.N, self.R))

        # Assign the cued ones now
        #   stimuli_correct: N x T x R
        #   cued_features:      N x (recall_feature, recall_time)
        self.theta[np.arange(self.N), self.data_gen.cued_features[:, 0]] = self.data_gen.stimuli_correct[np.arange(self.N), self.data_gen.cued_features[:, 1], self.data_gen.cued_features[:, 0]]

        # Construct the list of uncued features, which should be sampled
        self.theta_to_sample = np.array([[r for r in xrange(self.R) if r != self.data_gen.cued_features[n, 0]] for n in xrange(self.N)], dtype='int')

        if self.R == 2:
            # Just for convenience (in compute_angle_error), flatten the theta_to_sample
            self.theta_to_sample = self.theta_to_sample.flatten()


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

        # Precompute parameters
        for t in xrange(self.T):
            (self.ATtcB[t], self.mean_fixed_contrib[t], self.inv_covariance_fixed_contrib) = self.precompute_parameters(t, amplify_diag=amplify_diag)

        # Compute the normalization
        self.compute_normalization()


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


    def compute_normalization(self):
        '''
            Compute normalization factor for loglikelihood
        '''

        self.normalization = np.empty(self.N)

        for n in xrange(self.N):
            ## Pack parameters and integrate using scipy, super fast
            params = (self.theta[n, 1], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[self.tc[n]], self.sampled_feature_index, self.mean_fixed_contrib[self.tc[n]], self.inv_covariance_fixed_contrib)

            self.normalization[n] = np.log(spintg.quad(like_theta_fct_opt2d, -np.pi, np.pi, args=params)[0])


    #######

    def run_inference(self, parameters):
        '''
            Infer angles based on memory state

            Can either:
            1) Sample using Gibbs sampling / Slice sampler
            2) Set to Max Lik values

            Warning: needs around 200 samples to mix. Burn_samples set to 100.
        '''
        print "Running inference..."
        if parameters['inference_method'] == 'sample':
            # Sample thetas
            self.sample_theta(num_samples=parameters['num_samples'], burn_samples=parameters['burn_samples'], selection_method=parameters['selection_method'], selection_num_samples=parameters['selection_num_samples'], slice_width=parameters['slice_width'], integrate_tc_out=False, debug=True)
        elif parameters['inference_method'] == 'max_lik':
            # Just use the ML value for the theta
            self.set_theta_max_likelihood(num_points=100, post_optimise=True)


    def sample_all(self):
        '''
            Do one full sweep of sampling
        '''

        # t = time.time()
        self.sample_theta()
        # print "Sample_theta time: %.3f" % (time.time()-t)
        self.sample_tc()
        print "Likelihood: %.2f" % self.compute_likelihood()


    def sample_theta(self, num_samples=500, return_samples=False, burn_samples=100, integrate_tc_out=False, selection_method='last',selection_num_samples=250, subset_theta=None, slice_width=np.pi/16.0, slice_jump_prob=0.3, debug=True):
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

        if debug:
            if selection_method == 'last':
                print "Sampling theta: %d samples, %d burnin, select last" % (num_samples, burn_samples)
            else:
                print "Sampling theta: %d samples, %d selection, %d burnin" % (num_samples, selection_num_samples, burn_samples)

        if return_samples:
            all_samples = np.zeros((permuted_datapoints.size, num_samples))

        if debug:
            search_progress = progress.Progress(permuted_datapoints.size)

        # Do everything in log-domain, to avoid numerical errors
        i = 0
        # for n in progress.ProgressDisplay(permuted_datapoints, display=progress.SINGLE_LINE):
        for n in permuted_datapoints:

            # Sample all the non-cued features
            permuted_features = np.random.permutation(self.theta_to_sample[n, np.newaxis])

            for sampled_feature_index in permuted_features:
                # Get samples from the current distribution
                if integrate_tc_out:
                    samples = self.get_samples_theta_tc_integratedout(n, num_samples=num_samples, sampled_feature_index=sampled_feature_index, burn_samples=burn_samples)
                else:
                    (samples, _) = self.get_samples_theta_current_tc(n, num_samples=num_samples, sampled_feature_index=sampled_feature_index, burn_samples=burn_samples, slice_width=slice_width, slice_jump_prob=slice_jump_prob)

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

                if debug:
                    search_progress.increment()

                    if search_progress.done():
                        eol = '\n'
                    else:
                        eol = '\r'

                    line= "%.2f%%, %s - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())
                    sys.stdout.write("%s%s%s" % (line, " " * (78-len(line)), eol))
                    sys.stdout.flush()

            i+= 1

        if return_samples:
            return all_samples


    def get_samples_theta_current_tc(self, n, num_samples=2000, burn_samples=200, slice_width=np.pi/4., slice_jump_prob=0.2, sampled_feature_index=0):

        # Pack the parameters for the likelihood function.
        #   Here, as the loglike_function only varies one of the input, need to give the rest of the theta vector.
        params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[self.tc[n]], sampled_feature_index, self.mean_fixed_contrib[self.tc[n]], self.inv_covariance_fixed_contrib)

        # Sample the new theta
        # samples, llh = slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=burn_samples, widths=np.pi/8., loglike_fct_params=params, debug=False, step_out=True)
        samples, llh = slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=burn_samples, widths=slice_width, loglike_fct_params=params, debug=False, step_out=True, jump_probability=slice_jump_prob)
        # samples, llh = slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=burn_samples, widths=0.01, loglike_fct_params=params, debug=False, step_out=True)

        # samples, llh = slicesampler.sample_1D_circular(1, self.theta[n, sampled_feature_index], loglike_theta_fct, burn=100, widths=np.pi/3., thinning=2, loglike_fct_params=params, debug=False, step_out=True)

        return (samples, llh)


    def get_samples_theta_tc_integratedout(self, n, num_samples=2000, burn_samples=100, sampled_feature_index=0):
        '''
            Sample theta, with tc integrated out.
            Use rejection sampling (or something), discarding some samples.

            Note: the actual number of samples returned is random, but around num_samples.
        '''

        samples_integratedout = []
        for tc in xrange(self.T):
            params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[tc], sampled_feature_index, self.mean_fixed_contrib[tc], self.inv_covariance_fixed_contrib)

            # TODO> Should be starting from the previous sample here.
            # samples, _ = slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=burn_samples, widths=np.pi/8., loglike_fct_params=params, debug=False, step_out=True)
            samples, _ = slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=burn_samples, widths=np.pi/3., loglike_fct_params=params, debug=False, step_out=True)

            # Now keep only some of them, following p(tc)
            #   for now, p(tc) = 1/T
            filter_samples = np.random.random_sample(num_samples) < 1./self.T
            samples_integratedout.extend(samples[filter_samples])

        return np.array(samples_integratedout)


    def set_theta(self, new_thetas):
        '''
            Update thetas to a given value (most likely to experimentally measured ones)
        '''

        self.theta[:, self.sampled_feature_index] = new_thetas


    def set_theta_max_likelihood(self, num_points=100, post_optimise=True):
        '''
            Update theta to their Max Likelihood values.
            Should be faster than sampling.
        '''

        all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        llh = np.zeros(num_points)

        # Compute the array
        for n in progress.ProgressDisplay(np.arange(self.N), display=progress.SINGLE_LINE):

            # Pack the parameters for the likelihood function
            params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[self.tc[n]], self.sampled_feature_index, self.mean_fixed_contrib[self.tc[n]], self.inv_covariance_fixed_contrib)

            # Compute the loglikelihood for all possible first feature
            # Use this as initial value for the optimisation routine
            for i in xrange(num_points):
                # Give the correct cued second feature
                llh[i] = loglike_theta_fct_single(all_angles[i], params)

            # opt_angles[n] = spopt.fminbound(loglike_theta_fct_single_min, -np.pi, np.pi, params, disp=3)
            # opt_angles[n] = spopt.brent(loglike_theta_fct_single_min, params)
            # opt_angles[n] = wrap_angles(np.array([np.mod(spopt.anneal(loglike_theta_fct_single_min, np.random.random_sample()*np.pi*2. - np.pi, args=params)[0], 2.*np.pi)]))

            if post_optimise:
                self.theta[n, self.sampled_feature_index] = spopt.fmin(loglike_theta_fct_single_min, all_angles[np.argmax(llh)], args=params, disp=False)[0]
            else:
                self.theta[n, self.sampled_feature_index] = all_angles[np.argmax(llh)]


    def change_cued_features(self, t_cued):
        '''
            Change the cue.
                Modify time of cue, and pull it from data_gen again
        '''

        # The time of the cued feature
        self.data_gen.cued_features[:, 1] = t_cued

        # Reset the cued theta
        self.theta[np.arange(self.N), self.data_gen.cued_features[:, 0]] = self.data_gen.stimuli_correct[np.arange(self.N), self.data_gen.cued_features[:, 1], self.data_gen.cued_features[:, 0]]


    def compute_bic(self, K=None, integrate_tc_out=False):
        '''
            Compute the BIC score for the current model.

            Default K parameters:
                - Sigmax
                - M neurons
                - ratio_conj if code_type is mixed

            Usually, sigma_y is set to a super small value, and rc_scales are set automatically.
            Not sure if num_samples/burn_samples should count, I don't think so.
        '''

        if K is None:
            # Assume we set Sigmax and M.
            K = 2.

            if self.random_network.population_code_type == 'mixed':
                K += 1.

        LL = self.compute_loglikelihood(integrate_tc_out=integrate_tc_out)

        return bic(K, LL, self.N)


    def compute_loglikelihood(self, integrate_tc_out=False):
        '''
            Compute the summed loglikelihood for the current setting of thetas and using the likelihood defined in loglike_theta_fct_single

            - integrate_tc_out:  use the current tc, or should integrate over possible recall times?
        '''

        return np.nansum(self.compute_loglikelihood_N(integrate_tc_out))


    def compute_loglikelihood_N(self, integrate_tc_out=False):
        '''
            Compute the loglikelihood for each datapoint, using the current setting of thetas and likelihood functions.
            Uses the normalisation.

            - integrate_tc_out: use current tc, or should integrate over recall times?
        '''
        if integrate_tc_out:
            return self.compute_loglikelihood_tc_integratedout() - self.normalization
        else:
            return self.compute_loglikelihood_current_tc() - self.normalization

    def compute_loglikelihood_current_tc(self):
        '''
            Compute the loglikelihood for the current setting of thetas and tc and using the likelihood defined in loglike_theta_fct_single
        '''

        loglikelihood = np.empty(self.N)

        for n in xrange(self.N):
            # Pack the parameters for the likelihood function
            params = (self.theta[n].copy(), self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[self.tc[n]], self.sampled_feature_index, self.mean_fixed_contrib[self.tc[n]], self.inv_covariance_fixed_contrib)

            # Compute the loglikelihood for the current datapoint
            loglikelihood[n] = loglike_theta_fct_single(self.theta[n, self.sampled_feature_index], params)

        return loglikelihood


    def compute_loglikelihood_tc_integratedout(self):
        '''
            Compute the loglikelihood for the current setting of thetas and using the likelihood defined in loglike_theta_fct_single
            Integrates tc out.
        '''

        loglikelihood = np.empty((self.N, self.T))

        for n in xrange(self.N):
            for tc in xrange(self.T):
                # Pack the parameters for the likelihood function
                params = (self.theta[n].copy(), self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[tc], self.sampled_feature_index, self.mean_fixed_contrib[tc], self.inv_covariance_fixed_contrib)

                # Compute the loglikelihood for the current datapoint
                loglikelihood[n, tc] = loglike_theta_fct_single(self.theta[n, self.sampled_feature_index], params)

        return loglikelihood



    def compute_likelihood_fullspace(self, n=0, all_angles=None, num_points=1000, normalize=False, remove_mean=False, should_exponentiate=False):
        '''
            Computes and returns the (log)likelihood evaluated for a given datapoint on the entire space (e.g. [-pi,pi]).
        '''

        if all_angles is None:
            all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        else:
            num_points = all_angles.size

        likelihood = np.zeros((self.T, num_points))

        # Compute the array
        for t in xrange(self.T):

            # Compute the loglikelihood for all possible first feature
            for i in xrange(num_points):

                # Pack the parameters for the likelihood function
                params = (np.array([all_angles[i], self.data_gen.stimuli_correct[n, t, 1]]), self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[t], self.sampled_feature_index, self.mean_fixed_contrib[t], self.inv_covariance_fixed_contrib)

                # Give the correct cued second feature
                likelihood[t, i] = loglike_theta_fct_single(all_angles[i], params)
                # likelihood[t, i] = loglike_theta_fct_vect(np.array([all_angles[i], self.data_gen.stimuli_correct[n, t, 1]]), params)

        likelihood = likelihood.T

        # Center loglik
        if remove_mean:
            likelihood -= np.mean(likelihood, axis=0)

        # Normalise if required.
        if normalize:
            likelihood -= self.normalization[n]

        if should_exponentiate:
            # If desired, plot the likelihood, not the loglik
            likelihood = np.exp(likelihood)

        return likelihood


    ######################

    def plot_likelihood(self, n=0, t=0, amplify_diag = 1.0, should_sample=False, num_samples=2000, return_output=False, should_exponentiate = False, num_points=1000, should_normalize=False, ax_handle=None):


        # Pack the parameters for the likelihood function.
        #   Here, as the loglike_function only varies one of the input, need to give the rest of the theta vector.
        params = (self.theta[n].copy(), self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[t], self.sampled_feature_index, self.mean_fixed_contrib[t], self.inv_covariance_fixed_contrib)

        x = np.linspace(-np.pi, np.pi, num_points)

        ll_x = np.array([(loglike_theta_fct_single(a, params)) for a in x])

        if should_normalize:
            ll_x -= self.normalization[n]

        if should_exponentiate:
            ll_x = np.exp(ll_x)

        # ll_x -= np.mean(ll_x)
        # ll_x /= np.abs(np.max(ll_x))

        if ax_handle is None:
            f, ax_handle = plt.subplots()

        ax_handle.plot(x, ll_x)
        ax_handle.axvline(x=self.data_gen.stimuli_correct[n, self.data_gen.cued_features[n, 1], 0], color='r')
        ax_handle.axvline(x=self.theta[n, self.theta_to_sample[n]], color='k', linestyle='--')

        ax_handle.set_xlim((-np.pi, np.pi))

        if should_sample:
            samples, _ = slicesampler.sample_1D_circular(num_samples, np.random.rand()*2.*np.pi-np.pi, loglike_theta_fct_single, burn=500, widths=np.pi/4., loglike_fct_params=params, debug=False, step_out=True)
            x_edges = x - np.pi/num_points  # np.histogram wants the left-right boundaries...
            x_edges = np.r_[x_edges, -x_edges[0]]  # the rightmost boundary is the mirror of the leftmost one
            sample_h, left_x = np.histogram(samples, bins=x_edges)
            ax_handle.bar(x_edges[:-1], sample_h/np.max(sample_h).astype('float'), facecolor='green', alpha=0.75, width=np.pi/num_points)

        ax_handle.get_figure().canvas.draw()

        if return_output:
            if should_sample:
                return (ll_x, x, samples)
            else:
                return (ll_x, x)


    def plot_likelihood_variation_twoangles(self, num_points=100, amplify_diag=1.0, should_plot=True, should_return=False, should_exponentiate = False, remove_mean=False, n=0, t=0, interpolation='nearest', normalize=False, colormap=None):
        '''
            Compute the likelihood, varying two angles around.
            Plot the result
        '''


        all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        llh_2angles = np.zeros((num_points, num_points))

        # Compute the array
        for i in xrange(num_points):
            print "%d%%" % (i/float(num_points)*100)
            for j in xrange(num_points):

                # Pack the parameters for the likelihood function
                params = (np.array([all_angles[i], all_angles[j]]), self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[t], self.sampled_feature_index, self.mean_fixed_contrib[t], self.inv_covariance_fixed_contrib)

                # llh_2angles[i, j] = loglike_theta_fct_vect(np.array([all_angles[i], all_angles[j]]), params)
                llh_2angles[i, j] = loglike_theta_fct_single(all_angles[i], params)

        if remove_mean:
            llh_2angles -= np.mean(llh_2angles)

        # Normalise if required.
        if normalize:
            llh_2angles -= self.normalization[n]

        if should_exponentiate:
            llh_2angles = np.exp(llh_2angles)

        if should_plot:
            # Plot the obtained landscape
            f = plt.figure()
            ax = f.add_subplot(111)
            im= ax.imshow(llh_2angles.T, origin='lower', cmap=colormap)
            im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
            im.set_interpolation(interpolation)
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
            color_gen = [colmap(1.*(i)/self.T) for i in xrange(self.T)][::-1]  # use 22 colors

            for t in xrange(self.T):
                w = plt_patches.Wedge((correct_angles[t, 0], correct_angles[t, 1]), 0.25, 0, 360, 0.10, color=color_gen[t], alpha=0.9)
                ax.add_patch(w)

            # plt.annotate('O', (correct_angles[1, 0], correct_angles[1, 1]), color='blue', fontweight='bold', fontsize=30, horizontalalignment='center', verticalalignment='center')


        if should_return:
            return llh_2angles


    def plot_likelihood_correctlycuedtimes(self, n=0, amplify_diag=1.0, all_angles=None, num_points=500, should_plot=True, should_return=False, should_exponentiate = False, show_legend=True, show_current_theta=True, debug=True, ax_handle=None):
        '''
            Plot the log-likelihood function, over the space of the sampled theta, keeping the other thetas fixed to their correct cued value.
        '''

        num_points = int(num_points)

        if all_angles is None:
            all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)

        # Compute the likelihood
        llh_2angles = self.compute_likelihood_fullspace(n=n, all_angles=all_angles, num_points=num_points, should_exponentiate=should_exponentiate, remove_mean=True)

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
            if ax_handle is None:
                f = plt.figure()
                ax_handle = f.add_subplot(111)

            lines = ax_handle.plot(all_angles, llh_2angles)
            ax_handle.set_xlim((-np.pi, np.pi))

            legends = ['-%d' % x for x in np.arange(self.T)[::-1]]
            legends[-1] = 'Last'

            for t in xrange(self.T):
                # Put the legends
                if show_legend:
                    ax_handle.legend(lines, legends, loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=self.T, fancybox=True, shadow=True)

                # Put a vertical line at the true answer
                ax_handle.axvline(x=self.data_gen.stimuli_correct[n, t, 0], color=lines[t].get_c())  # ax_handle[t] returns the plotted line


            # Put a dotted line at the current theta sample, for tc
            if show_current_theta:
                ax_handle.axvline(x=self.theta[n, self.theta_to_sample[n]], color='k', linestyle="--")

            ax_handle.get_figure().canvas.draw()

        # Print the answers
        if debug:
            print "True angles: %s >> Inferred: %s" % (' | '.join(['%.3f' % x for x in self.data_gen.stimuli_correct[n, :, 0]]),  ' | '.join(['%.3f' % x for x in all_angles[opt_angles]]))

        # if self.T == 2:
        #     plt.legend(('First', 'Second'), loc='best')
        #     print "True angles: %.3f | %.3f >> Inferred: %.3f | %.3f" % (self.data_gen.stimuli_correct[n, 0, 0], self.data_gen.stimuli_correct[n, 1, 0], all_angles[opt_angles[0]], all_angles[opt_angles[1]])
        #     plt.axvline(x=self.data_gen.stimuli_correct[n, 1, 0], color='g')
        # elif self.T == 3:
        #     plt.axvline(x=self.data_gen.stimuli_correct[n, 2, 0], color='r')
        #     plt.legend(('First', 'Second', 'Third'), loc='best')
        #     print "True angles: %.3f | %.3f | %.3f >> Inferred: %.3f | %.3f | %.3f" % (self.data_gen.stimuli_correct[n, 0, 0], self.data_gen.stimuli_correct[n, 1, 0], self.data_gen.stimuli_correct[n, 2, 0], all_angles[opt_angles[0]], all_angles[opt_angles[1]], all_angles[opt_angles[2]])

        plt.show()

        if should_return:
            return llh_2angles_out

        return ax_handle


    def plot_likelihood_comparison(self, n=0):
        '''
            Plot both likelihood and loglikelihoods of all t

            Allows to check how the posterior behaves
        '''

        f, axes = plt.subplots(2, 1)

        self.plot_likelihood_correctlycuedtimes(should_exponentiate=True, n=n, ax_handle=axes[0])

        self.plot_likelihood(should_normalize=True, n=n, ax_handle=axes[1])

    ########

    def fit_mixture_model(self, compute_responsibilities=False, use_all_targets=False):
        '''
            Fit Paul Bays' Mixture model.

            Can provide responsibilities as well if required
        '''

        if use_all_targets:
            em_circular_mixture_to_use = em_circularmixture_allitems_uniquekappa
        else:
            em_circular_mixture_to_use = em_circularmixture

        params_fit = em_circular_mixture_to_use.fit(*self.collect_responses())

        if compute_responsibilities:
            params_fit['resp'] = em_circular_mixture_to_use.compute_responsibilities(*(self.collect_responses() + (params_fit,) ))

        params_fit.setdefault('mixt_nontargets_sum', np.sum(params_fit['mixt_nontargets']))

        return params_fit


    def estimate_fisher_info_from_posterior(self, n=0, all_angles=None, num_points=500):
        '''
            Look at the curvature of the posterior to estimate the Fisher Information
        '''

        if all_angles is None:
            all_angles = np.linspace(-np.pi, np.pi, num_points)

        log_posterior = self.compute_likelihood_fullspace(n=n, all_angles=all_angles, num_points=num_points, should_exponentiate=False, remove_mean=True)[:, -1].T

        # Look if it seems Gaussian enough

        # log_posterior = np.log(posterior.T)
        log_posterior[np.isinf(log_posterior)] = 0.0
        log_posterior[np.isnan(log_posterior)] = 0.0

        dx = np.diff(all_angles)[0]

        # posterior /= np.sum(posterior*dx)

        np.seterr(all='raise')
        try:
            # Incorrect here, see Issue #23
            # FI_estim_curv = np.trapz(-np.gradient(np.gradient(log_posterior))*posterior/dx**2., x)

            ml_index = np.argmax(log_posterior)
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


    def estimate_fisher_info_from_posterior_avg_randomsubset(self, subset_size=1, num_points=500, full_stats=False):
        '''
            Estimate the Fisher Information from the curvature of the posterior.

            Takes the mean over all datapoints.
        '''

        mean_FI = np.zeros(subset_size)
        random_subset = np.random.randint(self.N, size=subset_size)

        all_angles = np.linspace(-np.pi, np.pi, num_points)

        for i in progress.ProgressDisplay(np.arange(subset_size), display=progress.SINGLE_LINE):
            mean_FI[i] = self.estimate_fisher_info_from_posterior(n=random_subset[i], all_angles=all_angles)

        if full_stats:
            return dict(mean=nanmean(mean_FI), std=nanstd(mean_FI), median=nanmedian(mean_FI), all=mean_FI)
        else:
            return nanmean(mean_FI)


    def compute_covariance_theoretical(self, precision=50, ignore_cache=False):
        '''
            Compute and returns the theoretical covariance, found from KL minimization.
        '''
        return self.random_network.compute_covariance_KL(sigma_2=(self.data_gen.sigma_x**2. + self.data_gen.sigma_y**2.), T=self.T, beta=1.0, precision=precision, ignore_cache=ignore_cache)


    def estimate_fisher_info_theocov(self, use_theoretical_cov=True, kappa_different=True):
        '''
            Compute the theoretical Fisher Information, using the KL-derived covariance matrix if desired
        '''

        if use_theoretical_cov:
            # Get the computed covariance
            computed_cov = self.compute_covariance_theoretical(precision=50, ignore_cache=False)

            # Check if it seems correctly similar to the current measured one.
            if np.mean((self.noise_covariance-computed_cov)**2.) > 0.01:
                print np.mean((self.noise_covariance-computed_cov)**2.)
                print "M: %d, rcscale: %.3f, sigmax: %.3f, sigmay: %.3f" % (self.M, self.random_network.rc_scale, self.data_gen.sigma_x, self.data_gen.sigma_y)

                pcolor_2d_data(computed_cov)
                pcolor_2d_data(self.noise_covariance)
                plt.show()

                raise ValueError('Big divergence between measured and theoretical divergence!')
        else:
            # Use the measured one...
            computed_cov = self.noise_covariance

        # Compute the theoretical FI
        return self.random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=computed_cov, kappa_different=kappa_different)


    def estimate_fisher_info_theocov_largen(self, use_theoretical_cov=True):
        '''
            Compute the theoretical Fisher Information, using the KL-derived covariance matrix if desired
        '''

        if use_theoretical_cov:
            # Get the computed covariance
            computed_cov = self.compute_covariance_theoretical(precision=50, ignore_cache=False)
            sigma = np.mean(np.diag(computed_cov))**0.5
        else:
            # Use the measured one...
            computed_cov = self.noise_covariance
            sigma = np.mean(np.diag(computed_cov))**0.5

        # Compute the theoretical FI
        return self.random_network.compute_fisher_information_theoretical(sigma=sigma)


    def estimate_marginal_inverse_fisher_info_montecarlo(self):
        '''
            Compute a Monte Carlo estimate of the Marginal Inverse Fisher Information.

            Marginalise over stimuli values for all items, estimating the Inverse Fisher Information for item 1, feature 1.

            Usually allows to account for closeby interactions better. Doesn't really work for Feature codes though.

            Return marginal inverse Fisher Information (units of variance, not like other functions)
        '''

        FI_estimates = self.random_network.compute_marginal_inverse_FI(self.T, self.inv_covariance_fixed_contrib, min_distance=self.data_gen.enforce_min_distance)

        return FI_estimates


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
            # TODO Precision withouth square?
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


    def estimate_precision_from_posterior_avg_randomsubset(self, subset_size=1, num_points=1000, full_stats=False):
        '''
            Estimate the precision from the posterior.

            Takes the mean over a subset of datapoints.
        '''

        random_subset = np.random.randint(self.N, size=subset_size)

        precisions = np.zeros(subset_size)

        for i in progress.ProgressDisplay(xrange(subset_size), display=progress.SINGLE_LINE):
            precisions[i] = self.estimate_precision_from_posterior(n=random_subset[i], num_points=num_points)

        if full_stats:
            return dict(mean=nanmean(precisions), std=nanstd(precisions), median=nanmedian(precisions), all=precisions)
        else:
            return nanmean(precisions)


    def estimate_precision_from_samples(self, n=0, num_samples=1000, num_repetitions=1, selection_method='median', return_samples=False):
        '''
            Take samples of theta for a particular datapoint, and estimate the precision from their distribution.
        '''

        all_precisions = np.zeros(num_repetitions)

        if return_samples:
            all_samples = np.zeros((num_repetitions, num_samples))

        for repet_i in xrange(num_repetitions):

            # Get samples
            samples = self.sample_theta(num_samples=num_samples, integrate_tc_out=False, return_samples=True, subset_theta=[n], selection_method=selection_method)[0]

            # Estimate the circular standard deviation of those samples
            circ_std_dev = angle_circular_std_dev(samples)

            # And now get the precision (uncorrected for chance level)
            all_precisions[repet_i] = compute_angle_precision_from_std(circ_std_dev)

            if return_samples:
                all_samples[repet_i] = samples

        output = dict(mean=np.mean(all_precisions), std=np.std(all_precisions), all=all_precisions)

        if return_samples:
            output['samples'] = all_samples

        return output


    def estimate_precision_from_samples_avg(self, num_samples=1000, num_repetitions=1, full_stats=False, selection_method='median', return_samples=False):
        '''
            Estimate precision from the samples. Get it for every datapoint.
        '''

        all_precision = np.zeros(self.N)
        all_precision_everything = np.zeros((self.N, num_repetitions))

        if return_samples:
            all_samples = np.zeros((self.N, num_repetitions, num_samples))

        for i in progress.ProgressDisplay(xrange(self.N), display=progress.SINGLE_LINE):
            # print i
            res = self.estimate_precision_from_samples(n=i, num_samples=num_samples, num_repetitions=num_repetitions, selection_method=selection_method, return_samples=return_samples)

            all_precision[i] = res['mean']
            all_precision_everything[i] = res['all']

            if return_samples:
                all_samples[i] = res['samples']

        if full_stats:
            return dict(mean=nanmean(all_precision), std=nanstd(all_precision), median=nanmedian(all_precision), all=all_precision_everything)
        else:
            return nanmean(all_precision)


    def estimate_precision_from_samples_avg_randomsubset(self, subset_size=1, num_samples=1000, num_repetitions=1, full_stats=False, selection_method='median', return_samples=False):
        '''
            Estimate precision from the samples. Get it for every datapoint.

            Takes the mean over a subset of datapoints.
        '''

        random_subset = np.random.randint(self.N, size=subset_size)

        all_precision = np.zeros(subset_size)
        all_precision_everything = np.zeros((subset_size, num_repetitions))
        if return_samples:
            all_samples = np.zeros((subset_size, num_repetitions, num_samples))


        for i in progress.ProgressDisplay(xrange(subset_size), display=progress.SINGLE_LINE):
            res = self.estimate_precision_from_samples(n=random_subset[i], num_samples=num_samples, num_repetitions=num_repetitions, selection_method=selection_method, return_samples=return_samples)

            all_precision[i] = res['mean']
            all_precision_everything[i] = res['all']
            if return_samples:
                all_samples[i] = res['samples']

        if full_stats:
            return dict(mean=nanmean(all_precision), std=nanstd(all_precision), median=nanmedian(all_precision), all=all_precision_everything)
        else:
            return nanmean(all_precision)


    def estimate_truevariance_from_posterior(self, n=0, t=-1, all_angles=None, num_points=500, return_mean=False):
        '''
            Estimate the variance from the empirical posterior.
        '''

        if all_angles is None:
            all_angles = np.linspace(-np.pi, np.pi, num_points)

        posterior = self.compute_likelihood_fullspace(n=n, all_angles=all_angles, num_points=num_points, should_exponentiate=True, normalize=True, remove_mean=True)[:, t]

        mean_ = np.trapz(posterior*all_angles, all_angles)
        variance_ = np.trapz(posterior*(all_angles - mean_)**2., all_angles)

        if return_mean:
            return (variance_, mean_)
        else:
            return variance_


    def estimate_truevariance_from_posterior_avg(self, all_angles=None, num_points=500, full_stats=False):
        '''
            Get the mean estimated variance from the empirical posterior
        '''

        if all_angles is None:
            all_angles = np.linspace(-np.pi, np.pi, num_points)

        truevariances = np.zeros(self.N)

        for i in progress.ProgressDisplay(xrange(self.N), display=progress.SINGLE_LINE):
            # print i

            truevariances[i] = self.estimate_truevariance_from_posterior(n=i, all_angles=all_angles, num_points=num_points)

        if full_stats:
            return dict(mean=nanmean(truevariances), std=nanstd(truevariances), median=nanmedian(truevariances), all=truevariances)
        else:
            return nanmean(truevariances)


    def estimate_gaussianity_from_samples(self, significance_level=0.025, n=0, num_samples=500, num_repetitions=1, selection_method='median'):
        '''
            Take samples of theta for a particular datapoint, and estimate their gaussianity
        '''

        all_samples = np.zeros((num_repetitions, num_samples))
        all_gaussianity = np.zeros(num_repetitions, dtype=int)

        for repet_i in xrange(num_repetitions):

            # Get samples
            all_samples[repet_i] = self.sample_theta(num_samples=num_samples, integrate_tc_out=False, return_samples=True, subset_theta=[n], selection_method=selection_method, burn_samples=100)[0]

            # fit_gaussian_samples(all_samples[repet_i])

            # Check if gaussian or not.
            # normaltest returns (chi^2 stat, p-value). If p-value small => reject H0 coming from gaussian.
            # (so we inverse the result, if p-value smaller than significance level => non-gaussian)
            all_gaussianity[repet_i] = spst.normaltest(all_samples[repet_i])[1] >= significance_level

        # Result is strangely sensitive to very small variations...
        # Try to get a median over multiple tries
        print all_gaussianity
        return np.median(all_gaussianity) == 1


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
        posterior = self.plot_likelihood_correctlycuedtimes(n=n, num_points=num_points, should_plot=False, should_return=True, should_exponentiate = True, debug=False)[:, -1]

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

        for i in xrange(iterations):
            # Do a full sampling sweep
            self.sample_all()

            # Get the likelihoods
            (log_y[i+1], log_z[i+1], log_joint[i+1]) = self.compute_all_loglike()

            # Print report
            if verbose:
                print "Sample %d: likelihoods = y %.3f, z %.3f, joint: %.3f" % (i+1, log_y[i+1], log_z[i+1], log_joint[i+1])

        return (log_z, log_y, log_joint)


    def compute_angle_error(self, return_errors=False, return_groundtruth=False):
        '''
            Compute the mean angle error for the current assignment of Z
            output: dict(mean, std, population_vector)
                    (dict(...), [angle_errors], [true_angles])
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
                return (compute_mean_std_circular_data(angle_errors), angle_errors, true_angles)
            else:
                return (compute_mean_std_circular_data(angle_errors), angle_errors)
        else:
            if return_groundtruth:
                return (compute_mean_std_circular_data(angle_errors), true_angles)
            else:
                return compute_mean_std_circular_data(angle_errors)


    def get_precision(self, remove_chance_level=False, correction_theo_fit=1.0):
        '''
            Compute the precision, inverse of the variance of the errors.
            This is our target metric

            Approximately half of Fisher Information estimate, because of doubly stochastic process and near-gaussianity.
        '''

        # Compute precision
        precision = compute_angle_precision_from_std(self.compute_angle_error()['std'], square_precision=True)
        precision *= correction_theo_fit

        if remove_chance_level:
            # Remove the chance level
            precision -= compute_precision_chance(self.N)

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

            for i in xrange(self.N):
                print "% 4.3f" % (groundtruth[i]) + ''.join(["\t\t% 4.3f" % x for x in nontargets[i]]) + "\t\t % 4.3f \t % 4.3f" % (self.theta[i, 0], angle_errors[i])
        else:
            print "Target \t Inferred \t Error"
            for i in xrange(self.N):
                print "% 4.3f \t\t % 4.3f \t % 4.3f" % (self.theta[i, 0], groundtruth[i], angle_errors[i])

        print "======================================="
        print "  Precision:\t %.3f" % (self.get_precision())
        print "======================================="


    def plot_histogram_errors(self, bins=20, in_degrees=False, norm='max', nice_xticks=False, ax_handle=None):
        '''
            Compute the errors and plot a histogram.

            Should see a Gaussian + background.
        '''

        (_, errors) = self.compute_angle_error(return_errors=True)

        if ax_handle is None:
            f, ax_handle = plt.subplots()

        hist_angular_data(errors, bins=bins, title='Errors between response and target', norm=norm, in_degrees=in_degrees, ax_handle=ax_handle)

        if nice_xticks:
            plt.xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi), (r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=15)


    def plot_histogram_responses(self, bins=20, in_degrees=False, norm='max', show_angles=False, fignum=None, nice_xticks=False):
        '''
            Plot a histogram of the responses
        '''

        # Plot the responses
        (responses, target, nontargets) = self.collect_responses()

        hist_angular_data(responses, bins=bins, title='Distribution of responses', norm=norm, in_degrees=in_degrees, fignum=fignum)

        if show_angles:
            # Add lines for the target and non targets
            plt.axvline(x=target[0], color='b', linewidth=2)

            for nontarget_i in xrange(nontargets.shape[1]):
                plt.axvline(x=nontargets[0, nontarget_i], color='r', linewidth=2)

        if nice_xticks:
            plt.xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi), (r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=15)


    def plot_histogram_bias_nontarget(self, bins=31, in_degrees=False, dataio=None):
        '''
            Get an histogram of the errors between the response and all non targets

            If biased towards 0-values, indicates misbinding errors.

            [from Ed Awh's paper]
        '''

        assert self.T > 1, "No nontarget for a single object..."

        (responses, _, nontargets) = self.collect_responses()

        # Now check the error between the responses and nontargets.
        # Flatten everything, we want the full histogram.
        errors_nontargets = wrap_angles((responses[:, np.newaxis] - nontargets).flatten())

        # Errors between the response the best nontarget.
        errors_best_nontarget = wrap_angles((responses[:, np.newaxis] - nontargets))
        errors_best_nontarget = errors_best_nontarget[np.arange(errors_best_nontarget.shape[0]), np.argmin(np.abs(errors_best_nontarget), axis=1)]

        # Do the plots
        angle_space = np.linspace(-np.pi, np.pi, bins)

        # Get histograms of bias to nontargets.
        hist_samples_density_estimation(errors_nontargets, bins=angle_space, title='Errors between response and non-targets, N=%d' % (self.T), filename='hist_bias_nontargets_%ditems_{label}_{unique_id}.pdf' % (self.T), dataio=dataio)

        # Compute Vtest score
        vtest_dict = V_test(errors_nontargets)
        print vtest_dict

        # Gest histogram of bias to best non target
        hist_samples_density_estimation(errors_best_nontarget, bins=angle_space, title='Errors between response and best non-target, N=%d' % (self.T), filename='hist_bias_bestnontarget_%ditems_{label}_{unique_id}.pdf' % (self.T), dataio=dataio)


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




####################################
if __name__ == '__main__':

    print "====> DEPRECATED, use experimentlauncher.py instead"

    import experimentlauncher
    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True)

    plt.show()


