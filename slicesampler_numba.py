#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
slicesampler.py

Created by Loic Matthey on 2011-08-03.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import numpy as np
# import scipy.special as scsp

import numba as nub

# # @nub.autojit(nub.double(nub.double, nub.double, nub.double))
# @nub.autojit()
# def loglike_vonmises(x, params):
#     mu = params[0]
#     kappa = params[1]
#     out = kappa*np.cos(x - mu) - np.log(2.*np.pi) - np.log(scsp.i0(kappa))

#     return out

# @nub.autojit(locals={'thetas':nub.double[:], 'datapoint':nub.double[:], 'ATtcB':nub.double, 'sampled_feature_index':nub.int_, 'mean_fixed_contrib':nub.double[:], 'inv_covariance_fixed_contrib':nub.double[:,:]})
@nub.jit(nub.f8(nub.f8, nub.f8[:], nub.f8[:], nub.object_, nub.f8, nub.int_, nub.f8[:], nub.f8[:, :]))
def loglike_fct(new_theta, thetas, datapoint, rn, ATtcB, sampled_feature_index, mean_fixed_contrib, inv_covariance_fixed_contrib):
    '''
        Compute the loglikelihood of: theta_r | n_tc theta_r' tc
    '''
    
    # print 'what?', params, len(params)

    # thetas = params[0]
    # datapoint = params[1]
    # # rn = params[2]
    # # theta_mu = params[3]
    # # theta_kappa = params[4]
    # ATtcB = nub.double(params[5])
    # sampled_feature_index = params[6]
    # mean_fixed_contrib = params[7]
    # inv_covariance_fixed_contrib = params[8]

    # Put the new proposed point correctly
    thetas[sampled_feature_index] = new_theta

    # print nub.typeof(mean_fixed_contrib), nub.typeof(inv_covariance_fixed_contrib)
    # print inv_covariance_fixed_contrib

    # a = rn.get_network_response_numba(thetas)

    like_mean = datapoint - mean_fixed_contrib - ATtcB*rn.get_network_response_numba(thetas)

    # Using inverse covariance as param
    # return theta_kappa*np.cos(thetas[sampled_feature_index] - theta_mu) - 0.5*np.dot(like_mean, np.dot(inv_covariance_fixed_contrib, like_mean))
    return -0.5*nub.double(np.dot(like_mean, np.dot(inv_covariance_fixed_contrib, like_mean)))
    # return like_mean
    # return -1./(2*0.2**2)*np.sum(like_mean**2.)

# loglike_fct = nub.jit(nub.f8(nub.f8, nub.f8[:], nub.f8[:], nub.object_, nub.f8, nub.int_, nub.f8[:], nub.f8[:, :]))(loglike_fct)

@nub.autojit(nub.double[:](nub.int32, nub.double, nub.list_of_obj, nub.int32, nub.double, nub.double))
def sample_1D_circular_numba(N, x_initial, loglike_fct_params, burn, widths, jump_probability):
    '''
        Simple implementation of slice sampling for 1D variable
        
        Inputs:
            N                   1x1     Number of samples to gather
            x_initial           1x1     initial state
            loglike_fct_params  list    any extra arguments are passed on to logdist
            burn                1x1     after burning period of this length
            widths              1x1     step sizes for slice sampling. Should correspond.
            jump_probability    1x1     probability of MCMC jump
        
        Outputs:
            samples  Nx1   samples
            
            Iain Murray May 2004, tweaks June 2009, a diagnostic added Feb 2010
            See Pseudo-code in David MacKay's text book p375
    '''
    

    # Initialisation
    # print 'numba!'
    # print loglike_fct
    # print loglike_fct_params

    debug=False
    thinning= 1
    loglike_min = -np.inf
    jump=True
    last_loglikehood = loglike_fct(x_initial, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4], loglike_fct_params[5], loglike_fct_params[6])
    step_out = True
    

    samples = np.zeros(N, dtype=np.float)
    
    x_new  = x_initial
    j = 0
    tot_accepted = 0
    tot_rejected = 0
    

    # N samples
    # print 'for'
    for i in xrange(thinning*N+burn):
        
        # Add a probabilistic jump with Metropolis-Hasting
        if jump and np.random.rand() < jump_probability:
            # print "Jump!"
            xprime = np.random.random_sample()*2.*np.pi - np.pi
            
            # MH ratio
            llh_x_prime = loglike_fct(x_initial, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4], loglike_fct_params[5], loglike_fct_params[6])
            if np.log(np.random.rand()) <  llh_x_prime - last_loglikehood:
                # Accepted!
                x_new = xprime
                last_loglikehood = llh_x_prime
                tot_accepted += 1
            else:
                # rejected, keep x_new
                tot_rejected += 1
        else:
            log_uprime  = last_loglikehood + np.log(np.random.rand())
            # print "log_uprime: %.3f, lastllh: %.3f" % (log_uprime, last_loglikehood)
            
            # Create a horizontal interval (x_l, x_r) enclosing x_new. Place it randomly.
            rr     = np.random.rand()
            x_l    = x_new -  rr*widths
            x_r    = x_new + (1.-rr)*widths
            
            # Grow the interval to get an unbiased slice
            if step_out:
                if log_uprime < loglike_min:
                    # The current slice is too small for the likelihood, step_out will just hit the bounds.
                    x_l = -np.pi
                    x_r = np.pi
                else:
                    llh_l = loglike_fct(x_initial, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4], loglike_fct_params[5], loglike_fct_params[6])
                    # s = 0
                    while llh_l > log_uprime:
                        # print "stepping out left: [%.3f < %.3f < %.3f] %.3f %.3f" % (x_l, x_new, x_r, log_uprime, llh_l)
                        x_l     -= widths
                        llh_l   = loglike_fct(x_initial, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4], loglike_fct_params[5], loglike_fct_params[6])
                        if x_l <= -np.pi:
                            x_l = -np.pi
                            break
                    
                        # s+=1
                    # print s
                
                    llh_r = loglike_fct(x_initial, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4], loglike_fct_params[5], loglike_fct_params[6])
                    while llh_r > log_uprime:
                        x_r     += widths
                        llh_r   = loglike_fct(x_initial, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4], loglike_fct_params[5], loglike_fct_params[6])
                        if x_r >= np.pi:
                            x_r = np.pi
                            break
            
            
            # Sample a new point, shrinking the interval
            while True:
                xprime = np.random.random_sample()*(x_r - x_l) + x_l
                
                last_loglikehood = loglike_fct(x_initial, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4], loglike_fct_params[5], loglike_fct_params[6])

                if last_loglikehood > log_uprime:
                    # Accept this sample
                    # print 'Accept', i
                    x_new = xprime
                    break
                elif xprime > x_new:
                    x_r  = x_new
                elif xprime < x_new:
                    x_l = x_new
                else:
                    print "Slice sampler shrank too far."
                    return -1
        
        # Store this sample
        if i >= burn and (i % thinning == 0):
            if debug:
                print "Sample {}: {:.3f}".format(j+1, x_new)
            samples[j] = x_new
            j += 1
    
    # print "Done. \nJumps:%d, %d" % (tot_accepted, tot_rejected)
    
    return samples


# def test_sample():
    
#     loglike_fct_params = np.array([0.0, 0.1])
    
#     # Get samples
#     samples, last_llh = sample_1D_circular(1000, np.random.rand(), loglike_fct_params, 50, np.pi/8., 0.3)
    
#     # print samples

# sample_1D_circular_numba = nub.jit(nub.double[:](nub.int32, nub.double, nub.double[:], nub.int32, nub.double, nub.double))(sample_1D_circular)

if __name__ == '__main__':
    
    if True:
        # test von mises
        loglike_fct_params = [0.0, 4.0]
        
        # Get samples
        samples = sample_1D_circular_numba(5000, 0.1, loglike_fct_params, 50, np.pi/8., 0.3)

    # if True:
        # test full loglike
        # params = (self.theta[n], self.NT[n], self.random_network, self.theta_gamma, self.theta_kappa, self.ATtcB[self.tc[n]], sampled_feature_index, self.mean_fixed_contrib[self.tc[n]], self.inv_covariance_fixed_contrib)
    
    
    
