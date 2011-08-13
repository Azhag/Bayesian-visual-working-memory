#!/usr/bin/env python
# encoding: utf-8
"""
slicesampler.py

Created by Loic Matthey on 2011-08-03.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.special as scsp

class SliceSampler:
    def __init__(self):
        pass
        
    
    
    def sample(self, N, x_initial, loglike_fct, burn=100, widths=1., last_loglikehood=None, loglike_fct_params=None, step_out=True):
        
        '''
            Simple axis-aligned implementation of slice sampling for vectors
            
            Inputs:
                N                   1x1     Number of samples to gather
                x_initial           Dx1     initial state (or array with D elements)
                loglike_fct         @fn     function logprobstar = logdist(x, loglike_fct_params)
                burn                1x1     after burning period of this length
                widths              Dx1 or 1x1, step sizes for slice sampling. Should correspond.
                last_loglikehood    1x1     precomputed last loglikehood
                step_out            bool    set to true if widths may sometimes be far too small
                loglike_fct_params  dict    any extra arguments are passed on to logdist
            
            Outputs:
                samples  NxD   samples
                
                Iain Murray May 2004, tweaks June 2009, a diagnostic added Feb 2010
                See Pseudo-code in David MacKay's text book p375
        '''
        
        
        # Initialisation
        if np.isscalar(x_initial):
            x_initial = np.array(x_initial)
        D = x_initial.size
        all_D_permuted = np.random.permutation(np.arange(D))
        if np.isscalar(widths):
            widths = np.tile(widths, D)
        samples = np.zeros((N, D))
        
        if last_loglikehood is None:
            last_loglikehood = loglike_fct(x_initial, loglike_fct_params)
        
        x_new  = x_initial.copy()
        
        # N samples
        for i in np.arange(N+burn):
            
            log_uprime  = last_loglikehood + np.log(np.random.rand())
            x_l    = x_new.copy()
            x_r    = x_new.copy()
            xprime = x_new.copy()
            
            # Loop over dimensions
            for d in all_D_permuted:
                
                # Create a horizontal interval (x_l, x_r) enclosing x_new
                rr     = np.random.rand()
                x_l[d] -= rr*widths[d]
                x_r[d] += (1.-rr)*widths[d]
                
                # Adapt the interval, if the width could be too small (speedup: remove that)
                if step_out:
                    llh_l = loglike_fct(x_l, loglike_fct_params)
                    while llh_l > log_uprime:
                        x_l[d] = x_l[d] - widths[d]
                        llh_l  = loglike_fct(x_l, loglike_fct_params)
                    llh_r = loglike_fct(x_r, loglike_fct_params)
                    while llh_r > log_uprime:
                        x_r[d] = x_r[d] + widths[d]
                        llh_r  = loglike_fct(x_r, loglike_fct_params)
                
                # Sample a new point, shrinking the interval
                while True:
                    xprime[d] = np.random.rand()*(x_r[d] - x_l[d]) + x_l[d]
                    last_loglikehood = loglike_fct(xprime, loglike_fct_params)
                    if last_loglikehood > log_uprime:
                        # Accept this sample
                        x_new[d] = xprime[d]
                        break
                    elif xprime[d] > x_new[d]:
                        x_r[d] = x_new[d]
                    elif xprime[d] < x_new[d]:
                        x_l[d] = x_new[d]
                    else:
                        raise RuntimeException("Slice sampler shrank too far.")
            
            # Store this sample
            if i > burn:
                samples[i-burn] = x_new
        
        return samples, last_loglikehood
    
    
    def sample_1D(self, N, x_initial, loglike_fct, burn=100, widths=1., last_loglikehood=None, loglike_fct_params=None, step_out = True, bounds=(-np.inf, np.inf), thinning=1, debug=False):
        
        '''
            Simple implementation of slice sampling for 1D variable
            
            Inputs:
                N                   1x1     Number of samples to gather
                x_initial           1x1     initial state
                loglike_fct         @fn     function logprobstar = logdist(x, loglike_fct_params)
                burn                1x1     after burning period of this length
                widths              1x1     step sizes for slice sampling. Should correspond.
                last_loglikehood    1x1     precomputed last loglikehood
                step_out            bool    set to true if widths may sometimes be far too small
                loglike_fct_params  dict    any extra arguments are passed on to logdist
            
            Outputs:
                samples  Nx1   samples
                
                Iain Murray May 2004, tweaks June 2009, a diagnostic added Feb 2010
                See Pseudo-code in David MacKay's text book p375
        '''
        
        
        # Initialisation
        samples = np.zeros(N)
        
        if last_loglikehood is None:
            last_loglikehood = loglike_fct(x_initial, loglike_fct_params)
        
        x_new  = x_initial
        j = 0
        
        # N samples
        for i in np.arange(thinning*N+burn, dtype='int32'):
            log_uprime  = last_loglikehood + np.log(np.random.rand())
            # print log_uprime
            
            # Create a horizontal interval (x_l, x_r) enclosing x_new. Place it randomly.
            rr     = np.random.rand()
            x_l    = x_new -  rr*widths
            x_r    = x_new + (1.-rr)*widths
            
            # Grow the interval to get an unbiased slice
            if step_out:
                llh_l = loglike_fct(x_l, loglike_fct_params)
                while llh_l > log_uprime:
                    x_l     -= widths
                    llh_l   = loglike_fct(x_l, loglike_fct_params)
                    if x_l <= bounds[0]:
                        x_l = bounds[0]
                        break
                llh_r = loglike_fct(x_r, loglike_fct_params)
                while llh_r > log_uprime:
                    x_r     += widths
                    llh_r   = loglike_fct(x_r, loglike_fct_params)
                    if x_r >= bounds[1]:
                        x_r = bounds[1]
                        break
            
            # Sample a new point, shrinking the interval
            while True:
                xprime = np.random.rand()*(x_r - x_l) + x_l
                last_loglikehood = loglike_fct(xprime, loglike_fct_params)
                if last_loglikehood > log_uprime:
                    # Accept this sample
                    x_new = xprime
                    break
                elif xprime > x_new:
                    x_r  = x_new
                elif xprime < x_new:
                    x_l = x_new
                else:
                    raise RuntimeError("Slice sampler shrank too far.")
            
            # Store this sample
            if i >= burn and (i % thinning == 0):
                if debug:
                    print "Sample %d" % (j+1)
                samples[j] = x_new
                j += 1
        
        return samples, last_loglikehood
    
    
    def sample_1D_circular(self, N, x_initial, loglike_fct, burn=100, widths=1., last_loglikehood=None, loglike_fct_params=None, step_out = True, thinning=1, debug=False, loglike_min=-np.inf, jump=True, jump_probability=0.1):
        
        '''
            Simple implementation of slice sampling for 1D variable
            
            Inputs:
                N                   1x1     Number of samples to gather
                x_initial           1x1     initial state
                loglike_fct         @fn     function logprobstar = logdist(x, loglike_fct_params)
                burn                1x1     after burning period of this length
                widths              1x1     step sizes for slice sampling. Should correspond.
                last_loglikehood    1x1     precomputed last loglikehood
                step_out            bool    set to true if widths may sometimes be far too small
                loglike_fct_params  dict    any extra arguments are passed on to logdist
            
            Outputs:
                samples  Nx1   samples
                
                Iain Murray May 2004, tweaks June 2009, a diagnostic added Feb 2010
                See Pseudo-code in David MacKay's text book p375
        '''
        
        
        # Initialisation
        samples = np.zeros(N)
        
        if last_loglikehood is None:
            last_loglikehood = loglike_fct(x_initial, loglike_fct_params)
        
        x_new  = x_initial
        j = 0
        tot_accepted = 0
        tot_rejected = 0
        
        # N samples
        for i in np.arange(thinning*N+burn, dtype='int32'):
            
             # Add a probabilistic jump with Metropolis-Hasting
            if jump and np.random.rand() < jump_probability:
                # print "Jump!"
                xprime = np.random.rand()*2.*np.pi - np.pi
                
                # MH ratio
                llh_x_prime = loglike_fct(xprime, loglike_fct_params)
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
                        llh_l = loglike_fct(x_l, loglike_fct_params)
                        # s = 0
                        while llh_l > log_uprime:
                            # print "stepping out left: [%.3f < %.3f < %.3f] %.3f %.3f" % (x_l, x_new, x_r, log_uprime, llh_l)
                            x_l     -= widths
                            llh_l   = loglike_fct(x_l, loglike_fct_params)
                            if x_l <= -np.pi:
                                x_l = -np.pi
                                break
                        
                            # s+=1
                        # print s
                    
                        llh_r = loglike_fct(x_r, loglike_fct_params)
                        while llh_r > log_uprime:
                            x_r     += widths
                            llh_r   = loglike_fct(x_r, loglike_fct_params)
                            if x_r >= np.pi:
                                x_r = np.pi
                                break
                
                
                # Sample a new point, shrinking the interval
                while True:
                    xprime = np.random.rand()*(x_r - x_l) + x_l
                    
                    last_loglikehood = loglike_fct(xprime, loglike_fct_params)
                    if last_loglikehood > log_uprime:
                        # Accept this sample
                        x_new = xprime
                        break
                    elif xprime > x_new:
                        x_r  = x_new
                    elif xprime < x_new:
                        x_l = x_new
                    else:
                        raise RuntimeError("Slice sampler shrank too far.")
            
            # Store this sample
            if i >= burn and (i % thinning == 0):
                if debug:
                    print "Sample %d: %.3f" % (j+1, x_new)
                samples[j] = x_new
                j += 1
        
        print "%d, %d" % (tot_accepted, tot_rejected)
        
        return samples, last_loglikehood


def test_sample():
    
    loglike_theta_fct = lambda x, (mu, kappa): kappa*np.cos(x - mu) - np.log(2.*np.pi) - np.log(scsp.i0(kappa))
    loglike_fct_params = np.array([0.0, 0.1])
    
    # Get samples
    slicesampler = SliceSampler()
    samples, last_llh = slicesampler.sample_1D_circular(5000, np.random.rand(), loglike_theta_fct, burn=500, widths=np.pi/2., thinning=1, loglike_fct_params=loglike_fct_params, step_out=True, debug=False)
    
    # print samples



if __name__ == '__main__':
    
    if False:
        # Try it with a gaussian
        loglike_fct = lambda x, p=[0.0, 1.0]: (-0.5/p[1]**2.)*((x-p[0])**2.)
        loglike_fct_params = [0.0, 1.0]
        
        # Get samples
        slicesampler = SliceSampler()
        samples, last_llh = slicesampler.sample_1D(50000, np.random.rand(), loglike_fct, burn=100, widths=0.2, loglike_fct_params=loglike_fct_params, step_out=True)
        
        
        # Plot the results
        x = np.linspace(-10., 10., 100)
        n, left_x = np.histogram(samples, bins=x)
        
        # like_out = np.exp(loglike_fct(x, p=loglike_fct_params))
        # like_out /= np.sum(like_out)
        like_out = mlab.normpdf( x, loglike_fct_params[0], loglike_fct_params[1])
        
        plt.figure()
        plt.bar(x[:-1], n/np.sum(n.astype(float)), facecolor='green', alpha=0.75, width=np.diff(x)[0])
        plt.plot(x, like_out/5, 'r')
        
        print "Original parameters: %f, %f" % (loglike_fct_params[0], loglike_fct_params[1])
        print "Fitted parameters: %f, %f" % (np.mean(samples), np.cov(samples))
    
    if True:
        # Try with Von Mises
        loglike_theta_fct = lambda x, (mu, kappa): kappa*np.cos(x - mu) - np.log(2.*np.pi) - np.log(scsp.i0(kappa))
        loglike_fct_params = [0.0, 0.1]
        
        # Get samples
        slicesampler = SliceSampler()
        samples2, last_llh = slicesampler.sample_1D_circular(50000, np.random.rand(), loglike_theta_fct, burn=500, widths=0.01, loglike_fct_params=loglike_fct_params, step_out=True, debug=True, loglike_min = -np.log((2./2.0)*np.pi*scsp.i0(loglike_fct_params[1])))
        # samples2, last_llh = slicesampler.sample_1D_circular(50000, np.random.rand(), loglike_theta_fct, burn=500, widths=0.01, loglike_fct_params=loglike_fct_params, step_out=True, debug=True)
        
        # Plot
        x = np.linspace(-4., 4., 100)
        n, left_x = np.histogram(samples2, bins=x)
        
        like_out = np.exp(loglike_theta_fct(x, loglike_fct_params))
        like_out /= np.abs(np.max(like_out))
        # like_out -= np.mean(like_out)
        
        plt.figure()
        plt.bar(x[:-1], n/np.max(n.astype(float)), facecolor='green', alpha=0.75, width=np.diff(x)[0])
        plt.plot(x, like_out, 'r')
    
    
    
    
