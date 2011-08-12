# encoding: utf-8

"""
CYTHON VERSION

slicesampler.pyx

Created by Loic Matthey on 2011-08-03.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import numpy as np
cimport numpy as np

from libc.math cimport cos, log, M_PI, exp

cimport cython

# Set DTYPE
DFLOAT = np.float
DINT = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float64_t DFLOAT_t
ctypedef np.int_t DINT_t

@cython.boundscheck(False)
cpdef np.ndarray sample_1D_circular(DINT_t N, DFLOAT_t x_initial, DINT_t burn, DFLOAT_t widths, list loglike_fct_params, DINT_t step_out, DINT_t thinning, DINT_t debug):
        
    '''
        Simple implementation of slice sampling for 1D variable
        
        Inputs:
            N                   1x1     Number of samples to gather
            x_initial           1x1     initial state
            burn                1x1     after burning period of this length
            widths              1x1     step sizes for slice sampling. Should correspond.
            step_out            bool    set to true if widths may sometimes be far too small
            loglike_fct_params  dict    any extra arguments are passed on to logdist
        
        Outputs:
            samples  Nx1   samples
            
            Iain Murray May 2004, tweaks June 2009, a diagnostic added Feb 2010
            See Pseudo-code in David MacKay's text book p375
    '''
    
    
    # Initialisation
    cdef np.ndarray[DFLOAT_t, ndim=1] samples = np.zeros(N, dtype=DFLOAT)
    cdef DINT_t j = 0
    cdef DINT_t i
    cdef DINT_t all_n = thinning*N+burn
    cdef DFLOAT_t xprime
    cdef DFLOAT_t llh_l, llh_r, llh_x_prime
    cdef DFLOAT_t rr
    cdef DFLOAT_t x_l, x_r
    cdef DFLOAT_t log_uprime
    cdef DINT_t tot_accepted = 0
    cdef DINT_t tot_rejected = 0
    
    cdef DFLOAT_t last_loglikehood = loglike_collapsed(x_initial, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4],
    loglike_fct_params[5], loglike_fct_params[6], loglike_fct_params[7])
    
    cdef DFLOAT_t x_new  = x_initial
    
    # N samples
    for i in range(all_n):
        
        # Add a probabilistic jump with Metropolis-Hasting
        if np.random.rand() < 0.001:
            # print "Jump!"
            xprime = np.random.rand()*2.*M_PI - M_PI
            
            # MH ratio
            llh_x_prime = loglike_collapsed(xprime, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4],
    loglike_fct_params[5], loglike_fct_params[6], loglike_fct_params[7])
            if log(np.random.rand()) <  llh_x_prime - last_loglikehood:
                # Accepted!
                x_new = xprime
                last_loglikehood = llh_x_prime
                tot_accepted += 1
            else:
                # rejected, keep x_new
                tot_rejected += 1
        else:
            # Normal slice sampling
            log_uprime  = last_loglikehood + log(np.random.rand())
            # print "log_uprime: %.3f, lastllh: %.3f" % (log_uprime, last_loglikehood)
            
            # Create a horizontal interval (x_l, x_r) enclosing x_new. Place it randomly.
            rr     = np.random.rand()
            x_l    = x_new -  rr*widths
            x_r    = x_new + (1.-rr)*widths
            
            # Grow the interval to get an unbiased slice
            if step_out==1:
                llh_l = loglike_collapsed(x_l, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4],
    loglike_fct_params[5], loglike_fct_params[6], loglike_fct_params[7])
                # s = 0
                while llh_l > log_uprime:
                    # print "stepping out left: [%.3f < %.3f < %.3f] %.3f %.3f" % (x_l, x_new, x_r, log_uprime, llh_l)
                    x_l     -= widths
                    llh_l   = loglike_collapsed(x_l, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4],
    loglike_fct_params[5], loglike_fct_params[6], loglike_fct_params[7])
                    if x_l <= -M_PI:
                        x_l = -M_PI
                        break
                    
                    # s+=1
                # print s
                
                llh_r = loglike_collapsed(x_r, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4],
    loglike_fct_params[5], loglike_fct_params[6], loglike_fct_params[7])
                while llh_r > log_uprime:
                    x_r     += widths
                    llh_r   = loglike_collapsed(x_r, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4],
    loglike_fct_params[5], loglike_fct_params[6], loglike_fct_params[7])
                    if x_r >= M_PI:
                        x_r = M_PI
                        break
                
            
            # Sample a new point, shrinking the interval
            while True:
                xprime = np.random.rand()*(x_r - x_l) + x_l
                
                last_loglikehood = loglike_collapsed(xprime, loglike_fct_params[0], loglike_fct_params[1], loglike_fct_params[2], loglike_fct_params[3], loglike_fct_params[4],
    loglike_fct_params[5], loglike_fct_params[6], loglike_fct_params[7])
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
    
    # print tot_accepted/float(tot_accepted+tot_rejected)
    
    return samples

@cython.boundscheck(False)
cpdef DFLOAT_t loglike_collapsed(DFLOAT_t x, np.ndarray[DFLOAT_t, ndim=1] datapoint, object rn, DINT_t sampled_feature_index, DFLOAT_t theta_mu, DFLOAT_t theta_kappa, DFLOAT_t ATtcB, np.ndarray[DFLOAT_t, ndim=1] mean_fixed_contrib, np.ndarray[DFLOAT_t, ndim=2] covariance_fixed_contrib):
    '''
        Logposterior of the fullcollapsed model
    '''
    
    cdef np.ndarray[DFLOAT_t, ndim=1] like_mean
    cdef np.ndarray[DFLOAT_t, ndim=1] tmp_mean
    cdef DFLOAT_t output
    
    like_mean = datapoint - mean_fixed_contrib - ATtcB*np.dot(rn.popcodes[sampled_feature_index].mean_response(x), rn.W[sampled_feature_index].T)
    
    output = theta_kappa*cos(x - theta_mu)
    tmp_mean = np.linalg.solve(covariance_fixed_contrib, like_mean)
    output -= 0.5*np.dot(like_mean, tmp_mean)
    
    return output


cdef DFLOAT_t loglike_vonmises(DFLOAT_t x, DFLOAT_t mu, DFLOAT_t kappa):
    return kappa*cos(x - mu)
    # return log(exp(-40.*(x-3.)*(x-3.)) + exp(-30.*(x+3.)*(x+3)))


def test_sample(N):
    
    loglike_fct_params = np.array([0.0, 0.1])
    #sample_1D_circular(int N, DFLOAT_t x_initial, int burn, DFLOAT_t widths, np.ndarray loglike_fct_params, int step_out, int thinning, int debug):
    samples = sample_1D_circular(N, np.random.rand(), 500, 3.0, loglike_fct_params, 1, 3, 0)
    
    return samples


if __name__ == '__main__':
    
    
    if True:
        # Try with Von Mises
        loglike_fct_params = np.array([0.0, 0.1])
        
        # Get samples
        # samples2, last_llh = sample_1D_circular(50000, np.random.rand(), burn=500, widths=0.01, loglike_fct_params=loglike_fct_params, step_out=True, debug=True, loglike_min = -np.log((2./2.0)*np.pi*scsp.i0(loglike_fct_params[1])))
        # samples2, last_llh = slicesampler.sample_1D_circular(50000, np.random.rand(), loglike_theta_fct, burn=500, widths=0.01, loglike_fct_params=loglike_fct_params, step_out=True, debug=True)
        
        # Plot
        # x = np.linspace(-4., 4., 100)
        #         n, left_x = np.histogram(samples2, bins=x)
        #         
        #         like_out = np.exp(loglike_theta_fct(x, loglike_fct_params))
        #         like_out /= np.abs(np.max(like_out))
        #         # like_out -= np.mean(like_out)
        #         
        #         plt.figure()
        #         plt.bar(x[:-1], n/np.max(n.astype(float)), facecolor='green', alpha=0.75, width=np.diff(x)[0])
        #         plt.plot(x, like_out, 'r')
    
    
    
    
